import type { Diagnostic, ReactiveHost, UserFn, CellRef } from '@metael/lang';
import { descriptorOf, generationOf, makeDiagnostic, makeTypedArray, markFrozen } from '@metael/lang';
import { gateKernel, checkMultiOutputShape, synthOutputKernel } from './gate.ts';
import { gateReducer, cpuReduce } from './reduce.ts';
import { gateBinMapper, cpuHistogram } from './histogram.ts';
import { checkStaticBounds } from './bounds.ts';
import { buildBindingTable, collectFreeNames, type BindingTable } from './binding.ts';
import { checkCost, type DeviceLimits } from './cost.ts';
import { emitCpu } from './emit-cpu.ts';
import { emitWgsl, emitReduceWgsl, emitHistogramWgsl, HISTOGRAM_WORKGROUP } from './emit-wgsl.ts';
import { emitGlsl, emitReduceGlsl, REDUCE_TILE } from './emit-glsl.ts';
import { checkMatch, checkReduceMatch, type MatchVerdict } from './oracle.ts';
import { kernelHash, bufferFingerprint } from './hash.ts';
import { makeCpuBackend } from './device/cpu.ts';
import type { Backend, BackendKind, DispatchInput } from './device/index.ts';
import { selectBackend } from './device/index.ts';
import { makeGpuBufferHandle, residentInfo, disposeHandle } from './handle.ts';
import { compsOf, type OutputElement } from './output.ts';

export interface GpuConfig {
  readonly output: readonly number[];
  readonly precision?: 'f16' | 'f32';
  readonly backend?: 'auto' | BackendKind;
  /** Opt-in: after the GPU dispatch, re-run a SAMPLE of output cells through the interpreter and
   *  tolerance-check the GPU output against it (populates `match`). Off by default — a CPU-side interpreter
   *  sweep on every dispatch defeats the point of offloading to the GPU; enable it only when you want the
   *  correctness proof (e.g. a demo, or while validating a new kernel). */
  readonly verify?: boolean;
  /** Opt-in: also run the whole sweep on the CPU backend to time a baseline (populates `cpuMs`, and
   *  `speedup` when the primary backend is a real GPU). Off by default — a second full dispatch per run
   *  doubles the work and defeats the speedup it measures; enable it only for a GPU-vs-CPU race demo. */
  readonly benchmark?: boolean;
  /** The shape of `resource.value` when the dispatch settles. `'array'` (default) → a plain `number[]`
   *  (freeze-safe, backward-compatible, ergonomic for the collection builtins). `'buffer'` → a frozen `f32`
   *  custom value wrapping the readback Float32Array with NO copy — freeze-safe via the same `markFrozen` a
   *  const typed array uses, and re-usable as a kernel input. `'gpu-buffer'` → a `GpuBufferHandle`: a
   *  resident-buffer custom value that keeps the output ON-DEVICE and re-uses it as a kernel input for
   *  pipelining (the next stage binds it directly on the same backend; a foreign/CPU reader lazily reads it
   *  back). Like `'buffer'` it is a typed-array-like custom value, so the collection builtins + emit read it. */
  readonly outputType?: 'array' | 'buffer' | 'gpu-buffer';
  /** The per-cell output element. `'f32'` (default) → ONE value per cell (`output[c]`, back-compat). A
   *  `'vec2'`/`'vec3'`/`'vec4'` → N values per cell in a FLAT-INTERLEAVED buffer (cell `c` component `k` at
   *  `output[c * N + k]`), and requires the kernel's every `return` to yield a vecN of that width (else the
   *  gate rejects with MLGPU-OUTPUT-SHAPE). All three backends produce this identical normalized layout.
   *  MUTUALLY EXCLUSIVE with `outputs` (a scalar single-output run vs a named multi-output run). */
  readonly outputElement?: OutputElement;
  /** MULTI-OUTPUT: a named set of output buffers. When set, the kernel must return an OBJECT LITERAL whose
   *  keys exactly match these names and whose values are lowerable scalars/vecs (each shaped by its
   *  `element`, default `'f32'`). The `output` dims still set the grid; each named output is written to its
   *  own buffer, settled into `resource.outputs[<name>]` (a `number[]`) with `resource.value = null`. v1
   *  supports ARRAY-mode only (`outputType` other than the default is rejected for multi-output). MUTUALLY
   *  EXCLUSIVE with `outputElement`. Implemented as N single-output dispatches (one per key) over the proven
   *  single-output path — a named-object return is equivalent to N kernels, one per named output. */
  readonly outputs?: Record<string, { element?: OutputElement }>;
}
/** A REDUCTION config: a 2-arg ASSOCIATIVE + COMMUTATIVE reducer folds an input buffer to a scalar, seeded by
 *  `identity`. A DISTINCT kernel kind from the map `gpu()` (whose kernel's params are thread coords), so it
 *  flows through the dedicated `gpuReduce` head + `gateReducer`. The reducer must be COMMUTATIVE (not merely
 *  associative): the WGSL workgroup tree fold reorders operands across lanes (`op(op(s0,s2), op(s1,s3))`), so a
 *  non-commutative associative reducer diverges across backends — `verify: true` catches it (see `cpuReduce`).
 *  The GPU reduction legs ship: a WebGL2 ping-pong multi-pass + a WGSL workgroup-shared tree, with the CPU
 *  linear fold as the oracle floor. Only the SCALAR fold ships — `scan:true` (a prefix-scan buffer output) is
 *  a follow-on and is REJECTED (`MLGPU-NOT-LOWERABLE`), never silently folded to the scalar. */
export interface ReduceConfig {
  readonly input: unknown;   // a typed-array custom value (an f32 buffer) — the values to fold
  /** The fold seed. It MUST be the reducer's NEUTRAL element — the value `e` for which `reduce(e, x) === x`
   *  for every `x` (0 for a sum, 1 for a product, a very-negative sentinel like -1e30 / -Infinity for max, a
   *  very-large one like 1e30 / +Infinity for min). This is a CALLER CONTRACT, not enforced: neutrality is not
   *  statically checkable in general. It matters because the CPU oracle (`cpuReduce`) is a LINEAR fold that
   *  applies `identity` EXACTLY ONCE (seed, then fold each element), whereas the GPU tree reduction re-seeds
   *  the identity into EVERY tile on EVERY pass — folding it a backend-dependent number of times. For a neutral
   *  identity that is harmless (`e ∘ e ∘ … ∘ x === x`), so CPU and GPU agree; a NON-neutral identity is folded
   *  a different number of times on each backend, so the results DIVERGE (e.g. sum with identity 5 → the CPU
   *  adds 5 once, the GPU adds it once per tile per pass). `verify: true` catches such a divergence
   *  (`match.ok === false`); with verify off the wrong answer is silent — hence the contract. */
  readonly identity: number;
  readonly scan?: boolean;   // UNBUILT: a prefix-scan (a buffer output) is a follow-on; `scan:true` is REJECTED (MLGPU-NOT-LOWERABLE) — only the scalar fold ships
  readonly backend?: 'auto' | BackendKind;
  readonly verify?: boolean;
  readonly benchmark?: boolean;
}
/** A HISTOGRAM config: a 1-arg bin-mapper maps each input element to a BIN INDEX; the result is a per-bin
 *  COUNT array of length `bins`. A DISTINCT kernel kind from the map `gpu()` (thread coords) and the reduce
 *  (fold to a scalar) — a DATA-DEPENDENT ATOMIC SCATTER, so it flows through the dedicated `gpuHistogram` head
 *  + `gateBinMapper`. Per-backend at run time: cpu → the exact `cpuHistogram` oracle; webgpu →
 *  `atomicAdd(&bins[b], 1)`; webgl2 → cpuHistogram (WebGL2 has NO fragment-shader atomics, so a histogram
 *  falls to the CPU oracle — settled `backend: 'cpu'` + a note). An out-of-range bin index (< 0 or >= bins) is
 *  DROPPED (not counted), the standard histogram bounds behavior; CPU + WGSL agree via a bounds guard. */
export interface HistogramConfig {
  readonly input: unknown;   // a typed-array custom value (an f32 buffer) — the values to bin
  readonly bins: number;     // the number of buckets — the result `value` is a number[] of this length (the counts)
  readonly backend?: 'auto' | BackendKind;
  readonly verify?: boolean;
  readonly benchmark?: boolean;
}
export interface GpuResource {
  core: boolean; reasons: Diagnostic[]; wgsl: string; glsl: string; backend: BackendKind;
  /** The settled result. A map `gpu()` sets a `number[]` (default) or a typed-array/handle `object`; a
   *  multi-output run leaves it `null` (see `outputs`); a scalar `gpuReduce` sets a bare `number`. Widened to
   *  admit that scalar — additive + back-compat (every prior consumer read an array/object/null). */
  pending: boolean; value: number[] | object | number | null; error: Diagnostic | null;
  /** MULTI-OUTPUT result: `{ <name>: number[] }` when `cfg.outputs` was used (with `value === null` — a
   *  multi-output run has no single primary value); `null` for a single-output run (back-compat — then
   *  `value` carries the single output as before). */
  outputs: Record<string, number[] | object> | null;
  gpuMs: number | null; cpuMs: number | null; speedup: number | null; match: MatchVerdict | null;
  /** A human-readable notice about the settled run, or `null`. Today it carries a PRECISION-FALLBACK notice:
   *  an `f16` request that ran at `f32` (because the acquired backend lacked shader-f16, or the kernel has a
   *  scalar uniform whose f16 uniform-packing path isn't supported yet) — the values are correct f32, and the
   *  note explains the downgrade so a playground can surface it. `null` for every f32 run (back-compat). */
  note: string | null;
}

const MAX_LIVE = 8;

/** The RANK GATE shared by the single-output `gpu()` and the multi-output `gpuMulti()` paths: a kernel's
 *  params ARE its thread coordinates, so the arity must match the output's dimension count, and at most 3
 *  thread dimensions (x, y, z) are dispatchable. Returns the reasons to reject LOUDLY (a rank>3 kernel, or an
 *  arity≠dims mismatch) rather than let one fall through silently-wrong. Both paths fold these into `core`.
 *  Identical logic in both — deduped here so the two paths cannot drift in codes or wording. */
function rankGate(kernel: UserFn, dims: readonly number[]): Diagnostic[] {
  const rank = kernel.params.length;
  const dimsRank = dims.length;
  const reasons: Diagnostic[] = [];
  if (rank > 3) reasons.push(makeDiagnostic('MLGPU-NOT-LOWERABLE', `a kernel of rank ${rank} is not lowerable — at most 3 thread dimensions (x, y, z) are supported`));
  else if (rank !== dimsRank) reasons.push(makeDiagnostic('MLGPU-OUTPUT-SHAPE', `kernel arity (${rank}) must match the output dimension count (${dimsRank})`));
  return reasons;
}

export interface GpuEngineDeps {
  tryWebGpu: () => Promise<Backend | null>;
  tryWebGl2: () => Backend | null;
  limitsHint: DeviceLimits;
}

export class GpuEngine {
  private readonly host: ReactiveHost;
  private readonly deps: GpuEngineDeps;
  private readonly memo = new Map<string, { resource: GpuResource; cell: CellRef }>();
  // One backend instance per `requested` config, acquired once and reused across every dispatch — so a
  // producer's resident output buffer can be bound directly by a later consumer on the SAME instance
  // (residency), and an N-stage pipeline uses ONE device, not N. The engine owns these; they're disposed
  // only at teardown (dispose()), never per-dispatch or per-eviction.
  private readonly backendPool = new Map<string, Promise<Backend>>();
  // Per-object identity for engine-minted 'buffer'-mode outputs. A frozen 'buffer' output carries a
  // generation that reads 0 for every fresh buffer, so keying the memo off generationOf would let two
  // distinct same-length outputs collide (a consumer returns stale data, never re-dispatches). This map
  // gives each minted output a distinct id the memo keys on. Only engine outputs are in it — a user's own
  // f32 input stays on the generation path (back-compat), a gpu-buffer stays on the nonce path.
  private readonly outputIdentity = new WeakMap<object, number>();
  private outputNonce = 0;
  private readonly queue: (() => Promise<void>)[] = [];
  private draining = false;
  private disposed = false;

  constructor(host: ReactiveHost, deps: GpuEngineDeps) { this.host = host; this.deps = deps; }

  gpu(kernel: UserFn, cfg: GpuConfig): GpuResource {
    const precision = cfg.precision ?? 'f32';
    const requested: 'auto' | BackendKind = cfg.backend ?? 'auto';
    if (cfg.outputs !== undefined) return this.gpuMulti(kernel, cfg, precision, requested);
    const comps = compsOf(cfg.outputElement);   // f32→1 (default), vec2/3/4→2/3/4
    // RANK GATE (before the kernel gate): a kernel's params ARE its thread coordinates, so the arity must
    // match the output's dimension count, and at most 3 thread dimensions (x, y, z) are dispatchable. Reject
    // LOUDLY here rather than let a rank>3 kernel or an arity≠dims mismatch fall through silently-wrong. The
    // SAME gate is applied to the multi-output path (gpuMulti) via this shared helper so the two can't drift.
    const rankReasons = rankGate(kernel, cfg.output);
    const { reasons: gateReasons, bindings } = gateKernel(kernel, this.host, comps);
    // A CONSERVATIVE static bounds proof over the kernel's buffer-index expressions, bounded by the output
    // dims (which the dims-agnostic gate never sees). It rejects (MLGPU-INDEX-STATIC) ONLY an index provably
    // out of range for EVERY coord; a data-dependent / partially-in-range index is left to the sampled oracle.
    // Run as a separate pass so the gate's signature stays dims-agnostic + this stays isolated.
    checkStaticBounds(kernel, bindings, cfg.output, gateReasons);
    const { gens, inputBytes } = this.computeGens(bindings);
    const outT = cfg.outputType ?? 'array';
    // Config rejection: a vecN output (comps>1) is INCOMPATIBLE with the RESIDENT 'gpu-buffer' output type.
    // A resident output is a cell-indexed RGBA32F texture on WebGL2 (one texel per cell, the N components
    // packed into R/G/B/A), but a downstream consumer reads a FLAT-INTERLEAVED buffer via `_fetch`'s R channel
    // at flat index `i*comps+k` — so a same-instance resident vecN buffer would be silently misread cross-stage
    // (a metadata patch can't fix it: the resident layout is fundamentally cell-indexed-RGBA). 'array' (a flat
    // number[]) and 'buffer' (a flat NON-resident f32 handle, never a resident texture) are both coherent with
    // a vecN result; only 'gpu-buffer' (the resident path) is rejected — mirroring the multi-output deferral of
    // a non-array outputType. Reject BEFORE dispatch so no incoherent resident handle is ever minted.
    const cfgReasons: Diagnostic[] = [];
    if (comps > 1 && outT === 'gpu-buffer') {
      cfgReasons.push(makeDiagnostic('MLGPU-NOT-LOWERABLE', "a vecN outputElement is not supported with outputType 'gpu-buffer' (a resident vecN buffer is a follow-on) — use outputType 'array' or 'buffer'"));
    }
    const reasons = [...gateReasons, ...cfgReasons, ...rankReasons];
    // core iff nothing was flagged — by the gate OR the static-bounds pass (which pushed into gateReasons) OR
    // the config check OR the rank gate. (The gate's own `core` is subsumed by `gateReasons.length === 0`.)
    const core = gateReasons.length === 0 && cfgReasons.length === 0 && rankReasons.length === 0;
    // Fold the output width into the flags so a vecN run is a DISTINCT resource from a scalar run of the same
    // kernel (append `cN` for comps>1; a scalar comps=1 stays blank → the pre-vecN key is byte-identical).
    const flags = `${cfg.verify ? 'v' : ''}${cfg.benchmark ? 'b' : ''}${outT === 'buffer' ? 'B' : outT === 'gpu-buffer' ? 'G' : ''}${comps > 1 ? `c${comps}` : ''}`;   // distinct flags → distinct resource
    const key = `${kernelHash(kernel, bindings)}::${JSON.stringify(cfg.output)}::${precision}::${requested}::${flags}::${gens.join(',')}`;
    const hit = this.memo.get(key);
    if (hit) return this.host.readCell(hit.cell) as GpuResource;

    const wgsl = core ? emitWgsl(kernel, bindings, precision, comps) : '';
    const glsl = core ? emitGlsl(kernel, bindings, precision, comps) : '';
    const outputBytes = cfg.output.reduce((a, b) => a * b, 1) * comps * 4;   // N values per cell for a vecN output
    const costErr = core ? checkCost(outputBytes, inputBytes, cfg.output, this.deps.limitsHint) : null;

    const resource: GpuResource = {
      core, reasons, wgsl, glsl, backend: 'cpu', pending: core && !costErr,
      value: null, outputs: null,   // single-output: outputs is always null (value carries the one output)
      error: costErr ?? (core ? null : makeDiagnostic('MLGPU-NOT-LOWERABLE', 'kernel is not GPU-lowerable')),
      gpuMs: null, cpuMs: null, speedup: null, match: null, note: null,
    };
    if (!core || costErr) { resource.pending = false; }
    const cell = this.host.allocateCell(resource);
    this.memo.set(key, { resource, cell });
    // A synchronous input-read failure (reading a resident-handle input whose producer was already freed —
    // its cache never materialized) must NOT throw out of gpu(): gpu() runs inside the reader's derive
    // (interpreter resolveCall), which is not wrapped in a BufferError catch, so a throw escapes to the
    // top-level catch → ML-LANG-INTERNAL → the WHOLE component tree is lost (sibling nodes too) AND this
    // memo entry is left permanently pending (its dispatch never enqueued → a spinner forever on re-derive).
    // Instead: settle THIS resource with a local error and skip the dispatch, keeping the failure local to
    // this gpu node so the rest of the tree derives normally. `resource` IS the object the memo + cell hold,
    // so mutating it in place is what a later cell reader (and the `subscribed` we return) observes — the
    // same in-place-settle the non-pending (`!core || costErr`) path above uses; no writeCell needed.
    const settleInputError = (e: unknown): void => {
      resource.pending = false;
      resource.error = makeDiagnostic('MLGPU-INPUT-UNAVAILABLE', String((e as Error)?.message ?? e));
      this.memo.set(key, { resource, cell });
    };
    // Before LRU eviction can dispose a producer handle THIS dispatch is about to read, materialize that
    // handle's CPU cache — so eviction frees only the producer's GPU buffer, never strands the values a
    // consumer needs. Cost-neutral: the input-resolution loop below reads the same bufferView (now a cache
    // hit). A no-op for a non-handle buffer input. (A producer evicted+disposed by an UNRELATED prior
    // dispatch — cache still null — throws MLGPU-USE-AFTER-DISPOSE here; the catch settles it as a local
    // MLGPU-INPUT-UNAVAILABLE instead of letting it collapse the tree.)
    if (resource.pending) {
      try { this.materializeResidentCaches(bindings); } catch (e) { settleInputError(e); }
    }
    this.evictLru();
    // Read through the cell so a reader (a mount walk-effect) SUBSCRIBES to it — the settle writeCell then
    // re-runs that reader. Returning the raw `resource` would leave the reader unsubscribed (no re-render).
    const subscribed = this.host.readCell(cell) as GpuResource;

    if (resource.pending) {
     try {
      const cpuRun = emitCpu(kernel, bindings, this.host, comps);
      const { inputs, residentInputs, scalars: scalarUniforms } = this.resolveInputs(bindings);
      this.enqueue(async () => {
        if (this.disposed) return;   // engine torn down before this task ran → don't acquire a device
        // BORROW the pooled backend (acquired once per `requested`, reused across dispatches) — never dispose
        // it here. The pool owns it; dispose() frees it exactly once at teardown. Reusing the same instance is
        // what lets a producer's resident buffer be bound by a later consumer (residency), and keeps an N-stage
        // pipeline on ONE device. Safe because the queue drains SERIALLY (scheduleDrain awaits each task), so a
        // pooled backend is never used concurrently.
        let backend = await this.acquireBackend(requested);
        // dispose() may have run DURING the await above (device acquisition is async): if so, bail instead of
        // writing into the cleared memo. dispose() itself frees the pooled backend once acquisition resolves.
        if (this.disposed) return;
        // PRECISION FALLBACK (computed ONCE from the FIRST-acquired backend, before the dispatch loop). An
        // f16 request runs at f16 ONLY when the backend advertises shader-f16 AND the kernel has no scalar
        // uniform (the f16 uniform-packing path isn't supported yet — see below); otherwise it cleanly
        // downgrades to f32 with a note. The interpreter (cpuRun) is precision-agnostic (it computes in f64).
        // A dispatch-throw re-ladder to a LOWER rung uses effGlsl (which mirrors the effective precision —
        // mediump for f16, highp for f32; WebGL2 uses R32F textures regardless, so no packing mismatch) or
        // the precision-agnostic cpuRun; the f16 WGSL is only ever consumed by a webgpu dispatch. effPrecision
        // is fixed here, never recomputed per rung, so the fallback composes cleanly with the re-ladder.
        const { effPrecision, note } = this.effectivePrecision(precision, backend, scalarUniforms.length > 0);
        const effWgsl = effPrecision === precision ? wgsl : emitWgsl(kernel, bindings, effPrecision, comps);
        const effGlsl = effPrecision === precision ? glsl : emitGlsl(kernel, bindings, effPrecision, comps);
        const dims = cfg.output;
        let next: GpuResource;
        try {
          const dispatchInput: DispatchInput = { kernel, bindings, dims, precision: effPrecision, wgsl: effWgsl, glsl: effGlsl, cpuRun, outputComps: comps, inputs, scalars: scalarUniforms,
            retainOutput: outT === 'gpu-buffer', residentInputs };
          // Try the acquired backend; on a RUNTIME dispatch throw, re-ladder DOWN (webgpu→webgl2→cpu) and retry
          // ONCE per rung, recording the ACTUAL backend that succeeds. cpu is the true floor at DISPATCH time (a
          // device can fault mid-run, not just at acquisition), so a GPU dispatch failure degrades to the next
          // rung instead of a terminal error. The helper re-checks `disposed` after every await → a null return
          // means dispose landed mid-re-ladder: bail without settling (the dispose-race guard). If every rung
          // threw (cpu is the floor → a cpu throw is terminal), it throws the LAST error → the catch below
          // settles the terminal MLGPU-DISPATCH exactly as before.
          const settled = await this.dispatchReladder(backend, (b) => b.dispatch(dispatchInput));
          if (settled === null) return;   // disposed mid-re-ladder → bail without writing a stale cell
          backend = settled.backend;      // the rung that SUCCEEDED — benchmark/verify/value-wrap/memo use it
          const gpuRes = settled.result;
          const onCpu = backend.kind === 'cpu';
          // BENCHMARK (opt-in): time a CPU baseline. On the CPU floor the dispatch already IS the CPU run,
          // so reuse it (no second sweep); on a real GPU run a fresh CPU dispatch to get cpuMs + speedup.
          // Off by default — a second full sweep per dispatch doubles the work and defeats the speedup.
          let cpuMs: number | null = null;
          let speedup: number | null = null;
          if (cfg.benchmark) {
            const cpuRes = onCpu ? gpuRes : await makeCpuBackend().dispatch(dispatchInput);
            cpuMs = cpuRes.ms;
            // Guard the divide: a coarse/zero GPU timer (gpuRes.ms === 0) would otherwise yield Infinity/NaN.
            speedup = onCpu || gpuRes.ms === 0 ? null : cpuRes.ms / gpuRes.ms;
          }
          // dispose() (or an LRU eviction of THIS key) may have landed during the dispatch/benchmark awaits
          // above. If the engine is gone, or this key is no longer the live memo entry (evicted while
          // pending, or superseded by a newer dispatch), bail WITHOUT writing a stale cell. Never dispose the
          // backend here — it's pooled (engine-owned). Checked after every await, before any mutation.
          if (this.disposed || this.memo.get(key)?.cell !== cell) return;
          // VERIFY (opt-in): re-check a sample of cells against the interpreter oracle. Off by default — a
          // CPU-side interpreter sweep on every GPU dispatch is exactly the cost the GPU exists to avoid.
          const match = cfg.verify
            ? checkMatch({ fn: kernel, host: this.host, bindings, output: gpuRes.output, dims, precision: effPrecision, sampleCount: 256, comps })
            : null;
          const value =
            outT === 'gpu-buffer' && gpuRes.resident
              ? makeGpuBufferHandle({
                  backendKind: backend.kind,
                  length: dims.reduce((a, b) => a * b, 1) * comps,   // the flat interleaved readback is total*comps long
                  gpuBuffer: gpuRes.resident.gpuBuffer,
                  readback: () => gpuRes.output,   // the CPU-side readback (always present) — the correct-values fallback
                  dispose: gpuRes.resident.dispose,
                })
              : outT === 'buffer'
              // The f32 handle wraps the N-wide Float32Array (length total*comps). A future vecN-TYPED handle
              // (a typed array whose element is a vecN) is a further seam; today it is a flat f32 view.
              ? (() => { const h = makeTypedArray('f32', gpuRes.output, this.host.allocateGeneration()); markFrozen(h); this.outputIdentity.set(h, ++this.outputNonce); return h; })()
              // gpu-buffer requested but no resident handle returned (should not happen — every backend supports
              // retainOutput): degrade to a plain array (safe; no resident buffer to leak).
              : Array.from(gpuRes.output);
          next = { ...resource, backend: backend.kind, pending: false, value,
            // Show the shaders that ACTUALLY ran (the effective precision) + the note explaining any downgrade.
            wgsl: effWgsl, glsl: effGlsl, note,
            gpuMs: onCpu ? null : gpuRes.ms, cpuMs, speedup, match };
          this.memo.set(key, { resource: next, cell });
        } catch (e) {
          if (this.disposed) return;
          // Null-safe error formatting (mirrors settleInputError): a terminal throw of `null`/`undefined`
          // (a non-Error) must not itself throw a TypeError reading `.message` and escape as an unhandled
          // rejection — use optional chaining so it degrades to String(e).
          next = { ...resource, pending: false, error: makeDiagnostic('MLGPU-DISPATCH', String((e as Error)?.message ?? e)) };
          if (this.memo.get(key)?.cell === cell) this.memo.set(key, { resource: next, cell });
          else return;   // key evicted mid-flight → don't re-insert a stale entry
        }
        this.host.writeCell(cell, next);
      });
     } catch (e) {
       // A synchronous input read failed (e.g. a resident-handle producer freed before the consumer read
       // it). Settle a local error + skip the dispatch instead of throwing out of gpu() and collapsing the
       // reader's whole tree / stranding this entry pending. `subscribed` reads the same in-place resource.
       settleInputError(e);
     }
    }
    return subscribed;
  }

  // ─── The REDUCTION kernel kind: a 2-arg associative + commutative reducer folds an input buffer to a scalar ───
  // A DISTINCT kernel kind from the map `gpu()` (whose kernel's params are thread coords) — so it flows
  // through the dedicated `gpuReduce` head + `gateReducer` (arity + a scalar-lowerable, PURE-over-(acc,x)
  // body), never the map-kernel gate. It MIRRORS `gpu`'s reactive-resource shape (memo → allocateCell →
  // subscribe → enqueue an async task → settle), keying the memo off the reducer hash + the input buffer's
  // generation/nonce + the identity/scan/backend/flags so a distinct input (a generation bump or a fresh
  // handle) re-dispatches.
  //
  // v1 SCOPE (the oracle floor + the head/gate/engine scaffold): the fold runs on the CPU — the EXACT linear
  // left-fold `cpuReduce` that is the correctness oracle — regardless of the requested `backend`, and settles
  // `backend: 'cpu'`. The GPU reduction legs (a WebGL2 ping-pong, a WGSL workgroup-shared tree) are follow-on
  // work: they acquire a real device (via the same acquireBackend/dispatchReladder ladder the map path uses)
  // and verify their reordered tree fold against THIS linear fold. `scan` (a prefix-scan buffer output) is a
  // further follow-on; `scan:true` is rejected (MLGPU-NOT-LOWERABLE) — only the scalar reduce path ships here.
  gpuReduce(reducer: UserFn, cfg: ReduceConfig): GpuResource {
    const requested: 'auto' | BackendKind = cfg.backend ?? 'auto';
    const { reasons: gateReasons, bindings } = gateReducer(reducer, this.host);
    // Config validation: the input must be a foldable buffer (a typed-array custom value with `iterate`).
    const desc = descriptorOf(cfg.input);
    const cfgReasons: Diagnostic[] = [];
    if (!desc?.iterate) cfgReasons.push(makeDiagnostic('MLGPU-BAD-INPUT', 'gpuReduce input must be a typed-array buffer (f32/i32/u32) to fold'));
    // Reject only NaN — it serializes ambiguously in the memo key (see below) and is never a meaningful fold
    // seed. ±Infinity IS a valid identity: the TRUE neutral element for max (-Infinity) / min (+Infinity),
    // and a valid f32 uniform. (Rejecting it would contradict the neutral-identity contract, which recommends
    // -Infinity/+Infinity as the exact neutral for max/min.)
    if (Number.isNaN(cfg.identity)) cfgReasons.push(makeDiagnostic('MLGPU-BAD-INPUT', `gpuReduce identity must not be NaN`));
    // A prefix-scan (a running-total buffer output) is unbuilt: only the scalar fold ships. Reject `scan:true`
    // LOUDLY rather than silently returning the scalar fold (which would look like a working scan).
    if (cfg.scan) cfgReasons.push(makeDiagnostic('MLGPU-NOT-LOWERABLE', 'gpuReduce scan (prefix-scan) is not supported — only the scalar fold is available; omit `scan` for the reduction'));

    const { seg, bytes: inputBytes } = this.bufferGenSegment('input', cfg.input);
    // DISPATCH-LIMIT BOUND on the reduce INPUT (the map path guards this via cost.ts's output-dim check; the
    // reduce path can't). The scalar cost gate below is `checkCost(4, inputBytes, [1], …)` — its dimension
    // limit only inspects the OUTPUT dims `[1]` (`1 > maxWG*256` never fires), so it never bounds the INPUT.
    // But the FIRST reduction pass dispatches `ceil(inputLen / REDUCE_TILE)` workgroups DIRECTLY (webgpu's
    // `dispatchWorkgroups`, one per partial) / the same count of output texels (webgl2), NOT the map path's
    // `ceil(dim / workgroupSize)` (whose *256 headroom is why cost.ts multiplies by 256). So the reduce grid
    // is `ceil(inputLen / REDUCE_TILE)` groups and its limit is `≤ maxComputeWorkgroupsPerDimension` (NO *256).
    // An input past that would `dispatchWorkgroups(> limit)` → an async validation error → a silently-wrong
    // scalar (the re-ladder only fires on a THROWN dispatch). Reject at the gate (a cfgReason → core=false → no
    // dispatch) instead. REDUCE_TILE (256) is the SAME constant the WebGL2 tile (via `dispatchReduce`'s `tile`),
    // the WGSL `@workgroup_size`, and webgpu's per-pass grid use — they cannot drift, so this bound is correct
    // for every backend. Threshold: inputLen > maxComputeWorkgroupsPerDimension * REDUCE_TILE (≈16.7M for the
    // 65535 CPU hint). Guarded on a valid buffer input (a bad input is already flagged MLGPU-BAD-INPUT above).
    const inputLen = inputBytes / 4;   // bufferGenSegment counts f32/i32/u32 (all 4-byte) as length*4
    const maxWg = this.deps.limitsHint.maxComputeWorkgroupsPerDimension;
    if (desc?.iterate && Math.ceil(inputLen / REDUCE_TILE) > maxWg) {
      cfgReasons.push(makeDiagnostic('MLGPU-ALLOC', `reduce input (${inputLen} elements) exceeds the device dispatch limit — the first-pass grid ceil(${inputLen}/${REDUCE_TILE}) workgroups exceeds maxComputeWorkgroupsPerDimension (${maxWg})`));
    }
    const reasons = [...gateReasons, ...cfgReasons];
    const core = reasons.length === 0;
    const flags = `${cfg.verify ? 'v' : ''}${cfg.benchmark ? 'b' : ''}${cfg.scan ? 's' : ''}`;
    // Encode the identity with String(), NOT JSON.stringify: `JSON.stringify(Infinity)` === `"null"` ===
    // `JSON.stringify(-Infinity)`, so ±Infinity identities would COLLIDE in the memo (a max-reduce seeded
    // -Infinity would read a min-reduce's cached scalar). `String(±Infinity)` → the distinct strings
    // "Infinity"/"-Infinity"; a finite id keys exactly as before relative to itself (String(0) === "0").
    const key = `R::${kernelHash(reducer, bindings)}::id=${String(cfg.identity)}::${requested}::${flags}::${seg}`;
    const hit = this.memo.get(key);
    if (hit) return this.host.readCell(hit.cell) as GpuResource;

    // Cost gate: the output is one scalar (4 bytes); the input is the folded buffer, counted once.
    const costErr = core ? checkCost(4, inputBytes, [1], this.deps.limitsHint) : null;

    // Emit BOTH reduction shaders synchronously (like the map path emits wgsl/glsl) so `resource.wgsl`/`glsl`
    // show the shaders each GPU leg runs: the WebGL2 reducer-fold-over-a-tile fragment shader (a ping-pong tree
    // reduction) + the WebGPU workgroup-shared tree reduction (a compute shader with var<workgroup> + barriers).
    // The reducer's closed-over scalar constants (role:'scalar') are its only uniforms — both legs set them as
    // `_u_<name>` (GLSL) / `_p._u_<name>` (WGSL).
    const reduceScalars: { name: string; value: number }[] = [];
    for (const b of bindings.byName.values()) if (b.role === 'scalar') reduceScalars.push({ name: b.name, value: b.value });
    const reduceGlsl = core ? emitReduceGlsl(reducer, bindings) : '';
    const reduceWgsl = core ? emitReduceWgsl(reducer, bindings, cfg.identity) : '';

    const resource: GpuResource = {
      core, reasons, wgsl: reduceWgsl, glsl: reduceGlsl, backend: 'cpu', pending: core && !costErr,
      value: null, outputs: null,
      error: costErr ?? (core ? null : makeDiagnostic('MLGPU-NOT-LOWERABLE', 'reducer is not GPU-lowerable')),
      gpuMs: null, cpuMs: null, speedup: null, match: null, note: null,
    };
    if (!core || costErr) resource.pending = false;
    const cell = this.host.allocateCell(resource);
    this.memo.set(key, { resource, cell });

    // A synchronous input-read failure must not throw out of gpuReduce() (it runs inside the reader's derive)
    // — settle a local error + skip the fold instead, keeping the failure local to this node. Mirrors gpu().
    const settleInputError = (e: unknown): void => {
      resource.pending = false;
      resource.error = makeDiagnostic('MLGPU-INPUT-UNAVAILABLE', String((e as Error)?.message ?? e));
      this.memo.set(key, { resource, cell });
    };

    // Materialize the input values SYNCHRONOUSLY (before enqueue), mirroring the map path's resolveInputs — so
    // a read failure is caught locally here, and the fold runs against a stable snapshot.
    let values: number[] = [];
    if (resource.pending) {
      try { values = Array.from(desc!.iterate!(cfg.input), (v) => Number(v)); } catch (e) { settleInputError(e); }
    }
    this.evictLru();
    const subscribed = this.host.readCell(cell) as GpuResource;

    if (resource.pending) {
      const inputValues = Float32Array.from(values);
      this.enqueue(async () => {
        if (this.disposed) return;
        // Acquire the requested backend; a REAL GPU reduction runs the WebGL2 multi-pass ping-pong tree fold
        // (its `dispatchReduce`), re-laddering down (webgpu→webgl2→cpu) on absence-of-method / a dispatch throw.
        // The cpu floor IS `cpuReduce` — the exact linear left-fold + the correctness oracle. The interpreter
        // fold stays the reference every GPU leg is verified against (a tree fold reorders → a float-associativity
        // tolerance, not the map path's tight ulp bound). A cpu-requested run never touches the GPU legs.
        let backend = await this.acquireBackend(requested);
        if (this.disposed) return;
        let next: GpuResource;
        try {
          // Run the fold on the acquired rung, re-laddering DOWN on a throw OR a missing dispatchReduce. Each
          // rung: webgl2 → its ping-pong; cpu → cpuReduce (the floor). A backend without dispatchReduce (a WGSL
          // leg still scaffolded) is treated as a throw → re-ladder. Returns the actual backend + the scalar.
          const runReduce = async (b: Backend): Promise<number> => {
            if (b.kind === 'cpu') return cpuReduce(reducer, values, cfg.identity, this.host);   // the linear-fold floor
            if (!b.dispatchReduce) throw new Error(`MLGPU-NO-REDUCE: the ${b.kind} backend has no reduction path`);
            const res = await b.dispatchReduce({ glsl: reduceGlsl, wgsl: reduceWgsl, inputValues, identity: cfg.identity, tile: REDUCE_TILE, scalars: reduceScalars });
            return res.value;
          };
          const t0 = performance.now();
          const settled = await this.dispatchReladder(backend, runReduce);
          if (settled === null) return;   // disposed mid-re-ladder → bail without settling
          backend = settled.backend;
          const scalar = settled.result;
          const gpuMs = performance.now() - t0;
          const onCpu = backend.kind === 'cpu';
          if (this.disposed || this.memo.get(key)?.cell !== cell) return;   // superseded/evicted mid-flight
          // BENCHMARK (opt-in): on the cpu floor the fold already IS the cpu run (reuse gpuMs, no second sweep);
          // on a real GPU rung, time a cpuReduce baseline for the speedup. VERIFY (opt-in): compare the settled
          // scalar against the linear-fold oracle within the reduction tolerance (a tree fold reorders → ulps).
          let cpuMs: number | null = null;
          let speedup: number | null = null;
          if (cfg.benchmark) {
            if (onCpu) { cpuMs = gpuMs; }
            else { const c0 = performance.now(); cpuReduce(reducer, values, cfg.identity, this.host); cpuMs = performance.now() - c0; speedup = gpuMs > 0 ? cpuMs / gpuMs : null; }
          }
          const match = cfg.verify ? checkReduceMatch(scalar, cpuReduce(reducer, values, cfg.identity, this.host)) : null;
          next = { ...resource, backend: backend.kind, pending: false, value: scalar,
            gpuMs: onCpu ? null : gpuMs, cpuMs, speedup, match };
          this.memo.set(key, { resource: next, cell });
        } catch (e) {
          if (this.disposed) return;
          next = { ...resource, pending: false, error: makeDiagnostic('MLGPU-DISPATCH', String((e as Error)?.message ?? e)) };
          if (this.memo.get(key)?.cell === cell) this.memo.set(key, { resource: next, cell });
          else return;
        }
        this.host.writeCell(cell, next);
      });
    }
    return subscribed;
  }

  // ─── The HISTOGRAM kernel kind: a 1-arg bin-mapper scatters an input buffer into per-bin counts ───
  // A DISTINCT kernel kind from the map `gpu()` (thread coords) and the reduce (2-arg fold to a scalar) — a
  // DATA-DEPENDENT ATOMIC SCATTER. So it flows through the dedicated `gpuHistogram` head + `gateBinMapper`
  // (arity=1 + a scalar-lowerable, PURE-over-x body). MIRRORS `gpuReduce`'s reactive-resource shape (memo →
  // allocateCell → materialize input synchronously → subscribe → enqueue → settle), keying the memo off the
  // mapper hash + bins + backend + flags + the input buffer's generation/nonce so a distinct input or bin
  // count re-dispatches (the `H::` prefix keeps it distinct from map/reduce keys).
  //
  // PER-BACKEND ROUTING (the honest backend policy): cpu → `cpuHistogram` (the exact oracle floor); webgpu →
  // `dispatchHistogram` (the atomic scatter — adapter-gated, so it re-ladders/falls here since tryWebGpu is
  // null); webgl2 → `cpuHistogram` (WebGL2's fragment stage has NO atomics, so a histogram falls to the CPU
  // oracle — a documented per-backend difference). A webgl2 or auto-that-fell run therefore RUNS ON CPU and
  // settles `backend: 'cpu'` + a note explaining the fallback, so backend honesty holds (we never claim a
  // WebGL2 scatter it can't do).
  gpuHistogram(binMapper: UserFn, cfg: HistogramConfig): GpuResource {
    const requested: 'auto' | BackendKind = cfg.backend ?? 'auto';
    const { reasons: gateReasons, bindings } = gateBinMapper(binMapper, this.host);
    // Config validation: the input must be a foldable buffer (a typed-array custom value with `iterate`), and
    // `bins` must be a positive integer. Checked BEFORE bufferGenSegment reads the input length — but
    // bufferGenSegment is null-safe (the Phase-4 fix), so a vec input already coerces to length 0 there
    // without throwing; the MLGPU-BAD-INPUT reason below makes it non-core so it never dispatches.
    const desc = descriptorOf(cfg.input);
    const cfgReasons: Diagnostic[] = [];
    if (!desc?.iterate) cfgReasons.push(makeDiagnostic('MLGPU-BAD-INPUT', 'gpuHistogram input must be a typed-array buffer (f32/i32/u32) to bin'));
    if (!Number.isInteger(cfg.bins) || cfg.bins < 1) cfgReasons.push(makeDiagnostic('MLGPU-BAD-INPUT', `gpuHistogram bins must be a positive integer — got ${cfg.bins}`));

    const { seg, bytes: inputBytes } = this.bufferGenSegment('input', cfg.input);
    // DISPATCH-LIMIT BOUND on the histogram INPUT (parity with the reduce path — the map path guards this via
    // cost.ts's output-dim check; the histogram cost gate below is `checkCost(bins*4, inputBytes, [cfg.bins], …)`
    // whose dimension limit only inspects the OUTPUT dims `[cfg.bins]`, so it never bounds the INPUT). But the
    // WebGPU scatter dispatches `ceil(inputLen / HISTOGRAM_WORKGROUP)` workgroups DIRECTLY (dispatchHistogram,
    // one thread per input element), NOT the map path's `ceil(dim / workgroupSize)` (whose *256 headroom is why
    // cost.ts multiplies by 256). An input past `maxComputeWorkgroupsPerDimension * HISTOGRAM_WORKGROUP` would
    // `dispatchWorkgroups(> limit)` → an ASYNC validation error (not a thrown one, so dispatchReladder can't
    // rescue it) → the bins buffer stays zero → silently-wrong all-zero counts. Reject at the gate (a cfgReason
    // → core=false → no dispatch) instead. HISTOGRAM_WORKGROUP is the SAME constant the WGSL `@workgroup_size`
    // and webgpu's `dispatchHistogram` grid use (a single exported const — they cannot drift), so this bound is
    // correct for the scatter grid. Guarded on a valid buffer input (a bad input is already flagged
    // MLGPU-BAD-INPUT above, so it takes that path, not this one).
    const inputLen = inputBytes / 4;   // bufferGenSegment counts f32/i32/u32 (all 4-byte) as length*4
    const maxWg = this.deps.limitsHint.maxComputeWorkgroupsPerDimension;
    if (desc?.iterate && Math.ceil(inputLen / HISTOGRAM_WORKGROUP) > maxWg) {
      cfgReasons.push(makeDiagnostic('MLGPU-ALLOC', `histogram input (${inputLen} elements) exceeds the device dispatch limit — the scatter grid ceil(${inputLen}/${HISTOGRAM_WORKGROUP}) workgroups exceeds maxComputeWorkgroupsPerDimension (${maxWg})`));
    }
    const reasons = [...gateReasons, ...cfgReasons];
    const core = reasons.length === 0;
    const flags = `${cfg.verify ? 'v' : ''}${cfg.benchmark ? 'b' : ''}`;
    // The `H::` prefix keeps a histogram resource distinct from a map (no prefix) / reduce (`R::`) resource.
    const key = `H::${kernelHash(binMapper, bindings)}::bins=${cfg.bins}::${requested}::${flags}::${seg}`;
    const hit = this.memo.get(key);
    if (hit) return this.host.readCell(hit.cell) as GpuResource;

    // Cost gate: the output is `bins` u32 counts; the input is the scattered buffer, counted once.
    const outputBytes = (Number.isInteger(cfg.bins) && cfg.bins > 0 ? cfg.bins : 1) * 4;
    const costErr = core ? checkCost(outputBytes, inputBytes, [cfg.bins], this.deps.limitsHint) : null;

    // Emit the WGSL atomic-scatter shader synchronously (like the reduce path emits its shaders) so
    // `resource.wgsl` shows the shader the WebGPU leg runs. There is no histogram GLSL — WebGL2 has no
    // fragment atomics, so its leg falls to the CPU oracle (glsl stays ''). The bin-mapper's closed-over scalar
    // constants (role:'scalar') ride `_HParams` as `_u_<name>`.
    const histScalars: { name: string; value: number }[] = [];
    for (const b of bindings.byName.values()) if (b.role === 'scalar') histScalars.push({ name: b.name, value: b.value });
    const histWgsl = core ? emitHistogramWgsl(binMapper, bindings, cfg.bins) : '';

    const resource: GpuResource = {
      core, reasons, wgsl: histWgsl, glsl: '', backend: 'cpu', pending: core && !costErr,
      value: null, outputs: null,
      error: costErr ?? (core ? null : makeDiagnostic('MLGPU-NOT-LOWERABLE', 'bin-mapper is not GPU-lowerable')),
      gpuMs: null, cpuMs: null, speedup: null, match: null, note: null,
    };
    if (!core || costErr) resource.pending = false;
    const cell = this.host.allocateCell(resource);
    this.memo.set(key, { resource, cell });

    // A synchronous input-read failure must not throw out of gpuHistogram() (it runs inside the reader's
    // derive) — settle a local error + skip the scatter instead, keeping the failure local. Mirrors gpuReduce.
    const settleInputError = (e: unknown): void => {
      resource.pending = false;
      resource.error = makeDiagnostic('MLGPU-INPUT-UNAVAILABLE', String((e as Error)?.message ?? e));
      this.memo.set(key, { resource, cell });
    };

    // Materialize the input values SYNCHRONOUSLY (before enqueue), mirroring the reduce path — so a read
    // failure is caught locally here, and the scatter runs against a stable snapshot.
    let values: number[] = [];
    if (resource.pending) {
      try { values = Array.from(desc!.iterate!(cfg.input), (v) => Number(v)); } catch (e) { settleInputError(e); }
    }
    this.evictLru();
    const subscribed = this.host.readCell(cell) as GpuResource;

    if (resource.pending) {
      const inputValues = Float32Array.from(values);
      const bins = cfg.bins;
      this.enqueue(async () => {
        if (this.disposed) return;
        // Acquire the requested backend. A histogram scatters on WebGPU only (real storage atomics); the cpu
        // floor + the webgl2 leg BOTH run the exact `cpuHistogram` oracle (WebGL2 has no fragment atomics), so
        // the ladder degrades to CPU on any non-WebGPU rung. The interpreter count stays the reference.
        let backend = await this.acquireBackend(requested);
        if (this.disposed) return;
        let next: GpuResource;
        try {
          // Run the scatter on the acquired rung, re-laddering DOWN on a throw OR a missing dispatchHistogram.
          // Each rung: webgpu → its atomic scatter; cpu → cpuHistogram (the oracle floor); webgl2 → treated as
          // a throw (no dispatchHistogram) → re-ladder → cpu. Returns the actual backend + the counts.
          const runHistogram = async (b: Backend): Promise<number[]> => {
            if (b.kind === 'cpu') return cpuHistogram(binMapper, values, bins, this.host);   // the exact oracle floor
            if (!b.dispatchHistogram) throw new Error(`MLGPU-NO-HISTOGRAM: the ${b.kind} backend has no scatter path (no fragment atomics)`);
            const res = await b.dispatchHistogram({ wgsl: histWgsl, inputValues, bins, scalars: histScalars });
            return res.counts;
          };
          const t0 = performance.now();
          const settled = await this.dispatchReladder(backend, runHistogram);
          if (settled === null) return;   // disposed mid-re-ladder → bail without settling
          backend = settled.backend;
          const counts = settled.result;
          const gpuMs = performance.now() - t0;
          const onCpu = backend.kind === 'cpu';
          if (this.disposed || this.memo.get(key)?.cell !== cell) return;   // superseded/evicted mid-flight
          // BENCHMARK (opt-in): on the cpu floor the scatter already IS the cpu run (reuse gpuMs); on a real
          // GPU rung, time a cpuHistogram baseline for the speedup. VERIFY (opt-in): compare the settled counts
          // against the exact oracle (an integer count match — exact, not a tolerance).
          let cpuMs: number | null = null;
          let speedup: number | null = null;
          if (cfg.benchmark) {
            if (onCpu) { cpuMs = gpuMs; }
            else { const c0 = performance.now(); cpuHistogram(binMapper, values, bins, this.host); cpuMs = performance.now() - c0; speedup = gpuMs > 0 ? cpuMs / gpuMs : null; }
          }
          let match: MatchVerdict | null = null;
          if (cfg.verify) {
            const oracle = cpuHistogram(binMapper, values, bins, this.host);
            const ok = counts.length === oracle.length && counts.every((c, i) => c === oracle[i]);
            match = { ok, kind: 'exact', maxUlp: 0 };   // histogram counts are exact integers — no tolerance
          }
          // BACKEND HONESTY: a histogram that ran on the CPU (a cpu-requested run, OR a webgl2/auto run that
          // fell to CPU because it has no scatter path) settles `backend: 'cpu'` + a note explaining the
          // WebGL2 fallback when webgl2 was requested. We NEVER claim a WebGL2 scatter it cannot do.
          const note = onCpu && requested === 'webgl2'
            ? 'histogram runs on CPU on WebGL2 (no fragment-shader atomics)'
            : null;
          next = { ...resource, backend: backend.kind, pending: false, value: counts, note,
            gpuMs: onCpu ? null : gpuMs, cpuMs, speedup, match };
          this.memo.set(key, { resource: next, cell });
        } catch (e) {
          if (this.disposed) return;
          next = { ...resource, pending: false, error: makeDiagnostic('MLGPU-DISPATCH', String((e as Error)?.message ?? e)) };
          if (this.memo.get(key)?.cell === cell) this.memo.set(key, { resource: next, cell });
          else return;
        }
        this.host.writeCell(cell, next);
      });
    }
    return subscribed;
  }

  // ─── Shared input plumbing (used by BOTH the single-output gpu() path and the multi-output gpuMulti) ───

  /** Build the per-buffer generation-keying segments + the summed input bytes for a binding table — the
   *  memo-key discriminator that re-dispatches when an input's contents change. Extracted so the multi-output
   *  path keys off the SAME input state (a shared kernel closure → the same buffers) as the single path. */
  private computeGens(bindings: BindingTable): { gens: string[]; inputBytes: number } {
    const gens: string[] = [];
    let inputBytes = 0;
    for (const b of bindings.byName.values()) if (b.role === 'buffer') {
      const { seg, bytes } = this.bufferGenSegment(b.name, b.value);
      gens.push(seg); inputBytes += bytes;
    }
    return { gens, inputBytes };
  }

  /** The memo-key segment + byte count for ONE buffer value — the discriminator that re-dispatches when the
   *  buffer's contents change. Shared by `computeGens` (the map path, one segment per closed-over buffer) and
   *  the reduce path (its single `cfg.input` buffer). A resident handle keys off its per-handle nonce (its
   *  descriptor is immutable, no custom-type generation); an engine-minted 'buffer' output keys off its
   *  per-output identity (a fresh buffer reads generation 0, so two distinct same-length outputs would else
   *  collide); a user's own buffer keys off its custom-type generation (a stable const buffer → a memo hit is
   *  correct for an unchanging input; an in-place mutation bumps the generation → re-dispatch). */
  private bufferGenSegment(name: string, value: unknown): { seg: string; bytes: number } {
    const info = residentInfo(value);
    if (info !== null) return { seg: `${name}#H${info.nonce}:${info.length}`, bytes: info.length * 4 };
    // Null-safe length read: a real f32/i32/u32 buffer's getMember('length') returns a number; a
    // non-buffer custom value (a vec/mat) returns the NOT_HANDLED Symbol (NOT undefined, so `?? 0` won't
    // fire) → `Number(Symbol)` would THROW and escape gpuReduce()/gpu() into the reader's derive, collapsing
    // the whole component tree. Coerce a non-number (Symbol or undefined) to 0 — a bad input is already
    // flagged MLGPU-BAD-INPUT by its caller's `!desc?.iterate` check (core=false → no dispatch), so 0 here is
    // harmless. Leaves a valid buffer's key byte-identical (its length is a number).
    const lenOf = (v: unknown): number => { const raw = descriptorOf(v)?.getMember?.(v, 'length'); return typeof raw === 'number' ? raw : 0; };
    if (this.outputIdentity.has(value as object)) {
      const len = lenOf(value);
      return { seg: `${name}#O${this.outputIdentity.get(value as object)}:${len}`, bytes: len * 4 };
    }
    // Keep calling readGeneration(gen) even though the fingerprint already captures content: the read is a
    // load-bearing REACTIVE SIDE EFFECT (it subscribes the reader to this buffer's generation), so an in-place
    // mutation bumps the generation → the reader re-runs → this segment is recomputed → the fingerprint
    // changes → re-dispatch. Dropping the read would leave a mutated buffer's consumer un-subscribed.
    const gen = generationOf(value); const g = gen !== undefined ? this.host.readGeneration(gen) : 0;
    const len = lenOf(value);
    // Fold a CONTENT fingerprint into the key so two DISTINCT same-length buffers (both fresh → generation 0,
    // contents absent from kernelHash) don't collide → the consumer would else return the first buffer's stale
    // result. A per-object id can't be used (a `const x = f32([...])` is rebuilt fresh on each reactive
    // re-derive → a new id every derive → an infinite re-dispatch loop); a CONTENT hash is correct (different
    // content → different key) AND convergent (rebuilt identical content → same key → memo hit → fixpoint).
    // Read the values zero-copy via bufferView (falling back to iterate); a non-buffer coerces to len 0 above
    // and is already flagged MLGPU-BAD-INPUT, so an empty fingerprint here is harmless. O(n) hash — see
    // bufferFingerprint (the dispatch data transfer stays zero-copy; only this key hashing reads the values).
    const desc = descriptorOf(value);
    const view = desc?.bufferView?.(value);
    const fpData: ArrayLike<number> = view?.data ?? (desc?.iterate ? Array.from(desc.iterate(value), (v) => Number(v)) : []);
    const fp = bufferFingerprint(fpData);
    return { seg: `${name}#${g}:${len}:${fp}`, bytes: len * 4 };
  }

  /** Materialize each resident-handle input's CPU cache before LRU eviction can free the producer's GPU
   *  buffer (so eviction never strands the values a consumer needs). A no-op for a non-handle buffer input.
   *  May throw MLGPU-USE-AFTER-DISPOSE (a producer already freed) — the caller settles that as a local error. */
  private materializeResidentCaches(bindings: BindingTable): void {
    for (const b of bindings.byName.values()) {
      if (b.role === 'buffer' && residentInfo(b.value) !== null) descriptorOf(b.value)?.bufferView?.(b.value);
    }
  }

  /** Resolve a binding table's buffer inputs (CPU-fallback Float32Array + any same-instance resident buffer)
   *  and scalar uniforms — the exact `DispatchInput` payload the backends consume. Shared so a multi-output
   *  sub-dispatch resolves inputs identically to a single-output dispatch (same zero-copy + f32 conversion). */
  private resolveInputs(bindings: BindingTable): {
    inputs: { name: string; data: Float32Array }[]; residentInputs: Map<string, unknown>; scalars: { name: string; value: number }[];
  } {
    const inputs: { name: string; data: Float32Array }[] = [];
    const residentInputs = new Map<string, unknown>();
    const scalars: { name: string; value: number }[] = [];
    for (const b of bindings.byName.values()) {
      if (b.role === 'buffer') {
        // A resident-handle input carries a backend-native resident buffer — offer it as the (same-instance)
        // GPU fast-path. STILL resolve + push the CPU-fallback data below for EVERY buffer (including a
        // resident handle): when residency doesn't fire (a fresh backend per dispatch → foreign instance, or a
        // CPU/foreign backend), the backend uploads this data. Reading a handle's bufferView triggers its lazy
        // readback (a CPU-array closure, not a GPU read) → the correct values.
        // Only offer a LIVE resident handle's buffer as a same-instance fast-path bind. A DISPOSED handle
        // (its producer LRU-evicted before this consumer dispatched) still carries a non-undefined gpuBuffer,
        // but that native buffer/texture was FREED — binding it is a use-after-free (WebGL2 → silent 0s;
        // WebGPU → a validation throw → a re-ladder). Skip it so this input falls to the readback-cache upload
        // below (the cache was pre-filled before eviction by materializeResidentCaches), which is correct.
        const info = residentInfo(b.value);
        if (info !== null && info.gpuBuffer !== undefined && !info.disposed) residentInputs.set(b.name, info.gpuBuffer);
        const desc = descriptorOf(b.value);
        const view = desc?.bufferView?.(b.value);
        // Zero-copy when the store is ALREADY a Float32Array (an f32 buffer); else convert once to f32.
        const data = view && view.element === 'f32' && view.data instanceof Float32Array
          ? view.data
          : Float32Array.from((view?.data ?? desc?.iterate?.(b.value) ?? []) as ArrayLike<number>);
        inputs.push({ name: b.name, data });
      }
      else if (b.role === 'scalar') scalars.push({ name: b.name, value: b.value });
    }
    return { inputs, residentInputs, scalars };
  }

  // ─── Multi-output: a named-object return → N named output buffers ───
  // A kernel `return { sum: EXPR_s, diff: EXPR_d }` with `cfg.outputs = { sum, diff }` is equivalent to N
  // single-output kernels, one per named output, each returning that entry's EXPR. This composes the PROVEN
  // single-output path N times (gate → emit → dispatch → oracle), adding NO new emitter/backend surface:
  //   • Validate the object-return STRUCTURE (checkMultiOutputShape — the object-literal + exact-keys check
  //     the single-output gate can't do since it rejects object literals).
  //   • For each named output, synthesize the per-key kernel (synthOutputKernel), gate + emit it with that
  //     output's comps, and dispatch it INLINE in ONE async task — settling ONE resource holding all N
  //     outputs in `resource.outputs`. ONE cell + ONE memo entry (never a recursive this.gpu()), so no extra
  //     cells / re-derives. The synthesized single-output kernel IS the oracle input, so `verify` is
  //     automatically per-key correct.
  private gpuMulti(kernel: UserFn, cfg: GpuConfig, precision: 'f16' | 'f32', requested: 'auto' | BackendKind): GpuResource {
    const names = Object.keys(cfg.outputs ?? {});
    const compsByName: Record<string, number> = {};
    for (const name of names) compsByName[name] = compsOf(cfg.outputs![name]!.element);
    const outT = cfg.outputType ?? 'array';

    // Gate the WHOLE kernel body once against its closure (buffer inputs, no input writes, lowerable body) —
    // WITHOUT the single-output return-shape check (its `return { … }` would be rejected as an object literal).
    // gateFn (via buildBindingTable/collectFreeNames) walks the body but not the return SHAPE beyond emit
    // validity; the object-return structure is validated by checkMultiOutputShape below. We reuse gateKernel
    // for the reason set + bindings, then ADD the multi-output structural reasons. A per-return object literal
    // trips gateFn's `object` rejection (MLGPU-NOT-LOWERABLE) — expected + fine: a bad-shape multi-output
    // kernel is non-core either way. But a WELL-FORMED one also trips it, so we must gate the SYNTHESIZED
    // per-key kernels (which have scalar/vec returns) for the real lowerability verdict, not the object one.
    const free = collectFreeNames(kernel);
    const bindings = buildBindingTable(kernel, free, this.host);
    const structureReasons = checkMultiOutputShape(kernel, names);

    // RANK GATE (the SAME one the single-output path applies): the kernel's params ARE its thread coordinates
    // for multi-output too (each named output is written per-cell over the shared grid), so the arity must
    // match the output dims and at most 3 thread dimensions dispatch. The single-output pipeline the
    // synthesized per-key kernels flow through never re-checks this (synthOutputKernel preserves the params),
    // so without this a rank>3 / arity≠dims multi-output kernel would be silently accepted. Same helper →
    // identical MLGPU codes + wording as the single-output path.
    const rankReasons = rankGate(kernel, cfg.output);

    // Config validation. `outputs` + `outputElement` are mutually exclusive; an empty `outputs` is rejected;
    // v1 multi-output is ARRAY-mode only (buffer/gpu-buffer multi-output is a documented follow-on).
    const cfgReasons: Diagnostic[] = [];
    if (cfg.outputElement !== undefined) cfgReasons.push(makeDiagnostic('MLGPU-OUTPUT-SHAPE', "'outputs' (multi-output) and 'outputElement' (single vecN output) are mutually exclusive — set one, not both"));
    if (names.length === 0) cfgReasons.push(makeDiagnostic('MLGPU-OUTPUT-SHAPE', "'outputs' must declare at least one named output"));
    if (outT !== 'array') cfgReasons.push(makeDiagnostic('MLGPU-NOT-LOWERABLE', `multi-output supports outputType 'array' only in v1 (buffer/gpu-buffer multi-output is a follow-on) — got '${outT}'`));

    // Gate each synthesized per-key kernel (scalar/vec return) for the true per-output lowerability + width
    // verdict; collect every reason so a bad output surfaces. Emit the per-key WGSL/GLSL for display (the
    // first output's shaders represent the run — the panel shows one, all N run identically).
    const perOutput: { name: string; comps: number; synth: UserFn; synthBindings: BindingTable; wgsl: string; glsl: string }[] = [];
    const gateReasons: Diagnostic[] = [];
    if (cfgReasons.length === 0 && structureReasons.length === 0) {
      for (const name of names) {
        const comps = compsByName[name]!;
        const synth = synthOutputKernel(kernel, name);
        const v = gateKernel(synth, this.host, comps);
        gateReasons.push(...v.reasons);
        // The static bounds proof runs per synthesized single-output kernel (its OWN body + synthBindings),
        // bounded by the shared output dims — a provable-OOB index in that output's expression is rejected.
        checkStaticBounds(synth, v.bindings, cfg.output, gateReasons);
        perOutput.push({ name, comps, synth, synthBindings: v.bindings,
          wgsl: v.core ? emitWgsl(synth, v.bindings, precision, comps) : '',
          glsl: v.core ? emitGlsl(synth, v.bindings, precision, comps) : '' });
      }
    }
    const reasons = [...cfgReasons, ...structureReasons, ...gateReasons, ...rankReasons];
    const core = reasons.length === 0;

    // Memo key: the kernel + output dims + the outputs SPEC (names+elements) + input gens + flags + a `M`
    // marker so a multi-output run never collides with a single-output run of the same kernel/dims.
    const { gens, inputBytes } = this.computeGens(bindings);
    const spec = names.map((n) => `${n}:${compsByName[n]}`).join(',');
    const flags = `M${cfg.verify ? 'v' : ''}${cfg.benchmark ? 'b' : ''}`;
    const key = `${kernelHash(kernel, bindings)}::${JSON.stringify(cfg.output)}::${precision}::${requested}::${flags}::{${spec}}::${gens.join(',')}`;
    const hit = this.memo.get(key);
    if (hit) return this.host.readCell(hit.cell) as GpuResource;

    // Cost gate: the total output is Σ (grid × comps × 4) over every named output; inputs counted once.
    const cellCount = cfg.output.reduce((a, b) => a * b, 1);
    const totalOutputComps = names.reduce((a, n) => a + compsByName[n]!, 0);
    const outputBytes = cellCount * totalOutputComps * 4;
    const costErr = core ? checkCost(outputBytes, inputBytes, cfg.output, this.deps.limitsHint) : null;

    // The displayed WGSL/GLSL: the first output's shaders (all N are emitted; the panel shows one representative).
    const wgsl = core ? (perOutput[0]?.wgsl ?? '') : '';
    const glsl = core ? (perOutput[0]?.glsl ?? '') : '';

    const resource: GpuResource = {
      core, reasons, wgsl, glsl, backend: 'cpu', pending: core && !costErr,
      value: null, outputs: null,   // populated (each named buffer) on settle; value stays null for multi-output
      error: costErr ?? (core ? null : makeDiagnostic('MLGPU-NOT-LOWERABLE', 'kernel is not GPU-lowerable')),
      gpuMs: null, cpuMs: null, speedup: null, match: null, note: null,
    };
    if (!core || costErr) resource.pending = false;
    const cell = this.host.allocateCell(resource);
    this.memo.set(key, { resource, cell });

    const settleInputError = (e: unknown): void => {
      resource.pending = false;
      resource.error = makeDiagnostic('MLGPU-INPUT-UNAVAILABLE', String((e as Error)?.message ?? e));
      this.memo.set(key, { resource, cell });
    };
    if (resource.pending) {
      try { this.materializeResidentCaches(bindings); } catch (e) { settleInputError(e); }
    }
    this.evictLru();
    const subscribed = this.host.readCell(cell) as GpuResource;

    if (resource.pending) {
     try {
      // Resolve inputs PER synthesized output kernel. Each per-key shader (o.wgsl/o.glsl) is emitted from its
      // OWN synthBindings, whose buffer SUBSET + ORDER can differ from the whole-kernel order (a subset kernel
      // like `diff: a[i]` binds only `a`; a reorder like `x: b[i]` puts `b` first). The WebGPU backend binds
      // `input.inputs` POSITIONALLY (inputs[k] → the k-th declared `var<storage, read>`), so a sub-dispatch's
      // inputs array MUST be resolved from its OWN synthBindings — a shared whole-kernel `inputs` would feed
      // the wrong buffer (count mismatch → dispatch error; or reorder → silent wrong output). Resolved
      // synchronously here (before enqueue), same as the single-output path, so materializeResidentCaches
      // (run above over the whole-kernel bindings, a superset of every synthBindings' resident inputs) still
      // covers eviction timing.
      const jobs = perOutput.map((o) => {
        const resolved = this.resolveInputs(o.synthBindings);
        return {
          name: o.name, comps: o.comps, synth: o.synth, synthBindings: o.synthBindings, wgsl: o.wgsl, glsl: o.glsl,
          cpuRun: emitCpu(o.synth, o.synthBindings, this.host, o.comps),
          inputs: resolved.inputs, residentInputs: resolved.residentInputs, scalars: resolved.scalars,
        };
      });
      this.enqueue(async () => {
        if (this.disposed) return;
        let backend = await this.acquireBackend(requested);
        if (this.disposed) return;
        // PRECISION FALLBACK (computed ONCE from the FIRST backend, mirroring the single-output path). An f16
        // multi-output run downgrades to f32 with a note on a non-shader-f16 backend or a scalar-uniform
        // kernel. Re-emit each per-key shader at the effective precision when it changed (kernel+synthBindings
        // in scope via `jobs`). The interpreter (cpuRun) is precision-agnostic, so a re-ladder to a lower rung
        // uses f32-consistent glsl/cpu — the fallback composes with the whole-set dispatch-throw re-ladder.
        const anyScalar = jobs.some((j) => j.scalars.length > 0);
        const { effPrecision, note } = this.effectivePrecision(precision, backend, anyScalar);
        const effJobs = effPrecision === precision ? jobs : jobs.map((j) => ({
          ...j,
          wgsl: emitWgsl(j.synth, j.synthBindings, effPrecision, j.comps),
          glsl: emitGlsl(j.synth, j.synthBindings, effPrecision, j.comps),
        }));
        const dims = cfg.output;
        let next: GpuResource;
        // The full N-output sweep is ONE re-ladderable unit: on ANY sub-dispatch throw, the WHOLE set restarts
        // on the next lower rung (webgpu→webgl2→cpu). A device fault affects every sub-dispatch on that device,
        // so re-running the whole set on a working rung is the clean semantic (mirrors the single-output
        // re-ladder). Accumulators are LOCAL to the closure → they reset on each retry. `BAIL` is returned when
        // a dispose/eviction lands mid-sweep → abort the whole task WITHOUT settling (the dispose/evict guard).
        type MultiAgg = { outputs: Record<string, number[]>; gpuMsTotal: number; cpuMsTotal: number; anyGpu: boolean; allMatch: boolean; sawMatch: boolean; aggUlp: number; allExact: boolean };
        const BAIL = Symbol('bail');
        try {
          const settled = await this.dispatchReladder(backend, async (b): Promise<MultiAgg | typeof BAIL> => {
            const outputs: Record<string, number[]> = {};
            let gpuMsTotal = 0; let cpuMsTotal = 0; let anyGpu = false; let allMatch = true; let sawMatch = false;
            // Track the aggregate verdict HONESTLY: the WORST maxUlp across outputs + whether EVERY output was
            // exact. (A prior version fabricated kind:'exact', maxUlp:0 whenever all outputs merely passed `ok`,
            // hiding a within-tolerance ulp match.)
            let aggUlp = 0; let allExact = true;
            // N SEQUENTIAL sub-dispatches, one per named output, run INLINE in this ONE task (never a recursive
            // this.gpu() — that would allocate its own cell + memo + re-derive). Each reuses the single-output
            // backend.dispatch over its synthesized single-output kernel; the interpreter oracle (verify) runs
            // the SAME per-key kernel, so the per-output match is automatically correct. WebGL2 realizes this as
            // N fragment passes (one per output); an MRT (gl.drawBuffers) fast path — write all N in one pass —
            // is a documented follow-on optimization, not needed for v1 correctness.
            for (const j of effJobs) {
              const di = { kernel: j.synth, bindings: j.synthBindings, dims, precision: effPrecision, wgsl: j.wgsl, glsl: j.glsl,
                cpuRun: j.cpuRun, outputComps: j.comps, inputs: j.inputs, scalars: j.scalars, retainOutput: false, residentInputs: j.residentInputs };
              const gpuRes = await b.dispatch(di);
              if (this.disposed || this.memo.get(key)?.cell !== cell) return BAIL;   // superseded/evicted mid-flight
              outputs[j.name] = Array.from(gpuRes.output);
              const onCpu = b.kind === 'cpu';
              if (!onCpu) { anyGpu = true; gpuMsTotal += gpuRes.ms; }
              if (cfg.benchmark) { const cpuRes = onCpu ? gpuRes : await makeCpuBackend().dispatch(di); cpuMsTotal += cpuRes.ms; }
              if (cfg.verify) {
                const m = checkMatch({ fn: j.synth, host: this.host, bindings: j.synthBindings, output: gpuRes.output, dims, precision: effPrecision, sampleCount: 256, comps: j.comps });
                sawMatch = true; if (!m.ok) allMatch = false; if (m.maxUlp > aggUlp) aggUlp = m.maxUlp; if (m.kind !== 'exact') allExact = false;
              }
            }
            return { outputs, gpuMsTotal, cpuMsTotal, anyGpu, allMatch, sawMatch, aggUlp, allExact };
          });
          if (settled === null) return;              // disposed mid-re-ladder → bail without settling
          if (settled.result === BAIL) return;       // disposed/evicted mid-sweep → bail without settling
          backend = settled.backend;                 // the rung that SUCCEEDED — settle uses its kind
          const { outputs, gpuMsTotal, cpuMsTotal, anyGpu, allMatch, sawMatch, aggUlp, allExact } = settled.result;
          if (this.disposed || this.memo.get(key)?.cell !== cell) return;
          const cpuMs = cfg.benchmark ? cpuMsTotal : null;
          const speedup = cfg.benchmark && anyGpu && gpuMsTotal > 0 ? cpuMsTotal / gpuMsTotal : null;
          next = { ...resource, backend: backend.kind, pending: false, value: null, outputs,
            // The displayed shaders reflect the precision that ran (the first output's, as elsewhere) + the note.
            wgsl: effJobs[0]?.wgsl ?? resource.wgsl, glsl: effJobs[0]?.glsl ?? resource.glsl, note,
            gpuMs: anyGpu ? gpuMsTotal : null, cpuMs, speedup,
            // ONE aggregate verdict across all outputs: ok iff every sampled output matched; kind:'exact' only
            // if EVERY output was exact (else 'ulp'); maxUlp = the true WORST-case ulp seen across outputs.
            // Off (null) when verify wasn't requested.
            match: sawMatch ? { ok: allMatch, kind: allExact ? 'exact' : 'ulp', maxUlp: aggUlp } : null };
          this.memo.set(key, { resource: next, cell });
        } catch (e) {
          if (this.disposed) return;
          // Null-safe error formatting (mirrors settleInputError): a terminal throw of `null`/`undefined`
          // (a non-Error) must not itself throw a TypeError reading `.message` and escape as an unhandled
          // rejection — use optional chaining so it degrades to String(e).
          next = { ...resource, pending: false, error: makeDiagnostic('MLGPU-DISPATCH', String((e as Error)?.message ?? e)) };
          if (this.memo.get(key)?.cell === cell) this.memo.set(key, { resource: next, cell });
          else return;
        }
        this.host.writeCell(cell, next);
      });
     } catch (e) { settleInputError(e); }
    }
    return subscribed;
  }

  /** Decide the precision a dispatch ACTUALLY runs at, given the requested precision + the FIRST-acquired
   *  backend + whether the kernel has any scalar uniform. An `f16` request runs at f16 ONLY on a backend that
   *  advertises `features.f16` (a WebGPU device with shader-f16) AND when the kernel has NO scalar uniform;
   *  otherwise it downgrades to f32 with a note (correct f32 values, not wrong f16). The scalar-uniform guard
   *  is a deliberate v1 scope: the f16 WGSL declares scalar uniforms as f16 (`_u_<name>: f16`) in the _Params
   *  struct, but the backend packs the uniform block as f32 — the f16 uniform-packing (correct offset +
   *  16-byte alignment for half members) is fiddly and untestable here (no adapter), so a uniform-bearing f16
   *  kernel cleanly falls back to f32 rather than risk wrong scalar reads. A non-f16 request is untouched. */
  private effectivePrecision(precision: 'f16' | 'f32', backend: Backend, hasScalarUniform: boolean): { effPrecision: 'f16' | 'f32'; note: string | null } {
    if (precision !== 'f16') return { effPrecision: precision, note: null };
    if (!backend.features?.f16) return { effPrecision: 'f32', note: 'f16 requested but shader-f16 is unavailable on this backend — ran at f32' };
    if (hasScalarUniform) return { effPrecision: 'f32', note: 'f16 with scalar uniforms is not yet supported — ran at f32' };
    return { effPrecision: 'f16', note: null };
  }

  /** The ordered list of LOWER rungs to try after a backend of `kind` throws a RUNTIME dispatch (a device can
   *  fail mid-run, not only at acquisition). cpu is the true floor at DISPATCH time, so a GPU dispatch fault
   *  degrades down the ladder (webgpu→webgl2→cpu, webgl2→cpu) instead of becoming a terminal error; cpu has no
   *  lower rung (a cpu dispatch throw is terminal). */
  private lowerRungs(kind: BackendKind): BackendKind[] {
    if (kind === 'webgpu') return ['webgl2', 'cpu'];
    if (kind === 'webgl2') return ['cpu'];
    return [];   // cpu is the floor — no lower rung
  }

  /** Run a dispatch with RUNTIME-failure re-laddering. Tries `run` on the initially-acquired `first` backend;
   *  on a THROW it re-ladders down the rungs (webgpu→webgl2→cpu), acquiring each LOWER rung from the POOL
   *  (never re-acquiring the failed `requested` key — that returns the SAME broken instance; a rung is pooled
   *  under its own kind key + disposed once at teardown, so no lifecycle leak) and retrying ONCE per rung
   *  until one succeeds or the rungs are exhausted. Returns the SUCCEEDING backend + its `run` result, or
   *  `null` if the engine was disposed mid-re-ladder (the caller must bail WITHOUT settling — the dispose-race
   *  guard). Throws the LAST error when every rung threw (cpu is the floor → a cpu throw is terminal, settled
   *  as MLGPU-DISPATCH by the caller as before). Never disposes a backend here — they're pooled (engine-owned).
   *  Each await re-checks `disposed` so a dispose landing mid-re-ladder bails instead of touching a lower rung. */
  private async dispatchReladder<T>(first: Backend, run: (b: Backend) => Promise<T>): Promise<{ backend: Backend; result: T } | null> {
    let backend = first;
    const rungs = this.lowerRungs(first.kind);
    let rungIdx = 0;
    for (;;) {
      try {
        const result = await run(backend);
        if (this.disposed) return null;   // dispose landed during the (successful) dispatch await → bail
        return { backend, result };
      } catch (e) {
        if (this.disposed) return null;   // dispose landed during the (failed) dispatch await → bail
        if (rungIdx >= rungs.length) throw e;   // rungs exhausted (incl. cpu, the floor) → terminal (LAST error)
        backend = await this.acquireBackend(rungs[rungIdx++]!);   // acquire the next LOWER rung (pooled by kind)
        if (this.disposed) return null;   // dispose landed during the acquire await → bail
      }
    }
  }

  /** Acquire the backend for a `requested` config ONCE and reuse it across dispatches — so a producer's
   *  resident output buffer can be bound directly by a later consumer on the SAME instance (residency),
   *  and an N-stage pipeline uses ONE device, not N. The engine owns the pooled backends; they are disposed
   *  only at engine teardown (dispose()), never per-dispatch or per-eviction. */
  private acquireBackend(requested: 'auto' | BackendKind): Promise<Backend> {
    let p = this.backendPool.get(requested);
    if (!p) { p = selectBackend(requested, makeCpuBackend, this.deps.tryWebGpu, this.deps.tryWebGl2); this.backendPool.set(requested, p); }
    return p;
  }

  private enqueue(task: () => Promise<void>): void { this.queue.push(task); this.scheduleDrain(); }
  private scheduleDrain(): void {
    if (this.draining) return;
    this.draining = true;
    void Promise.resolve().then(async () => {
      while (this.queue.length) { const t = this.queue.shift()!; await t(); }
      this.draining = false;
    });
  }
  private evictLru(): void {
    while (this.memo.size > MAX_LIVE) {
      const oldest = this.memo.keys().next().value as string | undefined;
      if (oldest === undefined) break;
      const entry = this.memo.get(oldest);
      // Free the evicted entry's resident GPU buffer (disposeHandle is a no-op on a non-handle value). The
      // BACKEND is pooled (engine-owned) and shared across entries, so it is NOT disposed here — only at
      // teardown. Disposing it per-eviction would tear down a device other live entries still bind to.
      if (entry) disposeHandle(entry.resource.value);
      this.memo.delete(oldest);
    }
  }
  /** True if ANY resource dispatched on this engine is still pending (across every memo key, not just the
   *  ones a caller kept a reference to). The memo entry's `resource` is updated in lockstep with its cell on
   *  settle, so this reflects the current settle state. Lets a driver await every DECLARED resource's settle
   *  even when the returned value only projected out of one (`{ value: r.value }`) and carries no resource. */
  anyPending(): boolean {
    for (const e of this.memo.values()) if (e.resource.pending) return true;
    return false;
  }

  [Symbol.dispose](): void {
    // Mark disposed FIRST so any queued/in-flight drained task bails instead of writing into the cleared memo.
    // Then drop pending work + free every resident handle + dispose the pooled backends exactly once.
    // Idempotent — a second dispose runs over the now-empty memo/queue/pool.
    this.disposed = true;
    this.queue.length = 0;
    // Free each gpu-buffer handle's resident device buffer before the backend device it lives on is destroyed
    // (disposeHandle is a no-op on a non-handle value).
    for (const e of this.memo.values()) disposeHandle(e.resource.value);
    this.memo.clear();
    // Dispose every pooled backend exactly once. An acquisition may still be IN FLIGHT (dispose landed during
    // device acquisition) — attach to the promise so the resolved backend is freed, not orphaned. The second
    // arg swallows a rejected acquisition (selectBackend never rejects today, but be defensive). Then clear
    // the pool so a second dispose is a no-op.
    for (const p of this.backendPool.values()) void p.then((b) => b[Symbol.dispose](), () => {});
    this.backendPool.clear();
  }
}
