// Target dispatch: the same source, two backends. A UI run mounts a real @metael/vdom app into the given
// container (its live DOM is the preview). A compute run evaluates the source to a pure value and returns
// its pretty-printed string. Both return the diagnostics they produced so the caller can drive the shared
// diagnostics UI. The caller owns the "only swap the preview when diagnostics are empty" policy (see
// create.ts); a run always reports what it found.
import { mount, VDomHostEnv } from '@metael/vdom';
import type { VDomHandle } from '@metael/vdom';
import { evaluateProgram, PlainStorageHost, RecordingHostEnv } from '@metael/lang';
import type { Diagnostic, HostEnvironment, ReactiveHost, Arg, HostValue, SourceSpan } from '@metael/lang';
import { GpuHostEnv, tryWebGpuBackend, tryWebGl2Backend, CPU_LIMITS } from '@metael/gpu';
import { RuntimeReactiveHost, change } from '@metael/runtime';
import { prettyValue } from './compute-view.ts';
import type { Target } from './examples.ts';

export interface RunOptions {
  data?: unknown;
  seed?: number;
  // A 'gpu' run only: called when a gpu resource surfaces a NEW gate/dispatch reason — including on a LATE
  // re-derive (a kernel guarded behind an async-settled resource). Lets the caller show a reason that
  // arrives after the initial mount returns. The argument is the full deduped set seen so far.
  onGpuIssues?: (diags: Diagnostic[]) => void;
}

// A composite HostEnvironment: the `gpu` head resolves to a GpuResource; every other head delegates to the
// vdom display env (div/span/p/button/pre/code/…). bindHost binds BOTH. This app-level composition keeps
// @metael/gpu free of any vdom dependency — the two vocabularies meet only here, in the app that uses both.
// The backend ladder is WebGPU→CPU (the WebGL2 rung is not wired); in a headless browser the resource
// settles on the CPU floor, still correct (it matches the interpreter oracle).
//
// LIFECYCLE: ONE GpuUiEnv per mount (created in the 'gpu' branch below, returned by the envFactory on
// every pass). The gpu ENGINE is bound exactly once — its dispatch-memo is the loop-breaker that lets the
// resource settle (a fresh engine per pass would re-enqueue a dispatch every re-derive → an infinite loop),
// so it must persist across the mount's re-derive passes. The vdom env is re-bound each pass (its leaf
// effects belong to that pass's fresh host). The backend ladder is the full WebGPU→WebGL2→CPU: a browser
// without WebGPU still gets a real GPU dispatch via WebGL2 (compute-via-fragment), the CPU floor otherwise.
export class GpuUiEnv implements HostEnvironment {
  private readonly gpu = new GpuHostEnv({ tryWebGpu: tryWebGpuBackend, tryWebGl2: tryWebGl2Backend, limitsHint: CPU_LIMITS });
  private readonly vdom = new VDomHostEnv();
  private gpuBound = false;
  // A gpu resource that fails the lowerability/cost gate (core===false) or carries a dispatch error reports
  // its reason ON THE RESOURCE (r.reasons / r.error), NOT in the walk's diagnostics — the resource is a pure
  // value the program is expected to render. A program that never renders r.core/r.reasons (e.g. any pasted
  // kernel, where there is no authored badge) would fail SILENTLY. Collect those reasons here, keyed by
  // (code, message) so the same non-core resource read across re-derive passes is reported once.
  //
  // CRUCIAL: a gpu resource can go non-core on a LATER pass, not just the first — a kernel whose creation is
  // guarded behind an async-settled resource (`if (rA.value == null) … else gpu(b, …)`) is only derived once
  // stage A settles, i.e. on a re-derive AFTER the initial mount. A one-shot read at mount time misses it.
  // So we NOTIFY on every newly-seen issue (`onIssues`), letting the caller surface a late-arriving reason.
  private readonly gpuIssues = new Map<string, Diagnostic>();
  private readonly onIssues?: (diags: Diagnostic[]) => void;
  constructor(onIssues?: (diags: Diagnostic[]) => void) { this.onIssues = onIssues; }
  bindHost(host: ReactiveHost): void {
    if (!this.gpuBound) { this.gpu.bindHost(host); this.gpuBound = true; }   // bind the engine ONCE (memo persists)
    this.vdom.bindHost(host);
  }
  // Free the gpu engine (→ destroys any acquired WebGPU device) when the mount is torn down. mount()'s
  // unmount() calls this via the env's optional dispose; without it a real adapter leaks a GPUDevice per
  // re-mount (invisible on the CPU floor, where dispose is a no-op).
  [Symbol.dispose](): void { this.gpu[Symbol.dispose](); }
  /** The distinct gate/dispatch reasons of every non-lowerable or errored gpu resource seen so far. */
  gpuDiagnostics(): Diagnostic[] { return [...this.gpuIssues.values()]; }
  resolveCall(head: string, key: string, args: Arg[], children: HostValue[], span: SourceSpan):
    { handled: true; value: HostValue; kind?: 'value' } | { handled: false } {
    const g = this.gpu.resolveCall(head, key, args, children, span);
    if (g.handled) { this.recordGpuIssue(g.value); return g; }
    return this.vdom.resolveCall(head, key, args, children, span);
  }
  private recordGpuIssue(value: HostValue): void {
    const r = value as { core?: unknown; reasons?: Diagnostic[]; error?: Diagnostic | null } | null;
    if (!r || typeof r !== 'object' || typeof r.core !== 'boolean') return;   // not a GpuResource
    const ds: Diagnostic[] = [];
    if (r.core === false && Array.isArray(r.reasons)) ds.push(...r.reasons);
    if (r.error) ds.push(r.error);
    let added = false;
    for (const d of ds) { const k = `${d.code} ${d.message}`; if (!this.gpuIssues.has(k)) { this.gpuIssues.set(k, d); added = true; } }
    if (added) this.onIssues?.(this.gpuDiagnostics());   // a NEW issue appeared (possibly on a late re-derive)
  }
}

export interface UiRun { kind: 'ui'; handle: VDomHandle; diagnostics: Diagnostic[] }
export interface ComputeRun { kind: 'compute'; text: string; value: unknown; diagnostics: Diagnostic[] }
export type TargetRun = UiRun | ComputeRun;

/** Run `source` against `target`. For 'ui', `container` receives the live mount; for 'compute' it is unused. */
export function runTarget(target: Target, source: string, container: Element | undefined, opts: RunOptions): TargetRun {
  if (target === 'ui') {
    const handle = mount(source, container, { data: opts.data, seed: opts.seed });
    return { kind: 'ui', handle, diagnostics: handle.diagnostics };
  }
  if (target === 'gpu') {
    // A 'gpu' run IS a ui run: it mounts a vdom app that happens to call the `gpu` head, so it reuses the
    // UiRun shape + the ui preview path. The composite env resolves `gpu`, else the vdom display heads.
    // ONE env instance for the whole mount (its gpu engine + dispatch-memo must persist across re-derive
    // passes — see GpuUiEnv), returned by the factory on every pass.
    // The env NOTIFIES on a newly-seen gpu issue (onIssues), which covers a LATE re-derive — a kernel
    // created only after an async-settled resource (`if (rA.value == null) … else gpu(b, …)`) is derived
    // once stage A settles, AFTER this function returns; a mount-time snapshot alone would miss it.
    const env = new GpuUiEnv(opts.onGpuIssues);
    const handle = mount(source, container, { data: opts.data, seed: opts.seed, envFactory: () => env });
    // Also fold the reasons seen during the SYNCHRONOUS derive into the returned diagnostics — a non-core
    // resource read on the first frame surfaces immediately (dedup: a reason already in the walk isn't
    // repeated). Late reasons arrive via onGpuIssues.
    const walkKeys = new Set(handle.diagnostics.map((d) => `${d.code} ${d.message}`));
    const gpuDiags = env.gpuDiagnostics().filter((d) => !walkKeys.has(`${d.code} ${d.message}`));
    return { kind: 'ui', handle, diagnostics: [...handle.diagnostics, ...gpuDiags] };
  }
  const res = evaluateProgram(source, {
    host: new PlainStorageHost(), env: new RecordingHostEnv(), data: opts.data, seed: opts.seed,
  });
  return { kind: 'compute', text: prettyValue(res.value), value: res.value, diagnostics: res.diagnostics };
}

// A DOM-FREE env for the compute path: the gpu/gpuReduce/gpuHistogram heads resolve to a GpuResource
// value; every other head is declined (a pure compute program renders no display nodes). No vdom, no DOM.
class GpuComputeEnv implements HostEnvironment, Disposable {
  private readonly gpu = new GpuHostEnv({ tryWebGpu: tryWebGpuBackend, tryWebGl2: tryWebGl2Backend, limitsHint: CPU_LIMITS });
  bindHost(host: ReactiveHost): void { this.gpu.bindHost(host); }
  /** True while any gpu resource declared in the program is still pending (see GpuHostEnv.anyPending). */
  anyPending(): boolean { return this.gpu.anyPending(); }
  [Symbol.dispose](): void { this.gpu[Symbol.dispose](); }
  resolveCall(head: string, key: string, args: Arg[], children: HostValue[], span: SourceSpan):
    { handled: true; value: HostValue; kind?: 'value' } | { handled: false } {
    return this.gpu.resolveCall(head, key, args, children, span);
  }
}

export interface ComputeSettledResult { value: unknown; text: string; diagnostics: Diagnostic[] }

/** Evaluate a compute program to a value on a DOM-FREE path, awaiting any gpu resource's async settle.
 *  A program that reads a `GpuResource` returns it pending on the first eval; we drain macrotasks and
 *  re-evaluate on the SAME host/env until no pending resource remains (or a small iteration bound), then
 *  pretty-print. Non-gpu compute programs settle on the first eval (no pending resource → one pass). */
export async function runComputeSettled(source: string, opts: RunOptions): Promise<ComputeSettledResult> {
  const host = new RuntimeReactiveHost();
  const env = new GpuComputeEnv();
  env.bindHost(host);
  let value: unknown; let diagnostics: Diagnostic[];
  let iters = 0;
  // Re-evaluate until no gpu resource is still pending. The engine's memo (persistent on `host`+`env`) returns
  // the settled resource once its async dispatch drains — the same change→drain→re-read cycle the engine tests
  // use, driven here by re-evaluating the program. We check BOTH the returned value (`hasPendingResource`, which
  // walks it to any depth) AND the engine's declared-resource state (`env.anyPending()`): the latter catches a
  // program that only returns a PROJECTION of a resource (`{ value: r.value }`) — the returned value carries no
  // resource to detect, but the engine still has one pending, so we keep draining until it settles.
  //
  // DISPOSE ON EVERY EXIT PATH: `change()` can throw (the runtime's converge guard on arbitrary pasted
  // source), so the dispose lives in a finally — a normal return OR a propagating throw both free the engine
  // (→ any acquired GPU device). Without it a failed run leaks a device per invocation.
  try {
    for (;;) {
      let res!: { value: unknown; diagnostics: Diagnostic[] };
      change(() => { res = evaluateProgram(source, { host, env, data: opts.data, seed: opts.seed }); });
      value = res.value; diagnostics = res.diagnostics;
      if ((!hasPendingResource(value) && !env.anyPending()) || ++iters > 1000) break;
      await new Promise<void>((r) => setTimeout(r, 0));
    }
    return { value, text: prettyValue(value), diagnostics };
  } finally {
    env[Symbol.dispose]();
  }
}

/** True if `value` is, or contains at ANY depth, a GpuResource still pending. Walks objects AND arrays to
 *  any depth (`{ out: { r } }`, `{ items: [r] }`, `[[r]]`), so a settle loop never exits early on a nested
 *  resource. Depth-bounded defensively (the value is a frozen deep-immutable metael result, so cycles are not
 *  expected, but the bound stops any pathological structure from running away). */
function hasPendingResource(value: unknown): boolean {
  const isPending = (v: unknown): boolean => !!v && typeof v === 'object'
    && typeof (v as { pending?: unknown }).pending === 'boolean' && (v as { pending: boolean }).pending === true;
  const walk = (v: unknown, depth: number): boolean => {
    if (isPending(v)) return true;
    if (depth >= 100 || !v || typeof v !== 'object') return false;
    // Object.values handles arrays too (their elements), covering both `[r]` and `{ k: r }` uniformly.
    for (const child of Object.values(v as Record<string, unknown>)) if (walk(child, depth + 1)) return true;
    return false;
  };
  return walk(value, 0);
}
