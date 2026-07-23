# @metael/gpu — Agent Guidelines

`@metael/gpu` = an eval-free, verifiable GPU-compute engine over the metael kernel. A compute kernel authored as a metael `component` is gated for GPU-lowerability, lowered by **three emitters** (one AST → WGSL / GLSL-ES-3.0 / an eval-free CPU closure), dispatched down a **WebGPU → WebGL2 → CPU** ladder, and returned as a reactive `GpuResource`. The shipped metael interpreter is the correctness **oracle**: the CPU emitter runs the same code the interpreter does, and `verify` re-runs a sample through it. The package CONSUMES the language via `@metael/lang` + `@metael/math` and reactivity via `@metael/runtime`; it re-implements no language fact.

Load-bearing invariants + editing conventions live in the root [../../AGENTS.md](../../AGENTS.md); this file is the src-map + gpu-specific guidance.

## Architecture / src map

The pipeline is **gate → emit → cost → dispatch → oracle → reactive resource**, with two subpaths (`./lang`, `./builder`) feeding kernels in. The public barrel (`src/index.ts`) exports only the API-first surface; the gate/binding/emitter/oracle/bounds/hash pieces are implementation detail the tests reach by relative path (they carry no public stability contract).

### The lowerability gate + static analysis

- **`gate.ts`** — `gateKernel(kernel, host, comps)` → `GateVerdict` (`{ core, reasons, bindings }`): walks the kernel body + closures against the supported construct/builtin set and decides GPU-lowerability. It is **NOT `classifyProfile`** — that rejects array-index / for-of / member / every user call and would reject every kernel this engine runs. Instead it recognizes builtins via a composed **`GPU_CATALOG`** = `composeProfiles(mathProfile, coreIntrinsicsProfile).builtins` (the numeric math vocabulary + the `range` loop intrinsic) plus descriptor lowering (`descriptorOf(v).lower`) for custom values, with a local `RAND_SPEC` inserted *only* so a `rand()` call routes to its precise rejection. Also exports `checkMultiOutputShape`, `synthOutputKernel`, `normalizeImplicitReturn` (an implicit trailing-expr return the interpreter wouldn't propagate is a loud reject, not silent zeros).
- **`bounds.ts`** — `checkStaticBounds(kernel, bindings, dims, reasons)` + `intervalOf` / `Interval`: the static out-of-bounds prover — an interval analysis that rejects a provably out-of-range index (`MLGPU-INDEX-STATIC`), leaving data-dependent indices to the sampled oracle.
- **`binding.ts`** — `buildBindingTable(kernel, freeNames, host)` + `collectFreeNames(kernel)` → `Binding` / `BindingTable`: resolves each free name to a coord / buffer / scalar / uniform / callee role. The gate, all three emitters, and the oracle reuse the SAME table so name resolution never diverges.
- **`cost.ts`** — `checkCost(outputBytes, inputBytes, output, limits)` → the resource-cost gate (`MLGPU-ALLOC` before any dispatch, on an over-limit or malformed output shape). Exports `MAX_GPU_ALLOC` (512 MiB), `CPU_LIMITS`, `DeviceLimits`.
- **`hash.ts`** — `kernelHash(kernel, bindings)` + `bufferFingerprint(data)`: the kernel-hash + content-fingerprint ingredients of the dispatch-memo key, so distinct kernels / distinct buffer contents never collide on the same output shape.

### The three emitters (one AST → three targets)

- **`emit-wgsl.ts`** — `emitWgsl(kernel, bindings, precision, comps)` → the WGSL compute shader; `emitReduceWgsl` (a workgroup-shared tree reduction) + `emitHistogramWgsl` (an `atomic<u32>` scatter) + `HISTOGRAM_WORKGROUP`. WGSL has no native `inverse()` or quaternion type, so those are **hand-emitted** per-size `_invN` / `_qslerp` / `_qmat` prelude helpers — byte-checked against the oracle.
- **`emit-glsl.ts`** — `emitGlsl(kernel, bindings, precision, comps)` → the GLSL-ES-3.0 compute-via-fragment shader; `emitReduceGlsl` (the ping-pong tile fold) + `REDUCE_TILE`. WebGL2 has no compute stage, so a kernel runs as a fullscreen-quad fragment shader over `R32F` input textures into an `RGBA32F` render target read back with `readPixels`.
- **`emit-cpu.ts`** — `emitCpu(kernel, bindings, host, comps)` → an eval-free JS closure `(coords) => number[]` that delegates to the descriptor handlers, so CPU-emit ≡ the interpreter. This is the oracle floor.
- **`output.ts`** — `compsOf(el)` + `OutputElement`: the per-cell output element (`f32`/`vec2..4`) → component count for the flat-interleaved layout (`output[cell*N + k]`) every backend produces identically.

### The device layer / backend ladder

- **`device/index.ts`** — the backend seam: the `Backend` interface (`dispatch` + optional `dispatchReduce`/`dispatchHistogram`), `BackendKind`, `DispatchInput`/`DispatchResult` (+ reduce/histogram variants), and `selectBackend(prefer, makeCpu, tryWebGpu, tryWebGl2)` which verifies a **real adapter/device** at each rung (navigator.gpu truthy ≠ working) and falls WebGPU → WebGL2 → CPU.
- **`device/webgpu.ts`** — `tryWebGpuBackend()`: probes a real WebGPU device (resolves `null` when none); the storage-buffer compute path + `atomic<u32>` histogram + workgroup-tree reduction.
- **`device/webgl2.ts`** — `tryWebGl2Backend()`: the compute-via-fragment fallback (no fragment atomics → a histogram falls to the CPU oracle).
- **`device/cpu.ts`** — `makeCpuBackend()`: the always-available floor; runs the `emitCpu` closure / the CPU reduce+histogram oracles.
- **`device/pipeline-cache.ts`** — a per-backend compiled-pipeline cache keyed by shader text, so a re-dispatched or re-bound kernel with identical emitted source reuses its pipeline.
- **`handle.ts`** — `makeGpuBufferHandle` / `residentInfo` / `disposeHandle` + `HandleSpec`: the resident `GpuBufferHandle` (kept on-device for pipelining — a producer's output binds directly as the next same-backend kernel's input with no readback).

### The kernel kinds: reduce + histogram

- **`reduce.ts`** — `gateReducer(reducer, host)` (the DISTINCT 2-arg associative+commutative reducer gate) + `cpuReduce(reducer, inputValues, identity, host)` (the linear-fold oracle). `identity` must be the reducer's NEUTRAL element (the tree re-seeds it per tile per pass).
- **`histogram.ts`** — `gateBinMapper(binMapper, host)` (the DISTINCT 1-arg bin-mapper gate) + `cpuHistogram(binMapper, inputValues, bins, host)` (the exact scatter oracle). Out-of-range bin indices are dropped.

### The oracle

- **`oracle.ts`** — `checkMatch(input)` (a sampled interpreter re-run + ULP-tolerance check of the GPU output) + `checkReduceMatch(gpu, oracle)` → `MatchVerdict` (`{ ok, kind: 'exact'|'ulp', maxUlp }`); `OracleInput`. This is what `verify: true` populates as `resource.match`.

### The reactive resource + settle loop

- **`resource.ts`** — `GpuEngine` (one per reactive host) + `GpuConfig`/`ReduceConfig`/`HistogramConfig`/`GpuResource`/`GpuEngineDeps`. Its `gpu` / `gpuReduce` / `gpuHistogram` methods turn a head call into a **memoized reactive `GpuResource`**: gate → emit the three shaders → cost-check → dispatch on the host-driven async queue → re-ladder DOWN on a runtime dispatch fault. The resource starts `pending`, then settles in place with `value`/`outputs`/`backend`/timing — every field readable from metael source. The dispatch-memo key is keyed by kernel-hash + config + input buffer-fingerprint so distinct dispatches never collide.
- **`settle.ts`** — the FREE helpers over a `() => GpuResource` re-dispatch thunk: `settle(dispatch, opts?)` (await a settled resource — dispatch → drain a macrotask → re-dispatch until `!pending`; a disposed engine's dispatch throws, ending the loop), `subscribe(dispatch, onValue)` (a tracked effect that fires on pending then settled; returns a stop disposer), `settled(r)` (narrows `r.pending === false` — but `value` may still be `null` on a non-core/errored settle, so null-check it).
- **`buffer.ts`** — `gpuBuffer(data, host)`: wrap a plain `Float32Array | number[]` as a reduce/histogram input buffer without hand-boxing.

### `createGpuEngine` façade (`.` — the API-first core)

- **`api.ts`** — `createGpuEngine(opts?)` → a `GpuEngineFacade` (a fresh `RuntimeReactiveHost` + a `GpuEngine` over the real device ladder, or `cpuOnly`/custom `deps`). Its `dispatch(kernel, cfg)` wires the `change()` boundary and routes by `cfg.mode` (`'map'` default → `engine.gpu`, `'reduce'` → `engine.gpuReduce`, `'histogram'` → `engine.gpuHistogram`); returns the pending resource synchronously. `CreateGpuEngineOptions` / `GpuEngineFacade` / `DispatchConfig`. The façade is a native `Disposable` (`using gpu = createGpuEngine()`); dispatch throws after dispose. **This core carries NO interpreter dependency** — turning source text into a kernel is the `./lang` binding's job.

### `./lang` — the DSL binding (pulls the interpreter)

- **`lang/compile-kernel.ts`** — `compileKernel(src, host)`: `evaluateProgram` the snippet against `host` (so the kernel's `const a = f32(…)` closure lives on the same host the engine reads) and return the resulting `UserFn`; throws if the value isn't a function/component. This is the subpath that pulls `evaluateProgram`.
- **`lang/host-env.ts`** — `GpuHostEnv`: the `HostEnvironment` resolving the `gpu` / `gpuReduce` / `gpuHistogram` heads to reactive resources (`kind: 'value'` so a program reads `r.core`/`r.wgsl`/`r.value`), declining every other call; `bindHost` creates the engine, `anyPending()` lets a headless driver await declared resources.
- **`lang/profile.ts`** — `gpuProfile`: the vocabulary `Profile` for the three heads (for completion/hover/lens in tooling).

### `./builder` — the JS kernel builder

- **`builder/node.ts`** — `KNode` (a chainable wrapper over a metael AST `Expr`: `add`/`sub`/`mul`/`div`/`mod`, comparisons `lt`/`le`/`gt`/`ge`/`eq`/`ne`, `at(...)` computed index / `member(prop)`) + `lit(n)` / `param(name)` / `call(head, …args)` / `toExpr(x)`. JS has no operator overloading, so arithmetic/comparison are methods; each builds the SAME node the parser emits.
- **`builder/kernel.ts`** — `kernel(fn)` → the assembled `UserFn` (runs `fn` once with fresh param KNodes and captures the traced statements); `kernelAst(fn)` → the `Stmt`; the statement verbs `letVar` / `set` / `forRange` / `ifThen` / `ret` (a builder-trace over a module-level frame stack). The idiomatic form is the arrow-return `kernel((row, col) => row.add(col))` — a returned `KNode` is the per-cell output write. Proven AST-equivalent to the parser, so builder kernels inherit the identical gate/emit/oracle/dispatch path.

## The lowerable kernel vocabulary

A kernel is a `component k(…)` whose parameters are **thread coordinates** — 1-D `k(i)` over `output: [N]`, 2-D `k(x, y)` over `[W, H]`, or **3-D `k(x, y, z)` over `[W, H, D]`**. Rank > 3, or an arity that doesn't match the output's dimension count, is a loud gate reject. A kernel body may use:

- **Control + indexing** — scalar arithmetic, `if`/ternary, a bounded `for (… of range(n))`, typed-array indexing `a[i]` + `a.length`, a `let` accumulator (hence a `component`, not a plain `function`).
- **Scalar math** — `min`/`max`/`abs`/`sign`/`floor`/`ceil`/`round`/`clamp`/`trunc`/`degrees`/`radians`, `sqrt`/`pow`/`exp`/`exp2`/`log`/`log2`/`inverseSqrt`/`fract`/`step`/`mix`/`smoothstep`, trig `sin`/`cos`/`tan`, inverse-trig `asin`/`acos`/`atan`/`atan2`, hyperbolic `sinh`/`cosh`/`tanh`. Each maps to a shader intrinsic (with a domain guard where the interpreter has one) and applies **componentwise** to a `vec`.
- **vec/mat** — `vec2/3/4`, the square `mat2/3/4` + the six non-square `mat2x3`…`mat4x3` (`matCxR`, column-major), componentwise `+ - * /`, vec/mat–scalar scaling, column-major `mat*vec` / `mat*mat`, swizzles, plus `dot`/`cross`/`normalize`/`length`, `transpose`/`determinant`/`inverse`/`distance`/`reflect`/`refract`/`faceforward`.
- **quaternion** — the `vec4`-convention family `qmul`/`qconj`/`qinvert`/`qaxisangle`/`qrotate`/`qslerp`/`qmat`. WGSL has no native `inverse()` or quaternion type → hand-emitted prelude helpers, byte-checked against the oracle.

**Rejected** (span-anchored `MLGPU-*`, `resource.core === false`, `resource.reasons` says why): strings, objects, dynamic arrays, `while`, helper calls into non-lowerable bodies, and **`rand()`** (it cannot match the deterministic oracle). A **static bounds-prover** additionally rejects a provably out-of-range index (`MLGPU-INDEX-STATIC`).

### Output shapes, precision, and I/O

- **`outputType`** — `'array'` (default `number[]`), `'buffer'` (a frozen zero-copy `f32` handle), or `'gpu-buffer'` (a resident `GpuBufferHandle` for on-device **pipelining** — kernel A's output feeds kernel B with no CPU round-trip on the same backend; otherwise it lazily reads back).
- **`outputElement`** — `'f32'` (default) or `'vec2'`/`'vec3'`/`'vec4'`, written flat-interleaved (`output[cell*N + k]`).
- **`outputs: { name: {…} }`** — multi-output: a kernel returning a named object writes several buffers (`resource.outputs.<name>`), one per key. Mutually exclusive with `outputElement`; array-mode only.
- **`precision`** — `'f32'` (default) or `'f16'` (WebGPU `shader-f16`; falls back to f32 with a `resource.note` on a backend without the feature).
- **Inputs** — zero-copy for an `f32` buffer (the live store is handed to the backend); `i32`/`u32`/`f64` convert once to the `f32` GPU storage type. The collection builtins accept a typed array directly.

### `verify` / `benchmark`

A default dispatch is **GPU-only** (returns `value` + `gpuMs`). Two opt-in flags add cost: `verify: true` re-runs a **sample** of output cells through the interpreter and ULP-checks the GPU output (populates `match`); `benchmark: true` also runs the whole sweep on CPU to time a baseline (populates `cpuMs`, and `speedup` on a real GPU). Both off by default — a CPU sweep + a second full dispatch on every run would defeat the offload.

### Reductions + histograms

- **`gpuReduce(reducer, cfg)`** — a 2-arg associative **and commutative** reducer + an input buffer + a **neutral** `identity`, folded to a scalar (`0` sum, `1` product, `±Infinity` max/min). GPU legs: a WebGL2 ping-pong tile reduction + a WGSL workgroup-shared tree; the CPU linear fold is the oracle. `scan: true` (prefix-scan) is REJECTED, never silently folded.
- **`gpuHistogram(binMapper, cfg)`** — a 1-arg bin-mapper scattering each input into `cfg.bins` buckets (`atomicAdd` on WGSL; WebGL2 has no fragment atomics so it runs the CPU oracle with a `resource.note`). Out-of-range bin indices are dropped.

## Key invariants specific to gpu

- **Gate ↔ emitter lockstep.** A gate-accepted kernel must NEVER hit an emitter's no-lowering throw. This is exactly the class of bug adversarial review found (e.g. `rand()` in value position, `range()` mis-classified) — the gate and the three emitters must agree on every construct. When you widen one, widen all.
- **The interpreter is the correctness oracle.** The CPU emitter runs the same descriptor handlers the interpreter runs (CPU-emit ≡ interpreter by construction), and `verify` samples + ULP-checks the GPU output against it. Any shader-side special case (a domain guard, a hand-emitted helper) must match the interpreter's behavior, or `verify` diverges.
- **Memo-key completeness.** The dispatch memo is keyed by kernel-hash + config + buffer-fingerprint. A new field that changes the emitted shader or the dispatch result MUST enter the key, or distinct dispatches collide on a stale resource.
- **Diagnostics.** This package OWNS the `MLGPU-*` codes: `MLGPU-NOT-LOWERABLE`, `MLGPU-ALLOC`, `MLGPU-DISPATCH`, `MLGPU-INDEX-STATIC`, `MLGPU-OUTPUT-SHAPE`, `MLGPU-BAD-INPUT` / `MLGPU-INPUT-WRITE` / `MLGPU-INPUT-UNAVAILABLE`, `MLGPU-REDUCER-ARITY` / `MLGPU-BINMAPPER-ARITY`, `MLGPU-USE-AFTER-DISPOSE` / `MLGPU-READBACK-SHORT`, plus the fail-closed runtime codes `MLGPU-EMIT` (a gate↔emitter drift surfaces as a local diagnostic, not a tree collapse) and `MLGPU-NO-REDUCE` / `MLGPU-NO-HISTOGRAM` (an acquired backend rung lacks the reduction / scatter path → re-ladder DOWN).
- **Dependency boundary.** Imports ONLY `@metael/lang`, `@metael/math`, and `@metael/runtime` (enforced by `boundary.test.ts`), NEVER `@metael/vdom`. The device backends emit shader *strings* for the driver — they never `eval` a kernel-derived string in JS. The `.` core has no interpreter dep; only `./lang` pulls `evaluateProgram`.

## When you add/change — the drift checklist

- **A builtin's lowering.** Touch all four: the gate's `GPU_CATALOG` (so it's recognized), `emit-wgsl.ts` + `emit-glsl.ts` + `emit-cpu.ts` (all three targets), and confirm the interpreter oracle already handles it (it's the reference — add a parity test). A gate entry with no emitter path is the lockstep bug.
- **A new backend.** Implement the `Backend` interface (`device/*.ts`), add its probe to the `selectBackend` ladder, and give it a real-adapter browser proof.
- **A new kernel kind.** Add its gate (like `gateReducer`/`gateBinMapper`), its emitter(s), its CPU oracle, a `DispatchConfig` `mode` + the `dispatch` routing in `api.ts` and the engine method in `resource.ts`, and its head in `GpuHostEnv`.
- **A config field that affects the shader or result.** Add it to the memo key (`hash.ts` inputs / the engine's key composition), or dispatches collide.

## Testing / build

Two layers. Node unit tests assert the gate/emit/oracle/bounds/binding shapes and the API façade directly; `*.browser.test.ts` run the real-adapter WebGPU/WebGL2 proofs in Chromium (the empirical parity the no-adapter shader-string tests can't get). `boundary.test.ts` enforces the dependency boundary; `builder/equivalence.test.ts` pins builder ↔ parser AST equivalence. Keep them green and add a test with any change — especially a browser parity proof for a new lowering.

`npm run -w @metael/gpu typecheck` / `build`; `npx vitest run packages/gpu` (node) + `npx vitest run --project browser packages/gpu` (Chromium). From the repo root, `npm run docs:api:check` is the doc-coverage gate (every exported symbol needs a doc comment) and `npm run prepublishOnly` runs the full one-shot gate — `clean → build:packages → typecheck → lint → test → docs:api:check`.

---

Root [AGENTS.md](../../AGENTS.md) — kernel invariants + editing guardrails. [README.md](./README.md) — install + the runnable host-API surface.
