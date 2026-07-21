# @metael/gpu

[![metael](https://img.shields.io/badge/project-metael-blueviolet.svg?style=flat-square&logo=github)](https://github.com/andykswong/metael)
[![npm](https://img.shields.io/npm/v/@metael/gpu?style=flat-square&logo=npm)](https://www.npmjs.com/package/@metael/gpu)
[![license: MIT](https://img.shields.io/badge/License-MIT-red.svg?style=flat-square)](./LICENSE)

**An eval-free, verifiable GPU-compute engine** built on the [metael](../../README.md) reactive kernel
(`@metael/lang` + `@metael/runtime`). Write a compute kernel as a metael `component`; `@metael/gpu` decides
whether it is GPU-lowerable, emits a real compute shader, runs it on a real adapter, and returns a reactive
resource. The shipped metael interpreter is the correctness **oracle**.

Unlike a `.toString()`-and-transpile approach, the kernel is a **real parsed AST** from an eval-free
language — so there is **no `new Function`, no code-injection surface** (user source is *data*, never executed
as host code), the engine can point at *exactly why* a kernel is not lowerable (with source spans), and it can
check the GPU result against the language's own interpreter.

```
component ──gate──▶ lowerable?  ──emit──▶  WGSL  │  GLSL-ES-3.0  │  eval-free CPU
   kernel   │ over BUILTINS +      │ one AST, three targets, type codegen from each value's Lowering
            │ descriptor lowering  │
            └──cost gate──▶ dispatch ──▶ WebGPU → WebGL2 → CPU ──▶ reactive resource
                                          (each verifies a REAL adapter; cpu is the floor)
```

## The kernel is a metael `component`

A map kernel takes its output coordinates as parameters and returns one scalar per cell. Buffers (typed
arrays) + scalars are read from the closure; a `component` (not a plain `function`) so a `let` accumulator
works:

```ts
import { evaluateProgram, RecordingHostEnv } from '@metael/lang';
import { RuntimeReactiveHost, change } from '@metael/runtime';
import { GpuEngine, tryWebGpuBackend, tryWebGl2Backend, CPU_LIMITS } from '@metael/gpu';

const host = new RuntimeReactiveHost();
const { value: kernel } = evaluateProgram(`
  const N = 256
  const x = f32(N, (i) => i)
  const y = f32(N, (i) => 2 * i)
  component saxpy(i) { return 3 * x[i] + y[i] }
  saxpy`, { host, env: new RecordingHostEnv() });

const engine = new GpuEngine(host, {
  tryWebGpu: tryWebGpuBackend, tryWebGl2: tryWebGl2Backend, limitsHint: CPU_LIMITS,
});

let r;
change(() => { r = engine.gpu(kernel, { output: [256] }); });
// r.core === true, r.pending === true, r.wgsl is the emitted shader.
// The dispatch settles asynchronously; a reader inside change() re-runs when the resource cell is written.
```

`gpu(kernel, cfg)` returns a `GpuResource` immediately (classification + shader emission are synchronous); the
device dispatch settles on a later microtask and writes the resource cell, re-running any reactive reader.

A kernel is a `component k(…)` whose parameters are **thread coordinates** — 1-D `k(i)` over `output: [N]`,
2-D `k(x, y)` over `[W, H]`, or **3-D `k(x, y, z)` over `[W, H, D]`** (rank > 3, or an arity that does not
match the output's dimension count, is a loud gate reject).

A kernel body may use scalar arithmetic, `if`/ternary, a bounded `for (… of range(n))`, typed-array indexing
`a[i]` + `a.length`, and the full **`core`-profile numeric/vec/mat vocabulary**:

- **Scalar math** — `min`/`max`/`abs`/`sign`/`floor`/`ceil`/`round`/`clamp`/`trunc`/`degrees`/`radians`,
  `sqrt`/`pow`/`exp`/`exp2`/`log`/`log2`/`inverseSqrt`/`fract`/`step`/`mix`/`smoothstep`, the trig
  `sin`/`cos`/`tan`, inverse-trig `asin`/`acos`/`atan`/`atan2`, and hyperbolic `sinh`/`cosh`/`tanh`. Each maps
  to a shader intrinsic (with a domain guard where the interpreter has one) and applies **componentwise** to a
  `vec` argument.
- **vec/mat** — `vec2/3/4`, the square `mat2/3/4` + the six non-square `mat2x3`…`mat4x3` (`matCxR`, column-major),
  componentwise `+ - * /`, vec/mat–scalar scaling, **column-major** `mat*vec` / `mat*mat`, swizzles, plus
  `dot`/`cross`/`normalize`/`length`, `transpose`/`determinant`/`inverse`/`distance`/`reflect`/`refract`/`faceforward`,
  and the `vec4`-convention quaternion family `qmul`/`qconj`/`qinvert`/`qaxisangle`/`qrotate`/`qslerp`/`qmat`. WGSL
  has no native `inverse()` or quaternion type, so those are **hand-emitted** (a per-size `_invN`/`_qslerp`/`_qmat`
  prelude helper) — byte-checked against the interpreter oracle.

Strings, objects, dynamic arrays, `while`, helper calls, and **`rand()`** (it cannot match the deterministic
oracle) are **rejected** by the gate with a span-anchored `MLGPU-*` diagnostic (`resource.core === false`,
`resource.reasons` says why). A **static bounds-prover** additionally rejects a provably out-of-range index
(`MLGPU-INDEX-STATIC`), leaving data-dependent indices to the sampled oracle.

## Host API — `createGpuEngine()`

Drive the engine from host TypeScript without hand-wiring a host + engine + the settle dance. The kernel is
still authored in metael source (the emitters lower its AST — there is no JS-closure→shader path), but
everything around it is a plain function call.

```ts
import { createGpuEngine, settle } from '@metael/gpu';        // the API-first core — no interpreter dep
import { compileKernel } from '@metael/gpu/lang';             // the DSL binding — this subpath pulls the interpreter

const gpu = createGpuEngine();                       // real WebGPU→WebGL2→CPU ladder ({ cpuOnly: true } for tests)
const kernel = compileKernel(`
  const a = f32(1024, (i) => i)
  const b = f32(1024, (i) => i * 2)
  component add(i) { return a[i] + b[i] }
  add
`, gpu.host);
const r = await settle(() => gpu.dispatch(kernel, { output: [1024], verify: true }));
console.log(r.backend, r.value, r.wgsl, r.match?.ok);
gpu[Symbol.dispose]();
```

The core façade (`@metael/gpu`) is a thin, interpreter-free front door: one `dispatch` + the free settle helpers.
Turning source text into a kernel is the DSL binding, so `compileKernel` lives on the `./lang` subpath.

- `dispatch(kernel, cfg)` → the pending `GpuResource` synchronously; `cfg.mode` selects the kind (`'map'`
  default → a map kernel, `'reduce'` → a fold, `'histogram'` → a scatter). `.wgsl`/`.glsl`/`.core`/`.reasons`
  filled; `.value` after settle.
- the FREE `settle(() => gpu.dispatch(k, cfg))` (from `@metael/gpu`) → awaits the settled resource; `settled(r)`
  narrows `r.pending === false`; `subscribe(() => gpu.dispatch(k, cfg), onValue)` (also from `@metael/gpu`) → fires
  on pending then settled (guard `if (!r.pending)`), returns a stop disposer.
- `compileKernel(src, gpu.host)` (from `@metael/gpu/lang`) → the kernel `UserFn` (throws if the source isn't a
  function/component). NOTE it is on the `./lang` subpath — the core façade carries no interpreter.
- `gpuBuffer(data, gpu.host)` (from `@metael/gpu`) → wrap a plain `Float32Array | number[]` as a reduce/histogram
  input buffer without hand-boxing.
- `[Symbol.dispose]()` → frees the device + memo; `dispatch` throws afterward. The façade is a native
  `Disposable`, so a `using gpu = createGpuEngine()` declaration frees it at scope exit.

### Author a kernel in JS — `@metael/gpu/builder`

A host that would rather not write DSL source text can build the kernel in JS instead. The `./builder` subpath is a
TSL-style JS kernel builder — it authors the **same kernel AST the DSL parser produces** (proven AST-equivalent), so
a builder-authored kernel inherits the identical gate / emit / oracle / dispatch path:

```ts
import { kernel, call } from '@metael/gpu/builder';

const k = kernel((row, col) => row.add(col));        // the arrow-RETURN form: returns the per-cell value
const r = await settle(() => gpu.dispatch(k, { output: [W, H] }));
```

For loops/branches, the statement helpers `letVar` / `set` / `forRange` / `ifThen` / `ret` trace a body — e.g.
`kernel((n) => { const acc = letVar('acc', lit(0)); forRange(4, (i) => set(acc, acc.add(i))); ret(acc); })`.
`call('dot', a, b)` invokes a builtin/component head. The idiomatic form is the arrow-return
`kernel((row, col) => row.add(col))` — a returned `KNode` is the per-cell output write.

### Reading `r.value` — mind the pending window

`r.value` is `null` while `r.pending` is `true`, and only becomes the settled `number[]` after the dispatch
resolves and writes the resource cell. So only read `r.value` at a point where the resource is known settled:

- **Host API:** `settle()` already awaited, so `r.value` is the array. Reduce it in plain JS:
  ```ts
  const r = await settle(() => gpu.dispatch(kernel, { output: [8] }));   // kernel: x[i] * 2 over f32(8, i=>i)
  const sum = (r.value as number[]).reduce((a, b) => a + b, 0);   // 56
  ```
- **In a metael program (the DSL / a compute story):** reducing over `r.value` works. The robust,
  driver-independent idiom is to **keep the resource reachable from the value the program returns**, so any
  settle loop that inspects the returned value keeps re-evaluating until it is no longer pending:
  ```js
  const r = gpu(k, { output: [8] })
  { result: r, sum: reduce(r.value, (a, b) => a + b, 0) }   // ✅ keep `r` in the value
  ```
  Returning only a *projection* of a still-pending resource — e.g. `{ value: r.value }` — reads `r.value` as
  `null` on the pending pass, and the projected record carries no `pending` field for a value-inspecting loop to
  detect. A driver that gates on the *engine's* declared-resource state settles it anyway (the showcase
  playground's compute target does), but a value-inspecting driver would return `null` — so keep the resource
  reachable unless your driver tracks engine-pending.
- **Pure reduction? Prefer `gpuReduce`** — fold on the GPU with no per-element readback, rather than mapping then
  host-reducing (`identity` must be the reducer's neutral element: `0` sum, `1` product, `±Infinity` max/min):
  ```js
  const x = f32(8, (i) => i * 2)             // [0,2,4,6,8,10,12,14]
  component add(acc, v) { return acc + v }
  gpuReduce(add, { input: x, identity: 0 })  // → r.value === 56
  ```

### Reusable kernels bound to different inputs

Author the kernel once as a **factory** that closes over its inputs, then bind different buffers by calling
it again — the emitted shader is identical, so the pipeline cache reuses the compiled pipeline:

```ts
const makeKernel = compileKernel(`
  function makeKernel(a) { component ker(i) { return a[i] * 2 } return ker }
  makeKernel
`, gpu.host);
// makeKernel is a UserFn factory; curry it from host code with makeCallable (from @metael/lang), or from metael source.
```

For a **reactive** input, hold the current buffer in a signal and re-derive the kernel on change (a rebind
is a new kernel value, dispatched inside your own effect) — the engine re-dispatches with the new input.

The same engine powers both this host API and the in-DSL `gpu` / `gpuReduce` / `gpuHistogram` heads below —
the façade just wraps a `GpuEngine` + `RuntimeReactiveHost` and drives the `change()`/settle dance for you.

## Output shapes, precision, and I/O

`gpu(kernel, cfg)` shapes its result via `cfg`:

- **`outputType`** — `'array'` (default, a plain `number[]`), `'buffer'` (a frozen `f32` typed-array handle,
  zero-copy over the readback), or `'gpu-buffer'` (a **resident** `GpuBufferHandle` kept on-device for
  **pipelining** — kernel A's output feeds kernel B with no CPU round-trip when they run on the same backend;
  otherwise it lazily reads back).
- **`outputElement`** — `'f32'` (default) or `'vec2'`/`'vec3'`/`'vec4'` for a per-cell vector, written
  flat-interleaved (`output[cell*N + component]`).
- **`outputs: { name: {…} }`** — **multi-output**: a kernel returning a named object writes several output
  buffers (`resource.outputs.<name>`), one per key.
- **`precision`** — `'f32'` (default) or `'f16'` (WebGPU `shader-f16`; falls back to `f32` with a
  `resource.note` on a backend without the feature).

Inputs are **zero-copy** for an `f32` buffer (the live store is handed straight to the backend); `i32`/`u32`/
`f64` inputs convert once to the `f32` GPU storage type. The collection builtins (`map`/`filter`/`reduce`/…)
accept a typed array directly. On a runtime dispatch failure the engine **re-ladders** to the next backend
(a device can fault mid-run, not just at selection) rather than failing terminally.

## Beyond map kernels: reductions + histograms

Two dedicated heads add distinct kernel *kinds* (cross-thread cooperation the per-cell map can't express):

- **`gpuReduce(reducer, cfg)`** — a 2-arg associative **reducer** (`component add(acc, x) { return acc + x }`)
  + an input buffer + an `identity` (its **neutral** element), folded to a scalar. GPU legs are a WebGL2
  ping-pong tile reduction + a WGSL workgroup-shared tree; the CPU linear fold is the oracle. The reducer must
  be associative **and commutative** (the tree reorders operands).
- **`gpuHistogram(binMapper, cfg)`** — a 1-arg **bin-mapper** (`component binOf(x) { return x % 4 }`) scattering
  each input into `cfg.bins` buckets (`atomicAdd` on WGSL; WebGL2 has no fragment-shader atomics so it runs the
  CPU oracle, with a `resource.note`). Out-of-range bin indices are dropped.

## `verify` / `benchmark` are opt-in

A default dispatch is **GPU-only** — it returns the value + `gpuMs`. Two flags add cost only when asked:

- `verify: true` — after the dispatch, re-run a **sample** of output cells through the interpreter and
  ULP-tolerance-check the GPU output against it (populates `match`). A correctness self-check.
- `benchmark: true` — also run the whole sweep on the CPU backend to time a baseline (populates `cpuMs`, and
  `speedup` on a real GPU). A GPU-vs-CPU race.

Both are **off by default** — a CPU-side interpreter sweep and a second full dispatch on every run would defeat
the point of offloading to the GPU.

## The backend ladder

`selectBackend` verifies a **real adapter/device** at each rung and falls **WebGPU → WebGL2 → CPU**
(`navigator.gpu` being truthy is not enough — a headless runner silently lacks a working device). The WebGL2
backend runs a compute kernel as a fullscreen-quad **fragment** shader (WebGL2 has no compute stage): inputs
are `R32F` textures, the output an `RGBA32F` render target read back with `readPixels`. CPU is always the
floor, and because CPU-emit is the same code the interpreter runs, the CPU result is identical to the oracle
by construction.

## What's in the box

| Export | What it is |
|---|---|
| `GpuHostEnv` | a `HostEnvironment` resolving the `gpu` / `gpuReduce` / `gpuHistogram` heads to reactive resources — the front door |
| `GpuEngine`, `GpuConfig`, `GpuResource`, `GpuEngineDeps` | the engine: gate → emit → dispatch → oracle → memoized reactive resource (its `gpu` / `gpuReduce` / `gpuHistogram` methods) |
| `ReduceConfig`, `HistogramConfig` | the reduce / histogram configs |
| `gateKernel`, `gateReducer`, `gateBinMapper`, `GateVerdict` | the compute-lowerability gates (map kernel / associative reducer / bin-mapper) |
| `cpuReduce`, `cpuHistogram` | the CPU fold / scatter oracles |
| `checkStaticBounds`, `intervalOf`, `Interval` | the static out-of-bounds bounds-prover |
| `buildBindingTable`, `collectFreeNames`, `Binding`, `BindingTable` | free-name → coord/buffer/scalar/uniform/callee resolution |
| `checkCost`, `MAX_GPU_ALLOC`, `CPU_LIMITS`, `DeviceLimits` | the resource-cost gate (`MLGPU-ALLOC` before any dispatch) |
| `emitWgsl`, `emitGlsl`, `emitCpu`, `emitReduceWgsl`, `emitHistogramWgsl` | the emitters (one AST → WGSL / GLSL-ES-3.0 / an eval-free JS closure; + the reduce/histogram shader emitters) |
| `checkMatch`, `MatchVerdict`, `OracleInput` | the sampled interpreter oracle |
| `selectBackend`, `Backend`, `BackendKind`, `DispatchInput`, `DispatchResult` | the device seam + the WebGPU→WebGL2→CPU ladder (+ `dispatchReduce`/`dispatchHistogram`/`retainOutput`/`residentInputs`) |
| `makeCpuBackend`, `tryWebGpuBackend`, `tryWebGl2Backend` | the three backends (each verifies a real adapter; CPU is the floor) |
| `compsOf`, `OutputElement` | the per-cell output element → component count |
| `kernelHash` | the kernel-hash ingredient of the dispatch-memo key (so distinct kernels never collide on the same output shape) |

## Diagnostics

This package owns the `MLGPU-*` diagnostic codes: `MLGPU-NOT-LOWERABLE` (a kernel the gate rejects, with the
reason + span), `MLGPU-ALLOC` (the cost/dispatch-limit gate — an over-limit or malformed output shape),
`MLGPU-DISPATCH` (a device-dispatch failure), `MLGPU-INDEX-STATIC` (a provably out-of-range index),
`MLGPU-OUTPUT-SHAPE` (a bad vec/multi-output return), `MLGPU-BAD-INPUT` / `MLGPU-INPUT-WRITE` /
`MLGPU-INPUT-UNAVAILABLE` (input validation), `MLGPU-REDUCER-ARITY` / `MLGPU-BINMAPPER-ARITY` (reduce/histogram
callable shape), `MLGPU-USE-AFTER-DISPOSE` / `MLGPU-READBACK-SHORT` (resident-buffer lifecycle).

## Develop

```shell
npm run -w @metael/gpu typecheck
npm run -w @metael/gpu build      # → dist/ (.js + .d.ts, one per source module)
npx vitest run packages/gpu       # the node suite
npx vitest run --project browser packages/gpu   # the real-adapter WebGPU/WebGL2 proofs (Chromium)
```

Imports **only** `@metael/lang`, `@metael/math` (the numeric vocabulary it lowers), and `@metael/runtime` (enforced by a boundary test). The device backends emit
shader strings for the GPU driver — they never `eval` a kernel-derived string in JS; the CPU emitter + the
oracle run the eval-free interpreter. See [../../AGENTS.md](../../AGENTS.md) for the load-bearing invariants.

## License

MIT.
