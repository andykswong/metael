# @metael/gpu

[![metael](https://img.shields.io/badge/project-metael-blueviolet.svg?style=flat-square&logo=github)](https://github.com/andykswong/metael)
[![npm](https://img.shields.io/npm/v/@metael/gpu?style=flat-square&logo=npm)](https://www.npmjs.com/package/@metael/gpu)
[![license: MIT](https://img.shields.io/badge/License-MIT-red.svg?style=flat-square)](./LICENSE)

**An eval-free, verifiable GPU-compute engine** built on the [metael](../../README.md) reactive kernel. Write a compute kernel as a metael `component`; `@metael/gpu` decides whether it is GPU-lowerable, emits a real compute shader, runs it on a real adapter, and returns a reactive resource. Because the kernel is a **real parsed AST** (not a `.toString()`-and-transpile trick), there is **no `new Function`, no code-injection surface** — user source is *data*, never executed as host code — and the shipped metael interpreter is the correctness **oracle** the GPU result can be checked against.

```
component ──gate──▶ lowerable? ──emit──▶ WGSL │ GLSL-ES-3.0 │ eval-free CPU
   kernel   (over builtins +     (one AST, three targets)
            descriptor lowering)         │
                                dispatch ──▶ WebGPU → WebGL2 → CPU ──▶ reactive resource
                                             (each verifies a REAL adapter; CPU is the oracle floor)
```

## Install

```shell
npm install @metael/gpu     # pulls @metael/{lang,math,runtime}
```

Requires Node 24+ / a 2024+ browser (native `Symbol.dispose`). ESM-only. The engine façade is a native `Disposable` — use it with `using` or call `[Symbol.dispose]()`.

## Usage

Drive the engine from host TypeScript with `createGpuEngine()` + the free `settle()` helper — no hand-wiring of a host, engine, and the reactive settle dance. The kernel is authored in metael source (`compileKernel` lowers its AST); everything around it is a plain function call.

```ts
import { createGpuEngine, settle } from '@metael/gpu';   // the API-first core — no interpreter dep
import { compileKernel } from '@metael/gpu/lang';        // the DSL binding — this subpath pulls the interpreter

using gpu = createGpuEngine();                           // real WebGPU→WebGL2→CPU ladder ({ cpuOnly: true } for tests)
const kernel = compileKernel(`
  const a = f32(1024, (i) => i)
  const b = f32(1024, (i) => i * 2)
  component add(i) { return a[i] + b[i] }
  add
`, gpu.host);

const r = await settle(() => gpu.dispatch(kernel, { output: [1024], verify: true }));
console.log(r.backend, r.value, r.match?.ok);            // e.g. 'webgpu', [0, 3, 6, …], true
```

`dispatch(kernel, cfg)` returns the pending `GpuResource` synchronously (classification + shader emission are synchronous); `settle()` awaits the device dispatch and re-reads until `r.pending === false`, so `r.value` is the settled output.

**Author a kernel in JS** — the `./builder` subpath is a TSL-style builder that emits the *same* kernel AST the DSL parser produces, so it inherits the identical gate/emit/oracle/dispatch path:

```ts
import { kernel } from '@metael/gpu/builder';

const k = kernel((row, col) => row.add(col));            // arrow-return: the returned KNode is the per-cell value
const r = await settle(() => gpu.dispatch(k, { output: [W, H] }));
```

**In a metael program**, the same engine backs the `gpu` / `gpuReduce` / `gpuHistogram` heads directly — see [AGENTS.md](./AGENTS.md) for the head vocabulary and `GpuHostEnv`.

## At a glance

- **Three subpaths** — `.` (the interpreter-free host API: `createGpuEngine`/`dispatch` + the free `settle`/`subscribe`/`settled`/`gpuBuffer`), `./lang` (the DSL binding — `compileKernel` + `GpuHostEnv`; importing it pulls the interpreter), `./builder` (the JS kernel builder).
- **What a kernel can contain** — scalar arithmetic, `if`/ternary, a bounded `for (… of range(n))`, typed-array indexing, and the full `core`-profile numeric/vec/mat/quat vocabulary. Strings, objects, `while`, helper calls, and `rand()` are rejected with a span-anchored `MLGPU-*` diagnostic. See [AGENTS.md](./AGENTS.md) for the full lowerable vocabulary.
- **Backend ladder** — `selectBackend` verifies a *real* adapter at each rung and falls **WebGPU → WebGL2 → CPU**; the eval-free CPU emitter runs the same code the interpreter does, so it is the oracle floor.
- **`verify` / `benchmark` are opt-in** — a default dispatch is GPU-only; `verify: true` re-runs a sample through the interpreter and ULP-checks it, `benchmark: true` times a CPU baseline. Both off by default so offloading isn't defeated.
- **Output shapes** — `outputType` (`array`/`buffer`/resident `gpu-buffer` for zero-copy pipelining), `outputElement` (`f32`/`vec2..4`), named multi-output `outputs`, and `precision` `f16`/`f32`. Detailed in [AGENTS.md](./AGENTS.md).

**Boundary:** imports **only** `@metael/lang`, `@metael/math`, and `@metael/runtime` (enforced by a boundary test); never `@metael/vdom`. The core façade (`.`) carries no interpreter dependency — only the `./lang` subpath does.

## Develop

```shell
npm run -w @metael/gpu typecheck
npm run -w @metael/gpu build      # → dist/ (.js + .d.ts, one per source module)
npx vitest run packages/gpu       # the node suite
npx vitest run --project browser packages/gpu   # the real-adapter WebGPU/WebGL2 proofs (Chromium)
```

From the repo root, `npm run docs:api:check` is the doc-coverage gate (every exported symbol needs a doc comment) and `npm run prepublishOnly` runs the full one-shot gate — `clean → build:packages → typecheck → lint → test → docs:api:check`.

See the root [README.md](../../README.md) for the package map, [AGENTS.md](./AGENTS.md) for this package's architecture and change guidance, and the root [AGENTS.md](../../AGENTS.md) for the load-bearing kernel invariants.

## License

MIT — see [LICENSE](./LICENSE).
