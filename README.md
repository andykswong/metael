# metael

[![metael](https://img.shields.io/badge/project-metael-blueviolet.svg?style=flat-square&logo=github)](https://github.com/andykswong/metael)
[![npm](https://img.shields.io/npm/v/@metael/lang?style=flat-square&logo=npm)](https://www.npmjs.com/package/@metael/lang)
[![license: MIT](https://img.shields.io/badge/License-MIT-red.svg?style=flat-square)](./LICENSE)
[![build](https://img.shields.io/github/actions/workflow/status/andykswong/metael/build.yaml?style=flat-square)](https://github.com/andykswong/metael/actions/workflows/build.yaml)
[![codecov](https://img.shields.io/codecov/c/github/andykswong/metael?style=flat-square&logo=codecov)](https://codecov.io/gh/andykswong/metael)

**A generic, eval-free, reactive scripting-language kernel — the language, reactivity, and host-injection seam that domain frameworks build on.**

metael owns the *domain-agnostic* core and nothing else: a legible JS/ES-syntax surface run by an **eval-free tree-walking interpreter**; a serializable, editable **reactive-component AST**; a **fine-grained reactive runtime**; and the **host-injection seam** by which a domain supplies *which words exist* and *what they build*. It knows how to declare, compose, resolve, and react — never which vocabulary exists or what it renders to. A domain framework = **metael + its vocabulary + its renderer**, so the same kernel can drive a virtual DOM, a scene graph, or a pure data pipeline.

- **Eval-free & sandbox-safe** — no `eval`/`new Function`; a program can't reach host globals, and it's budgeted so it can't hang or exhaust memory. Safe to run arbitrary source inline.
- **Deterministic** — `result = f(source, data, seed, state)`; the only randomness is seeded. Same inputs → same output, every time. Machine-verifiable.
- **Immutable by construction** — everything a program creates is deep-frozen; it can't mutate injected data.

📖 **New to the language? Read [GUIDE.md](./GUIDE.md)** — the practical, example-driven tour of the syntax, builtins, and AST.

## Packages

| Package | What it is | Depends on |
|---|---|---|
| **`@metael/lang`** | The eval-free interpreter kernel: lexer → parser → discriminated-union AST → tree-walking evaluator (fuel/time/depth budgets + prototype guards), the host-injection port **interfaces** + test doubles, the generic child-collection walk, and the registry seam (the `range` loop primitive is the kernel's only intrinsic; general vocabulary is injected). | nothing (zero runtime deps) |
| **`@metael/math`** | The numeric standard library: a zero-dependency **core** (scalar/vec/mat/quat/transform math, column-major, out-param, precision-follows-store) usable without the language, plus `@metael/math/lang` — the boxed/immutable/reactive custom-value binding, `MATH_BUILTINS`, the lowering catalog, and the capability classifier. | core: nothing · `@metael/math/lang`: `@metael/lang` |
| **`@metael/std`** | The general standard library: `STD_BUILTINS` — collections (`map`/`filter`/`reduce`/`sort`/…), string (`split`/`join`/`chars`/…), structural (`keys`/`values`/`entries`/`object`/`has`), random (`rand`), datetime (`now`/`monotonic` via an injected host clock). | `@metael/lang` |
| **`@metael/runtime`** | The fine-grained reactive core (`signal`/`memo`/`effect` + a synchronous `change()` batch/flush + a converge guard), the generic **keyed-list diff** (add/remove/move + teardown), the real `ReactiveHost`, and the one-shot `derive()` composition root. | `@metael/lang` + `@vue/reactivity` |
| **`@metael/vdom`** | A Preact-signals-style virtual DOM built entirely on the kernel. Two subpaths: `.` the API-first core (`render`/`h` over host-authored VNodes — no interpreter dep) + `./lang` the DSL binding (`renderSource` renders a `component` written in the metael DSL to real, live DOM). A worked example of a full domain on top of metael. | `@metael/{lang,runtime}` |
| **`@metael/gpu`** | An eval-free, verifiable GPU-compute engine: gates lowerability, emits WGSL/GLSL/CPU, runs on a real WebGPU→WebGL2→CPU adapter ladder, and returns a reactive resource verified against the interpreter oracle. Three subpaths: `.` the API-first core (`createGpuEngine`/`dispatch` — no interpreter dep) + `./lang` the DSL binding (`compileKernel`/`GpuHostEnv`) + `./builder` a TSL-style JS kernel builder. | `@metael/{lang,math,runtime}` |

## Install

```shell
npm install @metael/lang                 # the kernel alone (zero runtime deps)
npm install @metael/math                  # + the numeric stdlib (core = zero deps; @metael/math/lang pulls @metael/lang)
npm install @metael/std                   # + the general stdlib builtins (pulls @metael/lang)
npm install @metael/runtime               # + the reactive runtime (pulls @metael/lang)
npm install @metael/vdom                  # + the VDOM domain (pulls @metael/{lang,runtime})
npm install @metael/gpu                   # + the verifiable GPU-compute engine (pulls @metael/{lang,runtime})
```

The layering is `@metael/lang` (the kernel alone) → the standard libraries `@metael/math` + `@metael/std` (vocabulary injected at `evaluateProgram`) and `@metael/runtime` (+ reactivity) → `@metael/vdom` (+ the VDOM domain); install the layer you need and its dependencies come with it. `@metael/math`'s **core** subpath (`@metael/math`) is zero-dependency and usable without the language at all. To develop against the sources instead, clone this monorepo (see [Develop](#develop)) and `npm run build:packages`.

Requires Node 24+ / a 2024+ browser (uses native `Symbol.dispose`). ESM-only. `@metael/*` types that own resources (the gpu engine façade, its pipeline cache, the reactive host scopes) implement native `Disposable` — use them with `using` or call `[Symbol.dispose]()`. Runtimes without native `Symbol.dispose`/`DisposableStack` (e.g. Safari) need the `disposablestack/auto` polyfill (imported by the showcase app's entry points).

## Usage

### Evaluate a program to a value (`@metael/lang`)

Run source as a pure computation. You supply a `HostEnvironment` (resolves any non-builtin call), a `ReactiveHost` (cells/effects — a plain double is fine for pure eval), optional `data`, and a `seed`.

```ts
import { evaluateProgram, PlainStorageHost, RecordingHostEnv } from '@metael/lang';

const { value, diagnostics } = evaluateProgram(
  `map(data.items, (it) => it.price * 2)`,
  {
    data: { items: [{ price: 3 }, { price: 5 }] },
    seed: 1,
    host: new PlainStorageHost(),   // stores reactive cells; no domain needed for pure eval
    env: new RecordingHostEnv(),    // resolves calls; the doubles answer permissively
  },
);

value;          // → [6, 10]
diagnostics;    // → []  (a fail-loud diagnostic list — never thrown exceptions)
```

`evaluateProgram` **never throws**: author errors, budget trips, and unknown calls all come back as `ML-LANG-*` diagnostics plus a safe value (often `null`).

### Lex or parse for tooling

```ts
import { lex, parseProgram } from '@metael/lang';

lex('map(xs, (x) => x)');            // → tokens (drives syntax highlighting)
const { program, diagnostics } = parseProgram('const x = 1; x + 2');
program.stmts;                        // → the AST (discriminated-union nodes, each span-tagged)
```

### Provide your own vocabulary (a custom host)

The injection seam is the heart of metael: a domain implements a small `HostEnvironment` (plus, for stateful output, a `ReactiveHost` + `KeyMinter`) and gets the whole language/AST/reactivity/determinism substrate for free. `resolveCall` turns a head into a value — return `{ handled: true, value }` for a node, `{ handled: true, value, kind: 'value' }` for a pure scalar/record value usable in expression position, or `{ handled: false }` to let metael emit a wrapper.

```ts
import type { HostEnvironment, Arg } from '@metael/lang';

const env: HostEnvironment = {
  resolveCall(head, key, args: Arg[]) {
    if (head === 'rgb') {
      const [r, g, b] = args.map((a) => a.value as number);
      return { handled: true, kind: 'value', value: { r, g, b } };   // a pure value builtin
    }
    return { handled: false };
  },
};
// evaluateProgram('rgb(255, 0, 0).r', { host, env, … }) → 255
```

See [GUIDE.md](./GUIDE.md) sections 8–10 for the composition model and the port shapes, and the generated API docs (`npm run docs:api`). The two sections below — a reactive UI and GPU compute — are full domains built on exactly this seam. Each core package exposes a host-callable API-first surface; the metael-DSL binding lives behind that package's `./lang` subpath, so importing the core carries no interpreter dependency.

### Render a reactive UI (`@metael/vdom`)

Write a `component` in the metael DSL; `renderSource` (the DSL binding, on the `./lang` subpath) renders it to real DOM and keeps it live — a reactive `let` read by one attribute patches only that node; a change to the tree's shape reconciles by key. (The `@metael/vdom` core exposes the interpreter-free `render`/`h` API over host-authored VNodes.)

```ts
import { renderSource } from '@metael/vdom/lang';

// The entry component is named `Story` by default (override with the `entry` option).
const source = `
  component Story() {
    let count = 0
    div {
      button({ onClick: () => { count = count + 1 } }, "+")
      span("clicked " + count + " times")
    }
  }
`;

const container = document.getElementById('app')!;
const handle = renderSource(source, container, {});   // third arg = RenderSourceOptions (all fields optional; e.g. { seed, data, entry })
// … later:
handle.unmount();
```

*(`div`/`button`/`span` are `@metael/vdom`'s vocabulary — a lowercase head becomes an element; a different host (like the `rgb` one above) defines different words. `count` is read only by `span`, so a click patches just that text node — no re-render.)*

### Compute on the GPU (`@metael/gpu`)

Write a compute kernel as a metael `component`; the engine gates lowerability, emits WGSL/GLSL/CPU, dispatches on a real WebGPU→WebGL2→CPU ladder, and (with `verify`) checks the GPU output against the interpreter oracle.

```ts
import { createGpuEngine, settle } from '@metael/gpu';        // the API-first core — no interpreter dep
import { compileKernel } from '@metael/gpu/lang';             // the DSL binding — this subpath pulls the interpreter

const gpu = createGpuEngine();                       // real WebGPU→WebGL2→CPU ladder ({ cpuOnly: true } for tests)
const kernel = compileKernel(`
  const a = f32(1024, (i) => i)
  component double(i) { return a[i] * 2 }
  double
`, gpu.host);
const r = await settle(() => gpu.dispatch(kernel, { output: [1024], verify: true }));
r.backend;    // 'webgpu' | 'webgl2' | 'cpu'
r.value;      // the settled Float32 output as number[]
r.match?.ok;  // true — the GPU output matched the interpreter oracle (verify:true)
gpu[Symbol.dispose]();
```

See [`packages/gpu/README.md`](packages/gpu/README.md) for the full host API, the `./builder` JS kernel builder, and the in-DSL `gpu`/`gpuReduce`/`gpuHistogram` heads.

## Develop

```shell
npm install                 # workspace devDeps (TS 6, Vite 8, Vitest 4, ESLint 10); no runtime deps
npm run typecheck           # tsc --noEmit (root + every package)
npm run lint                # eslint (0 warnings)
npm run build:packages      # build @metael/* → dist/ (.js + .d.ts)
npm test                    # vitest run (node + Playwright/Chromium browser projects)
npm run docs:api            # generate the TypeDoc API reference
```

`@metael/{lang,runtime}` are pure logic, fully CPU-unit-tested. `@metael/vdom` adds a Chromium (Playwright) project for real-DOM proofs (a node survives a reorder, focus/selection persist, event delegation fires, unsafe URLs are dropped, removed subtrees are torn down). A `safety.test.ts` source-scan asserts the kernel stays eval-free; a `sandbox.test.ts` suite proves a program can't escape the sandbox; per-package `boundary.test.ts` files enforce the dependency seams. The tests are the conformance bar — keep them green and add one with any change.

See [AGENTS.md](./AGENTS.md) for architecture, conventions, and editing guardrails.

## License

MIT — see [LICENSE](./LICENSE).
