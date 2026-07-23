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

## Hero example

Run source as a pure computation. You supply a `HostEnvironment` (resolves any non-builtin call), a `ReactiveHost` (cells/effects — a plain double is fine for pure eval), optional `data`, and a `seed`:

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

`evaluateProgram` **never throws**: author errors, budget trips, and unknown calls all come back as `ML-LANG-*` diagnostics plus a safe value (often `null`). For tooling, `lex(source)` and `parseProgram(source)` (both from `@metael/lang`) hand back tokens and the span-tagged AST.

## Install

```shell
npm install @metael/lang      # the kernel alone (zero runtime deps)
npm install @metael/math      # + the numeric stdlib (core = zero deps; @metael/math/lang pulls @metael/lang)
npm install @metael/std       # + the general stdlib builtins (pulls @metael/lang)
npm install @metael/runtime   # + the reactive runtime (pulls @metael/lang)
npm install @metael/vdom      # + the VDOM domain (pulls @metael/{lang,runtime})
npm install @metael/gpu       # + the verifiable GPU-compute engine (pulls @metael/{lang,math,runtime})
npm install @metael/lsp       # + the language services (pulls @metael/lang + vscode-languageserver-protocol)
```

The layering is `@metael/lang` (the kernel alone) → the standard libraries `@metael/math` + `@metael/std` (vocabulary injected at `evaluateProgram`) and `@metael/runtime` (+ reactivity) → `@metael/vdom` (+ the VDOM domain); install the layer you need and its dependencies come with it. `@metael/math`'s **core** subpath is zero-dependency and usable without the language at all.

ESM-only; requires Node 24+ / a 2024+ browser (native `Symbol.dispose`). Types that own resources implement native `Disposable` (use `using` or `[Symbol.dispose]()`); runtimes lacking it (e.g. Safari) need the `disposablestack/auto` polyfill.

## Packages

| Package | What it is | Depends on |
|---|---|---|
| **`@metael/lang`** | The eval-free interpreter kernel: lexer → parser → discriminated-union AST → tree-walking evaluator (fuel/time/depth budgets + prototype guards), the host-injection port **interfaces** + test doubles, the generic child-collection walk, and the registry seam (`range` is the kernel's only intrinsic; vocabulary is injected). Its `@metael/lang/profile` subpath is the vocabulary-metadata layer for tooling (`Profile`/`composeProfiles`/`classifyProfile`/`defineBuiltin`). | nothing (zero runtime deps) |
| **`@metael/math`** | The numeric standard library: a zero-dependency **core** (scalar/vec/mat/quat/transform math, column-major, out-param) usable without the language, plus `@metael/math/lang` — the boxed/immutable/reactive custom-value binding, `MATH_BUILTINS`, and `mathProfile`. | core: nothing · `@metael/math/lang`: `@metael/lang` |
| **`@metael/std`** | The general standard library: `STD_BUILTINS` — collections (`map`/`filter`/`reduce`/`sort`/…), string (`split`/`join`/`chars`/…), structural (`keys`/`values`/`entries`/`object`/`has`), random (`rand`), datetime (`now`/`monotonic` via an injected host clock). | `@metael/lang` |
| **`@metael/runtime`** | The fine-grained reactive core (`signal`/`memo`/`effect` + a synchronous `change()` batch/flush + a converge guard), the generic **keyed-list diff** (add/remove/move + teardown), the real `ReactiveHost`, and the one-shot `derive()` composition root. | `@metael/lang` + `@vue/reactivity` |
| **`@metael/vdom`** | A Preact-signals-style virtual DOM built on the kernel. Two subpaths: `.` the API-first core (`render`/`h` over host-authored VNodes — no interpreter dep) + `./lang` the DSL binding (`renderSource` renders a `component` written in the metael DSL to real, live DOM). A worked example of a full domain on metael. | `@metael/{lang,runtime}` |
| **`@metael/gpu`** | An eval-free, verifiable GPU-compute engine: gates lowerability, emits WGSL/GLSL/CPU, runs on a real WebGPU→WebGL2→CPU adapter ladder, and returns a reactive resource verified against the interpreter oracle. Subpaths: `.` the API-first core (`createGpuEngine`/`dispatch`) + `./lang` the DSL binding + `./builder` a TSL-style JS kernel builder. | `@metael/{lang,math,runtime}` |
| **`@metael/lsp`** | metael language services — a **protocol-free analysis engine** (`@metael/lsp/service`: diagnostics/completion/hover/signature/semantic-tokens/folding/selection/format/capability-lens over a composed `Profile`, in char offsets) + an **LSP wire-protocol shell** (`createServer`) + a **browser Worker transport** (`@metael/lsp/worker`). | `@metael/lang` + `vscode-languageserver-protocol` |

## Usage

Each core package exposes a host-callable, API-first surface; the metael-DSL binding lives behind that package's `./lang` subpath, so importing the core carries no interpreter dependency. See each package's README for depth.

### Provide your own vocabulary (a custom host)

The injection seam is the heart of metael: a domain implements a small `HostEnvironment` (plus, for stateful output, a `ReactiveHost` + `KeyMinter`) and gets the whole language/AST/reactivity/determinism substrate for free. `resolveCall` turns a head into a value — `{ handled: true, value }` for a node, `{ handled: true, value, kind: 'value' }` for a pure value usable in expression position, or `{ handled: false }` to let metael emit a wrapper.

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

See [GUIDE.md](./GUIDE.md) sections 8–10 for the composition model and the port shapes.

### Render a reactive UI (`@metael/vdom`)

Write a `component` in the metael DSL; `renderSource` (the `./lang` binding) renders it to live DOM — a reactive `let` read by one attribute patches only that node; a shape change reconciles by key.

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

const handle = renderSource(source, document.getElementById('app')!, {});   // 3rd arg = RenderSourceOptions (all optional)
// … later: handle.unmount();
```

See [`packages/vdom/README.md`](packages/vdom/README.md) for the interpreter-free `render`/`h` core.

### Compute on the GPU (`@metael/gpu`)

Write a compute kernel as a metael `component`; the engine gates lowerability, emits WGSL/GLSL/CPU, dispatches on a real WebGPU→WebGL2→CPU ladder, and (with `verify`) checks the output against the interpreter oracle.

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

### Editor tooling (`@metael/lsp`)

Diagnostics, completion, hover, and semantic tokens over a composed vocabulary `Profile` — a protocol-free analysis engine, an LSP wire-protocol shell, and a browser Web Worker transport. See [`packages/lsp/README.md`](packages/lsp/README.md).

## Develop

```shell
npm install                 # workspace devDeps (TS 6, Vite 8, Vitest 4, ESLint 10); no runtime deps
npm run typecheck           # tsc --noEmit (root + every package)
npm run lint                # eslint (0 warnings)
npm run build:packages      # build every @metael/* package → dist/ (.js + .d.ts); the ":packages" name
                            #   is deliberate — it's the workspace-fanout build, distinct from a single
                            #   package's own `build` (and from `npm run build -w @metael/site`)
npm test                    # vitest run (node + Playwright/Chromium browser projects)
npm run docs:api            # generate the TypeDoc API reference
npm run docs:api:check      # gate: fail if any exported symbol is undocumented (0 required)
```

Run **`npm run prepublishOnly` as the one-shot final gate** once a piece of work is complete — it runs `clean → build:packages → typecheck → test → docs:api:check`, exactly the bar a release must clear (npm also runs it automatically before `npm publish`). Lint isn't a separate step there: each package's `build` is `prebuild`-linted, so `build:packages` already lints every published package.

Tests are Vitest (a `node` project + a Playwright/Chromium `browser` project) and are the conformance bar — keep them green and add one with any change. See [AGENTS.md](./AGENTS.md) for the architecture, the standing safety/sandbox/boundary guardrails, and the editing conventions.

## License

MIT — see [LICENSE](./LICENSE).
