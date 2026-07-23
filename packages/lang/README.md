# @metael/lang

[![metael](https://img.shields.io/badge/project-metael-blueviolet.svg?style=flat-square&logo=github)](https://github.com/andykswong/metael)
[![npm](https://img.shields.io/npm/v/@metael/lang?style=flat-square&logo=npm)](https://www.npmjs.com/package/@metael/lang)
[![license: MIT](https://img.shields.io/badge/License-MIT-red.svg?style=flat-square)](./LICENSE)

**The eval-free, port-injected JS/ES interpreter kernel** — the language layer of the [metael](../../README.md) substrate. Zero runtime dependencies; imports nothing domain-specific, so a domain gets the whole language by injecting *which heads exist* + *what they build*. (Its reactive runtime companion is [`@metael/runtime`](../runtime/README.md).)

```
source ──lex──▶ tokens ──parse──▶ AST (Expr/Stmt/Program) ──evaluate / lowerEntry──▶ host values
                                     │ discriminated unions        │ via HostEnvironment.resolveCall
                                     │ every Expr/Stmt span-tagged  │ reactive `let` → ReactiveHost cells
```

## Install

```shell
npm install @metael/lang    # zero runtime dependencies
```

## Usage

The shortest runnable program — parse and evaluate an expression to a value:

```ts
import { evaluateProgram, PlainStorageHost, RecordingHostEnv } from '@metael/lang';

const { value, diagnostics } = evaluateProgram('1 + 2 * 3;', {
  host: new PlainStorageHost(),
  env: new RecordingHostEnv(),
});
// value === 7, diagnostics === []
```

For tooling, `lex(source)` and `parseProgram(source)` are both **total** (they return diagnostics, never throw), so an editor or highlighter can consume tokens/AST off partial source. To run real domain vocabulary instead of the bundled test doubles, implement the three host ports (below) and pass them as `host` / `env`.

## At a glance

- **The `.` entry is the kernel.** `evaluateProgram` does lex → parse → AST → eval-free tree-walk (fuel/time/depth **budgeted**, and **never-throwing** — author errors, budget trips, unknown calls, and forbidden-key access all become diagnostics + a safe `null`). Evaluation is genuinely eval-free (a source-scan test asserts no `eval`/`new Function`/string-timers/`GeneratorFunction`). `lowerEntry` is the generic, view-free child-collection walk for component-shaped programs.
- **`@metael/lang/profile` is the vocabulary-metadata / tooling layer** — a separate subpath (`Profile` / `composeProfiles` / `classifyProfile` / `defineBuiltin`) that describes a vocabulary *by name* for language services + codegen gates. Kept off the `.` entry so the kernel stays lean and privileges no vocabulary (a boundary test guards it).
- **The host-injection seam** is the package's whole point: a domain implements `HostEnvironment` / `ReactiveHost` / `KeyMinter` and gets the language for free.

See [AGENTS.md](./AGENTS.md) for the source map (key exports by purpose) + the deep host-seam contract.

## The host-injection seam

`@metael/lang` never builds a domain value — `HostValue` is opaque. A domain supplies three port interfaces (bundled test doubles implement all three, so the kernel is unit-testable with no domain present):

- **`HostEnvironment`** — `resolveCall(head, …)` builds a value for a head, or defers so the kernel emits a tagged `LangWrapper`.
- **`ReactiveHost`** — allocates/reads/writes reactive `let` cells and runs leaf effects within owner scopes.
- **`KeyMinter`** — mints identity keys so a re-derive reconciles by identity.

## Develop

```shell
npm run -w @metael/lang typecheck
npm run -w @metael/lang build     # → dist/ (.js + .d.ts, one per source module)
npx vitest run packages/lang      # the suite
```

From the repo root, `npm run docs:api:check` is the doc-coverage gate (0 undocumented exported symbols — every export needs a doc comment), and `npm run prepublishOnly` is the full pre-publish gate (`clean → build:packages → typecheck → lint → test → docs:api:check`).

Domain-neutral by construction: the generic child-collection walk `lowerEntry` ships in THIS package (view-free lang machinery). Only domain-specific lowering (a domain's own View/scene-graph construction) and the reactive re-derive + keyed-diff belong to [`@metael/runtime`](../runtime/README.md). See [AGENTS.md](./AGENTS.md) for the package architecture and [../../AGENTS.md](../../AGENTS.md) for the load-bearing invariants and editing guardrails.

## License

MIT — see [LICENSE](./LICENSE).
