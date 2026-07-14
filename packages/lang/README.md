# @metael/lang

[![metael](https://img.shields.io/badge/project-metael-blueviolet.svg?style=flat-square&logo=github)](https://github.com/andykswong/metael)
[![npm](https://img.shields.io/npm/v/@metael/lang?style=flat-square&logo=npm)](https://www.npmjs.com/package/@metael/lang)
[![license: MIT](https://img.shields.io/badge/License-MIT-red.svg?style=flat-square)](./LICENSE)

**The eval-free, port-injected JS/ES interpreter kernel** — the language layer of the [metael](../../README.md) substrate. Zero runtime dependencies; imports nothing domain-specific. (Its reactive runtime companion is [`@metael/runtime`](../runtime/README.md).)

`@metael/lang` turns source text into a reactive-component AST and evaluates it against a domain's host, staying entirely vocabulary-agnostic:

```
source ──lex──▶ tokens ──parse──▶ AST (Expr/Stmt/Program) ──evaluate / lowerEntry──▶ host values
                                     │ discriminated unions        │ via HostEnvironment.resolveCall
                                     │ every Expr/Stmt span-tagged  │ reactive `let` → ReactiveHost cells
```

It ships the **interpreter** + the generic **child-collection walk** (`lowerEntry` — instantiate the entry component, child-collect its body, resolve heads, mint keys, emit `Region`/`Wrapper`) + the host-injection port **interfaces** + **test doubles**. The port *implementations* and the fine-grained reactive runtime (which drives `lowerEntry` inside a `change()` boundary) live in the separate `@metael/runtime` package.

## What's in the box

| Export | What it is |
|---|---|
| `lex`, `Token`, `TokenType`, `LexResult` | the tokenizer (fail-soft; `ML-LANG-LEX` diagnostics) |
| `parseExpr`, `parseProgram`, `Parser`, `ParseProgramResult` | recursive-descent parser (`MAX_PARSE_DEPTH` guard; `ML-LANG-PARSE`) |
| `Expr`, `Stmt`, `Program`, `Pattern`, `BinOp`, `FORBIDDEN_KEYS` | the discriminated-union AST + the member/key security guard |
| `evaluateProgram`, `EvalOptions`, `EvalResult` | the eval-free tree-walker + fuel/time/depth budgets (never-throw); binds intrinsic seeded `rand`/`range` from `EvalOptions.seed` |
| `lowerEntry`, `LowerOptions`, `LowerResult` | the generic child-collection walk — instantiate the entry component, child-collect its body, resolve heads via `HostEnvironment`, mint keys via `KeyMinter`, emit `Region`/`Wrapper`; view-free (builds no domain node) |
| `DEFAULT_MAX_STEPS`, `DEFAULT_MAX_TIME_MS`, `DEFAULT_MAX_DEPTH`, `MAX_STRING_LENGTH` | the budget constants |
| `Environment`, `BindingMeta` | chained lexical scope + binding metadata |
| `makeSeededRng`, `range`, `MAX_RANGE` | the seeded-PRNG primitive (mulberry32) + bounded range |
| `makeDiagnostic`, `Diagnostic`, `SourceSpan`, `LiteralValue` | the diagnostic model |
| `HostEnvironment`, `ReactiveHost`, `KeyMinter` | the three host-injection port **interfaces** a domain implements |
| `Arg`, `Region`, `LangWrapper`, `Scope`, `HostValue`, `CellRef`, `EffectRegion` | the value contract across the seam |
| `region`, `isRegion`, `wrapper`, `isWrapper`, `didYouMean` | port helpers (`didYouMean` = Levenshtein-≤2 suggestion for fail-loud head resolution) |
| `PlainStorageHost`, `RecordingHostEnv`, `PathKeyMinter` | **test doubles** — run the kernel with no domain present |

## The host-injection seam

`@metael/lang` never builds a domain value — `HostValue` is opaque. A domain supplies three interfaces:

- **`HostEnvironment.resolveCall(head, key, args: Arg[], children, span)`** → `{ handled: true; value }` (the domain built a value) or `{ handled: false }` (metael emits a tagged `LangWrapper`). Each `Arg` keeps the parser's `{ value; name?; reactive? }` — the name-vs-position info is preserved, not discarded; the domain interprets roles. An optional `knownHeads` set enables fail-loud + `didYouMean` on a typo'd head.
- **`ReactiveHost`** — `allocateCell` / `readCell` / `writeCell` for reactive `let` state; `runLeafEffect(region, sink): Disposable` for a reactive prop (pipes the initial value synchronously, then on each dependent write; the returned native `Disposable` stops it); `scope<T>(run): Scope<T>` — an owner boundary (backed by `DisposableStack`) whose disposal tears down every cell + effect allocated inside it.
- **`KeyMinter`** — mints identity keys so a re-derive reconciles by identity.

The bundled test doubles (`PlainStorageHost` / `RecordingHostEnv` / `PathKeyMinter`) implement all three so the kernel is unit-testable in isolation — which is how this package proves it stays domain-agnostic.

## Usage

```ts
import { evaluateProgram, PlainStorageHost, RecordingHostEnv } from '@metael/lang';

const { value, diagnostics } = evaluateProgram('1 + 2 * 3;', {
  host: new PlainStorageHost(),
  env: new RecordingHostEnv(),
});
// value === 7, diagnostics === []
```

Evaluation is **eval-free** (a source-scan test asserts no `eval`/`new Function`/string-timers/`GeneratorFunction`), **budgeted** (fuel/time/depth/string-growth all fail closed with `ML-LANG-BUDGET`), and **never-throwing** (author errors, budget trips, unknown calls, and forbidden-key access all become diagnostics + a safe `null`).

## Develop

```shell
npm run -w @metael/lang typecheck
npm run -w @metael/lang build     # → dist/ (.js + .d.ts, one per source module)
npx vitest run packages/lang      # the suite (10 files / 127 tests)
```

Domain-neutral by construction: the generic child-collection walk `lowerEntry` ships in THIS package (view-free lang machinery). Only domain-specific lowering (a domain's own View/scene-graph construction) and the reactive re-derive + keyed-diff belong to `@metael/runtime`. See [../../AGENTS.md](../../AGENTS.md) for the load-bearing invariants and the editing guardrails.

## License

MIT.
