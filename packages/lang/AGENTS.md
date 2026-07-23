# @metael/lang — Agent Guidelines

`@metael/lang` = the eval-free, port-injected JS/ES interpreter kernel: `source ──lex──▶ tokens ──parse──▶ AST ──evaluate / lowerEntry──▶ host values`, staying entirely vocabulary-agnostic (a domain injects *which heads exist* + *what they build* through the three host ports). Zero runtime deps; imports nothing domain-specific.

**The load-bearing invariants + conventions live in the root [../../AGENTS.md](../../AGENTS.md) — read it.** It owns the eval-free rule, the vocabulary-agnostic core, the fixed call-resolution order, immutability/deep-freeze, spread, the pure builtin set, capability profiles, the `head { }` wrap shorthand, the custom-value descriptor protocol, the `ML-*` diagnostics, and the "When Editing" guardrails (self-containment grep, `FORBIDDEN_KEYS` + budget constants + `MAX_PARSE_DEPTH` + never-throw + the eval-free scan, native-Disposable teardown, build-before-claiming). **This file does not restate them** — it's the src-map + the language-evolution drift guide for THIS package.

## Architecture / src map

Each source file owns one concern; the key exports are attached to their file with the role they play. This is the one place to read to understand the package — a flat export list lives in the barrels (`index.ts`).

`src/*.ts` (the kernel `.` entry):

- **`lexer.ts`** — the tokenizer (`lex`, fail-soft with `ML-LANG-LEX` diagnostics) + the private `KEYWORDS` record (the source of truth). `KEYWORDS_SET` + the exhaustive `LEXICAL_CATEGORY` `Record<TokenType, LexicalCategory>` (backing `lexicalCategory`) derive from it — the single source tooling reads for syntax highlighting / semantic tokens, so nobody re-encodes the keyword/operator/punct classification (adding a `TokenType` is a compile error until it's classified).
- **`parser.ts`** — recursive-descent parser → AST (`parseExpr`/`parseProgram`/`Parser`, results `ParseProgramResult`/`ParseExprResult`, `ML-LANG-PARSE`); `BINARY_TIERS` precedence ladder; the `head { }` wrap-shorthand + the groupDepth-scoped newline guard; the `MAX_PARSE_DEPTH` guard.
- **`ast.ts`** — the discriminated-union AST: `Expr`/`Stmt`/`Program`/`Pattern` (+ `ArrayElement`/`ObjectEntry`) and `BinOp`, every node span-tagged; `FORBIDDEN_KEYS` (the member/key security guard).
- **`evaluate.ts`** — the eval-free tree-walker: `evaluateProgram` (+ `EvalOptions`/`EvalResult`) with fuel/time/depth budgets (never-throw) and the fixed call-resolution order; the budget constants `DEFAULT_MAX_STEPS`/`DEFAULT_MAX_TIME_MS`/`DEFAULT_MAX_DEPTH`/`MAX_STRING_LENGTH`; resolves the sole kernel intrinsic `range` (the bounded-loop primitive) before the host, seeding it from `EvalOptions.seed` — `rand` is NOT here (it lives in the injectable standard library, registry-dispatched); the custom-value descriptor dispatch sites; the callable/closure surface (`makeCallable`/`callUserFn`/`isUserFn`/`readClosureValue`/`UserFn`) + the coercion helpers (`truthy`/`looseEquals`/`strOf`).
- **`environment.ts`** — `Environment` (+ `BindingMeta`): chained lexical scope + binding metadata.
- **`registry.ts`** — the pure-builtin seam: `Builtin` (`{ name, invoke }`) / `BuiltinModule` / `BuiltinCtx` + `buildRegistry` (deep-frozen N-way merge) and the shared `EMPTY_REGISTRY` (`BuiltinRegistry`) for a builtin-free run.
- **`custom-types.ts`** — the non-forgeable Symbol-keyed `TypeDescriptor` protocol (operators/accessors/iteration/`Lowering`); `descriptorOf`/`isCustomType`/`generationOf`, typed-array (`isTypedArray`) + frozen tagging (`tagCustom`/`markFrozen`/`isFrozenCustom`), the `NOT_HANDLED` sentinel, `BufferError`.
- **`determinism.ts`** — the seeded PRNG (`makeSeededRng`, mulberry32) + the bounded `range` loop primitive (`MAX_RANGE`).
- **`diagnostics.ts`** — the `ML-*` diagnostic model: `makeDiagnostic`, `Diagnostic`, `SourceSpan`, `LiteralValue`.
- **`print.ts`** — the canonical printer (`printProgram`/`printExpr`/`printStmt`/`printString`/`printBlock`, `stripSpans`); the parse→print→parse **conservation law** (round-trip test); `MAX_PRINT_DEPTH`/`PrintDepthError`.
- **`ports.ts`** — the `HostEnvironment`/`ReactiveHost`/`KeyMinter` **interfaces** + the value contract that crosses the seam (`Arg`/`Region`/`LangWrapper`/`Scope`/`HostValue`/`CellRef`/`GenerationRef`/`EffectRegion`/`Clock`/`BindableHostEnv`); the port helpers `region`/`isRegion`/`wrapper`/`isWrapper`/`frozenClock`/`didYouMean` (Levenshtein-≤2 suggestion for fail-loud head resolution); the test doubles `PlainStorageHost`/`RecordingHostEnv`/`PathKeyMinter`.
- **`lower.ts`** — `lowerEntry` (+ `LowerOptions`/`LowerResult`), the generic view-free child-collection walk: instantiate the entry component, child-collect its body, resolve heads via `HostEnvironment`, mint keys via `KeyMinter`, emit `Region`/`Wrapper` (builds no domain node).
- **`index.ts`** — the `.` barrel.

`src/profile/*.ts` — the **`@metael/lang/profile`** subpath (the vocabulary-metadata / tooling layer a language service or codegen gate reads, NOT on the `.` entry). A `Profile` describes a domain's builtins/heads/types **by name** (no runtime dispatch); a consumer classifies over whatever (possibly composed) `Profile` it is handed.

- **`types.ts`** — the metadata record types: `BuiltinSpec`/`BuiltinProfile`/`Portability`/`HeadSpec`/`HeadParam`/`MemberSpec`/`TypeDescriptorMeta`. `BuiltinSpec` carries optional editor-hover metadata (`doc`/`params`/`returnDoc`) alongside its capability fields, so signature help + hover cards derive from the same spec a classifier reads.
- **`define-builtin.ts`** — `defineBuiltin` (co-locate a builtin's `BuiltinSpec` + its `invoke`, as a `DefinedBuiltin`) + `toBuiltinModule`/`builtinSpecMap` (project the same defs two ways — a `BuiltinModule` for the runtime path and a spec map for the tooling path, so invoke and metadata never drift).
- **`members.ts`** — `swizzleMembers` (derive a custom type's swizzle member specs — `.xy`, `.rgb`, …).
- **`profile.ts`** — `Profile`/`ComposedProfile` + `composeProfiles` (keyed-union of N profiles into one).
- **`core-intrinsics.ts`** — `coreIntrinsicsProfile` (publishes the kernel's `range` intrinsic's spec — the only vocabulary the kernel itself carries).
- **`classify.ts`** — `classifyProfile(fn, profile)` + `ProfileResult`: the static core-lowerability classifier — decides a function's core-compliance from its AST against the active `Profile` (metadata + a pure classifier; no codegen/dispatch engine).
- **`index.ts`** — the `./profile` barrel.

Domain packages publish a `Profile` a consumer composes with `coreIntrinsicsProfile` (e.g. `@metael/math`'s `mathProfile`, `@metael/std`, `@metael/vdom`, `@metael/gpu`).

## The injection contract

`@metael/lang` never builds a domain value — `HostValue` is opaque. The three port interfaces in `ports.ts` are the whole seam; the bundled test doubles implement all three so the kernel is unit-testable with no domain present (which is how it proves it stays domain-agnostic).

- **`HostEnvironment.resolveCall(head, key, args: Arg[], children, span)`** → `{ handled: true; value }` (the domain built a value) or `{ handled: false }` (the kernel emits a tagged `LangWrapper`). An additive `{ handled: true; kind: 'value' }` return marks a plain value (vs a node). Each `Arg` keeps the parser's `{ value; name?; reactive? }` — the name-vs-position info is preserved, not discarded; the domain interprets roles. An optional `knownHeads` set enables fail-loud + `didYouMean` on a typo'd head.
- **`ReactiveHost`** — `allocateCell` / `readCell` / `writeCell` for reactive `let` state; `runLeafEffect(region, sink): Disposable` for a reactive prop (pipes the initial value synchronously, then on each dependent write; the returned native `Disposable` stops it); `scope<T>(run): Scope<T>` — an owner boundary (backed by `DisposableStack`) whose disposal tears down every cell + effect allocated inside it.
- **`KeyMinter`** — mints identity keys so a re-derive reconciles by identity.

The port *implementations* and the fine-grained reactive runtime (which drives `lowerEntry` inside a `change()` boundary) live in the separate [`@metael/runtime`](../runtime/README.md) package.

## The two boundaries (this package's own guards)

- **lang self-containment.** Non-test `src/` imports NOTHING `@`-scoped and no `../` parent-relative (a domain View/vocabulary/runtime never leaks in). The grep is in the root AGENTS.md.
- **core ↛ profile.** `src/*.ts` (excluding `src/profile/`) never imports `./profile/` — the kernel stays lean and privileges no vocabulary. Guarded by `src/profile/boundary.test.ts` (a source scan).

## When you evolve the language, update these — the drift checklist

Keyed to the kind of language change; the point is that tooling (highlighting, completion, semantic tokens in `@metael/lsp` + `apps/site`, and `classifyProfile`) derives from single sources here, so most changes need no copy elsewhere.

- **Add/remove a keyword.** The private `KEYWORDS` record in `lexer.ts` is the source. Add the word there AND classify it in the exhaustive `LEXICAL_CATEGORY` record (a new `TokenType` FAILS typecheck until classified). `KEYWORDS_SET` + `lexicalCategory` derive automatically; the parser's keyword-as-member/key handling + all tooling then follow from `KEYWORDS_SET`/`lexicalCategory` — no copy to maintain.
- **Add a `TokenType`** (operator/punct/etc.). Classify it in `LEXICAL_CATEGORY` (compile-forced), add its spelling to the lexer's `twoMap`/`oneMap`, and — if it's a binary operator — add it to `parser.ts`'s `BINARY_TIERS` (at its precedence tier) + `BinOp` in `ast.ts`.
- **Add an AST node** (an `Expr`/`Stmt` variant). Touch `ast.ts` (the union), `parser.ts` (produce it), `evaluate.ts` (eval it), `print.ts` (print it — the conservation-law test guards round-trip). Downstream AST walkers (`@metael/lsp`'s scope-model / scope-check / folding / selection / capability-lens + `classifyProfile`) switch on `kind`; they're guard-tolerant of unknown kinds, so add the case where relevant AND a test.
- **Add a builtin.** It goes in a standard-library package's `BuiltinModule` via `defineBuiltin` (which co-locates the `BuiltinSpec` for the `Profile`) — NOT in the kernel (the kernel privileges only `range`). The registry seam + the domain's `Profile` carry it; `@metael/lang` needs no change.
- **Add a diagnostic code.** `ML-LANG-*`; keep it fail-loud + never-throw (a diagnostic + a safe `null`, never a throw out of `evaluateProgram`).
- **Touch the custom-value protocol.** The descriptor dispatch sites in `evaluate.ts` fire **after** the number/primitive fast path and **after** `FORBIDDEN_KEYS` — preserve both. See `custom-types.ts` for the descriptor shape.

## Testing / build

TDD — red → green → commit; a logic change gets a test. `npx vitest run packages/lang` (the suite). Every exported symbol needs a doc comment — `npm run docs:api:check` (from the repo root) is the doc-coverage gate (0 undocumented exports). The final pre-publish gate from the repo root is `npm run prepublishOnly` (`clean → build:packages → typecheck → lint → test → docs:api:check`). See the root AGENTS.md for the full editing guardrails.

---

Root [../../AGENTS.md](../../AGENTS.md) — kernel invariants + editing guardrails. [README.md](./README.md) — install + the export surface.
