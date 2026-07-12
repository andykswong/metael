# metael — Agent Guidelines

## Project Overview

metael is the **generic, reusable, eval-free reactive-DSL substrate** — a language kernel that domain frameworks build on instead of hand-rolling their own. It owns exactly the *domain-agnostic* core and nothing domain-specific:

1. **The language** — a legible JS/ES-syntax surface (declarations · declarative-wrapping composition) run by an **eval-free tree-walking interpreter** (sandbox-safe, LLM-emit-safe, budgeted).
2. **The reactive-component AST** — the serializable, editable parse target (reka *State*): `function` (pure) / `component` (stateful, reactive `let`) / control flow / expressions, every node span-tagged.
3. **The reactive runtime** — fine-grained signals/memos/effects + a synchronous `change()` batch/flush boundary + a converge guard + the generic keyed-list diff + the real port implementations (`@metael/runtime`, built + green).
4. **The host-injection contract** — the seam by which a *domain* supplies its vocabulary + output: `HostEnvironment` (resolve a head → a host value), `ReactiveHost` (cells/effects), `KeyMinter` (identity keys). **metael knows how to declare, compose, resolve, and react — never *which* heads exist or *what* they build.**
5. **Determinism + diagnostics** — fuel/deadline/recursion budgets, a seeded-PRNG primitive, a fail-loud diagnostic model. `result = f(source, data, seed, state)`.

A domain = **metael + its vocabulary + its derived View/renderers**. What metael does **not** own: the concrete vocabulary (geometry/shape/chart/material/camera heads), the derived View/scene-graph a run produces, renderers, and backend codegen. Those live in the consuming domain.

> **Kernel, not framework.** metael is the language + reactivity + injection seam; a *domain* is the framework built on it and owns the output artifact. Do not add domain vocabulary, a concrete View type, or a renderer to this repo.

## Why it exists (the consolidation)

The same eval-free reactive-DSL kernel — lexer → parser → tree-walking interpreter → reactive AST → registry-resolved vocabulary → deterministic derive — keeps getting re-implemented per domain. Rather than hand-roll it again, the language/AST/runtime becomes **one durable, tested substrate** each domain instantiates with its own vocabulary. This is metael. The extraction boundary is a proven pattern: a `ports.ts` seam isolates `lang` from any domain output behind the three ports, so the kernel never imports a domain's View/runtime.

## Repo Structure

This is an npm **workspaces monorepo**. All three packages are built + green.

```
packages/
├── lang/     @metael/lang    — [BUILT + GREEN] the eval-free, port-injected JS/ES interpreter kernel:
│                               lexer → recursive-descent parser → discriminated-union AST →
│                               eval-free tree-walking evaluator (fuel/time/depth budgets + FORBIDDEN_KEYS)
│                               + the generic child-collection walk (lowerEntry) + intrinsic seeded rand/range
│                               + the host-injection port INTERFACES (HostEnvironment/ReactiveHost/KeyMinter)
│                               + test doubles (PlainStorageHost/RecordingHostEnv/PathKeyMinter).
│                               Zero runtime deps; imports NOTHING domain-specific (self-contained).
├── runtime/  @metael/runtime — [BUILT + GREEN] the reactive runtime + the real port implementations:
│                               reactive core (signal/memo/effect + synchronous change() + converge guard,
│                               over vendored @vue/reactivity) + the generic keyed-list diff (add/remove/move
│                               + teardown-by-identity on remove) + RuntimeReactiveHost (native-Disposable
│                               runLeafEffect + DisposableStack scope() + cellKey latch + cell-freeing) + the
│                               one-shot derive() composition root (ML-RT-CONVERGE). Imports ONLY @metael/lang
│                               + @vue/reactivity (enforced by an automated boundary test).
└── vdom/     @metael/vdom    — [BUILT + GREEN] a Preact-signals-style virtual DOM built ENTIRELY on the
                               kernel — the generality showcase AND the vehicle that hardens the runtime's
                               keyed-list diff under full add/remove/reorder. A thin domain layer: a vnode
                               HostEnvironment (lowercase head → element vnode; Capitalized → decline →
                               transparent fragment) + materialize/reconcile/DOM-patcher + an output
                               sanitizer. Two update paths, automatic: a reactive `let` read by ONE
                               attribute/text position patches only that DOM node in place (a leaf effect,
                               no re-render); a change to the tree's SHAPE re-derives + reconciles by key
                               (DOM identity + focus + selection survive). Imports ONLY @metael/lang +
                               @metael/runtime (enforced by an automated import-boundary test).
```

Planned (design-only, not built): landing + playground apps (both dogfooded on `@metael/vdom`).

`@metael/lang` source layout (`packages/lang/src/`), bottom-up dependency order:

```
diagnostics.ts   SourceSpan, Diagnostic, makeDiagnostic (dependency-graph root; zero imports)
ast.ts           Expr/Stmt/Program/Pattern/BinOp discriminated unions + ArrayElement/ObjectEntry (spread-carrying) + FORBIDDEN_KEYS
determinism.ts   makeSeededRng (mulberry32) + range + MAX_RANGE (pure seeded-PRNG primitive)
environment.ts   Environment — Map-based chained lexical scope + BindingMeta
ports.ts         the 3 host-injection port INTERFACES + Region/LangWrapper/Arg/Scope + 3 test doubles + didYouMean
lexer.ts         lex() → tokens (ML-LANG-LEX diagnostics); the `ellipsis` (...) token for spread
parser.ts        recursive-descent Parser: parseExpr/parseProgram (MAX_PARSE_DEPTH guard; ML-LANG-PARSE); spread in literals + the head{} wrap shorthand
evaluate.ts      evaluateProgram() — the eval-free tree-walker + fuel/time/depth budgets + never-throw contract + intrinsic seeded rand/range + the pure collection builtins + deep-freeze immutability
lower.ts         the generic child-collection walk (lowerEntry): entry-component instantiation → child collection → resolveCall/key-minting/Region+Wrapper emission
index.ts         the public barrel (exports the generic lowerEntry; excludes any domain-specific lowering)
```

`@metael/runtime` source layout (`packages/runtime/src/`), bottom-up dependency order:

```
reactive.ts      signal/memo/effect over @vue/reactivity + the synchronous change()/drain boundary + ReactiveFlushError converge guard
keyed-diff.ts    diffKeyed (pure add/remove/move ops) + applyKeyedDiff (reconcile + dispose-by-identity teardown); zero imports
reactive-host.ts RuntimeReactiveHost — the real ReactiveHost: native-Disposable runLeafEffect + DisposableStack scope() + cellKey latch + exportState + cell-freeing on scope disposal
derive.ts        derive() — the one-shot composition root: one change()-wrapped lowerEntry pass; ML-RT-CONVERGE on a non-converging flush; the onHost seam
index.ts         the public barrel (runtime API + convenience re-exports of the @metael/lang seam)
```

## The extraction boundary

**The load-bearing invariant: `@metael/lang` imports NOTHING domain-specific.** Its `src/` has zero `@`-scoped imports and zero `../` parent-relative imports — verified by the gate. A `call` node is identical whether the head is a user component or a domain vocabulary word; *which* heads exist is a host/registry concern resolved through `HostEnvironment.resolveCall`. Keep it that way: never import a domain View, vocabulary, or renderer into `lang`. The generic child-collection walk (`lowerEntry` — instantiate the entry component, child-collect bodies, resolve heads through the ports, mint keys, emit Region/Wrapper) lives HERE in `@metael/lang` (it is view-free lang machinery). What stays out of this package is any *domain-specific* lowering (a domain's own View/scene-graph construction) and the reactive *re-derive* + keyed-diff, which belong to `@metael/runtime`.

Diagnostics are `ML-*`; the wrapper/effect brands are `__ml*`. If you touch this kernel, preserve that domain-neutrality — no domain codes, no domain brands, no domain imports leak in.

## The three interface-review fixes (in `ports.ts`)

These were added on top of the faithful port (reviewed against reka.js / SolidJS / Vue / preact-signals / CEL / Starlark):

1. **Native TC39 `Disposable` disposal.** `ReactiveHost.runLeafEffect` returns a native `Disposable` (`{ [Symbol.dispose]() }`), not `void` — so a keyed-diff `remove` can tear down a subtree's leaf effects instead of leaking. Added `scope<T>(run): Scope<T>` (`Scope extends Disposable`, backed by a `DisposableStack`) as an owner boundary. `runLeafEffect` pipes the region's **initial** value to the sink synchronously at subscription, then on each dependent write.
2. **`resolveCall` takes an ordered `Arg[]`.** Each arg carries `{ value; name?; reactive? }` — the parser's name-vs-position info is *preserved*, not discarded (no precedent with named args throws it away). metael still doesn't *interpret* roles (the domain does); it just doesn't drop what it parsed.
3. **Optional `knownHeads` + `didYouMean`.** A domain that supplies `knownHeads: ReadonlySet<string>` gets CEL/Starlark-style fail-loud on an unknown head (with a pure Levenshtein-≤2 `didYouMean` suggestion). Absent `knownHeads`, metael stays permissive.

## Build & Test

```shell
npm install                 # install workspace devDeps (TS 6, vite 8, vitest 4, eslint 10); no runtime deps
npm run typecheck           # tsc --noEmit (root) + every package's typecheck (--ws)
npm run lint                # eslint (root + packages)
npm run build:packages      # build @metael/* packages → dist/ (.js + .d.ts, preserveModules)
npm test                    # vitest run (node project)
npx vitest run packages/lang      # the @metael/lang suite specifically
npx vitest run packages/runtime   # the @metael/runtime suite specifically
```

Test runner is **Vitest** (node project only — neither package has a browser surface). Both packages are pure logic and **fully CPU-unit-tested** (no GPU/visual path). The test suite is the conformance bar; keep it green and add a test with any logic change.

## Key Conventions

- **TypeScript, ESM-only** (`"type": "module"`), `moduleResolution: bundler`. Sources import with explicit **`.ts` extensions** (`allowImportingTsExtensions`). `strict` + `noUncheckedIndexedAccess` + `verbatimModuleSyntax` + `noImplicitOverride` are on. `lib: ["ESNext", ...]` — `ESNext` resolves `Disposable`/`DisposableStack`/`Symbol.dispose` (no separate `ESNext.Disposable` needed on TS 6).
- **Dependency discipline per package.** `@metael/lang` is a pure, self-contained kernel with **zero runtime dependencies** — do not add one (a signal library like `@vue/reactivity` is a `@metael/runtime` concern, not `lang`). `@metael/runtime` may import **only** `@metael/lang` + `@vue/reactivity` — nothing domain-specific, no other package.
- **Eval-free tree-walking interpreter** — the DSL is evaluated by an AST walk, **never** `eval`/`new Function`/string-timers/`GeneratorFunction` (sandbox-safe, LLM-emit-safe, deterministic). A `safety.test.ts` source-scan asserts this; do not defeat it.
- **Vocabulary-agnostic core.** The grammar/reactivity/composition/registry hardcode **no** concrete heads. A domain's vocabulary change needs **zero** grammar/AST change. Do not add domain keywords — vocabulary is identifiers resolved through `HostEnvironment.resolveCall`.
- **Immutable collections.** DSL-created arrays/objects (literals + builtin results) are **deep-frozen** at eval — immutable by construction. A member/index write (`o.a = 2`, `a[0] = 9`) is a fail-loud `ML-LANG-IMMUTABLE`; the update path is reassignment + spread/builtins. An **identifier**-LHS assign (a reactive `let` write / `ML-LANG-CONST`) is unaffected — only member/index LHS writes are rejected (a computed forbidden key still surfaces `ML-LANG-FORBIDDEN`). Injected `data` is bound as-is (not frozen) but the write path is blocked for every value, so the DSL cannot mutate it. Do not reintroduce an in-place member/index write.
- **Spread is supported in literals** (`[...a, x]`, `{ ...o, k: v }`) via the `ellipsis` token — array + object literals only, not call args. A spread of a non-array/non-object is a fail-loud `ML-LANG-SPREAD` + a safe skip.
- **Seven pure collection builtins** — `map`/`filter`/`reduce`/`keys`/`values`/`entries`/`fromEntries` — are bound **intrinsically** (the same unbound-head-only, budget-ticked, deterministic pattern as `rand`/`range`): a user `function` of the same name shadows them; each ticks the budget per call + per element (a large collection fails closed with `ML-LANG-BUDGET`); each returns a **new frozen** collection and never mutates an input; a wrong-shape arg is a fail-loud `ML-LANG-BUILTIN-ARG` + a frozen-empty result. Callbacks may be an arrow OR a user-declared `function`.
- **`head { … }` wrap shorthand.** A bare identifier followed by a **same-line** `{` is a zero-arg wrapping call (`group { … }` ≡ `group() { … }`) — the parser synthesizes the `call` node. A next-line `{` after a bare ident stays two statements (the newline guard); a `{` after a *call* wraps on either line (unchanged).
- **Diagnostics are `ML-*`** — `ML-LANG-*` for lex/parse/eval/budget (`@metael/lang`; including `ML-LANG-IMMUTABLE` for a member/index write, `ML-LANG-SPREAD` for a spread of a non-array/non-object, `ML-LANG-BUILTIN-ARG` for a wrong-shape collection-builtin arg), `ML-RT-*` for the runtime (`@metael/runtime`; `ML-RT-CONVERGE` on a non-converging flush). A domain owns its own prefix for its own diagnostics. Fail-loud.
- **TDD for everything** (there is no un-unit-testable surface here). Red → green → commit; a change to logic gets a test.
- **No comments unless the "why" is non-obvious.**

## When Editing

- **Never break the self-containment boundaries.** After any change to `packages/lang/src/`, `grep -rn "from '@" packages/lang/src/ ; grep -rn "from '\.\./" packages/lang/src/` must produce **no output** — `lang` imports nothing domain-specific and nothing from a sibling package. `@metael/runtime` has its own boundary (imports only `@metael/lang` + `@vue/reactivity`), enforced by `packages/runtime/src/boundary.test.ts` — do not weaken it.
- **The load-bearing guards are not negotiable.** Never weaken: `FORBIDDEN_KEYS = new Set(['__proto__','constructor','prototype'])`; the budget constants (`DEFAULT_MAX_STEPS=100_000`, `DEFAULT_MAX_TIME_MS=1000`, `DEFAULT_MAX_DEPTH=64`, `MAX_STRING_LENGTH=10_000_000`); `MAX_PARSE_DEPTH=512`; the never-throw contract (`evaluateProgram` catches budget/parse-overflow → diagnostics + `null`, never throws); or the `safety.test.ts` eval-free scan.
- **The tests are the conformance bar.** The existing suite pins the kernel's behavior. If a change would make a test need a *logic* edit to pass, treat that as a red flag — the behavior is load-bearing; change the test only when you're deliberately and correctly changing the contract, with the reasoning recorded.
- **Disposal uses the native TC39 protocol.** `runLeafEffect` → `Disposable`; `scope()` → `Scope<T> extends Disposable`; no bespoke `() => void` disposer. Tear-down on throw must not leak a subscription (regression-tested).
- **Build + verify before claiming.** From the repo root: `npm run typecheck && npm run lint && npm run build:packages && npm test` (all green). Add/adjust a test with any logic change.

## Status

**`@metael/{lang,runtime,vdom}` — BUILT & GREEN (the kernel + the first showcase consumer are complete).**
- **`@metael/lang`** — the eval-free interpreter + the 3 interface-review fixes + the generic child-collection walk (`lowerEntry`) + intrinsic seeded `rand`/`range` + the collections increment (spread in literals, the seven pure collection builtins, deep-freeze immutability with `ML-LANG-IMMUTABLE`, the `head { }` wrap shorthand); domain-specific lowering excluded. Self-contained (zero cross-imports); eval-free scan green.
- **`@metael/runtime`** — the reactive core (signal/memo/effect + synchronous `change()` + converge guard over vendored `@vue/reactivity`) + the generic keyed-list diff (teardown-by-identity) + `RuntimeReactiveHost` (native-`Disposable` `runLeafEffect` + `DisposableStack` scope() + cellKey latch + cell-freeing) + the one-shot `derive()` (`ML-RT-CONVERGE`). Imports only `@metael/lang` + `@vue/reactivity` (automated boundary test).
- **`@metael/vdom`** — a Preact-signals-style virtual DOM on the kernel: a vnode `HostEnvironment` (lowercase→element, Capitalized→decline→transparent fragment), `materialize` (component→fragment, unknown→`ML-VDOM-UNKNOWN`), a real-DOM keyed reconcile driven by `diffKeyed`'s ops (matched-in-place, move, teardown-on-remove), event delegation, an output sanitizer (attr-allowlist + URL-scheme block), and `mount()` — ONE tracked effect that auto-splits the two update paths (a value-only change fires only a leaf effect and patches that DOM node in place with NO re-render; a structural change re-derives on a fresh host, state latched via `exportState`/`priorState`, and keyed-reconciles the DOM). The example components + the dev demo harness are test/dev fixtures, NOT public API (an import-boundary test asserts it). Imports only `@metael/lang` + `@metael/runtime`.

**Node: 25 test files / 259 tests + Browser: 3 files / 14 tests, all green** (the vdom package contributes 40 node + 14 Playwright/Chromium browser tests across 10 test files); typecheck · lint · build:packages clean. Built TDD with a two-lens adversarial review per task + a final comprehensive whole-branch + FULL spec-conformance + preact-alignment + efficiency review. The recorded budget-lifetime limitation (a reactive re-run shares the derive-time evaluator budget) is **moot** for `@metael/vdom` and stays deferred: a value-only change fires only a leaf effect (no re-walk at all), and a structural change re-derives with a fresh `Runner` (budget resets each pass) — a multi-thousand-interaction regression test proves no `ML-LANG-BUDGET`. The epoch fix is reserved for a hypothetical future persistent-leaf-effect fast path only.

**Next: landing + playground apps** — a landing pitch + a CodePen/W3Schools-style multi-target playground, both dogfooded on `@metael/vdom`.
