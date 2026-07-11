# metael — Agent Guidelines

## Project Overview

metael is the **generic, reusable, eval-free reactive-DSL substrate** — a language kernel that domain frameworks build on instead of hand-rolling their own. It owns exactly the *domain-agnostic* core and nothing domain-specific:

1. **The language** — a legible JS/ES-syntax surface (declarations · declarative-wrapping composition) run by an **eval-free tree-walking interpreter** (sandbox-safe, LLM-emit-safe, budgeted).
2. **The reactive-component AST** — the serializable, editable parse target (reka *State*): `function` (pure) / `component` (stateful, reactive `let`) / control flow / expressions, every node span-tagged.
3. **The reactive runtime** — fine-grained signals/memos/effects + a synchronous `change()` batch/flush boundary + a converge guard *(design; `@metael/runtime`, not yet built)*.
4. **The host-injection contract** — the seam by which a *domain* supplies its vocabulary + output: `HostEnvironment` (resolve a head → a host value), `ReactiveHost` (cells/effects), `KeyMinter` (identity keys). **metael knows how to declare, compose, resolve, and react — never *which* heads exist or *what* they build.**
5. **Determinism + diagnostics** — fuel/deadline/recursion budgets, a seeded-PRNG primitive, a fail-loud diagnostic model. `result = f(source, data, seed, state)`.

A domain = **metael + its vocabulary + its derived View/renderers**. What metael does **not** own: the concrete vocabulary (geometry/shape/chart/material/camera heads), the derived View/scene-graph a run produces, renderers, and backend codegen. Those live in the consuming domain.

> **Kernel, not framework.** metael is the language + reactivity + injection seam; a *domain* is the framework built on it and owns the output artifact. Do not add domain vocabulary, a concrete View type, or a renderer to this repo.

## Why it exists (the consolidation)

The same eval-free reactive-DSL kernel — lexer → parser → tree-walking interpreter → reactive AST → registry-resolved vocabulary → deterministic derive — keeps getting re-implemented per domain. Rather than hand-roll it again, the language/AST/runtime becomes **one durable, tested substrate** each domain instantiates with its own vocabulary. This is metael. The extraction boundary is a proven pattern: a `ports.ts` seam isolates `lang` from any domain output behind the three ports, so the kernel never imports a domain's View/runtime.

## Repo Structure

This is an npm **workspaces monorepo**. Only the kernel's front half is built so far.

```
packages/
└── lang/     @metael/lang   — [BUILT + GREEN] the eval-free, port-injected JS/ES interpreter kernel:
                               lexer → recursive-descent parser → discriminated-union AST →
                               eval-free tree-walking evaluator (fuel/time/depth budgets + FORBIDDEN_KEYS)
                               + the host-injection port INTERFACES (HostEnvironment/ReactiveHost/KeyMinter)
                               + test doubles (PlainStorageHost/RecordingHostEnv/PathKeyMinter).
                               Zero runtime deps; imports NOTHING domain-specific (self-contained).
```

Planned (design-only, not built): `@metael/runtime` (the port *implementations* + the fine-grained reactive runtime + the generic keyed-list diff), `@metael/vdom` (a Preact-like signal-VDOM showcase consumer + the forcing function for the keyed diff), and landing + playground apps.

`@metael/lang` source layout (`packages/lang/src/`), bottom-up dependency order:

```
diagnostics.ts   SourceSpan, Diagnostic, makeDiagnostic (dependency-graph root; zero imports)
ast.ts           Expr/Stmt/Program/Pattern/BinOp discriminated unions + FORBIDDEN_KEYS
determinism.ts   makeSeededRng (mulberry32) + range + MAX_RANGE (pure seeded-PRNG primitive)
environment.ts   Environment — Map-based chained lexical scope + BindingMeta
ports.ts         the 3 host-injection port INTERFACES + Region/LangWrapper/Arg/Scope + 3 test doubles + didYouMean
lexer.ts         lex() → tokens (ML-LANG-LEX diagnostics)
parser.ts        recursive-descent Parser: parseExpr/parseProgram (MAX_PARSE_DEPTH guard; ML-LANG-PARSE)
evaluate.ts      evaluateProgram() — the eval-free tree-walker + fuel/time/depth budgets + never-throw contract
index.ts         the public barrel (excludes the domain-specific lowering)
```

## The extraction boundary

**The load-bearing invariant: `@metael/lang` imports NOTHING domain-specific.** Its `src/` has zero `@`-scoped imports and zero `../` parent-relative imports — verified by the gate. A `call` node is identical whether the head is a user component or a domain vocabulary word; *which* heads exist is a host/registry concern resolved through `HostEnvironment.resolveCall`. Keep it that way: never import a domain View, vocabulary, or renderer into `lang`. The domain-specific AST→View lowering is deliberately **not** in this package — generic child-collection/derive belongs to `@metael/runtime`.

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
npx vitest run packages/lang   # the @metael/lang suite specifically
```

Test runner is **Vitest** (node project only — `lang` has no browser surface). Everything in `@metael/lang` is pure logic and **fully CPU-unit-tested** (no GPU/visual path). The test suite is the conformance bar; keep it green and add a test with any logic change.

## Key Conventions

- **TypeScript, ESM-only** (`"type": "module"`), `moduleResolution: bundler`. Sources import with explicit **`.ts` extensions** (`allowImportingTsExtensions`). `strict` + `noUncheckedIndexedAccess` + `verbatimModuleSyntax` + `noImplicitOverride` are on. `lib: ["ESNext", ...]` — `ESNext` resolves `Disposable`/`DisposableStack`/`Symbol.dispose` (no separate `ESNext.Disposable` needed on TS 6).
- **Zero runtime dependencies.** `@metael/lang` is a pure, self-contained kernel. Do not add a dependency to it. (A signal library like `@vue/reactivity` is a *`@metael/runtime`* concern, not `lang`.)
- **Eval-free tree-walking interpreter** — the DSL is evaluated by an AST walk, **never** `eval`/`new Function`/string-timers/`GeneratorFunction` (sandbox-safe, LLM-emit-safe, deterministic). A `safety.test.ts` source-scan asserts this; do not defeat it.
- **Vocabulary-agnostic core.** The grammar/reactivity/composition/registry hardcode **no** concrete heads. A domain's vocabulary change needs **zero** grammar/AST change. Do not add domain keywords — vocabulary is identifiers resolved through `HostEnvironment.resolveCall`.
- **Diagnostics are `ML-*`** — `ML-LANG-*` for lex/parse/eval/budget (this package), `ML-RT-*` for the runtime (future package). A domain owns its own prefix for its own diagnostics. Fail-loud.
- **TDD for everything** (there is no un-unit-testable surface here). Red → green → commit; a change to logic gets a test.
- **No comments unless the "why" is non-obvious.**

## When Editing

- **Never break the self-containment boundary.** After any change to `packages/lang/src/`, `grep -rn "from '@" packages/lang/src/ ; grep -rn "from '\.\./" packages/lang/src/` must produce **no output**. `lang` imports nothing domain-specific and nothing from a sibling package.
- **The load-bearing guards are not negotiable.** Never weaken: `FORBIDDEN_KEYS = new Set(['__proto__','constructor','prototype'])`; the budget constants (`DEFAULT_MAX_STEPS=100_000`, `DEFAULT_MAX_TIME_MS=1000`, `DEFAULT_MAX_DEPTH=64`, `MAX_STRING_LENGTH=10_000_000`); `MAX_PARSE_DEPTH=512`; the never-throw contract (`evaluateProgram` catches budget/parse-overflow → diagnostics + `null`, never throws); or the `safety.test.ts` eval-free scan.
- **The tests are the conformance bar.** The existing suite pins the kernel's behavior. If a change would make a test need a *logic* edit to pass, treat that as a red flag — the behavior is load-bearing; change the test only when you're deliberately and correctly changing the contract, with the reasoning recorded.
- **Disposal uses the native TC39 protocol.** `runLeafEffect` → `Disposable`; `scope()` → `Scope<T> extends Disposable`; no bespoke `() => void` disposer. Tear-down on throw must not leak a subscription (regression-tested).
- **Build + verify before claiming.** From the repo root: `npm run typecheck && npm run lint && npm run build:packages && npx vitest run packages/lang` (all green). Add/adjust a test with any logic change.

## Status

**`@metael/lang` — BUILT & GREEN.** The eval-free interpreter + the 3 interface-review fixes; the domain-specific lowering excluded. **9 test files / 89 tests green**; typecheck · lint · build:packages clean; self-contained (zero cross-imports); eval-free scan green. Built TDD with a two-lens adversarial review per task + a final comprehensive review (0 real code defects).

**Next: `@metael/runtime`** — the port *implementations* + the fine-grained reactive runtime (signals/memos/effects + synchronous `change()` + converge guard) + the generic keyed-list diff (which owns disposal on `remove`).
