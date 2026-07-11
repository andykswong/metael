# metael

**The generic, reusable, eval-free reactive-DSL substrate — the language kernel, extracted once.**

metael is the language + reactivity + host-injection seam that domain frameworks build on. It owns exactly the *domain-agnostic* core: a legible JS/ES-syntax surface run by an **eval-free tree-walking interpreter**; a serializable, editable **reactive-component AST** (reka *State*); a **fine-grained reactive runtime**; and the **host-injection contract** by which a domain supplies *which heads exist* and *what they build*. metael knows how to declare, compose, resolve, and react — never which vocabulary exists or what it renders to.

A domain = **metael + its vocabulary + its derived View/renderers**. The same kernel serves a 3D scene graph, a compute shader, and a signal-VDOM.

> **Why it exists:** the same eval-free reactive-DSL kernel keeps getting re-implemented per domain. metael consolidates it into **one durable, tested substrate** each domain instantiates with its own vocabulary — instead of hand-rolling it again.

## Status — `@metael/lang` built & green

The kernel's front half — **`@metael/lang`** — is **built, tested, and merged-ready**: the eval-free, port-injected JS/ES interpreter (lexer → parser → discriminated-union AST → tree-walking evaluator with fuel/time/depth budgets + `__proto__`/`constructor`/`prototype` guards), plus the host-injection port **interfaces** (`HostEnvironment` / `ReactiveHost` / `KeyMinter`) and their test doubles.

It is domain-neutral by construction, with the domain-specific AST→View lowering left to the consuming domain, and three interface-review fixes applied to the host-injection ports (native-`Disposable` disposal, an ordered `Arg[]` on `resolveCall`, and an optional fail-loud `knownHeads`/`didYouMean`).

**9 test files / 89 tests green**; typecheck · lint · build clean; zero runtime dependencies; **self-contained** — `@metael/lang` imports nothing domain-specific. Built subagent-driven / TDD with adversarial review per task + a final comprehensive spec-conformance pass.

**Next:** `@metael/runtime` — the port implementations + the fine-grained reactive runtime (signals/memos/effects + a synchronous `change()`) + the generic keyed-list diff — followed by an `@metael/vdom` showcase consumer.

## Quick start

Requires Node 26+.

```shell
npm install
npm run typecheck
npm test
```

## Architecture

- **Eval-free & deterministic.** The AST is inert data; a tree-walking interpreter evaluates it — never `eval`/`new Function`. Budgeted (fuel/wall-clock/recursion/string-growth, all fail-closed) and seeded, so `result = f(source, data, seed, state)` and LLM-emitted source is machine-verifiable.
- **Vocabulary-agnostic core.** The grammar, reactivity, composition, and registry hardcode **no** concrete heads. A `call` node is identical whether the head is a user component or a domain vocabulary word — so a domain's vocabulary change needs zero grammar change. *This* is why one kernel serves a scene graph, a shader, and a dataframe query alike.
- **Host-injection seam.** A domain implements three interfaces — `HostEnvironment` (resolve a head → an opaque host value), `ReactiveHost` (cells + leaf effects + owner scopes), `KeyMinter` (identity keys for reconciliation) — and gets the whole DSL / AST / reactivity / determinism substrate for free. metael calls the ports; the domain builds the values.

```
packages/
└── lang/     @metael/lang   — [BUILT] the eval-free, port-injected interpreter kernel (zero deps, self-contained)
                               diagnostics · ast · determinism · environment · ports · lexer · parser · evaluate

(planned, design-only)
   runtime/   @metael/runtime — the port implementations + reactive runtime + keyed-list diff
   vdom/      @metael/vdom    — a Preact-like signal-VDOM showcase consumer
```

See [AGENTS.md](./AGENTS.md) for the architecture, conventions, and editing guardrails.

## Tests

```shell
npm run typecheck        # tsc --noEmit (root + every package)
npm run lint             # eslint (root + packages)
npm run build:packages   # build @metael/* → dist/ (.js + .d.ts)
npm test                 # vitest run
```

Everything in `@metael/lang` is pure logic and fully CPU-unit-tested — the ported tests are the conformance bar, and a `safety.test.ts` source-scan asserts the kernel stays eval-free.

## License

MIT — see [LICENSE](./LICENSE).
