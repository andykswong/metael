# metael

**The generic, reusable, eval-free reactive-DSL substrate — the language kernel, extracted once.**

metael is the language + reactivity + host-injection seam that domain frameworks build on. It owns exactly the *domain-agnostic* core: a legible JS/ES-syntax surface run by an **eval-free tree-walking interpreter**; a serializable, editable **reactive-component AST** (reka *State*); a **fine-grained reactive runtime**; and the **host-injection contract** by which a domain supplies *which heads exist* and *what they build*. metael knows how to declare, compose, resolve, and react — never which vocabulary exists or what it renders to.

A domain = **metael + its vocabulary + its derived View/renderers**. The same kernel serves a 3D scene graph, a compute shader, and a signal-VDOM.

> **Why it exists:** the same eval-free reactive-DSL kernel keeps getting re-implemented per domain. metael consolidates it into **one durable, tested substrate** each domain instantiates with its own vocabulary — instead of hand-rolling it again.

## Status — `@metael/{lang,runtime}` built & green

The kernel is **complete, tested, and merged-ready** across two packages:

- **`@metael/lang`** — the eval-free, port-injected JS/ES interpreter (lexer → parser → discriminated-union AST → tree-walking evaluator with fuel/time/depth budgets + `__proto__`/`constructor`/`prototype` guards), the host-injection port **interfaces** (`HostEnvironment` / `ReactiveHost` / `KeyMinter`) + their test doubles, the generic child-collection **walk** (`lowerEntry`) as view-free lang machinery, and intrinsic seeded `rand`/`range`. Zero runtime dependencies; imports nothing domain-specific.
- **`@metael/runtime`** — the fine-grained reactive core (`signal`/`memo`/`effect` + a synchronous `change()` batch/flush boundary + a converge guard, over vendored `@vue/reactivity`), the generic **keyed-list diff** (add/remove/move + teardown-by-identity on `remove`), the real **`RuntimeReactiveHost`** (native-`Disposable` `runLeafEffect` + `DisposableStack` owner scopes + cellKey latch + cell-freeing), and the one-shot **`derive()`** composition root. Depends only on `@metael/lang` + `@vue/reactivity`.

Three interface-review fixes are baked into the host-injection ports (native-`Disposable` disposal, an ordered `Arg[]` on `resolveCall`, an optional fail-loud `knownHeads`/`didYouMean`). Determinism is a language-level guarantee: `result = f(source, data, seed, state)`, with cross-consumer conformance fixtures (same source + seed → identical host-value trace) + disposal fixtures (a keyed `remove` leaves no lingering effect and no retained cell-key state).

**18 test files / 182 tests green** across both packages; typecheck · lint · build clean; each package **self-contained** behind the port seam (`lang` imports nothing; `runtime` imports only `@metael/lang` + `@vue/reactivity`, enforced by an automated boundary test). Built subagent-driven / TDD with a two-lens adversarial review per task + a final comprehensive whole-branch + spec-conformance pass.

**Next:** `@metael/vdom` — a Preact-like signal-VDOM showcase consumer that also hardens the runtime's keyed-reconciliation half (a VDOM forces full add/remove/reorder).

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
├── lang/     @metael/lang    — [BUILT] the eval-free, port-injected interpreter kernel (zero deps, self-contained)
│                               diagnostics · ast · determinism · environment · ports · lexer · parser · evaluate · lower
└── runtime/  @metael/runtime — [BUILT] the reactive runtime + port implementations (deps: @metael/lang + @vue/reactivity)
                               reactive · reactive-host · keyed-diff · derive

(planned, design-only)
   vdom/      @metael/vdom    — a Preact-like signal-VDOM showcase consumer + the keyed-diff forcing function
```

See [AGENTS.md](./AGENTS.md) for the architecture, conventions, and editing guardrails.

## Tests

```shell
npm run typecheck        # tsc --noEmit (root + every package)
npm run lint             # eslint (root + packages)
npm run build:packages   # build @metael/* → dist/ (.js + .d.ts)
npm test                 # vitest run
```

Both packages are pure logic and fully CPU-unit-tested — the tests are the conformance bar. `@metael/lang`'s `safety.test.ts` source-scan asserts the kernel stays eval-free; `@metael/runtime`'s `boundary.test.ts` asserts the package imports nothing beyond `@metael/lang` + `@vue/reactivity`.

## License

MIT — see [LICENSE](./LICENSE).
