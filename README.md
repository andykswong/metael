# metael

**The generic, reusable, eval-free reactive-DSL substrate — the language kernel, extracted once.**

metael is the language + reactivity + host-injection seam that domain frameworks build on. It owns exactly the *domain-agnostic* core: a legible JS/ES-syntax surface run by an **eval-free tree-walking interpreter**; a serializable, editable **reactive-component AST** (reka *State*); a **fine-grained reactive runtime**; and the **host-injection contract** by which a domain supplies *which heads exist* and *what they build*. metael knows how to declare, compose, resolve, and react — never which vocabulary exists or what it renders to.

A domain = **metael + its vocabulary + its derived View/renderers**. The same kernel serves a 3D scene graph, a compute shader, and a signal-VDOM.

> **Why it exists:** the same eval-free reactive-DSL kernel keeps getting re-implemented per domain. metael consolidates it into **one durable, tested substrate** each domain instantiates with its own vocabulary — instead of hand-rolling it again.

## Status — `@metael/{lang,runtime,vdom}` built & green

The kernel + the first showcase consumer are **complete, tested, and merged-ready** across three packages:

- **`@metael/lang`** — the eval-free, port-injected JS/ES interpreter (lexer → parser → discriminated-union AST → tree-walking evaluator with fuel/time/depth budgets + `__proto__`/`constructor`/`prototype` guards), the host-injection port **interfaces** (`HostEnvironment` / `ReactiveHost` / `KeyMinter`) + their test doubles, the generic child-collection **walk** (`lowerEntry`) as view-free lang machinery, and intrinsic seeded `rand`/`range`. Zero runtime dependencies; imports nothing domain-specific.
- **`@metael/runtime`** — the fine-grained reactive core (`signal`/`memo`/`effect` + a synchronous `change()` batch/flush boundary + a converge guard, over vendored `@vue/reactivity`), the generic **keyed-list diff** (add/remove/move + teardown-by-identity on `remove`), the real **`RuntimeReactiveHost`** (native-`Disposable` `runLeafEffect` + `DisposableStack` owner scopes + cellKey latch + cell-freeing), and the one-shot **`derive()`** composition root. Depends only on `@metael/lang` + `@vue/reactivity`.
- **`@metael/vdom`** — a **Preact-signals-style virtual DOM built entirely on the kernel**: write a `component` in the metael DSL, and it renders to real, live DOM. Two update paths, chosen automatically — a reactive `let` read by a single attribute/text position patches **only that node** (a leaf effect, no re-render); a change to the tree's **shape** re-derives the affected subtree and reconciles it **by key**, reusing existing DOM nodes so focus + selection survive. It is the *generality showcase* (a wildly-different-from-scene-graph target falls out of the substrate with a thin domain layer) **and** the forcing function that hardens the runtime's keyed-list diff under full add/remove/reorder. Depends only on `@metael/lang` + `@metael/runtime`.

Three interface-review fixes are baked into the host-injection ports (native-`Disposable` disposal, an ordered `Arg[]` on `resolveCall`, an optional fail-loud `knownHeads`/`didYouMean`). Determinism is a language-level guarantee: `result = f(source, data, seed, state)`, with cross-consumer conformance fixtures (same source + seed → identical host-value trace) + disposal fixtures (a keyed `remove` leaves no lingering effect and no retained cell-key state).

**Node: 25 test files / 259 tests + Browser: 3 files / 14 tests, all green** (the vdom package adds 40 node + 14 Playwright/Chromium browser tests); typecheck · lint · build clean; each package **self-contained** behind the port seam (`lang` imports nothing; `runtime` imports only `@metael/lang` + `@vue/reactivity`; `vdom` imports only `@metael/{lang,runtime}` — all enforced by automated boundary tests). Built subagent-driven / TDD with a two-lens adversarial review per task + a final comprehensive whole-branch + spec-conformance + preact-alignment + efficiency pass.

**Next:** landing + playground apps — a landing pitch + a CodePen/W3Schools-style multi-target playground, both dogfooded on `@metael/vdom`.

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
├── runtime/  @metael/runtime — [BUILT] the reactive runtime + port implementations (deps: @metael/lang + @vue/reactivity)
│                               reactive · reactive-host · keyed-diff · derive
└── vdom/     @metael/vdom    — [BUILT] a Preact-signals-style VDOM on the kernel — the generality showcase + the
                               keyed-diff forcing function (deps: @metael/lang + @metael/runtime)
                               vnode · sanitize · host-env · materialize · patch · reconcile · delegate · mount

(planned, design-only)
   landing + playground apps — a landing pitch + a multi-target playground, both dogfooded on @metael/vdom
```

See [AGENTS.md](./AGENTS.md) for the architecture, conventions, and editing guardrails.

## Tests

```shell
npm run typecheck        # tsc --noEmit (root + every package)
npm run lint             # eslint (root + packages)
npm run build:packages   # build @metael/* → dist/ (.js + .d.ts)
npm test                 # vitest run (the node project — the whole monorepo)
npm run test:browser     # vitest run --project browser (the Playwright/Chromium real-DOM proofs)
```

`@metael/{lang,runtime}` are pure logic and fully CPU-unit-tested. `@metael/vdom` adds a **browser** vitest project (Playwright/Chromium) for the real-DOM proofs — same DOM node survives a reorder, focus + input selection persist across add/remove/reorder, event delegation fires, `javascript:` hrefs are dropped, and a removed subtree is torn down. The tests are the conformance bar. `@metael/lang`'s `safety.test.ts` source-scan asserts the kernel stays eval-free; `@metael/runtime`'s `boundary.test.ts` and `@metael/vdom`'s `boundary.test.ts` assert each package imports nothing beyond its allowed dependencies.

## License

MIT — see [LICENSE](./LICENSE).
