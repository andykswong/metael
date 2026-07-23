# @metael/vdom — Agent Guidelines

`@metael/vdom` is a Preact-signals-style virtual DOM built entirely on the metael kernel — a worked example
of a full *domain* on the substrate. It renders to real, live DOM and drives fine-grained updates from plain
signals. Two front doors sit behind subpath exports: **`@metael/vdom`** (the main entry) is the API-first
core — `h()`/`render()`, no interpreter dependency; **`@metael/vdom/lang`** is the metael-DSL binding —
`renderSource()`, `render` driven by the eval-free interpreter. Both produce the identical VNode tree and
share ONE build/reconcile/delegation path. The package CONSUMES the language + reactivity via
`@metael/{lang,runtime}`; it never re-implements a kernel fact (the keyed-list diff, signals, `change()`).
Read the root [AGENTS.md](../../AGENTS.md) first for the kernel invariants this builds on — this file is the
src-map + vdom-specific guidance.

## Architecture / src map

The `src/` tree is a clean split: everything directly under `src/` is the API-first core (imports no
interpreter); everything under `src/lang/` is the DSL binding (pulls `derive`).

### `src/` — the API-first core (`.` subpath; NO interpreter dependency)

- **`render.ts`** — `render(producer, container, hooks?)`: the mount loop. THE tracked `effect` re-runs the
  host `producer` (a `() => RenderNode | RenderNode[]` returning the `h()`-built tree) on a structural signal
  write and reconciles the retained tree; a value-only write fires only a leaf effect and patches one DOM
  node with no re-derive. Returns a `RenderHandle` (adds `setState` to the shared base). Pass `undefined` for
  `container` to run headless (tree-only). `RenderCoreHooks` (`preKeyed`/`onPassHandlers`/`diagnostics`) is
  the additive seam the DSL path uses to reuse the loop.
- **`h.ts`** — `h(tag, props?, ...children)` + `Fragment`: the hyperscript builder. Splits `props` into
  static attributes, captured `on…` handlers, and reactive-attribute thunks; normalizes each child (a
  `Thunk` → reactive text, a primitive → static text, `null`/`undefined`/`false`/`true` → dropped). Produces
  EXACTLY what the DSL host env's `resolveCall` produces, so the downstream path can't tell the two apart.
  Exports `Child`/`Props`/`Thunk`.
- **`vnode.ts`** — the `VNode` type (the opaque host value the walk produces: a lowercase `tag` element, a
  `FRAGMENT` transparent wrapper, or a `TEXT` node) + `Handler`, `isVNode`, `textVNode`, `FRAGMENT`/`TEXT`
  sentinels.
- **`normalize.ts`** — `normalizeNodes(raw)` + the `RenderNode` type: arrayify a producer's result and drop
  conditional holes, so `() => null` and `() => [cond && node, ...]` are tolerated.
- **`handle.ts`** — `VDomHandleBase`: the handle members common to both drivers (`tree()`, `diagnostics`,
  `invokeHandler`, `unmount`, plus the test-only `hasHandler`/`passCount` probes). Each driver extends it
  with its own reactive-write lever.
- **`sanitize.ts`** — the output sanitizer: `safeAttrName` (drop `on*` + raw-HTML sinks), `safeAttrValue`
  (block dangerous URL schemes on URL-bearing attributes, normalizing tab/newline/control obfuscations),
  `escapeText` (for a future HTML-string path — the live patcher uses `Text` nodes, which never parse HTML).
- **`bind.ts`** — walks a keyed tree and wires each reactive/handler stash: a reactive TEXT thunk → a leaf
  `effect`, an element's reactive props → a per-key leaf effect, `node.handlers` → the registry. Records
  disposers per-vnode so a removed subtree is torn down (manual teardown — the core has no per-pass GC).
- **`keying.ts`** — `assignKeys`: positional key assignment for a host-authored tree (a caller supplies
  `key` only for list identity).
- **`reconcile.ts`** — `reconcile(...)` + `flattenFragments`: the DOM patcher. Drives its mutations off the
  runtime keyed diff's ops (via `planLevel`) — matched nodes patched + recursed (DOM identity preserved),
  created nodes inserted, order enforced positionally, removed subtrees torn down. Re-registers the live DOM
  node onto the fresh vnode so a surviving node's leaf path stays alive after a re-derive.
- **`patch.ts`** — `createDom`/`applyAttrs`/`setText`/`setAttr` + `planLevel` (calls `@metael/runtime`'s
  `diffKeyed` — the diff is the runtime's, not re-implemented here). `setText`/`setAttr` are the SAME sinks
  both front doors use: seed the vnode before build, patch the live DOM node after.
- **`delegate.ts`** — one root listener per event walks from target to the nearest ancestor whose `data-key`
  owns a handler and dispatches through the registry (delegation, not per-node listeners — so handlers
  survive keyed reconciliation).

### `src/lang/` — the DSL binding (`./lang` subpath; pulls the interpreter)

- **`render-source.ts`** — `renderSource(source, container, opts?)`: the DSL front door. It IS `render` (the
  core loop) driven by a producer built from `compileToProducer`, plus the DSL-only cross-pass state (the
  host latch across passes, the handler-registry swap, a data signal). Returns a `VDomHandle` (adds
  `updateData` to the shared base). `RenderSourceOptions` = `CompileOptions` (`data`/`seed`/`entry`/
  `reactiveData` + the walk budgets `maxSteps`/`maxTimeMs`/`maxDepth`/`maxStringLength`).
- **`compile.ts`** — `compileToProducer(source, opts)`: the pure "source → VNode tree" step, independently
  testable without a DOM. Each `produce(priorState?)` runs ONE `derive`+`materialize` pass on a FRESH
  reactive host (re-seeded → deterministic; `priorState` latches surviving instances). Exports
  `CompileOptions` + `CompiledPass`.
- **`host-env.ts`** — `VDomHostEnv` (a `BindableHostEnv`): the vnode HostEnvironment. Builds an element VNode
  for a lowercase head and declines a Capitalized one. It is the single place a reactive scalar (a `Region`
  arg) becomes a reactive TEXT node and a reactive prop registers a per-attribute leaf effect (including a
  `style` object with nested Regions).
- **`materialize.ts`** — `materialize(value, diagnostics, handlers)`: converts the raw lowered tree (VNodes +
  `LangWrapper`s) into a retained VNode tree, capturing each handler `${nodeKey}:${event}`. A `component`
  wrapper (a declined in-DSL component) → a transparent `FRAGMENT`; an `unknown` head → dropped with a
  diagnostic.
- **`profile.ts`** — `vdomProfile`: the vocabulary metadata `Profile` (heads/builtins) for LSP/tooling.

## The two update paths

This is the core mechanic an agent must understand. Both front doors choose between them automatically:

1. **Value-only → a leaf-effect patch (no re-render).** A reactive value (a signal in the API path, a
   reactive `let` in the DSL) read by exactly ONE attribute or text position is bound to a leaf `effect`
   (`bind.ts` for the core; `VDomHostEnv.runLeafEffect` for the DSL). A write to it fires only that effect,
   which patches the single live DOM node via `setText`/`setAttr`. The tracked pass is NOT subscribed, so it
   does not re-run — `passCount()` must not increment (the direct proof the fine-grained path is real).
2. **Structural → re-derive + keyed reconcile.** A value read in the producer body / a `let` whose identity
   is reassigned (a list rebuilt via spread/`filter`) subscribes the tracked pass. A write re-runs the
   producer (a fresh `h()` tree, or a fresh `derive` on a new host with the prior state latched), and
   `reconcile` diffs it against the retained tree **by key** (off `@metael/runtime`'s `diffKeyed`): matched
   keys reuse their DOM nodes, so DOM identity, focus, and text selection survive a reorder/insert/remove.

## Load-bearing invariants

- **The core↛interpreter boundary.** `src/*.ts` (the non-`lang` core) NEVER imports `./lang/`, `derive`, or
  `evaluateProgram`, and imports only `@metael/{lang,runtime}` (+ relative siblings) — guarded by
  `src/boundary.test.ts` (a source scan). This is what lets the `.` subpath ship without pulling the
  interpreter; the DSL cost is opt-in via `@metael/vdom/lang`. `src/lang/*.ts` may import the interpreter but
  is still limited to `@metael/{lang,lang/profile,runtime}` + siblings (`src/lang/boundary.test.ts`).
- **The vnode HostEnvironment convention.** A head is a DOM ELEMENT iff its first char is lowercase (the JSX
  rule) → `VDomHostEnv` builds a `VNode`. A Capitalized head is a component instance, which the host
  DECLINES so the walk emits a wrapper → `materialize` turns it into a transparent `FRAGMENT` (a component is
  not a DOM node; its children splice into the parent). `h(Fragment, …)` produces the same transparent
  wrapper in the API path.
- **One sanitizer, both paths.** Every render reaches the DOM through `sanitize.ts`: `on*` + raw-HTML
  attribute names are dropped, dangerous URL schemes are blocked on URL-bearing attributes, and text is
  written via real `Text` nodes (XSS-safe by construction). Do not add a second output path that bypasses it.
- **No re-encoded kernel facts.** The keyed-list diff is `@metael/runtime`'s `diffKeyed` (via `planLevel`),
  signals/effects/`change()` are the runtime's, the AST/`derive`/diagnostics are `@metael/lang`'s. Do not
  fork a copy here.
- **Depth is host-bounded.** The tree walks recurse per level with no depth cap — the API tree is
  host-authored TS, so its depth is the caller's to bound. The DSL path threads a `maxDepth` budget because
  it walks attacker-influenceable source.

## When you add/change — the drift checklist

- **A new element behavior** (an attribute convention, a reactive-prop shape). The source of truth is the
  vnode HostEnvironment (`src/lang/host-env.ts`) AND `h.ts` — they must agree, since both produce the same
  VNode. Add a test on both front doors.
- **A change to how the tree diffs.** The keyed-list diff is `@metael/runtime`'s — change it THERE, not in
  `reconcile.ts`/`patch.ts` (which only realize its ops as DOM mutations). Adjust `planLevel`/`reconcile`
  only for DOM-placement mechanics, not diff semantics.
- **A new/changed security or sanitizer rule.** Update `src/sanitize.ts` AND `src/sanitize.test.ts` — a
  forbidden attribute name, a blocked URL scheme, or an obfuscation normalization is only real with its test.
- **A new vocabulary head/builtin.** Add it to the vnode host env's `resolveCall` handling (if it is a new
  element shape) and to `src/lang/profile.ts` (so LSP/tooling see it).

## Testing

Two layers. Node unit tests assert the pure steps directly (`h`, `normalize`, `keying`, `patch`,
`reconcile`, `sanitize`, `style`, `api-surface`, `bind`, `compile`, `materialize`, `host-env`, `profile`,
and the `boundary.test.ts` / `lang/boundary.test.ts` import guards). The Chromium `*.browser.test.ts` files
(`render`, `mount`, `vdom`, `disposal`, `leaf-reconcile`, `buffer-reactivity`, and `lang/render-source`) are
the real-DOM proofs — they verify DOM identity/focus/selection survival, the fine-grained leaf patch (via
`passCount`), and teardown-with-no-resurrection. Keep them green and add a test with any change — a real-DOM
test for anything that touches the patch/reconcile path.

## Build / verify

`npm run -w @metael/vdom typecheck` / `build`; `npx vitest run packages/vdom` (node + the Chromium real-DOM
tests). Every exported symbol needs a doc comment, checked by the repo-root doc-coverage gate `npm run
docs:api:check`. For the final check when an effort is done, run `npm run prepublishOnly` from the repo root
— the full one-shot gate: `clean → build:packages → typecheck → lint → test → docs:api:check`.

---

Root [AGENTS.md](../../AGENTS.md) — kernel invariants + editing guardrails. [README.md](./README.md) —
install + the front-door surface.
