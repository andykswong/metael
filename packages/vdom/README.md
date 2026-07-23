# @metael/vdom

[![metael](https://img.shields.io/badge/project-metael-blueviolet.svg?style=flat-square&logo=github)](https://github.com/andykswong/metael)
[![npm](https://img.shields.io/npm/v/@metael/vdom?style=flat-square&logo=npm)](https://www.npmjs.com/package/@metael/vdom)
[![license: MIT](https://img.shields.io/badge/License-MIT-red.svg?style=flat-square)](./LICENSE)

A small, signal-driven virtual DOM built entirely on the metael reactive kernel (`@metael/lang` +
`@metael/runtime`). It renders to real, live DOM and drives fine-grained updates from plain signals — and
it doubles as a worked example of a full domain built on the metael substrate.

## Install

```shell
npm install @metael/vdom
```

Pulls the kernel (`@metael/lang`) and the reactive runtime (`@metael/runtime`) — nothing else.

## Usage

Build a tree in TypeScript with the `h()` hyperscript builder and mount it with `render()`. No language,
no compiler — just functions and signals. Read a signal inside the producer (or inside a child/prop thunk)
and the DOM stays in sync:

```ts
import { signal } from '@metael/runtime';
import { h, render } from '@metael/vdom';

const n = signal(0);

const handle = render(
  () => h('div', { class: 'counter' },
    h('button', { onClick: () => n.set(n.get() + 1) }, '+'),
    () => `count: ${n.get()}`,   // a THUNK child is reactive text: a value write patches just this node
  ),
  document.getElementById('app')!,
);

handle.unmount();   // stops the tracked pass + every leaf effect, clears the container
```

### The DSL front door — `renderSource()`

The other front door renders a metael `component` written in source. The eval-free interpreter builds the
**same** tree, so untrusted source is *data*, never executed as host code:

```ts
import { renderSource } from '@metael/vdom/lang';

renderSource(`
  component Story() {
    let n = 0
    div({ class: "counter" }) {
      button({ onClick: () => { n = n + 1 } }, "+")
      span(n)              // reads only n → a leaf effect (no re-render on click)
    }
  }`, document.getElementById('app')!, {});
```

**Which to use:** reach for `h()`/`render()` when you write the tree in TypeScript and want type-checked
construction with no compiler on the path; reach for `renderSource()` when you render source authored
elsewhere (a playground, a saved document, user/LLM-provided content) — it is parsed, interpreted, and
budgeted, never executed.

## At a glance

- **Two subpaths.** `@metael/vdom` (the `.` API-first core — `render`/`h`/`Fragment`, no interpreter
  dependency); `@metael/vdom/lang` (the DSL binding — `renderSource`, which pulls the interpreter).
  Both produce the identical vnode tree and share one build/reconcile/delegation path.
- **Two update paths, chosen automatically.** A reactive value read by a single attribute/text position
  patches **only that node** (a leaf effect, no re-render); a change to the tree's **shape** (a list
  grows/shrinks/reorders) re-derives the affected subtree and reconciles it **by key**, reusing the
  existing DOM nodes so focus + selection survive.
- **Output-safe.** Every render goes through one sanitizer: `on*`/raw-HTML attribute names are dropped,
  dangerous URL schemes (`javascript:`/`data:`/`vbscript:`) are blocked, and text is written via real
  `Text` nodes — XSS-safe by construction.
- **Boundary.** Imports only `@metael/{lang,runtime}`; the `.` core carries no interpreter dependency
  (enforced by a boundary test).

It is a showcase for the kernel's generality — the vehicle that exercises the runtime's generic keyed-list
diff under full add/remove/reorder — not a general web framework. See [AGENTS.md](./AGENTS.md) for the
architecture, the two update paths in depth, and the full API surface.

## Develop

```shell
npm run -w @metael/vdom typecheck
npm run -w @metael/vdom build      # → dist/ (.js + .d.ts)
npx vitest run packages/vdom       # the suite (node + Chromium real-DOM proofs)
```

From the repo root, `npm run docs:api:check` is the doc-coverage gate (every exported symbol needs a doc
comment) and `npm run prepublishOnly` runs the full one-shot gate — `clean → build:packages → typecheck →
lint → test → docs:api:check`.

See the root [README.md](../../README.md) for the package map and [AGENTS.md](./AGENTS.md) for the
load-bearing invariants and editing guardrails.

## License

MIT — see [LICENSE](./LICENSE).
