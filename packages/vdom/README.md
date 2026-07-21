# @metael/vdom

[![metael](https://img.shields.io/badge/project-metael-blueviolet.svg?style=flat-square&logo=github)](https://github.com/andykswong/metael)
[![npm](https://img.shields.io/npm/v/@metael/vdom?style=flat-square&logo=npm)](https://www.npmjs.com/package/@metael/vdom)
[![license: MIT](https://img.shields.io/badge/License-MIT-red.svg?style=flat-square)](./LICENSE)

A small, signal-driven virtual DOM built entirely on the metael reactive kernel (`@metael/lang` +
`@metael/runtime`). It renders a component to real, live DOM and drives fine-grained updates from plain
signals.

There are **two front doors** onto the same rendering machinery — pick whichever suits the caller:

- **Host API** — build the tree in TypeScript with the `h()` hyperscript builder and mount it with
  `render()`. No language, no compiler; just functions and signals.
- **metael DSL** — write a `component` in metael source and render it with `renderSource()` (from
  `@metael/vdom/lang`). The eval-free interpreter builds the same tree, so untrusted source is *data*,
  never executed as host code.

Both produce the identical vnode tree and share one build/reconcile/delegation path, so their runtime
behaviour is the same.

Two update paths, chosen automatically (in both front doors):
- a reactive value read by a single attribute/text position patches **only that node** — a leaf effect,
  with no re-render;
- a change that alters the tree's **shape** (a list grows/shrinks/reorders) re-derives the affected
  subtree and reconciles it **by key**, reusing the existing DOM nodes so focus + selection survive.

It is a showcase for the kernel's generality and the vehicle that exercises the runtime's generic
keyed-list diff under full add/remove/reorder. It is not a general web framework.

## Host API — `h()` + `render()`

Build a vnode tree with `h(tag, props, ...children)` and mount it with `render(producer, container)`. The
`producer` is a host callback returning the tree; read signals inside it (or inside child/prop thunks) and
the DOM stays in sync.

```ts
import { signal } from '@metael/runtime';
import { h, render } from '@metael/vdom';

const n = signal(0);

const handle = render(
  () => h('div', { class: 'counter' },
    h('button', { onClick: () => n.set(n.get() + 1) }, '+'),
    // a THUNK child is reactive text: a value-only write patches just this text node (no re-render)
    () => `count: ${n.get()}`,
  ),
  document.getElementById('app')!,
);

// later, to tear it down (stops the tracked pass + all leaf effects, clears the container):
handle.unmount();
```

`h()` conventions:

| In `props` / children | Becomes |
|---|---|
| `onClick`, `onInput`, … (`on` + Capital, a function) | a delegated event handler |
| a **function** prop value, e.g. `class: () => cls.get()` | a **reactive attribute** (leaf effect; patches that one attribute) |
| any other prop value | a static attribute |
| `key: id` | list identity (used by keyed reconciliation) |
| a **thunk** child `() => v` | **reactive text** (leaf effect; patches that one text node) |
| a string / number child | static text |
| `null` / `undefined` / `false` | dropped — the JSX-conditional idiom (`cond && h(...)`) |
| a nested `h(...)` | a child element |

`Fragment` groups children without a wrapper element (they splice into the parent):

```ts
import { h, Fragment } from '@metael/vdom';
h(Fragment, {}, h('li', {}, 'a'), h('li', {}, 'b'));   // two <li>, no wrapper
```

A structural change is just a signal read in the producer body. Reassign the signal to a new value and the
producer re-runs; the tree is reconciled **by key**:

```ts
import { signal } from '@metael/runtime';
import { h, render } from '@metael/vdom';

const items = signal([{ id: 0, label: 'first' }, { id: 1, label: 'second' }]);

const handle = render(
  () => h('ul', {},
    ...items.get().map((it) =>
      h('li', { key: it.id },                       // key → stable identity across reorders
        it.label,
        h('button', { onClick: () => items.set(items.get().filter((r) => r.id !== it.id)) }, 'x'),
      ),
    ),
  ),
  document.getElementById('app')!,
);
// items.set([...]) re-runs the producer; matched keys reuse their DOM nodes, so focus/selection survive.
```

`render(producer, container, opts?)` returns a `RenderHandle`:

- `tree(): VNode | null` — the current retained tree (a fragment root unwraps to its first element).
- `setState(fn)` — run `fn` inside the render's reactive boundary (e.g. drive a signal write from outside a handler).
- `invokeHandler(nodeKey, event, arg)` / `hasHandler(nodeKey, event)` — fire / probe a captured handler by node key.
- `passCount()` — how many structural passes ran; a value-only write must **not** increment it (the fine-grained proof).
- `unmount()` — stop the tracked pass, dispose every leaf effect, and clear the container.

> **Tree depth is host-bounded.** The tree walks recurse per level with no depth cap — a host mapping
> *untrusted* deeply-nested data to nested `h()` calls must cap the nesting itself.

## metael DSL — `renderSource()`

Write a `component` in metael source and render it. The interpreter builds the same vnode tree; a reactive
`let` read by one position takes the fine-grained leaf path, and a `let` whose identity is reassigned (a
list rebuilt via spread/`filter`) takes the structural keyed path.

```ts
import { renderSource } from '@metael/vdom/lang';

const source = `
component Story() {
  let n = 0
  div({ class: "counter" }) {
    button({ onClick: () => { n = n + 1 } }, "+")
    span(n)              // reads only n → leaf effect (no re-render on click)
  }
}`;

const handle = renderSource(source, document.getElementById('app')!, {});
// handle: tree() / invokeHandler() / updateData() / unmount() (see VDomHandle)
```

A keyed list — add via spread, remove via `filter`; both **reassign** `items` (immutable rebuild), which the
structural path reconciles by `key`:

```ts
const source = `
component Story() {
  let items = [{ id: 0, label: "first" }, { id: 1, label: "second" }]
  let nextId = 2
  div {
    ul {
      for (const it of items) {
        li({ key: it.id }) {
          span(it.label)
          button({ onClick: () => { items = filter(items, (r) => r.id != it.id) } }, "x")
        }
      }
    }
    button({ onClick: () => { items = [...items, { id: nextId, label: "new" }]; nextId = nextId + 1 } }, "add")
  }
}`;

renderSource(source, document.getElementById('app')!, {});
```

`renderSource(source, container, opts)` takes `RenderSourceOptions` (`data`, `seed`, `entry`, `reactiveData`, and the
interpreter budgets `maxSteps` / `maxTimeMs` / `maxDepth` / `maxStringLength`) and returns a `VDomHandle`
(`tree()` / `invokeHandler()` / `updateData()` / `unmount()` / `passCount()`).

## Choosing a front door

- **Host API (`h`/`render`)** — you are writing TypeScript, want type-checked construction, and control the
  source yourself. No compiler on the path.
- **DSL (`renderSource`)** — you want to render source authored elsewhere (a playground, a saved document,
  user/LLM-provided content). Because the language is eval-free, running arbitrary source inline is safe:
  it is parsed and interpreted, never executed as host code, and is budgeted (fuel/deadline/recursion).

## Output safety

Both paths render through the same sanitizer: event-handler and raw-HTML attribute names
(`on*`, `innerHTML`, `srcdoc`, …) are dropped, and dangerous URL schemes (`javascript:` / `data:` /
`vbscript:`, including tab/newline/control-char obfuscations) are blocked on URL-bearing attributes. Text is
written via real `Text` nodes (which never parse HTML), so it is XSS-safe by construction.
