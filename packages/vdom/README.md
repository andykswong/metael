# @metael/vdom

A small, signal-driven virtual DOM built entirely on the metael reactive kernel (`@metael/lang` +
`@metael/runtime`). Write a `component` in the metael DSL; `@metael/vdom` renders it to real, live DOM.

Two update paths, chosen automatically:
- a reactive `let` read by a single attribute/text position patches **only that node** — a leaf effect,
  with no re-render;
- a change that alters the tree's **shape** (a list grows/shrinks/reorders) re-derives the affected
  subtree and reconciles it **by key**, reusing the existing DOM nodes so focus + selection survive.

It is a showcase for the kernel's generality and the vehicle that exercises the runtime's generic
keyed-list diff under full add/remove/reorder. It is not a general web framework.
