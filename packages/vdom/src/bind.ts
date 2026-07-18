// packages/vdom/src/bind.ts
// Walk a keyed vnode tree and wire its reactive/handler stashes:
//   • a reactive TEXT vnode (REACTIVE = thunk)     → effect(() => setText(node, thunk()))
//   • an element's reactive props (REACTIVE = map)  → effect(() => setAttr(node, k, thunk())) per key
//   • node.handlers                                 → registry.set(`${key}:${event}`, fn)
// Each leaf effect's disposer is pushed to `disposers` (the pass owns them) AND recorded per-vnode via a
// WeakMap so onRemove can dispose a removed subtree's effects (manual teardown — render() has no fresh-host
// GC to rely on). setText/setAttr are the SAME sinks the DSL path uses: they seed the vnode before build
// and patch the live DOM node after (fine-grained, no reconcile on a value-only change).
import { effect } from '@metael/runtime';
import { setText, setAttr } from './patch.ts';
import { REACTIVE, type Thunk } from './h.ts';
import type { VNode } from './vnode.ts';

const leafDisposers = new WeakMap<VNode, Array<() => void>>();

/** Dispose the leaf effects bound to a vnode subtree (called by render()'s onRemove). Depth-first. */
export function disposeLeaf(node: VNode): void {
  const ds = leafDisposers.get(node);
  if (ds) { for (const d of ds) d(); leafDisposers.delete(node); }
  for (const c of node.children) disposeLeaf(c);
}

export function bindReactive(
  nodes: readonly VNode[],
  disposers: Array<() => void>,
  registry: Map<string, (arg: unknown) => void>,
): void {
  for (const node of nodes) {
    const stash = (node as unknown as Record<symbol, unknown>)[REACTIVE];
    if (node.tag === '#text' && typeof stash === 'function') {
      const thunk = stash as Thunk;
      bindLeaf(node, (alive) => effect(() => { if (alive()) setText(node, thunk()); }), disposers);
    } else if (stash && typeof stash === 'object') {
      for (const [k, thunk] of Object.entries(stash as Record<string, Thunk>)) {
        bindLeaf(node, (alive) => effect(() => { if (alive()) setAttr(node, k, thunk()); }), disposers);
      }
    }
    if (node.handlers) for (const hd of node.handlers) registry.set(`${node.key}:${hd.event}`, hd.fn);
    if (node.children.length) bindReactive(node.children, disposers, registry);
  }
}

// Bind one leaf effect with an ALIVE GATE. stop() (vue's ReactiveEffect.stop) clears the ACTIVE flag but a
// run already scheduled into the current reactive batch STILL executes fn() once (untracked). When a
// structural re-derive is triggered by a signal also read inside a surviving leaf's thunk, that residual run
// would sink the OLD value into the reused DOM node AFTER the reconcile applied the new one. Gating fn() on a
// per-effect `alive` flag — cleared in the disposer before stop() — makes that residual run a no-op, so a
// disposed leaf can never clobber the reconciled DOM.
function bindLeaf(node: VNode, make: (alive: () => boolean) => (() => void), disposers: Array<() => void>): void {
  let alive = true;
  const stop = make(() => alive);
  record(node, () => { alive = false; stop(); }, disposers);
}

function record(node: VNode, stop: () => void, disposers: Array<() => void>): void {
  disposers.push(stop);
  const arr = leafDisposers.get(node) ?? [];
  arr.push(stop);
  leafDisposers.set(node, arr);
}
