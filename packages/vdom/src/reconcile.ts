import { type VNode, FRAGMENT, TEXT } from './vnode.ts';
import { createDom, applyAttrs, planLevel, textNodeOf, registerTextNode, registerElement } from './patch.ts';

/** A fragment has no DOM node, so its children reconcile against their real siblings. Flatten fragments
 *  into a single child sequence before diffing (recursively). */
export function flattenFragments(children: readonly VNode[]): VNode[] {
  const out: VNode[] = [];
  for (const c of children) { if (c.tag === FRAGMENT) out.push(...flattenFragments(c.children)); else out.push(c); }
  return out;
}

export interface ReconcileHooks {
  /** Called for each removed vnode subtree so the caller can dispose its leaf effects/cells (teardown). */
  onRemove: (vnode: VNode) => void;
}

/** Reconcile a parent element's children against the freshly-derived children, in place, BY KEY — driving
 *  the DOM mutations off the runtime keyed diff's ops (via planLevel). Matched nodes are patched + recursed
 *  (DOM identity preserved); created nodes are inserted; the final child order is enforced positionally
 *  (a move); removed nodes are torn down + detached. */
export function reconcile(parent: Element, prevChildren: VNode[], nextChildren: VNode[], doc: Document, index: Map<string, Element>, hooks: ReconcileHooks): VNode[] {
  const prev = flattenFragments(prevChildren);
  const next = flattenFragments(nextChildren);
  const plan = planLevel(prev, next);

  // Teardown removed subtrees (dispose leaf effects/cells + detach DOM + drop the index). Every prev
  // instance absent from next (plan.removed — identity-based, so a collapsed duplicate key is still torn
  // down) is disposed. This is the keyed diff's teardown-on-remove obligation, driven off the plan.
  for (const gone of plan.removed) tearDown(gone, index, hooks);

  // Build the reconciled next list: reuse matched instances (patched in place), create new ones.
  const out: VNode[] = [];
  for (const n of next) {
    const matched = plan.matched.get(n.key);
    if (matched) { patchNode(matched, n, doc, index, hooks); out.push(matched); }
    else { createDom(n, doc, index); out.push(n); }
  }

  // Enforce the next order positionally (this realizes the diff's `add`/`move` ops as DOM placement).
  for (let i = 0; i < out.length; i++) {
    const dom = domOf(out[i]!, index);
    if (!dom) continue;
    const ref = parent.childNodes[i] ?? null;
    if (ref !== dom) parent.insertBefore(dom, ref);
  }
  return out;
}

/** Patch a matched retained vnode in place from its fresh counterpart, then recurse children. Key/tag are
 *  stable (matched by key). Text nodes patch their data; elements patch attributes. Mutates the retained
 *  vnode's props/children so it stays the source of truth for the next diff.
 *
 *  Crucially, it also re-registers the live DOM node onto the FRESH vnode. This pass's leaf effects (bound
 *  during this pass's derive) close over the FRESH vnodes and patch through `textNodeOf`/`elementOf`; but
 *  the retained tree keeps the matched (prior) vnode. Without transferring the DOM registration to the
 *  fresh vnode, a later value-only write would sink into a fresh vnode with no DOM node (a silent no-op)
 *  and the fine-grained leaf path would go dead after any structural re-derive that preserved the node. */
function patchNode(target: VNode, fresh: VNode, doc: Document, index: Map<string, Element>, hooks: ReconcileHooks): void {
  if (target.tag === TEXT) {
    target.text = fresh.text;
    const n = textNodeOf(target);
    if (n) { n.data = fresh.text ?? ''; registerTextNode(fresh, n); }   // this pass's leaf effect patches `fresh`
    return;
  }
  const el = index.get(target.key);
  if (el) { applyAttrs(el, fresh.props); registerElement(fresh, el); }  // this pass's prop leaf effects patch `fresh`
  target.props = fresh.props;
  if (el) target.children = reconcile(el, target.children, fresh.children, doc, index, hooks);
}

/** Dispose + detach a removed subtree (depth-first): fire the teardown hook, drop the index, detach the DOM. */
function tearDown(vnode: VNode, index: Map<string, Element>, hooks: ReconcileHooks): void {
  hooks.onRemove(vnode);
  for (const c of vnode.children) tearDown(c, index, hooks);
  const el = index.get(vnode.key);
  if (el && el.parentNode) el.parentNode.removeChild(el);
  index.delete(vnode.key);
}

/** The DOM node for a vnode: an indexed element, else (a text vnode) its Text node held on the vnode. */
function domOf(vnode: VNode, index: Map<string, Element>): Node | null {
  if (vnode.tag === TEXT) return textNodeOf(vnode);
  return index.get(vnode.key) ?? null;
}
