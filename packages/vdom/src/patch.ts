import { diffKeyed, type KeyedOp } from '@metael/runtime';
import { FRAGMENT, TEXT, type VNode } from './vnode.ts';
import { safeAttrName, safeAttrValue } from './sanitize.ts';

/** The plan for one children level: the runtime keyed-diff ops (which DRIVE the DOM mutations — they carry
 *  the add index + move from/to), the matched prev instances by key (reused by identity → DOM node identity
 *  survives), and the removed prev instances (teardown targets). PURE — no DOM. */
export interface LevelPlan {
  ops: KeyedOp[];
  matched: Map<string, VNode>;   // next key → reused prev instance
  removed: VNode[];              // prev vnodes absent from next (dispose + detach)
}

/** Delegate op generation to the runtime's tree-agnostic keyed diff over the key sequences, then resolve
 *  which prev instances are reused (by identity, consume-once for duplicate keys) vs removed. */
export function planLevel(prev: readonly VNode[], next: readonly VNode[]): LevelPlan {
  const ops = diffKeyed(prev.map((v) => v.key), next.map((v) => v.key));
  const byKey = new Map<string, VNode>();
  for (const v of prev) if (!byKey.has(v.key)) byKey.set(v.key, v);
  const matched = new Map<string, VNode>();
  const consumed = new Set<string>();
  const reused = new Set<VNode>();
  for (const n of next) {
    const existing = byKey.get(n.key);
    if (existing && !consumed.has(n.key)) { consumed.add(n.key); matched.set(n.key, existing); reused.add(existing); }
  }
  const removed = prev.filter((v) => !reused.has(v));
  return { ops, matched, removed };
}

/** Create real DOM for a vnode. A text vnode → a raw Text node (Text nodes do not parse HTML → XSS-safe
 *  without escaping). A fragment → a DocumentFragment of its built children (transient; the reconcile splices
 *  children into the parent). An element → an Element with sanitized attributes + built children. `index`
 *  records key→node for delegation + reconcile lookups (fragments/text are not indexed by element key). */
export function createDom(vnode: VNode, doc: Document, index: Map<string, Element>): Node {
  if (vnode.tag === TEXT) { const t = doc.createTextNode(vnode.text ?? ''); registerTextNode(vnode, t); return t; }
  if (vnode.tag === FRAGMENT) {
    const frag = doc.createDocumentFragment();
    for (const c of vnode.children) frag.appendChild(createDom(c, doc, index));
    return frag;
  }
  const el = doc.createElement(vnode.tag);
  el.setAttribute('data-key', vnode.key);
  applyAttrs(el, vnode.props);
  for (const c of vnode.children) el.appendChild(createDom(c, doc, index));
  index.set(vnode.key, el);
  registerElement(vnode, el);
  return el;
}

/** Set/patch an element's attributes from props, applying the sanitizer. A null/undefined/false value or a
 *  blocked URL scheme removes the attribute. Handlers are never here (captured separately). */
export function applyAttrs(el: Element, props: Record<string, unknown>): void {
  for (const [k, v] of Object.entries(props)) applyAttr(el, k, v);
}

/** Serialize a style OBJECT to a CSS declaration string ("a: b; c: d"). camelCase property names
 *  become kebab-case; a `--custom` (or already-kebab) name is left as-is; null/undefined/false/true
 *  entries are dropped (no CSS meaning); other values are coerced with String(). An empty result is
 *  the empty string (the caller removes the attribute). This is the object-`style` path the minimal
 *  "coerce-to-string → setAttribute" prop model did not cover — an object style previously stringified
 *  to "[object Object]". Numeric values are emitted verbatim (`{ zIndex: 3 }` → "z-index: 3") — there is
 *  no `px` auto-unit, so a unit must be written explicitly (`{ width: "3px" }`). */
export function styleObjectToCss(style: Record<string, unknown>): string {
  const decls: string[] = [];
  for (const [prop, value] of Object.entries(style)) {
    if (value === null || value === undefined || value === false || value === true) continue;
    const name = prop.startsWith('--') ? prop : prop.replace(/[A-Z]/g, (m) => `-${m.toLowerCase()}`);
    decls.push(`${name}: ${String(value)}`);
  }
  return decls.join('; ');
}

// Form-control attributes the browser tracks as LIVE DOM PROPERTIES that diverge from the attribute once
// the control is "dirty" (user-edited): the attribute is only the *default*, so `setAttribute('value', …)`
// is ignored after the user types. A controlled input whose reactive `value` is reset (e.g. clearing a
// field after submit) must patch the `.value` PROPERTY; likewise a checkbox/radio `.checked`, a
// `<select>`/`<option>` `.selected`, and the boolean `.disabled`. We mirror onto the property AND keep the
// attribute in sync below (so createDom/reconcile + the a11y tree stay authoritative).
const DOM_PROPERTIES = new Set(['value', 'checked', 'selected', 'disabled']);

/** Set/patch ONE attribute on an element through the sanitizer (the single-attribute path a reactive-prop
 *  leaf effect drives — see setAttr). A null/undefined/false value or a blocked URL scheme removes it. */
export function applyAttr(el: Element, k: string, v: unknown): void {
  if (!safeAttrName(k)) return;
  // Mirror form-control live properties BEFORE the attribute paths (which early-return on a removal), so a
  // dirty field's displayed value/checked actually updates. `value` is a string; the rest are booleans.
  if (DOM_PROPERTIES.has(k) && k in el) {
    const cleared = v === null || v === undefined || v === false;
    if (k === 'value') (el as unknown as Record<string, unknown>).value = cleared ? '' : String(v);
    else (el as unknown as Record<string, unknown>)[k] = !cleared;
  }
  if (v === null || v === undefined || v === false) { el.removeAttribute(k); return; }
  // Object-valued `style`: serialize to CSS text instead of String(v) → "[object Object]". An empty
  // result removes the attribute. (A string style still flows through the scalar path below.)
  if (k === 'style' && typeof v === 'object' && !Array.isArray(v)) {
    const css = styleObjectToCss(v as Record<string, unknown>);
    if (css === '') { el.removeAttribute(k); return; }
    const safeCss = safeAttrValue(k, css);
    if (safeCss === null) { el.removeAttribute(k); return; }
    el.setAttribute(k, safeCss);
    return;
  }
  const raw = v === true ? '' : String(v);
  const safe = safeAttrValue(k, raw);
  if (safe === null) { el.removeAttribute(k); return; }
  el.setAttribute(k, safe);
}

// A vnode's live DOM node is stashed on the vnode itself via a WeakMap so a leaf effect can patch it in
// place with NO reconcile (the fine-grained fast path), and so the reconcile can find a text node (which
// carries no data-key attribute). Living here (not in reconcile.ts) avoids a patch↔reconcile import cycle:
// createDom registers the node; both the leaf-effect sinks (setText/setAttr) and the reconcile read it.
const textNodes = new WeakMap<VNode, Text>();
const elementNodes = new WeakMap<VNode, Element>();
export function registerTextNode(vnode: VNode, node: Text): void { textNodes.set(vnode, node); }
export function textNodeOf(vnode: VNode): Text | null { return textNodes.get(vnode) ?? null; }
export function registerElement(vnode: VNode, el: Element): void { elementNodes.set(vnode, el); }
export function elementOf(vnode: VNode): Element | null { return elementNodes.get(vnode) ?? null; }

/** A reactive-text leaf effect's sink: record the value on the retained vnode (so createDom/reconcile stay
 *  authoritative) AND — once the DOM Text node exists — patch it in place, so a value-only change updates
 *  the DOM with no re-render (the preact-signals fine-grained path). Before the first build the DOM node is
 *  absent, so this just seeds the vnode field, which createDom then reads. */
export function setText(vnode: VNode, value: unknown): void {
  const s = value === null || value === undefined ? '' : String(value);
  vnode.text = s;
  const node = textNodeOf(vnode);
  if (node) node.data = s;
}

/** A reactive-prop leaf effect's sink: record the value on the retained vnode's props (authoritative for
 *  createDom/reconcile) AND — once the element exists — patch that one attribute in place through the
 *  sanitizer, so a value-only change updates the DOM with no re-render. */
export function setAttr(vnode: VNode, key: string, value: unknown): void {
  vnode.props[key] = value;
  const el = elementOf(vnode);
  if (el) applyAttr(el, key, value);
}
