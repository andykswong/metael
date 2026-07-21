// packages/vdom/src/h.ts
// The API-first hyperscript builder: construct a VNode tree from host TS, mirroring exactly what
// VDomHostEnv.resolveCall produces so the downstream build/reconcile path can't tell the difference.
// Reactive values are PLAIN THUNKS (() => v): a thunk child → a reactive TEXT vnode; a thunk prop → a
// reactive attribute. render() binds them to leaf effects. Keys are left '' here and assigned in a
// positional post-build pass (see keying.ts), so a caller only supplies `key` for list identity.
import { FRAGMENT, TEXT, type VNode, type Handler } from './vnode.ts';

/** A stash symbol on the vnode: for a TEXT vnode → its reactive thunk; for an element → a
 *  { name: thunk } map of reactive props. Read by bindReactive; never serialized. */
export const REACTIVE: unique symbol = Symbol('metael.vdom.reactive');
const USER_KEY: unique symbol = Symbol.for('metael.vdom.userKey');

/** Sentinel tag for a transparent fragment (children splice into the parent). */
export const Fragment = FRAGMENT;

/** A reactive value passed to {@link h}: a zero-arg function read inside a leaf effect. As a child it
 *  becomes a reactive `TEXT` vnode; as a prop it becomes a reactive attribute — either way `render()` binds
 *  it so a value-only change patches one DOM node without a re-derive. */
export type Thunk = () => unknown;
/** An accepted child of {@link h}: a {@link VNode}, a primitive coerced to static text, a {@link Thunk} for
 *  reactive text, or `null`/`undefined`/`false`/`true` (skipped — the JSX-conditional idiom `cond && node`). */
export type Child = VNode | string | number | boolean | null | undefined | Thunk;
/** The props bag passed to {@link h}: an `on…` function is a captured event handler, any other function is a
 *  reactive attribute {@link Thunk}, `key` sets the reconcile identity, and every other value is a static
 *  attribute. */
export type Props = Record<string, unknown>;

function isHandlerName(name: string): boolean { return /^on[A-Z]/.test(name); }

/** Normalize one child to a VNode (or null to skip). A thunk → a reactive TEXT vnode; a primitive →
 *  a static TEXT vnode; a VNode passes through; null/undefined/false are skipped (JSX-conditional idiom). */
function toChild(c: Child): VNode | null {
  if (c === null || c === undefined || c === false || c === true) return null;
  if (typeof c === 'function') {
    const node: VNode = { tag: TEXT, props: {}, children: [], key: '', text: '' };
    (node as unknown as Record<symbol, unknown>)[REACTIVE] = c;
    return node;
  }
  if (typeof c === 'object') return c;                 // already a VNode
  return { tag: TEXT, props: {}, children: [], key: '', text: String(c) };
}

/** The API-first hyperscript builder: construct a {@link VNode} from host TypeScript, producing exactly what
 *  the display vocabulary's `resolveCall` produces so the downstream build/reconcile path can't tell the two
 *  apart. Splits `props` into static attributes, captured `on…` handlers, and reactive-attribute thunks;
 *  normalizes each child via the {@link Child} rules. Keys are left empty here and assigned positionally in a
 *  later pass — supply `key` only for list identity. */
export function h(tag: string, props: Props | null = {}, ...children: Child[]): VNode {
  const p = props ?? {};
  const handlers: Handler[] = [];
  const finalProps: Props = {};
  const reactiveProps: Record<string, Thunk> = {};
  let userKey: string | undefined;
  for (const [k, v] of Object.entries(p)) {
    if (k === 'key') { userKey = v as string; continue; }
    if (typeof v === 'function' && isHandlerName(k)) { handlers.push({ event: k, fn: v as Handler['fn'] }); continue; }
    if (typeof v === 'function') { reactiveProps[k] = v as Thunk; continue; }   // reactive attribute
    finalProps[k] = v;
  }
  const kids: VNode[] = [];
  for (const c of children) { const n = toChild(c); if (n) kids.push(n); }
  const node: VNode = { tag, props: finalProps, children: kids, key: '' };
  if (handlers.length) node.handlers = handlers;
  if (Object.keys(reactiveProps).length) (node as unknown as Record<symbol, unknown>)[REACTIVE] = reactiveProps;
  if (userKey !== undefined) (node as unknown as Record<symbol, unknown>)[USER_KEY] = userKey;
  return node;
}

/** Read the stashed caller key (used by the keyer). */
export function userKeyOf(node: VNode): string | undefined {
  return (node as unknown as Record<symbol, unknown>)[USER_KEY] as string | undefined;
}
