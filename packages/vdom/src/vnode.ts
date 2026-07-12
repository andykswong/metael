// The VNode — the opaque HostValue the child-collection walk produces. An element vnode has a lowercase
// tag; a FRAGMENT (a materialized component instance) has no tag and only splices its children into the
// parent (a component is not a DOM node); a TEXT vnode carries `text`. Reactive text is a TEXT vnode whose
// value a leaf effect patches in place.

export interface Handler { readonly event: string; readonly fn: (arg: unknown) => void }

export interface VNode {
  /** A lowercase DOM tag for an element; FRAGMENT for a transparent fragment; TEXT for a text node. */
  readonly tag: string;
  props: Record<string, unknown>;
  children: VNode[];
  readonly key: string;
  handlers?: Handler[];
  /** Text content when tag === TEXT (mutually exclusive with children). A leaf effect may patch this. */
  text?: string;
}

export const FRAGMENT = '';       // no DOM node; children splice into the parent
export const TEXT = '#text';      // a text node

export function isVNode(v: unknown): v is VNode {
  return typeof v === 'object' && v !== null && typeof (v as { tag?: unknown }).tag === 'string';
}

/** A text vnode with a positional key. */
export function textVNode(text: string, key: string): VNode {
  return { tag: TEXT, props: {}, children: [], key, text };
}
