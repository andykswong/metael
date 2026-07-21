// The VNode — the opaque HostValue the child-collection walk produces. An element vnode has a lowercase
// tag; a FRAGMENT (a materialized component instance) has no tag and only splices its children into the
// parent (a component is not a DOM node); a TEXT vnode carries `text`. Reactive text is a TEXT vnode whose
// value a leaf effect patches in place.

/** One captured DOM event handler on a {@link VNode}: the event name plus the function fired for it.
 *  Handlers are dispatched through the delegated listener inside the reactive `change()` boundary, never
 *  bound directly to the element. */
export interface Handler {
  /** The `on…`-prefixed event name as authored (e.g. `onClick`, `onInput`), matched by the delegator. */
  readonly event: string;
  /** The callback invoked when the event fires; `arg` is the normalized event payload (e.g. `{ value, key }`). */
  readonly fn: (arg: unknown) => void;
}

/** The virtual-DOM node — the opaque host value the child-collection walk produces and the build/reconcile
 *  path consumes. An element vnode carries a lowercase `tag`; a {@link FRAGMENT} (a materialized component
 *  instance) has no DOM node and only splices its children into the parent; a {@link TEXT} vnode carries
 *  `text`. Reactive text is a `TEXT` vnode whose value a leaf effect patches in place. */
export interface VNode {
  /** A lowercase DOM tag for an element; FRAGMENT for a transparent fragment; TEXT for a text node. */
  readonly tag: string;
  /** Static (non-reactive) element attributes/properties as an unordered map; reactive attributes are
   *  stashed off-band and bound to leaf effects, not stored here. */
  props: Record<string, unknown>;
  /** Child vnodes in document order. Empty for a `TEXT` node (mutually exclusive with `text`). */
  children: VNode[];
  /** The reconcile identity key — a caller-supplied list key or a positionally-assigned key. Stable across
   *  passes so the keyed diff can match, reuse, and reorder instances instead of recreating them. */
  readonly key: string;
  /** The captured event handlers for this element, if any (absent when the node has none). */
  handlers?: Handler[];
  /** Text content when tag === TEXT (mutually exclusive with children). A leaf effect may patch this. */
  text?: string;
}

/** The sentinel `tag` of a transparent fragment: it owns no DOM node of its own; its children splice
 *  directly into the parent (how a materialized component instance appears in the tree). */
export const FRAGMENT = '';       // no DOM node; children splice into the parent
/** The sentinel `tag` of a text node, whose content lives in {@link VNode.text}. */
export const TEXT = '#text';      // a text node

/** Type guard: is `v` a {@link VNode} (an object with a string `tag`)? Distinguishes a vnode from the raw
 *  host values the walk may otherwise carry. */
export function isVNode(v: unknown): v is VNode {
  return typeof v === 'object' && v !== null && typeof (v as { tag?: unknown }).tag === 'string';
}

/** A text vnode with a positional key. */
export function textVNode(text: string, key: string): VNode {
  return { tag: TEXT, props: {}, children: [], key, text };
}
