import type { BindableHostEnv, HostValue, Arg, ReactiveHost, Region, SourceSpan } from '@metael/runtime';
import { isRegion } from '@metael/runtime';
import { TEXT, type VNode } from '../vnode.ts';
import { setText, setAttr } from '../patch.ts';

/** on[A-Z] camelCase handler convention (onClick/onInput/…): a function value under such a name is
 *  captured as a delegated handler, never set as an attribute. */
function isHandlerName(name: string): boolean { return /^on[A-Z]/.test(name); }

/** A head is a DOM ELEMENT iff its first char is lowercase (the JSX rule); a Capitalized head is a
 *  component instance, which this host declines so the walk emits a wrapper (→ a fragment). */
function isElementHead(head: string): boolean {
  const c = head[0] ?? '';
  return c >= 'a' && c <= 'z';
}

/** True if `v` is a plain object with at least one nested reactive Region value (a `{ color: <let> }`
 *  style object). Such an object is not itself a Region, so it needs its own leaf-effect binding. */
function isStyleWithRegion(v: unknown): boolean {
  if (typeof v !== 'object' || v === null || Array.isArray(v) || isRegion(v)) return false;
  return Object.values(v as Record<string, unknown>).some((x) => isRegion(x));
}

/** Resolve a style object's nested Regions to their current values (running each thunk), leaving
 *  non-Region entries untouched. Returns a fresh plain object applyAttr can serialize to CSS text. */
function resolveStyleRegions(style: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const [k, x] of Object.entries(style)) out[k] = isRegion(x) ? (x as Region).run() : x;
  return out;
}

/**
 * The vnode HostEnvironment. Builds an element VNode for a lowercase head and declines a Capitalized one.
 * It is the single place a reactive scalar (a Region arg0) becomes a reactive TEXT node and a reactive
 * prop entry (a Region in the props object) registers a per-attribute leaf effect — so only that DOM
 * position patches on a value change (no re-render). Function-valued handler props are captured for the
 * delegated dispatcher.
 *
 * The reactive host is bound AFTER construction (bindHost) — the derive hands it over just before the
 * walk, and resolveCall (which runs during the walk) reads it then.
 */
export class VDomHostEnv implements BindableHostEnv {
  private host: ReactiveHost | null = null;
  /** Bind the reactive host handed over by the derive before the walk runs. */
  bindHost(host: ReactiveHost): void { this.host = host; }

  resolveCall(
    head: string,
    key: string,
    args: Arg[],
    children: HostValue[],
    _span: SourceSpan,
  ): { handled: true; value: HostValue } | { handled: false } {
    if (!isElementHead(head)) return { handled: false };   // Capitalized → component wrapper → fragment

    const node: VNode = { tag: head, props: {}, children: children as VNode[], key };
    const a0 = args[0]?.value;

    // A LEADING PROPS object (head({ … }) or head({ … }, "text")) configures the element; a Region or any
    // other leading value is CONTENT. So: if arg0 is a plain (non-Region) object it is props and the
    // remaining args are content; otherwise EVERY arg is content (span(n) / span("hi") / span("a", "b")).
    // A Region is a plain object, so the Region check must come first or a reactive scalar would mis-route
    // into props.
    let contentArgs: Arg[];
    if (a0 !== undefined && !isRegion(a0) && typeof a0 === 'object' && a0 !== null && !Array.isArray(a0)) {
      this.applyProps(node, a0 as Record<string, unknown>);
      contentArgs = args.slice(1);
    } else {
      contentArgs = args;
    }

    // Content args become TEXT children, in author order, prepended before any wrap-block children. A
    // reactive scalar (span(n) / a trailing reactive arg) binds a leaf effect that seeds the vnode text
    // before the build and patches the live Text node on every later write (fine-grained, no reconcile);
    // a static value is a raw text node. setText owns both the vnode field + (once built) the DOM node.
    const textChildren: VNode[] = [];
    contentArgs.forEach((arg, i) => {
      if (arg.value === undefined) return;
      const textNode: VNode = { tag: TEXT, props: {}, children: [], key: `${key}/#text#${i}`, text: '' };
      if (isRegion(arg.value)) {
        this.host!.runLeafEffect((arg.value as Region).run, (out) => { setText(textNode, out); });
      } else {
        setText(textNode, arg.value);
      }
      textChildren.push(textNode);
    });
    node.children = [...textChildren, ...node.children];
    return { handled: true, value: node };
  }

  /** Route each prop: a handler function → node.handlers; a reactive Region → a leaf effect patching
   *  node.props[k] in place (fine-grained); a static value → set directly. `key` is identity, not an attr. */
  private applyProps(node: VNode, props: Record<string, unknown>): void {
    for (const [k, v] of Object.entries(props)) {
      if (k === 'key') continue;
      if (typeof v === 'function' && isHandlerName(k)) {
        (node.handlers ??= []).push({ event: k, fn: v as (arg: unknown) => void });
        continue;
      }
      // A `style` OBJECT whose entries contain a reactive Region (e.g. { color: c }) is a plain object,
      // so the top-level isRegion(v) check below is false. Bind ONE leaf effect that re-reads the whole
      // object (resolving each nested Region) and patches the single `style` attribute in place — the
      // fine-grained path, whole-attribute granularity (style is one attribute). A static/no-Region
      // object falls through to the plain assignment, which createDom/applyAttr serialize at build.
      if (k === 'style' && isStyleWithRegion(v)) {
        this.host!.runLeafEffect(() => resolveStyleRegions(v as Record<string, unknown>), (out) => { setAttr(node, k, out); });
        continue;
      }
      if (isRegion(v)) {
        // Seed the vnode prop before the build (applyAttrs reads it) and patch that one attribute on the
        // live element in place on every later value write — a value change updates the DOM with NO
        // reconcile (fine-grained). setAttr owns both writes so the vnode props + DOM never diverge.
        this.host!.runLeafEffect((v as Region).run, (out) => { setAttr(node, k, out); });
        continue;
      }
      node.props[k] = v;
    }
  }
}
