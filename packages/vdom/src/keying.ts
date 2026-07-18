// packages/vdom/src/keying.ts
// Kind-namespaced keying: assign each vnode a stable key so that reconcile (which matches purely by key,
// and whose patchNode assumes a key match implies a stable tag) never aliases nodes of different kinds at
// the same position. Mirrors the DSL path's PathKeyMinter exactly:
//   • element (unkeyed): `${parentKey}/${tag}#${perTagOrdinal}`   — per-parent, per-tag counter
//   • element (caller key): `${parentKey}/${tag}[${authorKey}]`   — tag-namespaced bracket form
//   • text vnode:            `${parentKey}/#text#${textOrdinal}`   — its own ordinal space
//   • fragment (tag ''):     `${parentKey}/frag#${fragOrdinal}`    — its own ordinal space
// A flat positional index would let an unkeyed conditional sibling shift a <span>'s key onto a former
// <p>'s slot (patchNode would then apply span props to the <p> element). Kind-namespacing removes that
// cross-kind aliasing. The residual same-tag shift (remove the first of two <span>s) is inherent and equal
// to the DSL's behavior — callers pass an explicit `key` for dynamic same-kind lists. Runs as a post-build
// pass over the tree h() returns (h leaves key=''). Mutates keys in place.
import { FRAGMENT, TEXT, type VNode } from './vnode.ts';
import { userKeyOf } from './h.ts';

export function assignKeys(nodes: readonly VNode[], parentKey: string): void {
  const ordinals = new Map<string, number>();   // per-tag element counter
  let textOrdinal = 0;
  let fragOrdinal = 0;
  for (const n of nodes) {
    let seg: string;
    if (n.tag === TEXT) {
      seg = `#text#${textOrdinal++}`;
    } else if (n.tag === FRAGMENT) {
      seg = `frag#${fragOrdinal++}`;
    } else {
      const uk = userKeyOf(n);
      if (uk !== undefined) {
        seg = `${n.tag}[${uk}]`;
      } else {
        const next = ordinals.get(n.tag) ?? 0;
        ordinals.set(n.tag, next + 1);
        seg = `${n.tag}#${next}`;
      }
    }
    (n as { key: string }).key = `${parentKey}/${seg}`;
    if (n.children.length) assignKeys(n.children, n.key);
  }
}
