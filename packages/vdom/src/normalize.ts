import { type VNode } from './vnode.ts';

/** A producer result entry: a VNode, or a conditional hole (`cond && node` → false/null/undefined) that
 *  is dropped — the same JSX-conditional idiom `h()` accepts for children. */
export type RenderNode = VNode | null | undefined | boolean;

/** Normalize a producer's return value into the `VNode[]` the render core consumes: arrayify a single
 *  result and drop conditional holes (`null`/`undefined`/`false`/`true`), so `() => null` (empty) and
 *  `() => [cond && node, ...]` (a top-level JSX-conditional) are tolerated. */
export function normalizeNodes(raw: RenderNode | RenderNode[]): VNode[] {
  return (Array.isArray(raw) ? raw : [raw]).filter(
    (n): n is VNode => n !== null && n !== undefined && n !== false && n !== true,
  );
}
