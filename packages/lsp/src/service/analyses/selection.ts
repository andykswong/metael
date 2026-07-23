import type { Document } from '../document.ts';
import type { SvcSelection, SvcSpan } from '../results.ts';

/** Recursion ceiling for the AST walk, matching the parser's nesting bound so a pathological or partial
 *  (mid-edit) tree fails closed rather than overflowing the JS stack. */
const MAX_WALK_DEPTH = 512;

/** An empty subtree extent (no spanned descendants): a start greater than its end, so it contains
 *  nothing. */
const EMPTY: readonly [number, number] = [Infinity, -Infinity];

/**
 * Compute selection ranges for each requested offset — the widening chain of source ranges an editor
 * cycles through on "expand selection".
 *
 * @remarks
 * For each offset the walk collects the *source extent* of every AST node whose extent contains the
 * offset, then returns them innermost-first, widening outward. A node's extent is the min start / max
 * end over its whole subtree — not its bare `span`, which anchors only a keyword or operator — so a
 * parent's extent always encloses its children's and the containing extents form a true nesting chain.
 * The whole document is added as the outermost candidate. Duplicates are removed, candidates are ordered
 * narrowest-first, and a strict containment chain is built greedily so each range in `ranges` properly
 * contains the previous one (`ranges[0]` is the tightest enclosing span, the last is the widest). Each
 * requested offset yields one {@link SvcSelection}, in request order. Pure and total.
 */
export function computeSelection(doc: Document, offsets: readonly number[]): readonly SvcSelection[] {
  const docRange: SvcSpan = { start: 0, end: doc.text.length };
  return offsets.map((offset) => {
    const candidates: SvcSpan[] = [];
    const collect = (lo: number, hi: number): void => {
      if (lo <= hi && lo <= offset && offset <= hi) candidates.push({ start: lo, end: hi });
    };
    for (const stmt of doc.parse.program.stmts ?? []) extentOf(stmt, 0, collect);
    collect(docRange.start, docRange.end);
    return { ranges: chain(candidates) };
  });
}

/**
 * Return `node`'s subtree extent as `[start, end]`, reporting via `emit` every node extent along the
 * way. Statement/expression spans anchor only a keyword or operator, so the true extent is the min/max
 * over the node's own span and all its descendants' extents.
 */
function extentOf(node: unknown, depth: number, emit: (lo: number, hi: number) => void): readonly [number, number] {
  if (depth > MAX_WALK_DEPTH || !node || typeof node !== 'object') return EMPTY;
  let lo = Infinity;
  let hi = -Infinity;
  const span = (node as { span?: { start?: unknown; end?: unknown } }).span;
  if (span && typeof span.start === 'number' && typeof span.end === 'number') {
    lo = span.start;
    hi = span.end;
  }
  const d = depth + 1;
  for (const key of Object.keys(node as Record<string, unknown>)) {
    if (key === 'span') continue;
    const child = (node as Record<string, unknown>)[key];
    if (Array.isArray(child)) {
      for (const c of child) {
        const [clo, chi] = extentOf(c, d, emit);
        if (clo < lo) lo = clo;
        if (chi > hi) hi = chi;
      }
    } else if (child && typeof child === 'object') {
      const [clo, chi] = extentOf(child, d, emit);
      if (clo < lo) lo = clo;
      if (chi > hi) hi = chi;
    }
  }
  if (lo <= hi) emit(lo, hi);
  return [lo, hi];
}

/** Deduplicate, order narrowest-first, and reduce to a strict containment chain so each kept range
 *  properly encloses the one before it. */
function chain(candidates: readonly SvcSpan[]): readonly SvcSpan[] {
  const uniq = new Map<string, SvcSpan>();
  for (const c of candidates) uniq.set(`${c.start}:${c.end}`, c);
  const sorted = [...uniq.values()].sort((a, b) => (a.end - a.start) - (b.end - b.start) || a.start - b.start);
  const out: SvcSpan[] = [];
  for (const c of sorted) {
    const last = out[out.length - 1];
    if (!last) out.push(c);
    else if (c.start <= last.start && c.end >= last.end && (c.start < last.start || c.end > last.end)) out.push(c);
  }
  return out;
}
