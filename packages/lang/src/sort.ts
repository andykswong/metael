// A total, stable, deterministic ordering for the `sort` builtin. The DEFAULT order groups values by
// a fixed type rank (null < bool < number < string < object/array) and orders within a group; NaN is
// pinned to the end of the number group (never poisons, unlike the `<`-operator comparison). The sort
// itself is a stable bottom-up merge sort operating on a COPY (non-mutating).

/** Fixed cross-type rank: null(0) < bool(1) < number(2) < string(3) < everything-else/object(4). */
function typeRank(v: unknown): number {
  if (v === null || v === undefined) return 0;
  if (typeof v === 'boolean') return 1;
  if (typeof v === 'number') return 2;
  if (typeof v === 'string') return 3;
  return 4;
}

/** The default total order. Returns <0, 0, or >0. Total (never NaN): mixed types order by rank;
 *  NaN sorts after every real number but stays within the number group (before strings). */
export function defaultCompare(a: unknown, b: unknown): number {
  const ra = typeRank(a);
  const rb = typeRank(b);
  if (ra !== rb) return ra - rb;
  switch (ra) {
    case 1: return (a === b) ? 0 : (a === false ? -1 : 1);           // false < true
    case 2: {
      const na = a as number;
      const nb = b as number;
      const aNaN = Number.isNaN(na);
      const bNaN = Number.isNaN(nb);
      if (aNaN && bNaN) return 0;
      if (aNaN) return 1;      // NaN after any real number
      if (bNaN) return -1;
      return na < nb ? -1 : na > nb ? 1 : 0;
    }
    case 3: { const sa = a as string; const sb = b as string; return sa < sb ? -1 : sa > sb ? 1 : 0; }
    default: return 0;                                               // null==null, object==object (stable keeps order)
  }
}

/** A stable bottom-up merge sort over a COPY of `xs`. `cmp` must return a number; equal (0) elements
 *  keep their relative input order. Never mutates `xs`. */
export function stableSort<T>(xs: readonly T[], cmp: (a: T, b: T) => number): T[] {
  const a = xs.slice();
  const n = a.length;
  if (n < 2) return a;
  const buf = new Array<T>(n);
  for (let width = 1; width < n; width *= 2) {
    for (let lo = 0; lo < n; lo += 2 * width) {
      const mid = Math.min(lo + width, n);
      const hi = Math.min(lo + 2 * width, n);
      let i = lo;
      let j = mid;
      let k = lo;
      while (i < mid && j < hi) {
        // `<= 0` keeps left before right on ties → stable.
        if (cmp(a[i]!, a[j]!) <= 0) buf[k++] = a[i++]!;
        else buf[k++] = a[j++]!;
      }
      while (i < mid) buf[k++] = a[i++]!;
      while (j < hi) buf[k++] = a[j++]!;
    }
    for (let i = 0; i < n; i++) a[i] = buf[i]!;
  }
  return a;
}
