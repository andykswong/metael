// packages/math/src/core/scratch.ts
// A tiny reusable-buffer pool for read-before-write matrix ops (matmul / transpose / inverse).
//
// Why it exists: those ops must read every element of their inputs while producing the result, so
// writing the result directly into an output that ALIASES an input (e.g. `matmul(a, …, a)`) would
// corrupt the computation mid-flight. The fix is to compute the whole result into a private temporary
// and only then copy it into the caller's output — so aliasing is always safe. Allocating that
// temporary on every call would churn the heap, so we hand out a per-size buffer that each op fully
// overwrites before reading back.
//
// Correctness rests on this code being synchronous and single-threaded with no re-entrancy: a caller
// asks for one scratch buffer, fills it completely, copies out, and returns — no operation calls back
// into another that would need the same-size buffer at the same time. The pool therefore just memoises
// one array per requested size.

const pool = new Map<number, number[]>();

/**
 * Returns a reusable temporary buffer of the given length. The contents are undefined on entry — the
 * caller MUST fully overwrite the region it uses before reading it back. Not part of the public math
 * surface; internal to the read-before-write matrix ops.
 */
export function scratch(size: number): number[] {
  let buf = pool.get(size);
  if (buf === undefined) {
    buf = new Array<number>(size);
    pool.set(size, buf);
  }
  return buf;
}
