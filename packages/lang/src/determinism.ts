// Seeded determinism substrate: grammar-free pure fns feeding `result = f(source, data, seed, …)`.
// mulberry32-class PRNG — same seed → same [0,1) sequence.

/**
 * The inclusive upper bound on the argument to {@link range}: a request for more than this many
 * elements yields an empty array.
 *
 * @remarks Caps a value-producing op the step budget cannot preempt (one {@link range} call
 * materializes the whole array in a single step), so a large `range(n)` fails closed to `[]` rather
 * than allocating unboundedly.
 */
export const MAX_RANGE = 1_000_000;

/**
 * Build a deterministic pseudo-random number generator: the same `seed` always yields the same
 * `[0, 1)` sequence, so a seeded run reproduces exactly.
 *
 * @param seed - the seed value; the returned generator's sequence is a pure function of it.
 * @returns a zero-argument function that returns the next value in `[0, 1)` on each call.
 * @remarks A mulberry32-class generator — small, self-contained, and reseeded identically on a fresh
 * run, so a re-run is byte-stable.
 */
export function makeSeededRng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Build the integer array `[0, 1, …, n-1]`, bounded by a hard cap.
 *
 * @param n - the exclusive upper bound; fractional values are floored.
 * @returns the array `[0 .. floor(n))`, or an empty array when `n` is negative or exceeds
 *          {@link MAX_RANGE}.
 * @remarks Fails closed to `[]` past {@link MAX_RANGE} because a single call materializes the whole
 * array in one evaluation step, so the step budget cannot preempt an unbounded request.
 */
export function range(n: number): number[] {
  if (n > MAX_RANGE || n < 0) return [];
  return Array.from({ length: Math.floor(n) }, (_, i) => i);
}
