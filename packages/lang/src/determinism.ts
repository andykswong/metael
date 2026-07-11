// Seeded determinism substrate: grammar-free pure fns feeding `result = f(source, data, seed, …)`.
// mulberry32-class PRNG — same seed → same [0,1) sequence.
export const MAX_RANGE = 1_000_000;

/** Deterministic PRNG factory — same seed → same [0,1) sequence. */
export function makeSeededRng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** [0..n) with a hard cap (a value-producing op the step budget can't preempt — F determinism). */
export function range(n: number): number[] {
  if (n > MAX_RANGE || n < 0) return [];
  return Array.from({ length: Math.floor(n) }, (_, i) => i);
}
