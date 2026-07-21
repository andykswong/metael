// packages/math/src/core/bits.ts
// Integer bit operations on 32-bit unsigned values. Each op coerces its input with `x >>> 0` so the
// argument is treated as a 32-bit unsigned integer, matching the shading-language `countOneBits`/
// `reverseBits` semantics. Pure number→number; no host or diagnostic concerns here.

/** Population count: the number of set bits in `x >>> 0` (a 32-bit unsigned integer). */
export function countOneBits(x: number): number {
  // `>>>= 1` shifts in a zero (unsigned), and `n !== 0` guards the high-bit case that a signed
  // `n > 0` test would mishandle — so 0xFFFFFFFF correctly yields 32.
  let n = x >>> 0;
  let c = 0;
  while (n !== 0) {
    c += n & 1;
    n >>>= 1;
  }
  return c;
}

/** Reverse the 32 bits of `x >>> 0`, returning an unsigned 32-bit result (bit 0 ↔ bit 31). */
export function reverseBits(x: number): number {
  let n = x >>> 0;
  let r = 0;
  for (let i = 0; i < 32; i++) {
    r = (r << 1) | (n & 1);
    n >>>= 1;
  }
  // A left-shift produces a signed int32; `>>> 0` makes the final value a non-negative uint32.
  return r >>> 0;
}
