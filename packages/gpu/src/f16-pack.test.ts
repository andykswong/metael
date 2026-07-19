// Unit tests for the PURE f32 ↔ IEEE-754 binary16 (half) bit conversions + 4-byte alignment helpers in
// `f16-pack.ts`. No WebGPU device is needed: `f32ToF16`/`f16ToF32`/`unpackF16` are exported and directly
// callable, so they are asserted against known-correct half-precision bit patterns here (the `packF16`
// Float16Array fast path in Node otherwise never exercises the scalar conversions).
//
// Half layout: 1 sign / 5 exponent (bias 15) / 10 mantissa bits.
import { describe, it, expect, vi } from 'vitest';
import { align4, f32ToF16, f16ToF32, packF16, unpackF16 } from './f16-pack.ts';

describe('align4 — round a byte count up to the next multiple of 4', () => {
  it('rounds each input to the next multiple of 4', () => {
    expect([align4(0), align4(1), align4(2), align4(3), align4(4), align4(5)]).toEqual([0, 4, 4, 4, 4, 8]);
  });
});

describe('f32ToF16 — f32 → half bit pattern (round-to-nearest-even)', () => {
  it('encodes signed zero', () => {
    expect(f32ToF16(0)).toBe(0x0000);
    expect(f32ToF16(-0)).toBe(0x8000);
  });
  it('encodes exactly-representable normal values', () => {
    expect(f32ToF16(1.0)).toBe(0x3c00);
    expect(f32ToF16(2.0)).toBe(0x4000);
    expect(f32ToF16(0.5)).toBe(0x3800);
    expect(f32ToF16(-2.0)).toBe(0xc000);
    expect(f32ToF16(65504)).toBe(0x7bff);   // largest finite half (max normal)
  });
  it('encodes ±Infinity', () => {
    expect(f32ToF16(Infinity)).toBe(0x7c00);
    expect(f32ToF16(-Infinity)).toBe(0xfc00);
  });
  it('encodes NaN as a quiet NaN (exp all ones, mantissa nonzero)', () => {
    const h = f32ToF16(NaN);
    expect((h >>> 10) & 0x1f).toBe(0x1f);   // exponent field all ones
    expect(h & 0x3ff).not.toBe(0);          // mantissa nonzero → NaN, not Inf
    expect(h).toBe(0x7e00);                 // this module's quiet-NaN pattern
  });
  it('overflows a value above half-max to ±Inf', () => {
    expect(f32ToF16(70000)).toBe(0x7c00);    // > 65504 → +Inf
    expect(f32ToF16(-70000)).toBe(0xfc00);   // → -Inf
  });
  it('encodes subnormal (denormal) values', () => {
    expect(f32ToF16(2 ** -16)).toBe(0x0100);   // a mid-range subnormal
    expect(f32ToF16(2 ** -24)).toBe(0x0001);   // the smallest positive subnormal
    expect(f32ToF16(6e-8)).toBe(0x0001);       // ≈ 2**-24, rounds to the smallest subnormal
  });
  it('applies round-to-nearest-even when a subnormal value falls between grid points', () => {
    // Subnormal grid step is 2**-24. These exercise the round-UP branch of the subnormal path:
    expect(f32ToF16(1.75 * 2 ** -24)).toBe(0x0002);   // rem > halfway → round up
    expect(f32ToF16(1.5 * 2 ** -24)).toBe(0x0002);    // exact tie, half (0x0001) odd → round up to even
    expect(f32ToF16(2.5 * 2 ** -24)).toBe(0x0002);    // exact tie, half (0x0002) even → stay (round down)
  });
  it('underflows a too-small value to signed zero', () => {
    expect(f32ToF16(1e-10)).toBe(0x0000);    // below half's smallest subnormal → +0
    expect(f32ToF16(-1e-10)).toBe(0x8000);   // → -0
  });
  it('applies round-to-nearest-EVEN at a tie in the normal range', () => {
    // Half step at exponent 2^11 is 2. 2049 is a tie between 2048 (even mantissa) and 2050;
    // round-half-to-even picks 2048. 2051 ties between 2050 and 2052; even picks 2052.
    expect(f16ToF32(f32ToF16(2049))).toBe(2048);
    expect(f16ToF32(f32ToF16(2051))).toBe(2052);
    expect(f32ToF16(2049)).toBe(0x6800);   // 2048
    expect(f32ToF16(2051)).toBe(0x6802);   // 2052
  });
});

describe('f16ToF32 — half bit pattern → f32', () => {
  it('decodes signed zero (preserving -0)', () => {
    expect(f16ToF32(0x0000)).toBe(0);
    expect(Object.is(f16ToF32(0x8000), -0)).toBe(true);
  });
  it('decodes ±Infinity', () => {
    expect(f16ToF32(0x7c00)).toBe(Infinity);
    expect(f16ToF32(0xfc00)).toBe(-Infinity);
  });
  it('decodes NaN', () => {
    expect(Number.isNaN(f16ToF32(0x7e00))).toBe(true);
  });
  it('decodes subnormals via mantissa normalization', () => {
    expect(f16ToF32(0x0001)).toBe(2 ** -24);              // smallest positive subnormal
    expect(f16ToF32(0x0100)).toBe(2 ** -16);              // a mid-range subnormal
    expect(f16ToF32(0x03ff)).toBe((2 ** -14) * (1023 / 1024));   // largest subnormal
  });
  it('round-trips every exactly-representable value through f32→half→f32', () => {
    for (const x of [0, 1.0, 2.0, 0.5, -2.0, 65504, 2 ** -16, 2 ** -24]) {
      expect(f16ToF32(f32ToF16(x))).toBe(x);
    }
    expect(Object.is(f16ToF32(f32ToF16(-0)), -0)).toBe(true);
  });
});

describe('packF16 / unpackF16 — even-length alignment + round-trip', () => {
  it('rounds an odd-length input up to an even-length, 4-byte-aligned Uint16Array with a trailing 0 pad', () => {
    const packed = packF16(Float32Array.from([1, 2, 3]));
    expect(packed.length).toBe(4);              // 3 → 4 (even)
    expect(packed.length % 2).toBe(0);
    expect(packed.byteLength % 4).toBe(0);      // WebGPU writeBuffer requirement
    expect(packed[3]).toBe(0);                  // trailing pad slot
    expect(Array.from(unpackF16(packed, 3))).toEqual([1, 2, 3]);   // ignores the pad
  });
  it('leaves an even-length input length unchanged', () => {
    const packed = packF16(Float32Array.from([1, 2, 3, 4]));
    expect(packed.length).toBe(4);
    expect(packed.byteLength % 4).toBe(0);
    expect(Array.from(unpackF16(packed, 4))).toEqual([1, 2, 3, 4]);
  });
  it('round-trips half-representable values through packF16 → unpackF16', () => {
    const src = Float32Array.from([0.5, -2, 65504, 2 ** -16, 0]);
    const packed = packF16(src);
    const back = unpackF16(packed, src.length);
    expect(Array.from(back)).toEqual([0.5, -2, 65504, 2 ** -16, 0]);
    // -0 survives the round-trip (toEqual treats -0 as 0, so assert it separately with Object.is):
    expect(Object.is(unpackF16(packF16(Float32Array.from([-0])), 1)[0], -0)).toBe(true);
  });
});

describe('packF16 — scalar fallback loop (no Float16Array)', () => {
  it('packs via the scalar f32ToF16 loop when globalThis.Float16Array is absent', async () => {
    // HAS_FLOAT16 / Float16ArrayRef are captured at module load, so force a FRESH module instance whose
    // capture sees no Float16Array. `vi.stubGlobal(..., undefined)` makes `typeof Float16Array` === 'undefined'
    // (HAS_FLOAT16 → false); `vi.resetModules()` drops the cached instance so the re-import re-evaluates.
    vi.stubGlobal('Float16Array', undefined);
    vi.resetModules();
    try {
      const mod = await import('./f16-pack.ts');
      const odd = mod.packF16(Float32Array.from([1, 2, 3]));   // scalar loop (lines 87-89)
      expect(odd.length).toBe(4);
      expect(odd.byteLength % 4).toBe(0);
      expect(odd[3]).toBe(0);   // pad left 0 by the scalar path
      expect(Array.from(mod.unpackF16(odd, 3))).toEqual([1, 2, 3]);
      const even = mod.packF16(Float32Array.from([1, 2, 3, 4]));
      expect(even.length).toBe(4);
      expect(Array.from(mod.unpackF16(even, 4))).toEqual([1, 2, 3, 4]);
    } finally {
      vi.unstubAllGlobals();
      vi.resetModules();
    }
  });
});
