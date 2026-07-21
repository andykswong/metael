// packages/math/src/core/scalar.test.ts
import { describe, it, expect } from 'vitest';
import { mod, roundHalfEven, asinh, mix, smoothstep, log, log2, inverseSqrt, sqrt, asin, acos } from './scalar.ts';

describe('core scalar math', () => {
  it('mod is floored (differs from JS % for negatives)', () => {
    expect(mod(-1, 3)).toBe(2);       // floored: -1 - 3*floor(-1/3) = -1 - 3*(-1) = 2
    expect(-1 % 3).toBe(-1);          // JS truncated remainder, for contrast
    expect(mod(7, 3)).toBe(1);
  });
  it('roundHalfEven (banker)', () => { expect(roundHalfEven(0.5)).toBe(0); expect(roundHalfEven(1.5)).toBe(2); expect(roundHalfEven(2.5)).toBe(2); });
  it('asinh matches Math.asinh', () => { expect(asinh(2)).toBeCloseTo(Math.asinh(2), 12); });
  it('mix + smoothstep', () => { expect(mix(0, 10, 0.5)).toBe(5); expect(smoothstep(0, 1, 0.5)).toBeCloseTo(0.5, 12); });
  // Parity with the interpreter's internal NaN-clamp + degenerate-edge guard:
  it('domain-guarded funcs return NaN, not ±Inf', () => {
    expect(log(0)).toBeNaN(); expect(log2(0)).toBeNaN(); expect(inverseSqrt(0)).toBeNaN();
    expect(sqrt(-1)).toBeNaN(); expect(asin(2)).toBeNaN(); expect(acos(-2)).toBeNaN();
  });
  it('smoothstep with equal edges returns 0 (not NaN)', () => { expect(smoothstep(5, 5, 10)).toBe(0); });
});
