// packages/math/src/core/bits.test.ts
import { describe, it, expect } from 'vitest';
import { countOneBits, reverseBits } from './bits.ts';

describe('core bit ops (32-bit)', () => {
  it('countOneBits is the population count over x >>> 0', () => {
    expect(countOneBits(0b1011)).toBe(3);
    expect(countOneBits(0)).toBe(0);
    expect(countOneBits(1)).toBe(1);
    // High bit set: the loop must treat x as unsigned (0xFFFFFFFF has all 32 bits set).
    expect(countOneBits(0xffffffff)).toBe(32);
    expect(countOneBits(0x80000000)).toBe(1);
  });

  it('reverseBits reverses the 32 bits and returns an unsigned result', () => {
    // bit 0 → bit 31
    expect(reverseBits(1)).toBe(0x80000000 >>> 0);
    expect(reverseBits(1)).toBe(2147483648);
    // bit 31 → bit 0
    expect(reverseBits(0x80000000)).toBe(1);
    expect(reverseBits(0)).toBe(0);
    // All bits set is its own reverse.
    expect(reverseBits(0xffffffff)).toBe(0xffffffff);
    expect(reverseBits(0xffffffff)).toBe(4294967295);
    // Result is always non-negative (final >>> 0).
    expect(reverseBits(1)).toBeGreaterThanOrEqual(0);
  });

  it('reverseBits is an involution (round-trips)', () => {
    expect(reverseBits(reverseBits(0x12345678))).toBe(0x12345678);
  });
});
