import { describe, it, expect } from 'vitest';
import { makeSeededRng, range, MAX_RANGE } from './determinism.ts';

describe('determinism (grammar-free unit tests)', () => {
  it('same seed → identical sequence', () => {
    const a = makeSeededRng(42); const b = makeSeededRng(42);
    expect([a(), a(), a()]).toEqual([b(), b(), b()]);
  });
  it('different seeds → different sequences', () => {
    expect(makeSeededRng(1)()).not.toBe(makeSeededRng(2)());
  });
  it('rng output is in [0,1)', () => {
    const r = makeSeededRng(7);
    for (let i = 0; i < 100; i++) { const v = r(); expect(v).toBeGreaterThanOrEqual(0); expect(v).toBeLessThan(1); }
  });
  it('range produces [0..n)', () => { expect(range(4)).toEqual([0, 1, 2, 3]); });
  it('range caps at MAX_RANGE (returns empty + does not allocate unbounded)', () => {
    expect(range(MAX_RANGE + 1)).toEqual([]);
  });
});
