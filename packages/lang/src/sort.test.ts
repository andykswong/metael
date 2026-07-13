import { describe, it, expect } from 'vitest';
import { defaultCompare, stableSort } from './sort.ts';

describe('defaultCompare — total order null < bool < number < string < object', () => {
  it('orders across type ranks', () => {
    expect(defaultCompare(null, false)).toBeLessThan(0);
    expect(defaultCompare(false, 1)).toBeLessThan(0);
    expect(defaultCompare(1, 'a')).toBeLessThan(0);
    expect(defaultCompare('a', {})).toBeLessThan(0);
  });
  it('orders numbers ascending and pins NaN to the end of the number group', () => {
    expect(defaultCompare(1, 2)).toBeLessThan(0);
    expect(defaultCompare(NaN, 2)).toBeGreaterThan(0);   // NaN after any real number
    expect(defaultCompare(NaN, 'a')).toBeLessThan(0);    // still a number, before strings
    expect(defaultCompare(NaN, NaN)).toBe(0);
  });
  it('orders booleans false < true and strings lexicographically', () => {
    expect(defaultCompare(false, true)).toBeLessThan(0);
    expect(defaultCompare('a', 'b')).toBeLessThan(0);
    expect(defaultCompare('b', 'a')).toBeGreaterThan(0);
  });
});

describe('stableSort', () => {
  it('sorts with the default order and does not mutate the input', () => {
    const input = [3, 1, 2];
    const out = stableSort(input, defaultCompare);
    expect(out).toEqual([1, 2, 3]);
    expect(input).toEqual([3, 1, 2]);   // original untouched
  });
  it('is stable — equal elements keep input order', () => {
    const input = [{ k: 1, i: 'a' }, { k: 1, i: 'b' }, { k: 0, i: 'c' }];
    const out = stableSort(input, (a, b) => (a as { k: number }).k - (b as { k: number }).k);
    expect(out.map((x) => (x as { i: string }).i)).toEqual(['c', 'a', 'b']);
  });
});
