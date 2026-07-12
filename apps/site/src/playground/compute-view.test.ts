import { describe, it, expect } from 'vitest';
import { prettyValue } from './compute-view.ts';

describe('prettyValue', () => {
  it('pretty-prints scalars', () => {
    expect(prettyValue(42)).toBe('42');
    expect(prettyValue('hi')).toBe('"hi"');
    expect(prettyValue(true)).toBe('true');
    expect(prettyValue(null)).toBe('null');
  });

  it('keeps a short array compact on one line', () => {
    expect(prettyValue([1, 2, 3])).toBe('[1, 2, 3]');
  });

  it('keeps the fib-shaped flat array on one line (not one number per row)', () => {
    expect(prettyValue([0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]))
      .toBe('[0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]');
  });

  it('greedy-wraps a long flat array across lines (not one per line)', () => {
    const out = prettyValue(Array.from({ length: 40 }, (_, i) => i));
    expect(out.startsWith('[\n')).toBe(true);
    // multiple values share a line
    expect(out).toContain('0, 1, 2, 3');
    // but it did break onto more than one row
    expect(out.split('\n').length).toBeGreaterThan(2);
  });

  it('breaks a wide array of objects one object per line, each object compact', () => {
    // 3-field objects overflow the 72-char budget as a whole array → break per-object, each still compact.
    const out = prettyValue([
      { name: 'Ann', score: 91, grade: 'A' },
      { name: 'Cy', score: 73, grade: 'C' },
    ]);
    expect(out).toBe(
      '[\n  { "name": "Ann", "score": 91, "grade": "A" },\n  { "name": "Cy", "score": 73, "grade": "C" }\n]',
    );
  });

  it('keeps a small nested object compact when it fits', () => {
    expect(prettyValue({ a: 1, b: [2] })).toBe('{ "a": 1, "b": [2] }');
  });

  it('breaks a wide object across lines when it overflows', () => {
    const wide = { alpha: 111111, bravo: 222222, charlie: 333333, delta: 444444, echo: 555555 };
    const out = prettyValue(wide);
    expect(out.startsWith('{\n')).toBe(true);
    expect(out).toContain('\n  "alpha": 111111,');
  });

  it('renders a top-level function as a placeholder, never empty', () => {
    expect(prettyValue(() => 1)).toBe('<function>');
  });

  it('renders undefined as a placeholder', () => {
    expect(prettyValue(undefined)).toBe('undefined');
  });

  it('renders a nested function as an unquoted placeholder', () => {
    expect(prettyValue({ f: () => 1 })).toBe('{ "f": <function> }');
  });

  it('renders an empty array and empty object compactly', () => {
    expect(prettyValue([])).toBe('[]');
    expect(prettyValue({})).toBe('{}');
  });
});
