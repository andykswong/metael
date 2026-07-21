import { describe, it, expect } from 'vitest';
import { evaluateProgram, PlainStorageHost, RecordingHostEnv } from '@metael/lang';
import { STD_BUILTINS } from './index.ts';

const run = (src: string, data?: unknown) =>
  evaluateProgram(src, { data, host: new PlainStorageHost(), env: new RecordingHostEnv(), builtins: [STD_BUILTINS] });

describe('std collections builtins', () => {
  it('map applies a callback and returns a new frozen array', () => {
    expect(run('map([1, 2, 3], (x) => x * 2)').value).toEqual([2, 4, 6]);
    const r = run('const m = map([1], (x) => x); m[0] = 9; m');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-IMMUTABLE')).toBe(true);
  });
  it('map passes the index as the second callback argument', () => {
    expect(run('map(["a", "b"], (x, i) => i)').value).toEqual([0, 1]);
  });
  it('a user function works as a map callback (not only an arrow)', () => {
    expect(run('function dbl(x) { x * 2 } map([1, 2, 3], dbl)').value).toEqual([2, 4, 6]);
  });
  it('filter keeps truthy-predicate elements', () => {
    expect(run('filter([1, 2, 3, 4], (x) => x % 2 == 0)').value).toEqual([2, 4]);
  });
  it('reduce folds with an initial accumulator', () => {
    expect(run('reduce([1, 2, 3, 4], (acc, x) => acc + x, 0)').value).toBe(10);
  });
  it('some / every short-circuit on the predicate', () => {
    expect(run('some([1, 2, 3], (x) => x > 2)').value).toBe(true);
    expect(run('some([1, 2, 3], (x) => x > 9)').value).toBe(false);
    expect(run('every([2, 4, 6], (x) => x % 2 == 0)').value).toBe(true);
    expect(run('every([2, 3], (x) => x % 2 == 0)').value).toBe(false);
  });
  it('find returns the first match or null; findIndex returns the index or -1', () => {
    expect(run('find([1, 2, 3], (x) => x > 1)').value).toBe(2);
    expect(run('find([1, 2, 3], (x) => x > 9)').value).toBe(null);
    expect(run('findIndex([1, 2, 3], (x) => x > 1)').value).toBe(1);
    expect(run('findIndex([1, 2, 3], (x) => x > 9)').value).toBe(-1);
  });
  it('includes uses loose equality', () => {
    expect(run('includes([1, 2, 3], 2)').value).toBe(true);
    expect(run('includes([1, 2, 3], "2")').value).toBe(true);
    expect(run('includes([1, 2, 3], 9)').value).toBe(false);
  });
  it('sort has a total, stable, default order and does not mutate', () => {
    expect(run('sort([3, 1, 2])').value).toEqual([1, 2, 3]);
    expect(run('sort([3, 1, 2], (a, b) => b - a)').value).toEqual([3, 2, 1]);
  });
  it('reverse returns a new reversed array', () => {
    expect(run('reverse([1, 2, 3])').value).toEqual([3, 2, 1]);
  });
  it('slice clamps start/end like Array.prototype.slice, with negatives from the end', () => {
    expect(run('slice([1, 2, 3, 4, 5], 1, 3)').value).toEqual([2, 3]);
    expect(run('slice([1, 2, 3, 4, 5], -2)').value).toEqual([4, 5]);
    expect(run('slice([1, 2, 3], 1)').value).toEqual([2, 3]);
  });
  it('a wrong-shape arg is fail-loud ML-LANG-BUILTIN-ARG + a safe empty result', () => {
    const r = run('map(5, (x) => x)');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-BUILTIN-ARG')).toBe(true);
    expect(r.value).toEqual([]);
  });
  it('a user function shadows the intrinsic (unbound-head-only)', () => {
    expect(run('function map() { 99 } map([1, 2], (x) => x)').value).toBe(99);
  });
});
