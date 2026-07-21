// packages/math/src/core/vec.test.ts
import { describe, it, expect } from 'vitest';
import {
  add, sub, mul, div, scale, dot, cross, normalize, length, distance,
  reflect, refract, faceforward,
} from './vec.ts';
import { matColumn, matScale } from './mat.ts';

describe('core vec (componentwise + geometric)', () => {
  it('add / sub componentwise', () => {
    expect(add([1, 2, 3], [4, 5, 6])).toEqual([5, 7, 9]);
    expect(sub([4, 5, 6], [1, 2, 3])).toEqual([3, 3, 3]);
  });
  it('mul / div componentwise', () => {
    expect(mul([2, 3, 4], [5, 6, 7])).toEqual([10, 18, 28]);
    expect(div([10, 18, 28], [5, 6, 7])).toEqual([2, 3, 4]);
  });
  it('scale by a scalar', () => { expect(scale([1, 2, 3], 2)).toEqual([2, 4, 6]); });
  it('dot product', () => { expect(dot([1, 2, 3], [4, 5, 6])).toBe(32); }); // 4+10+18
  it('cross product (right-handed basis)', () => {
    expect(cross([1, 0, 0], [0, 1, 0])).toEqual([0, 0, 1]);
    expect(cross([0, 1, 0], [0, 0, 1])).toEqual([1, 0, 0]);
  });
  it('length + normalize (3-4-5)', () => {
    expect(length([3, 4])).toBe(5);
    expect(normalize([3, 4])).toEqual([0.6, 0.8]);
  });
  it('distance is the length of the difference', () => { expect(distance([0, 0], [3, 4])).toBe(5); });
  it('reflect about a normal', () => {
    // I - 2·dot(I,N)·N; dot([1,-1],[0,1]) = -1 → [1,-1] + 2·[0,1] = [1,1]
    expect(reflect([1, -1], [0, 1])).toEqual([1, 1]);
  });
  it('refract straight-through at eta=1, and total internal reflection → zero', () => {
    expect(refract([0, -1], [0, 1], 1)).toEqual([0, -1]); // k = 1, √k = 1, N-term cancels
    expect(refract([1, 0], [0, 1], 2)).toEqual([0, 0]);   // k = 1 - 4·(1-0) = -3 < 0 → all-zeros
  });
  it('faceforward orients N away from I', () => {
    // dot(Nref,I) ≥ 0 → -N; dot(Nref,I) < 0 → N
    expect(faceforward([1, 2], [0, 1], [0, 1])).toEqual([-1, -2]);  // dot(Nref,I)=1 ≥ 0 → -N
    expect(faceforward([1, 2], [0, 1], [0, -1])).toEqual([1, 2]);   // dot(Nref,I)=-1 < 0 → N
  });

  it('out-param writes into and returns the same array', () => {
    const out: number[] = [];
    const r = add([1, 2], [3, 4], out);
    expect(r).toBe(out);
    expect(out).toEqual([4, 6]);
  });
  it('aliasing an input as the output is safe (add)', () => {
    const a = [1, 2, 3];
    const r = add(a, a, a); // out === a
    expect(r).toBe(a);
    expect(a).toEqual([2, 4, 6]);
  });
  it('aliasing an input as the output is safe (cross)', () => {
    const a = [1, 0, 0];
    const r = cross(a, [0, 1, 0], a); // out === a; components staged before write
    expect(r).toBe(a);
    expect(a).toEqual([0, 0, 1]);
  });
  it('normalize into an aliased output stays correct', () => {
    const v = [3, 4];
    normalize(v, v);
    expect(v).toEqual([0.6, 0.8]);
  });
});

describe('core mat scalar/column helpers', () => {
  it('matScale multiplies every element by a scalar', () => {
    expect(matScale([1, 2, 3, 4], 3)).toEqual([3, 6, 9, 12]);
  });
  it('matScale into an aliased output is safe', () => {
    const m = [1, 2, 3, 4];
    matScale(m, 2, m);
    expect(m).toEqual([2, 4, 6, 8]);
  });
  it('matColumn extracts a fresh copy of a column (column-major)', () => {
    const m = [1, 2, 3, 4, 5, 6]; // 2 rows × 3 cols, col-major
    expect(matColumn(m, 2, 3, 0)).toEqual([1, 2]);
    expect(matColumn(m, 2, 3, 2)).toEqual([5, 6]);
    const col = matColumn(m, 2, 3, 1);
    col[0] = 99;                  // mutating the copy must not touch the source
    expect(m).toEqual([1, 2, 3, 4, 5, 6]);
  });
});
