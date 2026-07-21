// packages/math/src/core/mat.test.ts
import { describe, it, expect } from 'vitest';
import { matmul, transpose, determinant, inverse, matrixCompMult, matColumn } from './mat.ts';

describe('core mat (column-major, aliasing-safe)', () => {
  it('matmul into an aliased input is correct', () => {
    // A = [[1,2],[3,4]] column-major flat = [1,3,2,4]; A*A = [[7,10],[15,22]] col-major [7,15,10,22]
    const a = [1, 3, 2, 4];
    const out = matmul(a, 2, 2, a, 2, 2, a); // out === a (alias)
    expect(out).toEqual([7, 15, 10, 22]);
  });
  it('transpose 2x3 → 3x2', () => { expect(transpose([1,2,3,4,5,6], 2, 3)).toEqual([1,3,5,2,4,6]); });
  it('determinant 2x2', () => { expect(determinant([1,3,2,4], 2)).toBe(-2); });
  it('inverse 2x2 round-trips to identity via matmul', () => {
    const m = [4, 2, 7, 6]; const inv = inverse(m, 2); const p = matmul(m, 2, 2, inv, 2, 2);
    expect(p[0]).toBeCloseTo(1, 10); expect(p[3]).toBeCloseTo(1, 10);
  });
  it('matrixCompMult is Hadamard', () => { expect(matrixCompMult([1,2,3,4], [5,6,7,8])).toEqual([5,12,21,32]); });
  it('matColumn extracts the i-th column (column-major)', () => {
    // m4 (4x4) col-major: column c occupies flat [c*4 .. c*4+3]
    const m4 = [1,2,3,0, 4,5,6,0, 7,8,9,0, 10,11,12,1];
    expect(matColumn(m4, 4, 4, 0)).toEqual([1,2,3,0]);
    expect(matColumn(m4, 4, 4, 3)).toEqual([10,11,12,1]);  // translation column of a TRS mat4
  });
});
