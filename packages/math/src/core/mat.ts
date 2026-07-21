// packages/math/src/core/mat.ts
// Plain flat-store, COLUMN-MAJOR matrix math over `number[]` (f64 precision).
//
// Column-major flat index: element (row, col) lives at flat index `col*rows + row`, so a matrix's flat
// array is column 0 top-to-bottom, then column 1, and so on.
//
// Out-param convention: `out = op(...args, out?)`. `out` omitted → a fresh `number[]`; supplied → written
// into and returned. Aliasing an input as the output is always legal. The read-before-write ops
// (matmul / transpose / inverse) read every element of their inputs while producing the result, so they
// compute the whole result into an aliasing-safe scratch buffer first and only then copy it into `out` —
// an aliased output can never corrupt the computation mid-flight. Elementwise ops (matrixCompMult /
// matScale) touch only index `i` to write `out[i]`, so they are aliasing-safe without a temporary.
//
// Precision-agnostic: no f32 rounding here. Shape/zero-length validation is a language-surface concern;
// these assume well-formed inputs (matmul requires aCols === bRows). A singular `inverse` yields ±Inf/NaN
// with no guard — matching the shader targets, which neither guard nor define a singular inverse.

import { scratch } from './scratch.ts';

/**
 * Column-major matrix product A(aRows×aCols) · B(bRows×bCols) → (aRows×bCols). Requires aCols === bRows.
 * out[c*R + r] = Σ_k a[k*R + r] · b[c*K + k]. Computed through scratch, then copied to `out`, so aliasing
 * `out` with `a` or `b` is safe.
 */
export function matmul(
  a: readonly number[], aRows: number, aCols: number,
  b: readonly number[], _bRows: number, bCols: number,
  out?: number[],
): number[] {
  const R = aRows, K = aCols, C = bCols;
  const tmp = scratch(R * C);
  for (let c = 0; c < C; c++) {
    for (let r = 0; r < R; r++) {
      let sum = 0;
      for (let k = 0; k < K; k++) sum += (a[k * R + r] as number) * (b[c * K + k] as number);
      tmp[c * R + r] = sum;
    }
  }
  const o = out ?? new Array<number>(R * C);
  for (let i = 0; i < R * C; i++) o[i] = tmp[i] as number;
  return o;
}

/**
 * Transpose a rows×cols column-major matrix into a cols×rows column-major matrix:
 * out[r*cols + c] = m[c*rows + r]. Computed through scratch, then copied to `out` (aliasing-safe).
 */
export function transpose(m: readonly number[], rows: number, cols: number, out?: number[]): number[] {
  const n = rows * cols;
  const tmp = scratch(n);
  for (let c = 0; c < cols; c++) {
    for (let r = 0; r < rows; r++) tmp[r * cols + c] = m[c * rows + r] as number;
  }
  const o = out ?? new Array<number>(n);
  for (let i = 0; i < n; i++) o[i] = tmp[i] as number;
  return o;
}

/** Componentwise (Hadamard) product of two same-length matrices: out[i] = a[i]·b[i]. */
export function matrixCompMult(a: readonly number[], b: readonly number[], out?: number[]): number[] {
  const o = out ?? new Array<number>(a.length);
  for (let i = 0; i < a.length; i++) o[i] = (a[i] as number) * (b[i] as number);
  return o;
}

/** Scale every element by a scalar: out[i] = m[i]·s. */
export function matScale(m: readonly number[], s: number, out?: number[]): number[] {
  const o = out ?? new Array<number>(m.length);
  for (let i = 0; i < m.length; i++) o[i] = (m[i] as number) * s;
  return o;
}

/** A fresh copy of column `i` of a rows×cols column-major matrix = flat [i*rows .. i*rows + rows − 1]. */
export function matColumn(m: readonly number[], rows: number, _cols: number, i: number): number[] {
  const out = new Array<number>(rows);
  const base = i * rows;
  for (let r = 0; r < rows; r++) out[r] = m[base + r] as number;
  return out;
}

/**
 * Determinant of an n×n column-major matrix (n = 2, 3, or 4) by hardcoded cofactor expansion. Element
 * (row, col) lives at flat index col*n + row. The n=4 case expands along the first row with inline 3×3
 * minors. Matches the shader targets' native determinant().
 */
export function determinant(m: readonly number[], n: number): number {
  const e = (row: number, col: number): number => m[col * n + row] as number;
  if (n === 2) return e(0, 0) * e(1, 1) - e(0, 1) * e(1, 0);
  if (n === 3) {
    return e(0, 0) * (e(1, 1) * e(2, 2) - e(1, 2) * e(2, 1))
         - e(0, 1) * (e(1, 0) * e(2, 2) - e(1, 2) * e(2, 0))
         + e(0, 2) * (e(1, 0) * e(2, 1) - e(1, 1) * e(2, 0));
  }
  // n === 4 — cofactor expansion along the first row, each 3×3 minor expanded inline.
  const m3 = (r0: number, r1: number, r2: number, c0: number, c1: number, c2: number): number =>
      e(r0, c0) * (e(r1, c1) * e(r2, c2) - e(r1, c2) * e(r2, c1))
    - e(r0, c1) * (e(r1, c0) * e(r2, c2) - e(r1, c2) * e(r2, c0))
    + e(r0, c2) * (e(r1, c0) * e(r2, c1) - e(r1, c1) * e(r2, c0));
  return e(0, 0) * m3(1, 2, 3, 1, 2, 3)
       - e(0, 1) * m3(1, 2, 3, 0, 2, 3)
       + e(0, 2) * m3(1, 2, 3, 0, 1, 3)
       - e(0, 3) * m3(1, 2, 3, 0, 1, 2);
}

/**
 * Determinant of a k×k column-major matrix (element (row, col) at flat col*k + row) by cofactor expansion
 * along the first column. k=1 is the scalar; k=2 is the direct 2×2 formula; larger k recurses on the
 * (k−1)×(k−1) minors. General over k so it serves both the top-level inverse determinant and the smaller
 * minors the adjugate needs.
 */
function detFlat(m: readonly number[], k: number): number {
  if (k === 1) return m[0] as number;
  if (k === 2) return (m[0] as number) * (m[3] as number) - (m[2] as number) * (m[1] as number);
  let sum = 0;
  for (let row = 0; row < k; row++) {
    const sign = row % 2 === 0 ? 1 : -1;                        // cofactor sign along column 0
    sum += sign * (m[row] as number) * detFlat(subMat(m, k, row, 0), k - 1);  // element (row, 0) at flat index row
  }
  return sum;
}

/**
 * The (k−1)×(k−1) column-major submatrix of a k×k column-major matrix, deleting row `dr` and column `dc`.
 * Iterating remaining columns (outer) then remaining rows (inner) pushes elements in column-major order.
 */
function subMat(m: readonly number[], k: number, dr: number, dc: number): number[] {
  const out: number[] = [];
  for (let col = 0; col < k; col++) {
    if (col === dc) continue;
    for (let row = 0; row < k; row++) {
      if (row === dr) continue;
      out.push(m[col * k + row] as number);
    }
  }
  return out;
}

/**
 * Inverse of an n×n column-major matrix via the closed-form adjugate/determinant:
 * inv[i][j] = ((−1)^(i+j) · minor(j, i)) / det, stored column-major (element (row=i, col=j) at flat j*n+i).
 * Computed through scratch, then copied to `out`, so aliasing `out` with `m` is safe. A singular matrix
 * (det ≈ 0) yields ±Inf/NaN elements — undefined, no guard, matching the shader targets.
 */
export function inverse(m: readonly number[], n: number, out?: number[]): number[] {
  const det = detFlat(m, n);
  const tmp = scratch(n * n);
  for (let i = 0; i < n; i++) {          // i = row of the inverse
    for (let j = 0; j < n; j++) {        // j = column of the inverse
      const sign = (i + j) % 2 === 0 ? 1 : -1;
      const minor = detFlat(subMat(m, n, j, i), n - 1);
      tmp[j * n + i] = (sign * minor) / det;
    }
  }
  const o = out ?? new Array<number>(n * n);
  for (let idx = 0; idx < n * n; idx++) o[idx] = tmp[idx] as number;
  return o;
}
