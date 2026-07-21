// packages/math/src/core/transform.ts
// Affine transform composition/decomposition and camera projection/view matrices, all over plain flat
// `number[]` (f64 precision). Conventions throughout: COLUMN-MAJOR storage (element (row, col) at flat
// index `col*rows + row`), a RIGHT-HANDED, Y-up coordinate space, and the OpenGL-style `[-1, 1]` clip-z
// range for the projections (mapping the near plane to −1 and the far plane to +1). A `[0, 1]` clip-z
// (with any Y-flip) is a downstream normalization concern and is deliberately not baked in here.
//
// Out-param convention (shared with the rest of the core): `out = op(...args, out?)`. When `out` is
// omitted a fresh 16-element `number[]` is returned; when supplied it is written into and returned.
// `perspective`/`ortho` zero all 16 entries before setting the nonzero ones, so a reused (dirty) buffer
// is fully overwritten; `transformation`/`lookAt` write every one of the 16 entries directly.
//
// Precision-agnostic by design: no f32 rounding here (a caller/binding concern). Shape and degeneracy
// validation is a language-surface concern; these assume well-formed inputs (a well-formed camera for
// `lookAt` — coincident eye/center yields a zero forward vector whose normalize produces NaN, with no
// guard, matching the no-guard convention of the rest of the core).

import { qmat } from './quat.ts';
import { cross, normalize, sub, dot } from './vec.ts';

/**
 * Compose a translation (vec3), a rotation (unit quaternion xyzw), and a scale (vec3) into a single
 * column-major 4×4 affine matrix M = T · R · S. The rotation contributes a column-major 3×3 block; each
 * of its columns is scaled by the matching scale component, and the translation fills the last column.
 */
export function transformation(
  t: readonly number[], r: readonly number[], s: readonly number[], out?: number[],
): number[] {
  const R = qmat(r);                         // 9 flat, column-major 3×3 (element (row,col) at col*3+row)
  const sx = s[0] as number, sy = s[1] as number, sz = s[2] as number;
  const o = out ?? new Array<number>(16);
  // column 0 = rotation column 0 · sx
  o[0] = (R[0] as number) * sx; o[1] = (R[1] as number) * sx; o[2] = (R[2] as number) * sx; o[3] = 0;
  // column 1 = rotation column 1 · sy
  o[4] = (R[3] as number) * sy; o[5] = (R[4] as number) * sy; o[6] = (R[5] as number) * sy; o[7] = 0;
  // column 2 = rotation column 2 · sz
  o[8] = (R[6] as number) * sz; o[9] = (R[7] as number) * sz; o[10] = (R[8] as number) * sz; o[11] = 0;
  // column 3 = translation
  o[12] = t[0] as number; o[13] = t[1] as number; o[14] = t[2] as number; o[15] = 1;
  return o;
}

/**
 * Recover the translation, rotation (unit quaternion xyzw), and scale from a column-major 4×4 affine
 * matrix — the inverse of {@link transformation}. Translation is the last column. Scale is the length of
 * each of the three rotation/scale columns; when the upper-left 3×3 has a negative determinant (a
 * left-handed/mirrored basis) the first scale axis is negated so the residual rotation is proper
 * (determinant +1) — the standard single-axis sign convention. Rotation is then extracted from the
 * columns normalized by their (signed) scale via the trace-based matrix→quaternion algorithm (with the
 * largest-diagonal fallback for numerical robustness). Because a quaternion and its negation name the
 * same rotation, the recovered rotation is only defined up to sign.
 */
export function decompose(m: readonly number[]): { t: number[]; r: number[]; s: number[] } {
  const t = [m[12] as number, m[13] as number, m[14] as number];
  // The three basis columns (xyz of each).
  const c0x = m[0] as number, c0y = m[1] as number, c0z = m[2] as number;
  const c1x = m[4] as number, c1y = m[5] as number, c1z = m[6] as number;
  const c2x = m[8] as number, c2y = m[9] as number, c2z = m[10] as number;
  let sx = Math.hypot(c0x, c0y, c0z);
  const sy = Math.hypot(c1x, c1y, c1z);
  const sz = Math.hypot(c2x, c2y, c2z);
  // Determinant of the upper-left 3×3 (columns c0,c1,c2). A negative sign means a mirrored basis; fold
  // the reflection into the first scale axis so the normalized rotation stays proper (det +1).
  const det =
      c0x * (c1y * c2z - c1z * c2y)
    - c1x * (c0y * c2z - c0z * c2y)
    + c2x * (c0y * c1z - c0z * c1y);
  if (det < 0) sx = -sx;
  const s = [sx, sy, sz];
  // Normalized rotation matrix R (9 flat, column-major): each column divided by its (signed) scale.
  const r00 = c0x / sx, r10 = c0y / sx, r20 = c0z / sx; // column 0
  const r01 = c1x / sy, r11 = c1y / sy, r21 = c1z / sy; // column 1
  const r02 = c2x / sz, r12 = c2y / sz, r22 = c2z / sz; // column 2
  // Trace-based matrix→quaternion. `rIJ` is the entry at row I, column J of the rotation matrix.
  const trace = r00 + r11 + r22;
  let x: number, y: number, z: number, w: number;
  if (trace > 0) {
    const f = Math.sqrt(trace + 1) * 2; // f = 4·w
    w = 0.25 * f;
    x = (r21 - r12) / f;
    y = (r02 - r20) / f;
    z = (r10 - r01) / f;
  } else if (r00 > r11 && r00 > r22) {
    const f = Math.sqrt(1 + r00 - r11 - r22) * 2; // f = 4·x
    w = (r21 - r12) / f;
    x = 0.25 * f;
    y = (r01 + r10) / f;
    z = (r02 + r20) / f;
  } else if (r11 > r22) {
    const f = Math.sqrt(1 + r11 - r00 - r22) * 2; // f = 4·y
    w = (r02 - r20) / f;
    x = (r01 + r10) / f;
    y = 0.25 * f;
    z = (r12 + r21) / f;
  } else {
    const f = Math.sqrt(1 + r22 - r00 - r11) * 2; // f = 4·z
    w = (r10 - r01) / f;
    x = (r02 + r20) / f;
    y = (r12 + r21) / f;
    z = 0.25 * f;
  }
  return { t, r: [x, y, z, w], s };
}

/**
 * Right-handed perspective projection into `[-1, 1]` clip-z (OpenGL convention). `fovy` is the vertical
 * field of view in radians, `aspect` the width/height ratio, and `near`/`far` the clip-plane distances.
 * Column-major; all entries not set below are zero.
 */
export function perspective(fovy: number, aspect: number, near: number, far: number, out?: number[]): number[] {
  const f = 1 / Math.tan(fovy / 2);
  const nf = 1 / (near - far);
  const o = out ?? new Array<number>(16);
  for (let i = 0; i < 16; i++) o[i] = 0;
  o[0] = f / aspect;
  o[5] = f;
  o[10] = (far + near) * nf;
  o[11] = -1;
  o[14] = 2 * far * near * nf;
  return o;
}

/**
 * Right-handed orthographic projection into `[-1, 1]` clip-z (OpenGL convention) for the axis-aligned box
 * [left, right] × [bottom, top] × [near, far]. Column-major; all entries not set below are zero.
 */
export function ortho(
  left: number, right: number, bottom: number, top: number, near: number, far: number, out?: number[],
): number[] {
  const lr = 1 / (left - right);
  const bt = 1 / (bottom - top);
  const nf = 1 / (near - far);
  const o = out ?? new Array<number>(16);
  for (let i = 0; i < 16; i++) o[i] = 0;
  o[0] = -2 * lr;
  o[5] = -2 * bt;
  o[10] = 2 * nf;
  o[12] = (left + right) * lr;
  o[13] = (top + bottom) * bt;
  o[14] = (far + near) * nf;
  o[15] = 1;
  return o;
}

/**
 * Right-handed view matrix looking from `eye` toward `center` with the given `up` direction. Builds the
 * orthonormal camera basis (forward z = normalize(eye − center), right x = normalize(cross(up, z)), up
 * y = cross(z, x)) and stores the inverse rotation (basis vectors in the rows) with the eye translation
 * projected onto each axis in the last column. Column-major.
 */
export function lookAt(eye: readonly number[], center: readonly number[], up: readonly number[], out?: number[]): number[] {
  const z = normalize(sub(eye, center));  // forward (from center toward eye)
  const x = normalize(cross(up, z));       // right
  const y = cross(z, x);                   // recomputed up (already unit-length)
  const x0 = x[0] as number, x1 = x[1] as number, x2 = x[2] as number;
  const y0 = y[0] as number, y1 = y[1] as number, y2 = y[2] as number;
  const z0 = z[0] as number, z1 = z[1] as number, z2 = z[2] as number;
  const o = out ?? new Array<number>(16);
  o[0] = x0; o[1] = y0; o[2] = z0; o[3] = 0;                       // column 0
  o[4] = x1; o[5] = y1; o[6] = z1; o[7] = 0;                       // column 1
  o[8] = x2; o[9] = y2; o[10] = z2; o[11] = 0;                     // column 2
  o[12] = -dot(x, eye); o[13] = -dot(y, eye); o[14] = -dot(z, eye); o[15] = 1; // column 3
  return o;
}
