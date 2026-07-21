// packages/math/src/core/quat.ts
// Plain flat-store quaternion math over `number[]` (f64 precision). A quaternion is a vec4 laid out as
// (x, y, z, w) = imaginary xyz + real w; the identity rotation is [0, 0, 0, 1].
//
// Out-param convention: `out = op(...args, out?)`. `out` omitted → a fresh `number[]`; supplied → written
// into and returned. Aliasing an input as the output is legal: every op that reads several input
// components to form one output (qmul / qrotate / qslerp) stages the whole result in locals before the
// first write into `out`, so an aliased `out` (=== a or === b) can never corrupt a still-pending read.
//
// Precision-agnostic by design: no f32 rounding here (that is the caller/binding's concern). Shape and
// zero-length validation is a language-surface concern; these assume well-formed vec4 (and vec3) inputs.

/** Hamilton product a·b of two quaternions (xyzw). Non-commutative; applies b's rotation then a's. */
export function qmul(a: readonly number[], b: readonly number[], out?: number[]): number[] {
  const ax = a[0] as number, ay = a[1] as number, az = a[2] as number, aw = a[3] as number;
  const bx = b[0] as number, by = b[1] as number, bz = b[2] as number, bw = b[3] as number;
  // Stage all four components before writing so an aliased `out` (=== a or b) stays safe.
  const x = aw * bx + ax * bw + ay * bz - az * by;
  const y = aw * by - ax * bz + ay * bw + az * bx;
  const z = aw * bz + ax * by - ay * bx + az * bw;
  const w = aw * bw - ax * bx - ay * by - az * bz;
  const o = out ?? new Array<number>(4);
  o[0] = x; o[1] = y; o[2] = z; o[3] = w;
  return o;
}

/** Conjugate: negate the imaginary part, keep the real part → [-x, -y, -z, w]. */
export function qconj(q: readonly number[], out?: number[]): number[] {
  const o = out ?? new Array<number>(4);
  o[0] = -(q[0] as number); o[1] = -(q[1] as number); o[2] = -(q[2] as number); o[3] = q[3] as number;
  return o;
}

/** Inverse: conjugate divided by the squared norm → [-x, -y, -z, w] / (x²+y²+z²+w²). Unit quats: ≈ qconj. */
export function qinvert(q: readonly number[], out?: number[]): number[] {
  const x = q[0] as number, y = q[1] as number, z = q[2] as number, w = q[3] as number;
  const d = x * x + y * y + z * z + w * w;
  const o = out ?? new Array<number>(4);
  o[0] = -x / d; o[1] = -y / d; o[2] = -z / d; o[3] = w / d;
  return o;
}

/** Build a unit rotation quaternion from an axis (vec3) and an angle (radians): [axis·sin(θ/2), cos(θ/2)]. */
export function qaxisangle(axis: readonly number[], angle: number, out?: number[]): number[] {
  const s = Math.sin(angle / 2);
  const o = out ?? new Array<number>(4);
  o[0] = (axis[0] as number) * s; o[1] = (axis[1] as number) * s; o[2] = (axis[2] as number) * s;
  o[3] = Math.cos(angle / 2);
  return o;
}

/**
 * Rotate a vec3 by a quaternion using the optimized identity v + 2·cross(q.xyz, cross(q.xyz, v) + w·v),
 * expanded here without the intermediate w·v term as v + w·t + cross(q.xyz, t) where t = 2·cross(q.xyz, v).
 * The two cross products are inlined and the result staged in locals, so an aliased `out` (=== v) is safe.
 */
export function qrotate(q: readonly number[], v: readonly number[], out?: number[]): number[] {
  const x = q[0] as number, y = q[1] as number, z = q[2] as number, w = q[3] as number;
  const v0 = v[0] as number, v1 = v[1] as number, v2 = v[2] as number;
  // t = 2·cross(q.xyz, v)
  const t0 = 2 * (y * v2 - z * v1);
  const t1 = 2 * (z * v0 - x * v2);
  const t2 = 2 * (x * v1 - y * v0);
  // c2 = cross(q.xyz, t)
  const c0 = y * t2 - z * t1;
  const c1 = z * t0 - x * t2;
  const c2 = x * t1 - y * t0;
  const o = out ?? new Array<number>(3);
  o[0] = v0 + w * t0 + c0;
  o[1] = v1 + w * t1 + c1;
  o[2] = v2 + w * t2 + c2;
  return o;
}

/**
 * Spherical linear interpolation between two quaternions. An antipodal fix negates b when the dot product
 * is negative (so the shorter arc is taken); for nearly-parallel inputs (dot > 0.9995) it falls back to a
 * normalized lerp (nlerp) to avoid the division by sin(θ)→0. The result is staged in locals before the
 * first write, so an aliased `out` (=== a or b) is safe.
 */
export function qslerp(a: readonly number[], b: readonly number[], t: number, out?: number[]): number[] {
  const a0 = a[0] as number, a1 = a[1] as number, a2 = a[2] as number, a3 = a[3] as number;
  let b0 = b[0] as number, b1 = b[1] as number, b2 = b[2] as number, b3 = b[3] as number;
  let dot = a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
  if (dot < 0) { b0 = -b0; b1 = -b1; b2 = -b2; b3 = -b3; dot = -dot; }
  let r0: number, r1: number, r2: number, r3: number;
  if (dot > 0.9995) {
    // nlerp fallback: linear blend then renormalize. hypot (not sqrt-of-sum) for last-ULP consistency.
    r0 = a0 + t * (b0 - a0);
    r1 = a1 + t * (b1 - a1);
    r2 = a2 + t * (b2 - a2);
    r3 = a3 + t * (b3 - a3);
    const len = Math.hypot(r0, r1, r2, r3);
    r0 /= len; r1 /= len; r2 /= len; r3 /= len;
  } else {
    const th = Math.acos(dot);
    const s = Math.sin(th);
    const wa = Math.sin((1 - t) * th) / s;
    const wb = Math.sin(t * th) / s;
    r0 = wa * a0 + wb * b0;
    r1 = wa * a1 + wb * b1;
    r2 = wa * a2 + wb * b2;
    r3 = wa * a3 + wb * b3;
  }
  const o = out ?? new Array<number>(4);
  o[0] = r0; o[1] = r1; o[2] = r2; o[3] = r3;
  return o;
}

/**
 * Rotation matrix for a quaternion as a 9-element COLUMN-MAJOR 3×3 (element (row, col) at flat col*3+row).
 * Assumes a unit quaternion (as a rotation quaternion is). Reads all inputs before writing → aliasing-safe.
 */
export function qmat(q: readonly number[], out?: number[]): number[] {
  const x = q[0] as number, y = q[1] as number, z = q[2] as number, w = q[3] as number;
  const xx = x * x, yy = y * y, zz = z * z;
  const xy = x * y, xz = x * z, yz = y * z;
  const wx = w * x, wy = w * y, wz = w * z;
  const o = out ?? new Array<number>(9);
  o[0] = 1 - 2 * (yy + zz); o[1] = 2 * (xy + wz);     o[2] = 2 * (xz - wy);     // col 0
  o[3] = 2 * (xy - wz);     o[4] = 1 - 2 * (xx + zz); o[5] = 2 * (yz + wx);     // col 1
  o[6] = 2 * (xz + wy);     o[7] = 2 * (yz - wx);     o[8] = 1 - 2 * (xx + yy); // col 2
  return o;
}
