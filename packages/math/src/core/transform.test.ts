// packages/math/src/core/transform.test.ts
import { describe, it, expect } from 'vitest';
import { transformation, decompose, perspective, ortho, lookAt } from './transform.ts';
import { qaxisangle } from './quat.ts';

// Column-major 4×4 (element (r,c) at c*4+r) times a homogeneous 4-vector: out[r] = Σ_c M[c*4+r]·v[c].
function mat4TimesVec(m: readonly number[], v: readonly number[]): number[] {
  const out = [0, 0, 0, 0];
  for (let r = 0; r < 4; r++) {
    let s = 0;
    for (let c = 0; c < 4; c++) s += (m[c * 4 + r] as number) * (v[c] as number);
    out[r] = s;
  }
  return out;
}

const len3 = (a: readonly number[]): number =>
  Math.hypot(a[0] as number, a[1] as number, a[2] as number);
const dot3 = (a: readonly number[], b: readonly number[]): number =>
  (a[0] as number) * (b[0] as number) + (a[1] as number) * (b[1] as number) + (a[2] as number) * (b[2] as number);

// Two rotation quaternions are equal "up to sign" — q and −q name the same rotation.
function quatMatchesUpToSign(a: readonly number[], b: readonly number[], tol = 1e-9): boolean {
  const same = a.every((v, i) => Math.abs(v - (b[i] as number)) < tol);
  const opp = a.every((v, i) => Math.abs(v + (b[i] as number)) < tol);
  return same || opp;
}

describe('core transform + camera (column-major, RH, Y-up, [-1,1] clip-z)', () => {
  it('perspective fills the known OpenGL-style entries (f/aspect, f, -1, depth terms)', () => {
    const near = 1, far = 100;
    const p = perspective(Math.PI / 2, 1, near, far); // f = 1/tan(π/4) = 1
    expect(p[0]).toBeCloseTo(1);                       // f/aspect
    expect(p[5]).toBeCloseTo(1);                       // f
    expect(p[11]).toBe(-1);
    expect(p[10]).toBeCloseTo((far + near) / (near - far));
    expect(p[14]).toBeCloseTo((2 * far * near) / (near - far));
    expect(p[15]).toBe(0);
    // Off-diagonal / unused entries are zero.
    for (const i of [1, 2, 3, 4, 6, 7, 8, 9, 12, 13]) expect(p[i]).toBe(0);
  });

  it('perspective maps a point on the near plane to NDC z ≈ -1', () => {
    const near = 1, far = 100;
    const p = perspective(Math.PI / 2, 1, near, far);
    const clip = mat4TimesVec(p, [0, 0, -near, 1]); // eye-space point on the near plane
    const ndcZ = (clip[2] as number) / (clip[3] as number);
    expect(clip[3]).toBeCloseTo(near); // w_clip = -z_eye = near
    expect(ndcZ).toBeCloseTo(-1);
  });

  it('ortho fills the known OpenGL-style entries', () => {
    const [l, r, b, t, n, f] = [-2, 2, -1, 1, 1, 100];
    const o = ortho(l, r, b, t, n, f);
    expect(o[0]).toBeCloseTo(-2 / (l - r));
    expect(o[5]).toBeCloseTo(-2 / (b - t));
    expect(o[10]).toBeCloseTo(2 / (n - f));
    expect(o[12]).toBeCloseTo((l + r) / (l - r));
    expect(o[13]).toBeCloseTo((t + b) / (b - t));
    expect(o[14]).toBeCloseTo((f + n) / (n - f));
    expect(o[15]).toBe(1);
    for (const i of [1, 2, 3, 4, 6, 7, 8, 9, 11]) expect(o[i]).toBe(0);
  });

  it('decompose ∘ transformation round-trips translation, scale, and rotation (up to sign)', () => {
    const t = [1, 2, 3];
    const s = [2, 3, 4];
    const r = qaxisangle([0, 0, 1], 0.5); // unit rotation quat about +z
    const m = transformation(t, r, s);
    const d = decompose(m);
    for (let i = 0; i < 3; i++) expect(d.t[i]).toBeCloseTo(t[i] as number);
    for (let i = 0; i < 3; i++) expect(d.s[i]).toBeCloseTo(s[i] as number);
    expect(quatMatchesUpToSign(d.r, r)).toBe(true);
  });

  it('decompose ∘ transformation round-trips a general (non-axis-aligned) rotation', () => {
    const t = [-4, 0.5, 7];
    const s = [1, 1, 1];
    const r = qaxisangle([1, 2, 3], 0.9); // qaxisangle normalizes internally? no — but |[1,2,3]|≠1
    // Normalize the axis so r is a proper unit quaternion (transformation assumes a unit quat).
    const ax = [1, 2, 3];
    const al = Math.hypot(ax[0] as number, ax[1] as number, ax[2] as number);
    const runit = qaxisangle([(ax[0] as number) / al, (ax[1] as number) / al, (ax[2] as number) / al], 0.9);
    void r;
    const m = transformation(t, runit, s);
    const d = decompose(m);
    for (let i = 0; i < 3; i++) expect(d.t[i]).toBeCloseTo(t[i] as number);
    for (let i = 0; i < 3; i++) expect(d.s[i]).toBeCloseTo(s[i] as number);
    expect(quatMatchesUpToSign(d.r, runit)).toBe(true);
  });

  it('lookAt produces an orthonormal basis and the expected view translation', () => {
    const v = lookAt([0, 0, 5], [0, 0, 0], [0, 1, 0]);
    // The camera basis x/y/z lives in the ROWS of the upper-left 3×3 (a view matrix is the inverse rotation).
    const xr = [v[0] as number, v[4] as number, v[8] as number];
    const yr = [v[1] as number, v[5] as number, v[9] as number];
    const zr = [v[2] as number, v[6] as number, v[10] as number];
    expect(len3(xr)).toBeCloseTo(1);
    expect(len3(yr)).toBeCloseTo(1);
    expect(len3(zr)).toBeCloseTo(1);
    expect(dot3(xr, yr)).toBeCloseTo(0);
    expect(dot3(xr, zr)).toBeCloseTo(0);
    expect(dot3(yr, zr)).toBeCloseTo(0);
    // Homogeneous row + translation column.
    expect(v[3]).toBe(0);
    expect(v[7]).toBe(0);
    expect(v[11]).toBe(0);
    expect(v[15]).toBe(1);
    expect(v[12]).toBeCloseTo(0);  // -dot(x, eye)
    expect(v[13]).toBeCloseTo(0);  // -dot(y, eye)
    expect(v[14]).toBeCloseTo(-5); // -dot(z, eye); z=[0,0,1], eye=[0,0,5]
  });

  it('out-param writes into and returns the same array', () => {
    const out: number[] = [];
    const r = transformation([1, 2, 3], [0, 0, 0, 1], [1, 1, 1], out);
    expect(r).toBe(out);
    // identity rotation + unit scale ⇒ pure translation matrix.
    expect(out).toEqual([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 2, 3, 1]);
  });

  it('perspective/ortho out-param is zeroed then filled (a reused dirty buffer is fully overwritten)', () => {
    const dirty = new Array<number>(16).fill(9);
    const p = perspective(Math.PI / 3, 1.5, 0.1, 50, dirty);
    expect(p).toBe(dirty);
    for (const i of [1, 2, 3, 4, 6, 7, 8, 9, 12, 13, 15]) expect(p[i]).toBe(0);
  });
});
