// packages/math/src/core/quat.test.ts
import { describe, it, expect } from 'vitest';
import { qmul, qconj, qinvert, qaxisangle, qrotate, qslerp, qmat } from './quat.ts';

// Column-major 3×3 (element (r,c) at c*3+r) times a 3-vector: out[r] = Σ_c M[c*3+r]·v[c].
function mat3TimesVec(m: readonly number[], v: readonly number[]): number[] {
  const out = [0, 0, 0];
  for (let r = 0; r < 3; r++) {
    let s = 0;
    for (let c = 0; c < 3; c++) s += (m[c * 3 + r] as number) * (v[c] as number);
    out[r] = s;
  }
  return out;
}

describe('core quat (vec4 xyzw)', () => {
  it('qmul with the identity quat [0,0,0,1] is a no-op on both sides', () => {
    const q = [1, 2, 3, 4];
    expect(qmul([0, 0, 0, 1], q)).toEqual(q);
    expect(qmul(q, [0, 0, 0, 1])).toEqual(q);
  });

  it('qconj negates the imaginary part, keeps the real part', () => {
    expect(qconj([1, 2, 3, 4])).toEqual([-1, -2, -3, 4]);
  });

  it('qinvert of a unit quaternion equals its conjugate', () => {
    const q = qaxisangle([0, 0, 1], Math.PI / 2); // unit: |q| = 1
    const inv = qinvert(q);
    const conj = qconj(q);
    for (let i = 0; i < 4; i++) expect(inv[i]).toBeCloseTo(conj[i] as number, 12);
  });

  it('qinvert follows the -x/d,-y/d,-z/d, w/d formula on a non-unit quat', () => {
    const q = [1, 2, 3, 4];
    const d = 1 + 4 + 9 + 16; // 30
    expect(qinvert(q)).toEqual([-1 / d, -2 / d, -3 / d, 4 / d]);
  });

  it('qrotate: a 90° rotation about z sends +x to +y', () => {
    const q = qaxisangle([0, 0, 1], Math.PI / 2);
    const r = qrotate(q, [1, 0, 0]);
    expect(r[0]).toBeCloseTo(0, 12);
    expect(r[1]).toBeCloseTo(1, 12);
    expect(r[2]).toBeCloseTo(0, 12);
  });

  it('qrotate agrees with the rotation matrix from qmat (qrotate ≡ qmat·v)', () => {
    // A general, non-axis-aligned unit quaternion so every qmat entry participates.
    const q = qaxisangle([1, 2, 3], 0.7);
    const v = [0.3, -1.1, 2.4];
    const viaQuat = qrotate(q, v);
    const viaMat = mat3TimesVec(qmat(q), v);
    for (let i = 0; i < 3; i++) expect(viaQuat[i]).toBeCloseTo(viaMat[i] as number, 12);
  });

  it('qslerp endpoints reproduce the (normalized) inputs at t=0 and t=1', () => {
    const a = [0, 0, 0, 1];                       // unit
    const b = qaxisangle([0, 0, 1], Math.PI / 2); // unit
    const at0 = qslerp(a, b, 0);
    const at1 = qslerp(a, b, 1);
    for (let i = 0; i < 4; i++) expect(at0[i]).toBeCloseTo(a[i] as number, 12);
    for (let i = 0; i < 4; i++) expect(at1[i]).toBeCloseTo(b[i] as number, 12);
  });

  it('qslerp midpoint of two unit quats stays unit-length', () => {
    const a = [0, 0, 0, 1];
    const b = qaxisangle([0, 0, 1], Math.PI / 2);
    const mid = qslerp(a, b, 0.5);
    const len = Math.hypot(mid[0] as number, mid[1] as number, mid[2] as number, mid[3] as number);
    expect(len).toBeCloseTo(1, 12);
  });

  it('qslerp small-angle case falls back to normalized lerp and stays unit-length', () => {
    // dot > 0.9995 for nearly-parallel unit quats → nlerp branch.
    const a = [0, 0, 0, 1];
    const b = qaxisangle([0, 0, 1], 0.01); // tiny angle → dot ≈ 1
    const mid = qslerp(a, b, 0.5);
    const len = Math.hypot(mid[0] as number, mid[1] as number, mid[2] as number, mid[3] as number);
    expect(len).toBeCloseTo(1, 12);
  });

  it('qaxisangle builds [axis·sin(θ/2), cos(θ/2)]', () => {
    const q = qaxisangle([0, 0, 1], Math.PI / 2);
    const s = Math.SQRT1_2; // sin(π/4) = cos(π/4)
    expect(q[0]).toBeCloseTo(0, 12);
    expect(q[1]).toBeCloseTo(0, 12);
    expect(q[2]).toBeCloseTo(s, 12);
    expect(q[3]).toBeCloseTo(s, 12);
  });

  it('out-param writes into and returns the same array', () => {
    const out: number[] = [];
    const r = qmul([0, 0, 0, 1], [1, 2, 3, 4], out);
    expect(r).toBe(out);
    expect(out).toEqual([1, 2, 3, 4]);
  });

  it('qmul is aliasing-safe (out === a matches a fresh result)', () => {
    const a = [1, 2, 3, 4];
    const b = [5, 6, 7, 8];
    const fresh = qmul(a, b);
    const aliased = qmul(a, b, a); // out === a
    expect(aliased).toBe(a);
    expect(aliased).toEqual(fresh);
  });

  it('qrotate is aliasing-safe (out === v matches a fresh result)', () => {
    const q = qaxisangle([1, 0, 0], 0.9);
    const v = [1, 2, 3];
    const fresh = qrotate(q, [1, 2, 3]);
    const aliased = qrotate(q, v, v); // out === v
    expect(aliased).toBe(v);
    for (let i = 0; i < 3; i++) expect(aliased[i]).toBeCloseTo(fresh[i] as number, 12);
  });

  it('qslerp is aliasing-safe (out === a matches a fresh result)', () => {
    const b = qaxisangle([0, 1, 0], 1.2);
    const fresh = qslerp([0, 0, 0, 1], b, 0.4);
    const a = [0, 0, 0, 1];
    const r = qslerp(a, b, 0.4, a); // out === a
    expect(r).toBe(a);
    for (let i = 0; i < 4; i++) expect(r[i]).toBeCloseTo(fresh[i] as number, 12);
  });
});
