import { describe, it, expect } from 'vitest';
import { evaluateProgram, RecordingHostEnv, PlainStorageHost } from '@metael/lang';
import { vecStoreOf } from './custom-types.ts';

// Quaternions carry a vec4 layout `(x, y, z, w)` = imaginary xyz + real w. There is no distinct quat value
// type: every q* builtin consumes and produces a plain vecN (a vec4, or a vec3 for the rotated vector).
describe('quaternions (vec4 convention)', () => {
  const h = () => ({ host: new PlainStorageHost(), env: new RecordingHostEnv() });

  it('qmul identity: q * (0,0,0,1) === q', () => {
    const s = vecStoreOf(evaluateProgram('qmul(vec4(1,2,3,4), vec4(0,0,0,1))', h()).value);
    expect(Array.from(s.c)).toEqual([1, 2, 3, 4]);
  });

  it('qconj negates the imaginary part', () => {
    const s = vecStoreOf(evaluateProgram('qconj(vec4(1,2,3,4))', h()).value);
    expect(Array.from(s.c)).toEqual([-1, -2, -3, 4]);
  });

  it('qinvert of a unit quat equals its conjugate', () => {
    // For a unit quaternion the inverse IS the conjugate (dot(q,q) === 1). `+ 0` normalizes the -0 the
    // exact `-0/1` division produces (mathematically identical to 0) so the deep-equal is signed-zero-agnostic.
    const inv = vecStoreOf(evaluateProgram('qinvert(vec4(0,0,0,1))', h()).value);
    expect(Array.from(inv.c).map((x) => x + 0)).toEqual([0, 0, 0, 1]);
    // q * qinvert(q) === identity (0,0,0,1) for any non-zero quat.
    const prod = vecStoreOf(evaluateProgram('qmul(vec4(1,2,3,4), qinvert(vec4(1,2,3,4)))', h()).value);
    expect(Array.from(prod.c).map((x) => Math.round(x * 1000) / 1000 + 0)).toEqual([0, 0, 0, 1]);
  });

  it('qaxisangle + qrotate rotate a vector', () => {
    // A 90° rotation about +z carries (1,0,0) → (0,1,0). qaxisangle builds the rotation quat, qrotate applies it.
    const s = vecStoreOf(evaluateProgram('qrotate(qaxisangle(vec3(0,0,1), 1.5707963267948966), vec3(1,0,0))', h()).value);
    expect(Array.from(s.c).map((x) => Math.round(x))).toEqual([0, 1, 0]);
  });

  it('qaxisangle produces the (x,y,z,w) = (axis·sin(θ/2), cos(θ/2)) quat', () => {
    // A 180° rotation about +z: sin(π/2)=1, cos(π/2)=0 → (0,0,1,0).
    const s = vecStoreOf(evaluateProgram('qaxisangle(vec3(0,0,1), 3.141592653589793)', h()).value);
    expect(Array.from(s.c).map((x) => Math.round(x * 1000) / 1000 + 0)).toEqual([0, 0, 1, 0]);
  });

  it('qslerp endpoints: t=0 → a', () => {
    // Interpolating from the identity quat at t=0 returns the first endpoint (normalized).
    const a = vecStoreOf(evaluateProgram('qslerp(vec4(0,0,0,1), qaxisangle(vec3(0,0,1),1.0), 0)', h()).value);
    expect(Array.from(a.c).map((x) => Math.round(x * 1000) / 1000 + 0)).toEqual([0, 0, 0, 1]);
  });

  it('qslerp endpoints: t=1 → b (normalized)', () => {
    // At t=1 the result is the (unit) second endpoint. qaxisangle(z, 1.0) is already a unit quat.
    const b = vecStoreOf(evaluateProgram('qslerp(vec4(0,0,0,1), qaxisangle(vec3(0,0,1),1.0), 1)', h()).value);
    const expected = vecStoreOf(evaluateProgram('qaxisangle(vec3(0,0,1),1.0)', h()).value);
    Array.from(b.c).forEach((x, i) => expect(x).toBeCloseTo(expected.c[i] as number, 5));
  });

  it('qslerp midpoint is a unit quat', () => {
    // Slerp preserves unit norm at every t: |qslerp(a,b,0.5)| ≈ 1.
    const m = vecStoreOf(evaluateProgram('qslerp(qaxisangle(vec3(0,0,1),0.4), qaxisangle(vec3(0,0,1),1.2), 0.5)', h()).value);
    const len = Math.hypot(...Array.from(m.c));
    expect(len).toBeCloseTo(1, 5);
  });

  it('qmat(identity) is the column-major identity mat3', () => {
    const s = vecStoreOf(evaluateProgram('qmat(vec4(0,0,0,1))', h()).value);
    expect(Array.from(s.c)).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  });

  it('qmat(q) * v equals qrotate(q, v) — cross-checks the column-major layout', () => {
    // The rotation matrix built from q must transform v identically to applying q directly. If the
    // qmat column-major layout were transposed, this equivalence would break on a general axis/angle.
    const q = 'const q = qaxisangle(vec3(0.3,0.5,0.8), 0.9) ';
    for (const comp of ['x', 'y', 'z']) {
      const viaMat = evaluateProgram(`${q}(qmat(q) * vec3(1,2,3)).${comp}`, h()).value as number;
      const viaRot = evaluateProgram(`${q}qrotate(q, vec3(1,2,3)).${comp}`, h()).value as number;
      expect(viaMat).toBeCloseTo(viaRot, 5);
    }
  });
});
