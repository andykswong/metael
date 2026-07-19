import { describe, it, expect } from 'vitest';
import { evaluateProgram, RecordingHostEnv, makeCallable } from '@metael/lang';
import { RuntimeReactiveHost } from '@metael/runtime';
import { gateKernel } from './gate.ts';
import { emitCpu } from './emit-cpu.ts';

function assertParity(src: string, dims: number[], comps = 1): void {
  const host = new RuntimeReactiveHost();
  const fn = evaluateProgram(src, { host, env: new RecordingHostEnv() }).value as never;
  const { bindings, core } = gateKernel(fn, host, comps);
  expect(core).toBe(true);
  const cpu = emitCpu(fn, bindings, host, comps);
  const total = dims.reduce((a, b) => a * b, 1);
  for (let i = 0; i < total; i++) {
    // Reconstruct a cell's coords from its flat index under the shared row-major flatten (the LAST dim
    // varies fastest) — the same innermost-dim-fastest decomposition the CPU emitter, the verify oracle,
    // and every shader use. Covers any rank uniformly: rank 1 = [i], rank 2 = [floor(i/H), i%H].
    const coords = (() => {
      const c = new Array<number>(dims.length);
      let rem = i;
      for (let d = dims.length - 1; d >= 0; d--) { c[d] = rem % dims[d]!; rem = Math.floor(rem / dims[d]!); }
      return c;
    })();
    const call = makeCallable(fn, { host, env: { resolveCall: () => ({ handled: false }) }, maxSteps: 1_000_000 });
    const ref = call(...coords);
    const got = cpu(coords);
    const refComps = comps === 1 ? [Number(ref)] : Array.from({ length: comps }, (_, k) => Number((ref as { [k: string]: unknown })['xyzw'[k] as string] ?? 0));
    for (let k = 0; k < comps; k++) expect(got[k]).toBeCloseTo(refComps[k]!, 5);
  }
}

describe('parity: interpreter == CPU-emit (the gate⇒parity net)', () => {
  it('B1 mat*vec column-major', () => assertParity('const A = f32(8, (i)=>i) component k(i) { const m = mat2(A[0],A[1],A[2],A[3]) return (m * vec2(A[4],A[5])).x } k', [4]));
  it('B2 min on vec', () => assertParity('component k(i) { return min(vec2(i,5), vec2(3,2)).x } k', [4]));
  it('B3 mat*scalar', () => assertParity('component k(i) { return ((mat2(1,2,3,4) * 2) * vec2(1,1)).x } k', [4]));
  // Scalar-only rows (no vec/mat/dot/… name) — emit-cpu takes the INDEPENDENT hand-walk (evalE/applyBinary/
  // applyBuiltin), NOT the whole-cell interpreter delegate — so parity genuinely compares the interpreter
  // oracle against the hand-walk rather than self-comparing two runs of the same AST.
  it('scalar arithmetic parity (independent CPU hand-walk)', () => {
    assertParity('const a = f32(8,(i)=>i) const b = f32(8,(i)=>i*2) component k(i) { return a[i] * 3 + b[i] } k', [4]);
    assertParity('component k(x, y) { return (x > y ? x - y : y - x) } k', [3, 3]); // 2D, comparison/ternary/sub — hand-walked
  });
  // A rank-3 dispatch: the kernel encodes its own coords into the value (x*100 + y*10 + z), so every cell
  // round-trips its (x,y,z). A mismatch in the innermost-dim-fastest coord decomposition anywhere — the
  // harness, the CPU emitter, or the gate — scrambles a coord and diverges from the interpreter oracle.
  it('3D coords', () => assertParity('component k(x, y, z) { return x*100 + y*10 + z } k', [2, 2, 2]));
  it('math builtins scalar parity', () => {
    for (const b of ['tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh', 'exp2', 'log2', 'inverseSqrt', 'degrees', 'radians', 'trunc']) {
      assertParity(`component k(i) { return ${b}(0.5) } k`, [4]);
    }
    assertParity('component k(i) { return atan2(1, 2) } k', [4]);
  });
  it('math builtins vec parity', () => {
    assertParity('component k(i) { return sin(vec2(0.5, 1.0)).x } k', [4]);
    assertParity('component k(i) { return tan(vec2(0.2, 0.3)).y } k', [4]);
  });
  // P4 vec/mat ops: distance/reflect (vec geometry), determinant (square mat) and inverse (square-ctor).
  // The `inverse(M) * M ≈ I` identity is the strongest of the four — it exercises the full square-matrix
  // inverse against the interpreter oracle: `(inv·M·[1,0]).x` must round-trip to 1. Each kernel must
  // gate-pass core (assertParity asserts `core === true` before comparing per cell).
  it('P4 vec/mat op parity (node CPU==interp)', () => {
    assertParity('component k(i) { return distance(vec2(0,0), vec2(3,4)) } k', [4]);
    assertParity('component k(i) { return reflect(vec2(1,-1), vec2(0,1)).x } k', [4]);
    assertParity('component k(i) { return determinant(mat2(1,2,3,4)) } k', [4]);
    assertParity('component k(i) { return (inverse(mat2(4,2,7,6)) * mat2(4,2,7,6) * vec2(1,0)).x } k', [4]);
  });
  // Quaternion ops: qmul (Hamilton product), qaxisangle→qrotate (rotate a vec by an axis-angle quat),
  // and qmat(q)*v (rotate via the column-major mat3 the quat builds). The qmul/qrotate/qmat kernels name
  // quat heads that live in emit-cpu's VEC_NAMES, so emit-cpu delegates the WHOLE cell to the interpreter —
  // node parity is thus a self-consistency check (interpreter==interpreter) that each row gate-passes core
  // and its output shape is wired; the real cross-target proof (GLSL/WGSL qmat hand-emit == interpreter) is
  // the real-adapter row in parity.browser.test.ts.
  it('quat parity (node CPU==interp)', () => {
    assertParity('component k(i) { return qmul(vec4(1,2,3,4), vec4(0,0,0,1)).x } k', [4]);
    assertParity('component k(i) { return qrotate(qaxisangle(vec3(0,0,1), 1.57), vec3(1,0,0)).y } k', [4]);
    assertParity('component k(i) { return (qmat(qaxisangle(vec3(0,0,1),0.7)) * vec3(1,2,3)).x } k', [4]);
  });
});
export { assertParity };
