// Runs the real device ladder in Chromium. There is no adapter gate: absent a real adapter the ladder
// falls to the CPU backend, where `verify` would trivially match the interpreter and prove nothing about
// the shader. So this asserts a NON-CPU backend FIRST — if it ever runs adapter-less it fails loudly on the
// CPU fallback rather than false-greening — then asserts `verify` cross-checks the real WGSL/WebGL2 output
// against the interpreter oracle. Follows the repo's existing *.browser.test.ts convention (Chromium
// adapter + CPU fallback), not a novel skip mechanism.
import { describe, it, expect } from 'vitest';
import { createGpuEngine } from './api.ts';

describe('parity: interpreter == real WGSL/WebGL2', () => {
  it('B1 mat*vec matches on a real adapter', async () => {
    const gpu = createGpuEngine();
    const k = gpu.compile('const A = f32(8, (i)=>i) component k(i) { const m = mat2(A[0],A[1],A[2],A[3]) return (m * vec2(A[4],A[5])).x } k');
    const r = await gpu.settle(k, { output: [4], verify: true });
    expect(r.backend).not.toBe('cpu'); // a real shader path ran — not the CPU fallback (which proves nothing)
    expect(r.match?.ok).toBe(true);    // verify cross-checks the real GPU output vs the interpreter oracle
    gpu[Symbol.dispose]();
  });
  // Square-matrix inverse on a real adapter: `inverse(M) * M ≈ I`, so `(inv·M·[1,0]).x` reads back 1.
  // This env's real adapter is WebGL2, so this proves the GLSL-native `inverse` == the interpreter oracle.
  // The WGSL hand-emit is verified by construction + its emit-string test; a real WebGPU adapter would
  // prove the WGSL leg here too. `not.toBe('cpu')` fails loudly if the ladder ever falls to CPU (no adapter).
  it('inverse matches on a real adapter', async () => {
    const gpu = createGpuEngine();
    const k = gpu.compile('component k(i) { return (inverse(mat2(4,2,7,6)) * mat2(4,2,7,6) * vec2(1,0)).x } k');
    const r = await gpu.settle(k, { output: [4], verify: true });
    expect(r.backend).not.toBe('cpu');
    expect(r.match?.ok).toBe(true); // verify cross-checks the WGSL hand-emit / GLSL native inverse vs the interpreter
    gpu[Symbol.dispose]();
  });
  // Quat rotation on a real adapter: rotate v=[1,2,3] by the mat3 that qaxisangle(z, 0.7) builds via qmat.
  // This env's real adapter is WebGL2, so this proves the GLSL qmat hand-emit (its `_qmat` prelude helper)
  // == the interpreter oracle on a real WebGL2 device. The WGSL qmat leg is verified by construction + its
  // emit-string test; a real WebGPU adapter would prove it here too. `not.toBe('cpu')` fails loudly if the
  // ladder ever falls to CPU (no adapter), where verify would trivially match and prove nothing.
  it('qrotate vs qmat(q)*v equivalence matches on a real adapter', async () => {
    const gpu = createGpuEngine();
    const k = gpu.compile('component k(i) { return (qmat(qaxisangle(vec3(0,0,1),0.7)) * vec3(1,2,3)).x } k');
    const r = await gpu.settle(k, { output: [4], verify: true });
    expect(r.backend).not.toBe('cpu');
    expect(r.match?.ok).toBe(true); // verify cross-checks the GLSL/WGSL qmat hand-emit vs the interpreter oracle
    gpu[Symbol.dispose]();
  });
});
