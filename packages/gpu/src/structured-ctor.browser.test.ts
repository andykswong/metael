// Real-adapter parity for the three structured-constructor / column-access forms that are native to WGSL and
// GLSL: vecN composition (`vec3(vec2(a,b), c)`), matMxN from column vecs (`mat2(vec2(a,b), vec2(c,d))`), and
// mat column indexing (`m[i]`). The parity harness is `gate-accepted ⇒ interpreter == GPU`: each kernel is
// dispatched with `verify: true` so a SAMPLE of GPU cells is cross-checked against the interpreter oracle.
// Each runs on `backend: 'webgpu'` AND `backend: 'webgl2'` — a real-adapter mis-shape is a shader-compile
// error the no-adapter emit path can't catch, so `backend !== 'cpu'` fails loudly on the CPU fallback rather
// than false-greening (a mis-shaped kernel that fails to compile re-ladders to CPU, where verify trivially
// matches and proves nothing). Follows the repo's *.browser.test.ts convention (Chromium SwiftShader adapter).
import { describe, it, expect } from 'vitest';
import { createGpuEngine } from './api.ts';
import { settle } from './settle.ts';
import { compileKernel } from './lang/compile-kernel.ts';
import type { BackendKind } from './device/index.ts';

const REAL: readonly BackendKind[] = ['webgpu', 'webgl2'];

describe('parity: interpreter == real WGSL/WebGL2 for the structured constructors + column access', () => {
  // Form 1 — vecN composition: `vec3(vec2(row, row*2), row+7)` flattens a vec2 + a scalar into a vec3 output.
  // WGSL `vec3<f32>(vec2<f32>(..), ..)` / GLSL `vec3(vec2(..), ..)` are the native composing constructors.
  for (const backend of REAL) {
    it(`vecN composition (vec3 output) matches the oracle on ${backend}`, async () => {
      const gpu = createGpuEngine();
      const k = compileKernel('component k(i) { return vec3(vec2(i, i * 2), i + 7) } k', gpu.host);
      const r = await settle(() => gpu.dispatch(k, { output: [8], outputElement: 'vec3', backend, verify: true }));
      expect(r.backend).not.toBe('cpu');   // a real shader path ran — not the CPU fallback (which proves nothing)
      expect(r.match?.ok).toBe(true);      // verify cross-checks the real GPU output vs the interpreter oracle
      gpu[Symbol.dispose]();
    });
  }

  // Form 2 — matMxN from column vecs: `mat2(vec2(a,b), vec2(c,d))` builds the 2×2 from its two columns, then
  // multiplies a vec2. WGSL `mat2x2<f32>(vec2<f32>(..), vec2<f32>(..))` / GLSL `mat2(vec2(..), vec2(..))` are
  // the native column constructors; the matmul + `.x` read produce one scalar per cell.
  for (const backend of REAL) {
    it(`matMxN from column vecs (scalar output) matches the oracle on ${backend}`, async () => {
      const gpu = createGpuEngine();
      const k = compileKernel('component k(i) { const m = mat2(vec2(1, 2), vec2(3, 4)) const v = vec2(i, i + 1) return (m * v).x } k', gpu.host);
      const r = await settle(() => gpu.dispatch(k, { output: [8], backend, verify: true }));
      expect(r.backend).not.toBe('cpu');
      expect(r.match?.ok).toBe(true);
      gpu[Symbol.dispose]();
    });
  }

  // Form 3 — `m[i]` mat column indexing: a mat3 (built with i-dependent middle column) is indexed to its
  // column-1 vec3, swizzled `.xyz`, and returned as a vec3 output. WGSL `m[u32(round(i))]` / GLSL
  // `m[int(roundEven(i))]` yield a vecR (R = row count) NATIVELY. This is the shape-inference fix: today a
  // mat `index` reads as scalar, so a vec3 output over `m[i].xyz` is gate-rejected (→ CPU) or mis-shaped.
  for (const backend of REAL) {
    it(`m[i] mat column indexing (vec3 output) matches the oracle on ${backend}`, async () => {
      const gpu = createGpuEngine();
      const k = compileKernel('component k(i) { const m = mat3(1, 2, 3, 4 * i, 5 * i, 6 * i, 7, 8, 9) return m[1].xyz } k', gpu.host);
      const r = await settle(() => gpu.dispatch(k, { output: [4], outputElement: 'vec3', backend, verify: true }));
      expect(r.backend).not.toBe('cpu');
      expect(r.match?.ok).toBe(true);
      gpu[Symbol.dispose]();
    });
  }
});
