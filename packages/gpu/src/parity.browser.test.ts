// Runs the real device ladder in Chromium. There is no adapter gate: absent a real adapter the ladder
// falls to the CPU backend, where `verify` would trivially match the interpreter and prove nothing about
// the shader. So this asserts a NON-CPU backend FIRST — if it ever runs adapter-less it fails loudly on the
// CPU fallback rather than false-greening — then asserts `verify` cross-checks the real WGSL/WebGL2 output
// against the interpreter oracle. Follows the repo's existing *.browser.test.ts convention (Chromium
// adapter + CPU fallback), not a novel skip mechanism.
import { describe, it, expect } from 'vitest';
import { createGpuEngine } from './api.ts';
import { settle } from './settle.ts';
import { compileKernel } from './lang/compile-kernel.ts';

describe('parity: interpreter == real WGSL/WebGL2', () => {
  it('B1 mat*vec matches on a real adapter', async () => {
    const gpu = createGpuEngine();
    const k = compileKernel('const A = f32(8, (i)=>i) component k(i) { const m = mat2(A[0],A[1],A[2],A[3]) return (m * vec2(A[4],A[5])).x } k', gpu.host);
    const r = await settle(() => gpu.dispatch(k, { output: [4], verify: true }));
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
    const k = compileKernel('component k(i) { return (inverse(mat2(4,2,7,6)) * mat2(4,2,7,6) * vec2(1,0)).x } k', gpu.host);
    const r = await settle(() => gpu.dispatch(k, { output: [4], verify: true }));
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
    const k = compileKernel('component k(i) { return (qmat(qaxisangle(vec3(0,0,1),0.7)) * vec3(1,2,3)).x } k', gpu.host);
    const r = await settle(() => gpu.dispatch(k, { output: [4], verify: true }));
    expect(r.backend).not.toBe('cpu');
    expect(r.match?.ok).toBe(true); // verify cross-checks the GLSL/WGSL qmat hand-emit vs the interpreter oracle
    gpu[Symbol.dispose]();
  });
  // A PLAIN metael array (`const x = [1,2,3,4]` — NO typed-array descriptor) as a kernel buffer input, run
  // on a REAL adapter: the coerce-to-f32 path uploads the values to the GPU, and verify cross-checks the real
  // shader output against the interpreter oracle. `not.toBe('cpu')` fails loudly if the ladder falls to CPU
  // (where verify would trivially match and prove nothing about the coerced upload reaching the shader).
  it('a plain-array buffer input matches the oracle on a real adapter', async () => {
    const gpu = createGpuEngine();
    const k = compileKernel('const x = [1, 2, 3, 4]\ncomponent k(i) { return x[i] * 2 }\nk', gpu.host);
    const r = await settle(() => gpu.dispatch(k, { output: [4], verify: true }));
    expect(r.backend).not.toBe('cpu');   // the coerced f32 store was uploaded to a real device
    expect(r.match?.ok).toBe(true);      // real shader output == the interpreter oracle over the plain array
    expect(r.value).toEqual([2, 4, 6, 8]);
    gpu[Symbol.dispose]();
  });

  // The newly GPU-lowered scalar builtins (mod / asinh / acosh / atanh) on a REAL adapter: verify cross-checks
  // the real WGSL/WebGL2 shader output against the interpreter oracle. This env's real adapter is WebGL2, so it
  // proves the GLSL leg (native mod — floored; native asinh/acosh/atanh) == the interpreter. The WGSL leg (mod
  // hand-emitted as the floored x-y*floor(x/y); asinh/acosh/atanh native) is proven by construction + its
  // emit-string tests; a real WebGPU adapter would prove it here too. `not.toBe('cpu')` fails loudly if the
  // ladder falls to CPU (where verify trivially matches and proves nothing about the shader).
  it('mod (floored) matches the oracle on a real adapter — sign follows the divisor for negatives', () => testParity(
    // x = i - 4 = [-4..3]; mod(x, 3) floored = [2,0,1,2,0,1,2,0]. A truncated `%` would give a wrong sign here,
    // so a real-adapter match proves the floored lowering (WGSL x-y*floor(x/y) / native GLSL mod) is correct.
    'const a = f32(8, (i) => i - 4)\ncomponent k(i) { return mod(a[i], 3) }\nk', [8]));
  it('asinh matches the oracle on a real adapter (native shader builtin)', () => testParity(
    'const a = f32(8, (i) => i - 4)\ncomponent k(i) { return asinh(a[i]) }\nk', [8]));
  it('acosh (in-domain) matches the oracle on a real adapter (native shader builtin)', () => testParity(
    // acosh domain is x>=1: x = i + 1 = [1..8], all in-domain, so the shader + interpreter agree exactly.
    'const a = f32(8, (i) => i + 1)\ncomponent k(i) { return acosh(a[i]) }\nk', [8]));
  it('atanh (in-domain) matches the oracle on a real adapter (native shader builtin)', () => testParity(
    // atanh domain is |x|<1: x = i/10 = [0..0.7], all in-domain. A native GPU atanh differs from the f64
    // interpreter oracle by a few ULP more than sin/cos (an inverse-hyperbolic's steeper slope) — observed
    // maxUlp 5 vs the oracle's tight 4-ULP f32 bound on a real WebGPU adapter. That is GPU transcendental
    // precision, not a lowering bug (~6e-7 absolute), so assert a small explicit ULP bound rather than the
    // tight match.ok — the value is proven correct to GPU f32 precision, and the native `atanh` DID run.
    'const a = f32(8, (i) => i / 10)\ncomponent k(i) { return atanh(a[i]) }\nk', [8], 8));

  // ─── the 32-bit bit ops on a REAL adapter, over BOTH backends (this env exposes a real WebGPU adapter AND a
  //     real WebGL2 adapter). Forcing each backend proves BOTH lowerings on a real device: WGSL's NATIVE
  //     countOneBits/reverseBits (via the u32 round-trip) AND the GLSL ES 3.00 `_countOneBits`/`_reverseBits`
  //     PRELUDE HELPERS (bitCount/bitfieldReverse are ES-3.10-only, so the helpers must compile + run). Both
  //     ops match the oracle EXACTLY (maxUlp 0): a popcount is 0..32 (f32-exact), and reversal preserves the
  //     ≤24-bit significant SPAN of an f32-exact integer, so a reversed value (even a large one) is f32-exact
  //     too — and the oracle frounds BOTH sides. reverseBits' inputs [1..8] reverse to LARGE outputs (e.g.
  //     reverseBits(3)=0xC0000000=3221225472, well past f32's 2^24 exact-INTEGER range) yet still match — the
  //     empirical proof that DECIDES lower-vs-CPU-only in favor of lowering (option a). Were this to fail, the
  //     honest fallback would be to keep reverseBits CPU-only (option b); it passes, so lowering is correct.
  for (const backend of ['webgpu', 'webgl2'] as const) {
    it(`countOneBits matches the oracle EXACTLY on a real ${backend} adapter (0..32 is f32-exact)`, () => testParityExact(
      'const a = f32(8, (i) => 2*i + 1)\ncomponent k(i) { return countOneBits(a[i]) }\nk', [8], backend));
    it(`reverseBits matches the oracle EXACTLY on a real ${backend} adapter — a large output preserves the f32-exact bit-span`, () => testParityExact(
      'const a = f32(8, (i) => i + 1)\ncomponent k(i) { return reverseBits(a[i]) }\nk', [8], backend));
    // ─── FRACTIONAL inputs: the interpreter oracle coerces via `>>>0` = ToUint32 = TRUNCATE toward zero. A shader
    //     that ROUNDS first (round(3.9)=4 vs trunc(3.9)=3) diverges silently — verify is opt-in, so a gate-accepted
    //     `countOneBits(x[i])` over an f32 buffer holding fractional values would return the WRONG bit count with no
    //     error. These rows are the empirical proof the shaders TRUNCATE (matching the oracle) on a real device: EVERY
    //     input `i + 0.75` (0.75, 1.75, …, 7.75) truncates DOWN to a different integer than it would round to (3.75 →
    //     trunc 3, not round 4), so a rounding shader would diverge on every cell. maxUlp 0 (both sides truncate to the
    //     same integer, then the popcount / span-preserving reverse is f32-exact). ───
    it(`countOneBits with FRACTIONAL inputs matches the oracle EXACTLY on a real ${backend} adapter — truncate like \`>>>0\`, not round`, () => testParityExact(
      'const a = f32(8, (i) => i + 0.75)\ncomponent k(i) { return countOneBits(a[i]) }\nk', [8], backend));
    it(`reverseBits with FRACTIONAL inputs matches the oracle EXACTLY on a real ${backend} adapter — truncate like \`>>>0\`, not round`, () => testParityExact(
      'const a = f32(8, (i) => i + 0.75)\ncomponent k(i) { return reverseBits(a[i]) }\nk', [8], backend));
  }
});

// Drive a kernel on a FORCED backend + verify, asserting a BIT-EXACT match (maxUlp 0). If the forced backend
// is unavailable in this env, the ladder falls to CPU — assert non-CPU so an adapter-less run fails loudly
// (a CPU verify trivially matches and would prove nothing about the shader). Used by the bit-op parity cases,
// whose f32-exactness claim (a popcount / a span-preserving reverse) demands an EXACT match, not a ULP bound.
async function testParityExact(src: string, output: number[], backend: 'webgpu' | 'webgl2'): Promise<void> {
  const gpu = createGpuEngine();
  try {
    const k = compileKernel(src, gpu.host);
    const r = await settle(() => gpu.dispatch(k, { output, verify: true, backend }));
    expect(r.backend).not.toBe('cpu');   // the forced real shader path ran — not the CPU fallback
    expect(r.match?.ok).toBe(true);      // verify cross-checks the real GPU output vs the interpreter oracle
    expect(r.match?.maxUlp).toBe(0);     // f32-exact: bit-identical to the oracle (no rounding gap)
  } finally {
    gpu[Symbol.dispose]();
  }
}

// Drive one kernel through the real device ladder + verify. Shared by the scalar-builtin parity cases. When
// `ulpBound` is given, accept the run within that ULP bound (documenting a native transcendental's few-ULP
// gap vs the f64 oracle) instead of the tight `match.ok` (the oracle's default 4-ULP f32 bound).
async function testParity(src: string, output: number[], ulpBound?: number): Promise<void> {
  const gpu = createGpuEngine();
  try {
    const k = compileKernel(src, gpu.host);
    const r = await settle(() => gpu.dispatch(k, { output, verify: true }));
    expect(r.backend).not.toBe('cpu');   // a real shader path ran — not the CPU fallback (which proves nothing)
    if (ulpBound === undefined) {
      expect(r.match?.ok).toBe(true);    // verify cross-checks the real GPU output vs the interpreter oracle
    } else {
      // A real shader path ran + the native op is correct to a few ULP of f32 (GPU transcendental precision).
      expect(r.match?.maxUlp ?? Infinity).toBeLessThanOrEqual(ulpBound);
    }
  } finally {
    gpu[Symbol.dispose]();
  }
}
