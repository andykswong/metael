// packages/gpu/src/plain-array-input.test.ts
// A PLAIN metael array (`const x = [1, 2, 3, 4]` — no typed-array descriptor) used as a kernel buffer input:
// it is classified as a buffer, coerced ONCE to Float32Array, and that coercion is cached by CONTENT
// fingerprint (not object identity) so a plain array rebuilt fresh each derive with identical content is not
// re-coerced. The interpreter oracle is the reference: a plain-array kernel's dispatch must match it.
import { describe, it, expect, vi } from 'vitest';
import { RuntimeReactiveHost, change } from '@metael/runtime';
import { evaluateProgram, isUserFn, RecordingHostEnv } from '@metael/lang';
import { MATH_BUILTINS } from '@metael/math/lang';
import type { UserFn } from '@metael/lang';
import { GpuEngine } from './resource.ts';

function kernelOf(src: string, host: RuntimeReactiveHost): UserFn {
  const res = evaluateProgram(src, { host, env: new RecordingHostEnv(), builtins: [MATH_BUILTINS] });
  if (!isUserFn(res.value)) throw new Error('expected kernel');
  return res.value;
}
const cpuOnlyDeps = { tryWebGpu: async () => null, tryWebGl2: () => null, limitsHint: { maxStorageBufferBindingSize: 1 << 28, maxComputeWorkgroupsPerDimension: 65535 } };
const drain = () => new Promise((r) => setTimeout(r, 20));

describe('plain number[] as a kernel buffer input', () => {
  it('a plain array input is classified as a buffer, dispatches, and matches the CPU oracle', async () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const x = [1, 2, 3, 4]\ncomponent k(i) { return x[i] * 2 }\nk`, host);
    const engine = new GpuEngine(host, cpuOnlyDeps);
    let r!: ReturnType<GpuEngine['gpu']>;
    change(() => { r = engine.gpu(kernel, { output: [4], backend: 'cpu', verify: true }); });
    expect(r.core).toBe(true);              // the gate ACCEPTS a plain all-number array as a buffer input
    expect(r.error).toBeNull();
    await drain();
    change(() => { r = engine.gpu(kernel, { output: [4], backend: 'cpu', verify: true }); });
    expect(r.pending).toBe(false);
    expect(r.value).toEqual([2, 4, 6, 8]);  // NOT an empty/zero buffer — the coercion carried the values
    expect(r.match?.ok).toBe(true);         // CPU-emit ≡ the interpreter oracle for a plain-array kernel
  });

  it('a mixed / non-number array is NOT a buffer — the gate rejects it (MLGPU-BAD-INPUT)', async () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const x = [1, "two", 3]\ncomponent k(i) { return x[i] }\nk`, host);
    const engine = new GpuEngine(host, cpuOnlyDeps);
    let r!: ReturnType<GpuEngine['gpu']>;
    change(() => { r = engine.gpu(kernel, { output: [3], backend: 'cpu' }); });
    expect(r.core).toBe(false);
    expect(r.reasons.some((d) => d.code === 'MLGPU-BAD-INPUT')).toBe(true);
  });

  it('distinct plain-array content gets a distinct result (no fingerprint collision on one engine)', async () => {
    const host = new RuntimeReactiveHost();
    const engine = new GpuEngine(host, cpuOnlyDeps);
    const kA = kernelOf(`const x = [1, 2, 3, 4]\ncomponent k(i) { return x[i] }\nk`, host);
    const kB = kernelOf(`const x = [10, 20, 30, 40]\ncomponent k(i) { return x[i] }\nk`, host);
    change(() => { engine.gpu(kA, { output: [4], backend: 'cpu' }); });
    change(() => { engine.gpu(kB, { output: [4], backend: 'cpu' }); });
    await drain();
    let rA!: ReturnType<GpuEngine['gpu']>; let rB!: ReturnType<GpuEngine['gpu']>;
    change(() => { rA = engine.gpu(kA, { output: [4], backend: 'cpu' }); });
    change(() => { rB = engine.gpu(kB, { output: [4], backend: 'cpu' }); });
    expect(rA.value).toEqual([1, 2, 3, 4]);
    expect(rB.value).toEqual([10, 20, 30, 40]);   // NOT aliased to kA's cached result
  });

  it('coerce-once: identical plain-array content is coerced to Float32Array exactly ONCE across two distinct-key dispatches', async () => {
    // The load-bearing cache assertion. verify:false vs verify:true are DISTINCT memo keys → BOTH resolve
    // inputs (so the memo hit alone can't hide a re-coerce). With content-fingerprint caching the plain array
    // is `Float32Array.from`'d ONCE and reused; without the cache it would be coerced twice.
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const x = [5, 6, 7, 8]\ncomponent k(i) { return x[i] * 2 }\nk`, host);
    const engine = new GpuEngine(host, cpuOnlyDeps);
    const fromSpy = vi.spyOn(Float32Array, 'from');
    try {
      change(() => { engine.gpu(kernel, { output: [4], backend: 'cpu' }); });                  // key A
      change(() => { engine.gpu(kernel, { output: [4], backend: 'cpu', verify: true }); });    // key B (distinct)
      // Count only the plain-array coercions (first arg is the [5,6,7,8] content) — robust against any
      // unrelated Float32Array.from a backend/oracle might do.
      const coercions = fromSpy.mock.calls.filter((c) => Array.isArray(c[0]) && (c[0] as number[]).length === 4 && (c[0] as number[])[0] === 5);
      expect(coercions.length).toBe(1);
    } finally { fromSpy.mockRestore(); }
    await drain();
  });

  it('the coerce cache is FIFO-capped at MAX_LIVE distinct contents (oldest content re-coerces, newest stays cached)', async () => {
    // Proves the cap WITHOUT exposing internals: dispatch MAX_LIVE+1 distinct-content plain arrays (each a
    // distinct coerce key), which FIFO-evicts the FIRST content. Re-dispatching that evicted content under a
    // DISTINCT memo key (verify:true → memo miss → resolveInputs re-runs) must RE-coerce it (count 2), while
    // the newest content is still cached (count 1). Without a cap the oldest would still be cached (count 1).
    const MAX_LIVE = 8;   // mirrors resource.ts — the coerce cache holds at most this many distinct contents
    const host = new RuntimeReactiveHost();
    const engine = new GpuEngine(host, cpuOnlyDeps);
    // Distinct single-element contents [1], [2], ... [MAX_LIVE+1] — each a distinct first-element marker.
    const kernels = Array.from({ length: MAX_LIVE + 1 }, (_, k) => kernelOf(`const x = [${k + 1}]\ncomponent k(i) { return x[i] }\nk`, host));
    const coercions = (spy: ReturnType<typeof vi.spyOn>, marker: number) =>
      (spy.mock.calls as unknown[][]).filter((c) => Array.isArray(c[0]) && (c[0] as number[]).length === 1 && (c[0] as number[])[0] === marker).length;
    const fromSpy = vi.spyOn(Float32Array, 'from');
    try {
      // Coercion is synchronous in gpu() (before enqueue), so all MAX_LIVE+1 coercions happen here — the first
      // content ([1]) is inserted first, so appending the (MAX_LIVE+1)th content FIFO-evicts it.
      kernels.forEach((kern) => change(() => { engine.gpu(kern, { output: [1], backend: 'cpu' }); }));
      expect(coercions(fromSpy, 1)).toBe(1);              // oldest coerced exactly once so far
      // Re-dispatch the EVICTED oldest under a distinct memo key → memo miss → resolveInputs → cache miss → RE-coerce.
      change(() => { engine.gpu(kernels[0]!, { output: [1], backend: 'cpu', verify: true }); });
      expect(coercions(fromSpy, 1)).toBe(2);              // re-coerced ⇒ it WAS evicted (the cap fired)
      // Re-dispatch the NEWEST (never evicted) under a distinct memo key → cache HIT → no re-coerce.
      change(() => { engine.gpu(kernels[MAX_LIVE]!, { output: [1], backend: 'cpu', verify: true }); });
      expect(coercions(fromSpy, MAX_LIVE + 1)).toBe(1);   // still cached ⇒ FIFO kept the newest
    } finally { fromSpy.mockRestore(); }
    await drain();
  });
});
