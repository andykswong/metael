// The backend residency contract, headless: a backend asked to `retainOutput` returns a `resident` handle
// wrapping its output buffer + a disposer. The CPU backend is trivially resident — the output Float32Array
// IS the buffer (portable, zero-copy), a no-op dispose. (The GPU resident-bind path is proven on a real
// adapter in webgl2.browser.test.ts, where the SAME backend instance produces + binds the resident object.)
import { describe, it, expect } from 'vitest';
import { makeCpuBackend } from './device/cpu.ts';
import type { DispatchInput } from './device/index.ts';

describe('@metael/gpu — CPU backend residency (headless)', () => {
  it('cpu backend retains its output array as the resident buffer', async () => {
    const back = makeCpuBackend();
    // The CPU backend's dispatch touches only `dims` + `cpuRun` (+ now `retainOutput`); the remaining fields
    // are unread by this backend, so dummies satisfy the contract without a real kernel/binding table.
    const di = {
      kernel: undefined as never, bindings: undefined as never,
      dims: [4], precision: 'f32' as const, wgsl: '', glsl: '',
      cpuRun: (c: readonly number[]) => [c[0]! * 2],
      inputs: [], scalars: [],
    } satisfies DispatchInput;
    const res = await back.dispatch({ ...di, retainOutput: true });
    expect(res.resident).toBeDefined();
    expect(res.resident!.gpuBuffer).toBe(res.output);   // the SAME Float32Array — zero-copy resident
    expect(() => res.resident!.dispose()).not.toThrow();  // no-op
    // Sanity: the array actually holds the computed cells (coord[0] * 2 over dims [4]).
    expect(Array.from(res.output)).toEqual([0, 2, 4, 6]);
    back[Symbol.dispose]();
  });

  it('cpu backend omits `resident` when retainOutput is not set', async () => {
    const back = makeCpuBackend();
    const di = {
      kernel: undefined as never, bindings: undefined as never,
      dims: [4], precision: 'f32' as const, wgsl: '', glsl: '',
      cpuRun: (c: readonly number[]) => [c[0]! * 2],
      inputs: [], scalars: [],
    } satisfies DispatchInput;
    const res = await back.dispatch(di);
    expect(res.resident).toBeUndefined();
    back[Symbol.dispose]();
  });
});
