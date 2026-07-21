// packages/gpu/src/buffer-output.test.ts
import { describe, it, expect } from 'vitest';
import { MATH_BUILTINS } from '@metael/math/lang';
import { RuntimeReactiveHost, change } from '@metael/runtime';
import { evaluateProgram, isUserFn, RecordingHostEnv, descriptorOf, isTypedArray } from '@metael/lang';
import type { UserFn } from '@metael/lang';
import { GpuEngine } from './resource.ts';

function kernelOf(src: string, host: RuntimeReactiveHost): UserFn {
  const res = evaluateProgram(src, { host, env: new RecordingHostEnv(), builtins: [MATH_BUILTINS] });
  if (!isUserFn(res.value)) throw new Error('kernel'); return res.value;
}
const cpuDeps = { tryWebGpu: async () => null, tryWebGl2: () => null, limitsHint: { maxStorageBufferBindingSize: 1 << 28, maxComputeWorkgroupsPerDimension: 65535 } };

describe('buffer output mode (outputType: "buffer")', () => {
  it('default (no outputType) still returns a plain number[]', async () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const x = f32([0, 1, 2, 3])\ncomponent k(i) { return x[i] * 2 }\nk`, host);
    const engine = new GpuEngine(host, cpuDeps);
    change(() => { engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    await new Promise((r) => setTimeout(r, 20));
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    expect(Array.isArray(s.value)).toBe(true);
    expect(s.value).toEqual([0, 2, 4, 6]);
  });
  it('outputType:"buffer" returns a frozen f32 custom value wrapping the readback store', async () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const x = f32([0, 1, 2, 3])\ncomponent k(i) { return x[i] * 2 }\nk`, host);
    const engine = new GpuEngine(host, cpuDeps);
    const cfg = { output: [4], backend: 'cpu' as const, outputType: 'buffer' as const };
    change(() => { engine.gpu(kernel, cfg); });
    await new Promise((r) => setTimeout(r, 20));
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, cfg); });
    expect(isTypedArray(s.value)).toBe(true);                     // an f32 handle, not a plain array
    // It reads like a buffer through the descriptor (index + length + iterate):
    const d = descriptorOf(s.value)!;
    expect(d.getMember!(s.value, 'length')).toBe(4);
    expect(d.getIndex!(s.value, 2)).toBe(4);
    expect(Array.from(d.iterate!(s.value) as number[])).toEqual([0, 2, 4, 6]);
    // It is FROZEN via markFrozen — the descriptor REPORTS frozen (the interpreter's write gate reads this
    // at the write site to reject an in-place write). NOTE: the typed-array descriptor's setIndex itself
    // validates only bounds/type, NOT the frozen box — so d.setIndex(...) directly would NOT throw; freeze
    // is enforced one layer up. Assert the reported frozen flag, not a direct setIndex throw.
    expect(d.frozen!(s.value)).toBe(true);
  });
  it('outputType is part of the memo key — array and buffer runs are distinct resources', async () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const x = f32([0, 1])\ncomponent k(i) { return x[i] }\nk`, host);
    const engine = new GpuEngine(host, cpuDeps);
    change(() => { engine.gpu(kernel, { output: [2], backend: 'cpu' }); });
    change(() => { engine.gpu(kernel, { output: [2], backend: 'cpu', outputType: 'buffer' }); });
    await new Promise((r) => setTimeout(r, 20));
    let arr!: ReturnType<GpuEngine['gpu']>; let bufr!: ReturnType<GpuEngine['gpu']>;
    change(() => { arr = engine.gpu(kernel, { output: [2], backend: 'cpu' }); });
    change(() => { bufr = engine.gpu(kernel, { output: [2], backend: 'cpu', outputType: 'buffer' }); });
    expect(Array.isArray(arr.value)).toBe(true);
    expect(isTypedArray(bufr.value)).toBe(true);
  });
});
