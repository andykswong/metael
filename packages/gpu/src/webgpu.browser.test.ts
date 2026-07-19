import { describe, it, expect } from 'vitest';
import { RuntimeReactiveHost, change } from '@metael/runtime';
import { evaluateProgram, isUserFn } from '@metael/lang';
import type { UserFn } from '@metael/lang';
import { RecordingHostEnv } from '@metael/lang';
import { GpuEngine } from './resource.ts';
import { tryWebGpuBackend } from './device/webgpu.ts';
import { gateKernel } from './gate.ts';
import { gateReducer } from './reduce.ts';
import { gateBinMapper } from './histogram.ts';
import { emitWgsl, emitReduceWgsl, emitHistogramWgsl } from './emit-wgsl.ts';
import { createGpuEngine } from './api.ts';

function kernelOf(src: string, host: RuntimeReactiveHost): UserFn { const res = evaluateProgram(src, { host, env: new RecordingHostEnv() }); if (!isUserFn(res.value)) throw new Error('kernel'); return res.value; }
function reducerOf(src: string, host: RuntimeReactiveHost): UserFn { const res = evaluateProgram(src, { host, env: new RecordingHostEnv() }); if (!isUserFn(res.value)) throw new Error('reducer'); return res.value; }
function mapperOf(src: string, host: RuntimeReactiveHost): UserFn { const res = evaluateProgram(src, { host, env: new RecordingHostEnv() }); if (!isUserFn(res.value)) throw new Error('mapper'); return res.value; }
const deps = { tryWebGpu: tryWebGpuBackend, tryWebGl2: () => null, limitsHint: { maxStorageBufferBindingSize: 1 << 28, maxComputeWorkgroupsPerDimension: 65535 } };

describe('@metael/gpu — real WebGPU dispatch (Chromium)', () => {
  it('saxpy on the GPU matches the interpreter oracle + reports the actual backend', async () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`
      const N = 256
      const x = f32(N, (i) => i)
      const y = f32(N, (i) => 2 * i)
      component k(i) { return 3 * x[i] + y[i] }
      k`, host);
    const engine = new GpuEngine(host, deps);
    // verify + benchmark opt-in so this real-adapter test still exercises the oracle + the CPU race.
    const cfg = { output: [256], verify: true, benchmark: true };
    change(() => { engine.gpu(kernel, cfg); });
    await new Promise((r) => setTimeout(r, 300));
    let settled!: ReturnType<GpuEngine['gpu']>;
    change(() => { settled = engine.gpu(kernel, cfg); });
    expect(settled.pending).toBe(false);
    expect(['webgpu', 'webgl2', 'cpu']).toContain(settled.backend);   // never ASSUME webgpu (headless fallback)
    expect(settled.match?.ok).toBe(true);                             // GPU within tolerance of the oracle
    expect((settled.value as number[] | null)?.[3]).toBeCloseTo(3 * 3 + 6, 3);             // output[3] = 3*3 + 2*3 = 15
    if (settled.backend === 'webgpu') expect(settled.speedup).not.toBeNull();
  });

  it('the emitted WGSL actually COMPILES on a real device (catches i32/f32 type errors a substring test misses)', async (ctx) => {
    const gpu = (navigator as unknown as { gpu?: GPU }).gpu;
    if (!gpu) return ctx.skip('no WebGPU: navigator.gpu absent');
    const adapter = await gpu.requestAdapter();
    if (!adapter) return ctx.skip('no WebGPU: requestAdapter() returned null');
    const device = await adapter.requestDevice();
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`
      const N = 8
      const a = f32(N * N, (i) => i)
      const b = f32(N * N, (i) => i)
      component product(row, col) { let sum = 0; for (const k of range(N)) { sum = sum + a[row * N + k] * b[k * N + col] } return sum }
      product`, host);
    const { bindings } = gateKernel(kernel, host);
    const wgsl = emitWgsl(kernel, bindings, 'f32');
    const module = device.createShaderModule({ code: wgsl });
    const info = await module.getCompilationInfo();
    const errors = info.messages.filter((m) => m.type === 'error');
    expect(errors.map((e) => e.message)).toEqual([]);   // ZERO shader-compile errors — the real gate
    device.destroy();
  });

  it('the f16 WGSL compiles on a real device that supports shader-f16 (else skips)', async (ctx) => {
    const gpu = (navigator as unknown as { gpu?: GPU }).gpu;
    if (!gpu) return ctx.skip('no WebGPU: navigator.gpu absent');
    const adapter = await gpu.requestAdapter();
    if (!adapter || !adapter.features.has('shader-f16')) return ctx.skip('no adapter, or adapter lacks shader-f16');
    const device = await adapter.requestDevice({ requiredFeatures: ['shader-f16'] });
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const x = f32(8, (i) => i)\ncomponent k(i) { return x[i] * 2 }\nk`, host);
    const { bindings } = gateKernel(kernel, host);
    const wgsl = emitWgsl(kernel, bindings, 'f16');
    const module = device.createShaderModule({ code: wgsl });
    const info = await module.getCompilationInfo();
    expect(info.messages.filter((m) => m.type === 'error').map((e) => e.message)).toEqual([]);
    device.destroy();
  });

  it('the reduce WGSL (workgroup-shared tree reduction) compiles on a real WebGPU device (else skips)', async (ctx) => {
    const gpu = (navigator as unknown as { gpu?: GPU }).gpu;
    if (!gpu) return ctx.skip('no WebGPU: navigator.gpu absent');
    const adapter = await gpu.requestAdapter();
    if (!adapter) return ctx.skip('no WebGPU: requestAdapter() returned null');
    const device = await adapter.requestDevice();
    const host = new RuntimeReactiveHost();
    const reducer = reducerOf(`component add(acc, x) { return acc + x }\nadd`, host);
    const { bindings } = gateReducer(reducer, host);
    const wgsl = emitReduceWgsl(reducer, bindings, 0);
    const module = device.createShaderModule({ code: wgsl });
    const info = await module.getCompilationInfo();
    expect(info.messages.filter((m) => m.type === 'error').map((e) => e.message)).toEqual([]);
    device.destroy();
  });

  it('the histogram WGSL (atomic scatter) compiles on a real WebGPU device (else skips)', async (ctx) => {
    const gpu = (navigator as unknown as { gpu?: GPU }).gpu;
    if (!gpu) return ctx.skip('no WebGPU: navigator.gpu absent');
    const adapter = await gpu.requestAdapter();
    if (!adapter) return ctx.skip('no WebGPU: requestAdapter() returned null');
    const device = await adapter.requestDevice();
    const host = new RuntimeReactiveHost();
    const binOf = mapperOf(`component binOf(x) { return x % 4 }\nbinOf`, host);
    const { bindings } = gateBinMapper(binOf, host);
    const wgsl = emitHistogramWgsl(binOf, bindings, 4);
    const module = device.createShaderModule({ code: wgsl });
    const info = await module.getCompilationInfo();
    expect(info.messages.filter((m) => m.type === 'error').map((e) => e.message)).toEqual([]);
    device.destroy();
  });

  // A rank-3 kernel is dispatched with a 3D workgroup grid (ceil per axis / 4, matching emitWgsl's
  // @workgroup_size(4,4,4)). `output: [2, 2, 8]` is deliberately chosen so the z-axis needs MORE than one
  // 4-deep workgroup: ceil(8/4) === 2. A grid that omits the z divisor (dispatchWorkgroups(x, y) with z
  // defaulting to 1) covers only z=0..3, leaving z=4..7 uncomputed → the verify cross-check against the
  // interpreter oracle MISMATCHES. So this is RED before the 3D grid fix and GREEN after — on a real adapter.
  it('a 3D kernel dispatches over a multi-workgroup z-axis and matches the oracle (real adapter)', async () => {
    const gpu = createGpuEngine();
    const k = gpu.compile('component k(x, y, z) { return x * 100 + y * 10 + z } k');
    const r = await gpu.settle(k, { output: [2, 2, 8], verify: true });
    expect(r.backend).not.toBe('cpu');   // a real GPU path ran — not the CPU fallback (which proves nothing)
    expect(r.match?.ok).toBe(true);      // verify cross-checks EVERY cell (incl. z=4..7) vs the interpreter oracle
    gpu[Symbol.dispose]();
  });
});
