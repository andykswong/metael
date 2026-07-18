import { describe, it, expect } from 'vitest';
import { RuntimeReactiveHost, change } from '@metael/runtime';
import { evaluateProgram, isUserFn, RecordingHostEnv } from '@metael/lang';
import type { UserFn } from '@metael/lang';
import { GpuEngine } from './resource.ts';
import type { Backend, DispatchInput, DispatchResult } from './device/index.ts';

function kernelOf(src: string, host: RuntimeReactiveHost): UserFn {
  const res = evaluateProgram(src, { host, env: new RecordingHostEnv() });
  if (!isUserFn(res.value)) throw new Error('kernel: ' + JSON.stringify(res.diagnostics)); return res.value;
}
const cpuDeps = { tryWebGpu: async () => null, tryWebGl2: () => null, limitsHint: { maxStorageBufferBindingSize: 1 << 28, maxComputeWorkgroupsPerDimension: 65535 } };

describe('multi-output (outputs: { sum, diff })', () => {
  it('a named-object return writes several output buffers', async () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const a = f32(4, (i) => i + 1)
const b = f32(4, (i) => i)
component k(i) { return { sum: a[i] + b[i], diff: a[i] - b[i] } }
k`, host);
    const engine = new GpuEngine(host, cpuDeps);
    const cfg = { output: [4], backend: 'cpu' as const, outputs: { sum: {}, diff: {} } };
    change(() => { engine.gpu(kernel, cfg); });
    await new Promise((r) => setTimeout(r, 20));
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, cfg); });
    // a[i]=i+1, b[i]=i → sum = 2i+1 = [1,3,5,7]; diff = 1 = [1,1,1,1].
    expect(s.value).toBeNull();                        // multi-output has no single primary value
    expect(s.outputs).not.toBeNull();
    expect(s.outputs!.sum).toEqual([1, 3, 5, 7]);
    expect(s.outputs!.diff).toEqual([1, 1, 1, 1]);
  });

  it('a single-output run is unaffected — resource.outputs is null, resource.value is the array', async () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const x = f32(4, (i) => i)\ncomponent k(i) { return x[i] * 2 }\nk`, host);
    const engine = new GpuEngine(host, cpuDeps);
    change(() => { engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    await new Promise((r) => setTimeout(r, 20));
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    expect(s.outputs).toBeNull();
    expect(s.value).toEqual([0, 2, 4, 6]);
  });

  it('rejects an outputs kernel that does not return an object with the declared keys', async () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const x = f32(4, (i) => i)\ncomponent k(i) { return x[i] }\nk`, host);   // scalar, not { sum, diff }
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu', outputs: { sum: {}, diff: {} } }); });
    expect(s.core).toBe(false);
    expect(s.reasons.some((r) => r.code === 'MLGPU-OUTPUT-SHAPE')).toBe(true);
  });

  it('multi-output passes each sub-dispatch the inputs in ITS shader binding order (WebGPU positional-bind safety)', async () => {
    // A spy backend that binds POSITIONALLY like WebGPU (`layout:'auto'`, inputs[k] → the k-th declared
    // storage buffer). It records any sub-dispatch whose supplied `inputs` order diverges from the buffer
    // order its OWN shader declares — that divergence IS the misbind bug. Values are computed by NAME
    // (via cpuRun, which is descriptor/name-based → correct) so a correct binding yields correct output.
    const problems: string[] = [];
    const spy: Backend = {
      kind: 'webgpu',
      limits: { maxStorageBufferBindingSize: 1 << 28, maxComputeWorkgroupsPerDimension: 65535 },
      async dispatch(input: DispatchInput): Promise<DispatchResult> {
        // The buffer binding order this shader declares (WGSL `var<storage, read> NAME: array<...>`).
        const declared = [...input.wgsl.matchAll(/var<storage,\s*read>\s+(\w+)\s*:/g)].map((m) => m[1]);
        const supplied = input.inputs.map((i) => i.name);
        if (declared.join(',') !== supplied.join(',')) problems.push(`declared [${declared}] vs supplied [${supplied}]`);
        const total = input.dims.reduce((a, b) => a * b, 1);
        const comps = input.outputComps ?? 1;
        const out = new Float32Array(total * comps);
        for (let i = 0; i < total; i++) { const cell = input.cpuRun([i]); for (let k = 0; k < comps; k++) out[i * comps + k] = cell[k]!; }
        return { output: out, ms: 1 };
      },
      [Symbol.dispose]() {},
    };
    const host = new RuntimeReactiveHost();
    // 'x' uses only b; 'y' uses a and b in a different order than the whole-kernel first-reference order.
    const kernel = kernelOf(`const a = f32(4, (i) => i + 10)
const b = f32(4, (i) => i)
component k(i) { return { x: b[i], y: a[i] - b[i] } }
k`, host);
    const engine = new GpuEngine(host, { tryWebGpu: async () => spy, tryWebGl2: () => null, limitsHint: spy.limits });
    const cfg = { output: [4], backend: 'webgpu' as const, outputs: { x: {}, y: {} } };
    change(() => { engine.gpu(kernel, cfg); });
    await new Promise((r) => setTimeout(r, 30));
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, cfg); });
    expect(problems).toEqual([]);                       // every sub-dispatch's inputs order matched its shader's binding order
    expect(s.outputs!.x).toEqual([0, 1, 2, 3]);         // b[i]
    expect(s.outputs!.y).toEqual([10, 10, 10, 10]);     // a[i]-b[i] = (i+10)-i
  });

  it('rejects a duplicate-key object return (a typo like { sum, sum } must not silently zero the missing output)', async () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const a = f32(4, (i) => i + 1)
const b = f32(4, (i) => i)
component k(i) { return { sum: a[i] + b[i], sum: a[i] - b[i] } }
k`, host);
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu', outputs: { sum: {}, diff: {} } }); });
    expect(s.core).toBe(false);
    expect(s.reasons.some((r) => r.code === 'MLGPU-OUTPUT-SHAPE')).toBe(true);
  });

  it('a multi-output verify verdict reports the true aggregate maxUlp/kind (not a fabricated exact/0)', async () => {
    // A spy backend that returns each output value nudged by exactly 1 ULP from the interpreter reference —
    // so verify sees a WITHIN-TOLERANCE (ok) but NON-exact match (maxUlp 1). Pre-fix the aggregate fabricated
    // kind:'exact', maxUlp:0 whenever every output was `ok`; post-fix it must report the true worst-case ulp.
    const nudge1Ulp = (x: number): number => {
      const buf = new ArrayBuffer(4); const f = new Float32Array(buf); const i = new Int32Array(buf);
      f[0] = Math.fround(x); i[0]! += 1; return f[0]!;
    };
    const spy: Backend = {
      kind: 'webgpu',
      limits: { maxStorageBufferBindingSize: 1 << 28, maxComputeWorkgroupsPerDimension: 65535 },
      async dispatch(input: DispatchInput): Promise<DispatchResult> {
        const total = input.dims.reduce((a, b) => a * b, 1);
        const comps = input.outputComps ?? 1;
        const out = new Float32Array(total * comps);
        for (let idx = 0; idx < total; idx++) { const cell = input.cpuRun([idx]); for (let k = 0; k < comps; k++) out[idx * comps + k] = nudge1Ulp(cell[k]!); }
        return { output: out, ms: 1 };
      },
      [Symbol.dispose]() {},
    };
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const a = f32(4, (i) => i + 1)
const b = f32(4, (i) => i)
component k(i) { return { sum: a[i] + b[i], diff: a[i] - b[i] } }
k`, host);
    const engine = new GpuEngine(host, { tryWebGpu: async () => spy, tryWebGl2: () => null, limitsHint: spy.limits });
    const cfg = { output: [4], backend: 'webgpu' as const, verify: true, outputs: { sum: {}, diff: {} } };
    change(() => { engine.gpu(kernel, cfg); });
    await new Promise((r) => setTimeout(r, 30));
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, cfg); });
    expect(s.match).not.toBeNull();
    expect(s.match!.ok).toBe(true);        // 1 ULP is within the f32 tolerance → the run is ok
    expect(s.match!.kind).toBe('ulp');     // but NOT exact — honest, not fabricated 'exact'
    expect(s.match!.maxUlp).toBe(1);        // the true worst-case ulp across all outputs
  });
});
