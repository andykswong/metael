import { describe, it, expect } from 'vitest';
import { RuntimeReactiveHost, change } from '@metael/runtime';
import { evaluateProgram, isUserFn } from '@metael/lang';
import type { UserFn } from '@metael/lang';
import { RecordingHostEnv, PlainStorageHost } from '@metael/lang';
import { GpuEngine } from './resource.ts';
import { gateKernel } from './gate.ts';
import { emitWgsl } from './emit-wgsl.ts';

function kernelOf(src: string, host: RuntimeReactiveHost): UserFn { const res = evaluateProgram(src, { host, env: new RecordingHostEnv() }); if (!isUserFn(res.value)) throw new Error('kernel'); return res.value; }
const cpuDeps = { tryWebGpu: async () => null, tryWebGl2: () => null, limitsHint: { maxStorageBufferBindingSize: 1 << 28, maxComputeWorkgroupsPerDimension: 65535 } };

describe('vec math in kernels (CPU path, node)', () => {
  it('a per-cell length(vec3(...)) kernel computes on the CPU + matches the oracle', async () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`
      const a = f32(48, (i) => i)
      component k(i) { const v = vec3(a[i*3], a[i*3+1], a[i*3+2]); return length(v) }
      k`, host);
    const engine = new GpuEngine(host, cpuDeps);
    const cfg = { output: [16], backend: 'cpu' as const, verify: true };   // verify → run the oracle
    change(() => { engine.gpu(kernel, cfg); });
    await new Promise((r) => setTimeout(r, 30));
    let settled!: ReturnType<GpuEngine['gpu']>;
    change(() => { settled = engine.gpu(kernel, cfg); });
    expect(settled.pending).toBe(false);
    expect((settled.value as number[] | null)?.[1]).toBeCloseTo(Math.hypot(3, 4, 5), 4);   // cell 1 → v=(3,4,5)
    expect(settled.match?.ok).toBe(true);
  });
  it('a cross/normalize kernel (vec intermediate → scalar) computes + matches the oracle', async () => {
    const host = new RuntimeReactiveHost();
    // length(cross(u, w)) — cross produces a vec intermediate that flows into length → scalar out.
    const kernel = kernelOf(`
      const a = f32(48, (i) => i)
      component k(i) {
        const u = vec3(a[i*3], a[i*3+1], a[i*3+2])
        const w = vec3(1, 0, 0)
        return length(cross(u, w))
      }
      k`, host);
    const engine = new GpuEngine(host, cpuDeps);
    const cfg = { output: [16], backend: 'cpu' as const, verify: true };   // verify → run the oracle
    change(() => { engine.gpu(kernel, cfg); });
    await new Promise((r) => setTimeout(r, 30));
    let settled!: ReturnType<GpuEngine['gpu']>;
    change(() => { settled = engine.gpu(kernel, cfg); });
    expect(settled.pending).toBe(false);
    expect(settled.match?.ok).toBe(true);   // CPU delegate ≡ interpreter (cross/normalize now correct, not NaN)
    expect((settled.value as number[] | null)?.[1]).toBeGreaterThan(0);   // cell 1: u=(3,4,5), cross with (1,0,0) is nonzero
  });
});

describe('vec math lowers to native WGSL vecN via Lowering.ops', () => {
  it('a vec3 * vec3 emits a native multiply + dot emits native dot', () => {
    const host = new PlainStorageHost();
    const res = evaluateProgram(`
      const a = f32(48, (i) => i)
      component k(i) { const u = vec3(a[i*3], a[i*3+1], a[i*3+2]); const v = u * u; return dot(u, v) }
      k`, { host, env: new RecordingHostEnv() });
    const fn = res.value as UserFn;
    const wgsl = emitWgsl(fn, gateKernel(fn, host).bindings, 'f32');
    expect(wgsl).toContain('vec3<f32>');
    expect(wgsl).toContain('dot(');
  });
});
