import { describe, it, expect } from 'vitest';
import { MATH_BUILTINS } from '@metael/math/lang';
import { RuntimeReactiveHost, change } from '@metael/runtime';
import { evaluateProgram, isUserFn } from '@metael/lang';
import type { UserFn } from '@metael/lang';
import { RecordingHostEnv } from '@metael/lang';
import { GpuEngine } from './resource.ts';
import { tryWebGpuBackend } from './device/webgpu.ts';

function kernelOf(src: string, host: RuntimeReactiveHost): UserFn { const res = evaluateProgram(src, { host, env: new RecordingHostEnv(), builtins: [MATH_BUILTINS] }); if (!isUserFn(res.value)) throw new Error('kernel'); return res.value; }
const deps = { tryWebGpu: tryWebGpuBackend, tryWebGl2: () => null, limitsHint: { maxStorageBufferBindingSize: 1 << 28, maxComputeWorkgroupsPerDimension: 65535 } };

describe('@metael/gpu — vec-math kernel (Chromium)', () => {
  it('length(vec3(...)) matches the oracle + reports the actual backend', async () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`
      const a = f32(48, (i) => i)
      component k(i) { const v = vec3(a[i*3], a[i*3+1], a[i*3+2]); return length(v) }
      k`, host);
    const engine = new GpuEngine(host, deps);
    const cfg = { output: [16], verify: true };   // verify → run the oracle on the real adapter
    change(() => { engine.gpu(kernel, cfg); });
    await new Promise((r) => setTimeout(r, 300));
    let settled!: ReturnType<GpuEngine['gpu']>;
    change(() => { settled = engine.gpu(kernel, cfg); });
    expect(settled.pending).toBe(false);
    expect(['webgpu', 'webgl2', 'cpu']).toContain(settled.backend);   // headless → likely cpu (delegates)
    expect(settled.match?.ok).toBe(true);
    expect((settled.value as number[] | null)?.[1]).toBeCloseTo(Math.hypot(3, 4, 5), 3);
  });
});
