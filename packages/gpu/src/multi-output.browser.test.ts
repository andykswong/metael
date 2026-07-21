import { describe, it, expect } from 'vitest';
import { MATH_BUILTINS } from '@metael/math/lang';
import { RuntimeReactiveHost, change } from '@metael/runtime';
import { evaluateProgram, isUserFn, RecordingHostEnv } from '@metael/lang';
import type { UserFn } from '@metael/lang';
import { GpuEngine } from './resource.ts';
import { tryWebGl2Backend } from './device/webgl2.ts';

function kernelOf(src: string, host: RuntimeReactiveHost): UserFn { const res = evaluateProgram(src, { host, env: new RecordingHostEnv(), builtins: [MATH_BUILTINS] }); if (!isUserFn(res.value)) throw new Error('kernel'); return res.value; }

// Whether THIS runner can actually dispatch on WebGL2. When true the multi-output test REQUIRES the webgl2
// backend actually ran — the point is proving the N-sequential-passes (one fragment pass per named output)
// each match the CPU/interpreter oracle on a real adapter (the empirical proof a no-adapter emit test can't
// give). Else → the CPU floor. Either way EACH named output must match its oracle.
const webgl2Live = !!tryWebGl2Backend();
const deps = { tryWebGpu: async () => null, tryWebGl2: tryWebGl2Backend, limitsHint: { maxStorageBufferBindingSize: 1 << 28, maxComputeWorkgroupsPerDimension: 65535 } };

describe('@metael/gpu — multi-output on real WebGL2 (Chromium)', () => {
  it('a { sum, diff } kernel dispatches N passes; each named output matches the oracle', async () => {
    const host = new RuntimeReactiveHost();
    const N = 64;
    // a[i]=i+1, b[i]=i → sum = 2i+1, diff = 1. Each output is its own fragment pass over the same inputs.
    const kernel = kernelOf(`
      const N = ${N}
      const a = f32(N, (i) => i + 1)
      const b = f32(N, (i) => i)
      component k(i) {
        return { sum: a[i] + b[i], diff: a[i] - b[i] }
      }
      k`, host);
    const engine = new GpuEngine(host, deps);
    const cfg = { output: [N], backend: 'webgl2' as const, outputs: { sum: {}, diff: {} }, verify: true };
    change(() => { engine.gpu(kernel, cfg); });
    await new Promise((r) => setTimeout(r, 400));
    let settled!: ReturnType<GpuEngine['gpu']>;
    change(() => { settled = engine.gpu(kernel, cfg); });
    expect(settled.pending).toBe(false);
    // On a live WebGL2 adapter each named output MUST have run on the GPU (not fallen to CPU); else the floor.
    expect(settled.backend).toBe(webgl2Live ? 'webgl2' : 'cpu');
    // The aggregate per-output oracle verdict: every sampled output matched the interpreter.
    expect(settled.match?.ok).toBe(true);
    // Multi-output → value is null; outputs carries each named buffer as a plain number[] of length N.
    expect(settled.value).toBeNull();
    const outs = settled.outputs as Record<string, number[]> | null;
    expect(outs).not.toBeNull();
    const sum = outs!.sum!; const diff = outs!.diff!;
    expect(sum.length).toBe(N);
    expect(diff.length).toBe(N);
    // sum[i] = 2i+1: [1,3,5,...]; diff[i] = 1 everywhere.
    for (const i of [0, 1, 5, 31, 63]) expect(sum[i]!).toBeCloseTo(2 * i + 1, 3);
    for (const i of [0, 1, 5, 31, 63]) expect(diff[i]!).toBeCloseTo(1, 3);
  });
});
