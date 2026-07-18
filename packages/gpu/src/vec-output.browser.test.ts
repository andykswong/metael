import { describe, it, expect } from 'vitest';
import { RuntimeReactiveHost, change } from '@metael/runtime';
import { evaluateProgram, isUserFn, RecordingHostEnv } from '@metael/lang';
import type { UserFn } from '@metael/lang';
import { GpuEngine } from './resource.ts';
import { tryWebGl2Backend } from './device/webgl2.ts';

function kernelOf(src: string, host: RuntimeReactiveHost): UserFn { const res = evaluateProgram(src, { host, env: new RecordingHostEnv() }); if (!isUserFn(res.value)) throw new Error('kernel'); return res.value; }

// Whether THIS runner can actually dispatch on WebGL2. When true the vec-output test REQUIRES the webgl2
// backend actually ran — the point is proving the RGBA-channel gather matches the CPU/WGSL interleaved
// layout on a real adapter (the empirical proof a no-adapter emit test can't give). Else → the CPU floor.
const webgl2Live = !!tryWebGl2Backend();
const deps = { tryWebGpu: async () => null, tryWebGl2: tryWebGl2Backend, limitsHint: { maxStorageBufferBindingSize: 1 << 28, maxComputeWorkgroupsPerDimension: 65535 } };

describe('@metael/gpu — single vecN output on real WebGL2 (Chromium)', () => {
  it('a vec3-add kernel on WebGL2 produces the interleaved layout + matches the oracle per-component', async () => {
    const host = new RuntimeReactiveHost();
    const N = 64;
    // x is a flat [N*3] buffer; cell i reads x[i*3..i*3+2] and adds (1,2,3) — a per-cell vec3.
    const kernel = kernelOf(`
      const N = ${N}
      const x = f32(N * 3, (i) => i)
      component k(i) {
        return vec3(x[i*3], x[i*3+1], x[i*3+2]) + vec3(1, 2, 3)
      }
      k`, host);
    const engine = new GpuEngine(host, deps);
    const cfg = { output: [N], backend: 'webgl2' as const, outputElement: 'vec3' as const, verify: true };
    change(() => { engine.gpu(kernel, cfg); });
    await new Promise((r) => setTimeout(r, 300));
    let settled!: ReturnType<GpuEngine['gpu']>;
    change(() => { settled = engine.gpu(kernel, cfg); });
    expect(settled.pending).toBe(false);
    // On a live WebGL2 adapter the RGBA-channel gather MUST have run on the GPU (not fallen to CPU) — that's
    // the real-adapter proof; else the CPU floor. Either way the per-component oracle must agree.
    expect(settled.backend).toBe(webgl2Live ? 'webgl2' : 'cpu');
    expect(settled.match?.ok).toBe(true);   // per-component ULP match with the interpreter (proves the RGBA gather ≡ the CPU layout)
    // The value is a FLAT INTERLEAVED number[] of length N*3. cell c: [x[3c]+1, x[3c+1]+2, x[3c+2]+3].
    const out = settled.value as number[] | null;
    expect(out?.length).toBe(N * 3);
    // cell 0 = [0+1, 1+2, 2+3] = [1,3,5]; cell 1 = [3+1,4+2,5+3] = [4,6,8]; cell 5 = [15+1,16+2,17+3]=[16,18,20].
    for (const k of [0, 1, 2]) expect(out?.[k]).toBeCloseTo([1, 3, 5][k]!, 3);
    for (const k of [0, 1, 2]) expect(out?.[3 + k]).toBeCloseTo([4, 6, 8][k]!, 3);
    for (const k of [0, 1, 2]) expect(out?.[15 + k]).toBeCloseTo([16, 18, 20][k]!, 3);
  });

  it('a vec2 output on WebGL2 packs 2 channels per cell (comps=2 interleave)', async () => {
    const host = new RuntimeReactiveHost();
    const N = 32;
    // cell i → vec2(i, i*10). Flat: [0,0, 1,10, 2,20, ...].
    const kernel = kernelOf(`
      const N = ${N}
      const x = f32(N, (i) => i)
      component k(i) { return vec2(x[i], x[i] * 10) }
      k`, host);
    const engine = new GpuEngine(host, deps);
    const cfg = { output: [N], backend: 'webgl2' as const, outputElement: 'vec2' as const, verify: true };
    change(() => { engine.gpu(kernel, cfg); });
    await new Promise((r) => setTimeout(r, 300));
    let settled!: ReturnType<GpuEngine['gpu']>;
    change(() => { settled = engine.gpu(kernel, cfg); });
    expect(settled.pending).toBe(false);
    expect(settled.backend).toBe(webgl2Live ? 'webgl2' : 'cpu');
    expect(settled.match?.ok).toBe(true);
    const out = settled.value as number[] | null;
    expect(out?.length).toBe(N * 2);
    expect(out?.[0]).toBeCloseTo(0, 3); expect(out?.[1]).toBeCloseTo(0, 3);      // cell 0 → (0, 0)
    expect(out?.[6]).toBeCloseTo(3, 3); expect(out?.[7]).toBeCloseTo(30, 3);     // cell 3 → (3, 30)
  });
});
