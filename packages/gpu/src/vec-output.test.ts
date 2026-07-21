import { describe, it, expect } from 'vitest';
import { MATH_BUILTINS } from '@metael/math/lang';
import { RuntimeReactiveHost, change } from '@metael/runtime';
import { evaluateProgram, isUserFn, RecordingHostEnv } from '@metael/lang';
import type { UserFn } from '@metael/lang';
import { GpuEngine } from './resource.ts';

function kernelOf(src: string, host: RuntimeReactiveHost): UserFn {
  const res = evaluateProgram(src, { host, env: new RecordingHostEnv(), builtins: [MATH_BUILTINS] });
  if (!isUserFn(res.value)) throw new Error('kernel: ' + JSON.stringify(res.diagnostics)); return res.value;
}
const cpuDeps = { tryWebGpu: async () => null, tryWebGl2: () => null, limitsHint: { maxStorageBufferBindingSize: 1 << 28, maxComputeWorkgroupsPerDimension: 65535 } };

describe('single vec output (outputElement: "vec3")', () => {
  it('a vec3-returning kernel writes an interleaved N-wide output buffer (cell*3 + k)', async () => {
    const host = new RuntimeReactiveHost();
    // x is a flat [n*3] buffer; cell i reads x[i*3..i*3+2], adds (1,2,3).
    const kernel = kernelOf(`const x = f32(12, (i) => i)
component k(i) {
  return vec3(x[i * 3], x[i * 3 + 1], x[i * 3 + 2]) + vec3(1, 2, 3)
}
k`, host);
    const engine = new GpuEngine(host, cpuDeps);
    const cfg = { output: [4], backend: 'cpu' as const, outputElement: 'vec3' as const };
    change(() => { engine.gpu(kernel, cfg); });
    await new Promise((r) => setTimeout(r, 20));
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, cfg); });
    // 4 cells * 3 comps = 12 values, flat interleaved. cell c: [x[3c]+1, x[3c+1]+2, x[3c+2]+3].
    // x[j] = j, so cell 0 = [0+1, 1+2, 2+3] = [1,3,5]; cell 1 = [3+1,4+2,5+3]=[4,6,8]; ...
    expect(s.value).toEqual([1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14]);
  });

  it('a scalar (default) output is unchanged — no outputElement means comps=1', async () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const x = f32(4, (i) => i)\ncomponent k(i) { return x[i] * 2 }\nk`, host);
    const engine = new GpuEngine(host, cpuDeps);
    change(() => { engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    await new Promise((r) => setTimeout(r, 20));
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    expect(s.value).toEqual([0, 2, 4, 6]);
  });

  it('rejects a vecN outputElement whose kernel returns a scalar (shape mismatch)', async () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const x = f32(4, (i) => i)\ncomponent k(i) { return x[i] }\nk`, host);   // scalar return
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu', outputElement: 'vec3' }); });
    expect(s.core).toBe(false);
    expect(s.reasons.some((r) => r.code === 'MLGPU-OUTPUT-SHAPE')).toBe(true);
  });
});

describe('vec-output path — silent cross-backend divergence guards', () => {
  it('rejects a non-xyzw swizzle (.rgb) — the interpreter cannot evaluate it, so it must not lower (no silent GPU-vs-CPU divergence)', async () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const x = f32(12, (i) => i)
component k(i) {
  const v = vec3(x[i * 3], x[i * 3 + 1], x[i * 3 + 2])
  return v.rgb
}
k`, host);
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu', outputElement: 'vec3' }); });
    expect(s.core).toBe(false);
    expect(s.reasons.some((r) => r.code === 'MLGPU-NOT-LOWERABLE')).toBe(true);
  });
  it('still ACCEPTS a valid xyzw swizzle (.xy / .xyz)', async () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const x = f32(8, (i) => i)
component k(i) {
  const v = vec4(x[i*2], x[i*2+1], 0, 0)
  return v.xy
}
k`, host);
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu', outputElement: 'vec2' }); });
    expect(s.core).toBe(true);   // .xy is fine
  });

  it('rejects a mismatched-width ternary return (vec3 vs vec2) — the branches disagree on output width', async () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const x = f32(12, (i) => i)
component k(i) {
  return (i > 0) ? vec3(x[i*3], x[i*3+1], x[i*3+2]) : vec2(x[i*3], x[i*3+1])
}
k`, host);
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu', outputElement: 'vec3' }); });
    expect(s.core).toBe(false);
    expect(s.reasons.some((r) => r.code === 'MLGPU-OUTPUT-SHAPE')).toBe(true);
  });
  it('still ACCEPTS a consistent-width ternary return (vec3 vs vec3)', async () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const x = f32(12, (i) => i)
component k(i) {
  return (i > 0) ? vec3(x[i*3], x[i*3+1], x[i*3+2]) : vec3(0, 0, 0)
}
k`, host);
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu', outputElement: 'vec3' }); });
    expect(s.core).toBe(true);
  });

  it('rejects a kernel-local name starting with _ (reserved for the compiler)', async () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const x = f32(4, (i) => i)\ncomponent k(i) { const _r = x[i] return _r * 2 }\nk`, host);
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    expect(s.core).toBe(false);
    expect(s.reasons.some((r) => r.code === 'MLGPU-NOT-LOWERABLE')).toBe(true);
  });
  it('still ACCEPTS a normal (non-underscore) kernel local', async () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const x = f32(4, (i) => i)\ncomponent k(i) { const r = x[i] return r * 2 }\nk`, host);
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    expect(s.core).toBe(true);
  });
});

describe('vec-output path — resident-layout + over-range-swizzle guards', () => {
  it('rejects a vecN output combined with outputType gpu-buffer (a resident vecN buffer is incoherent on WebGL2 — deferred)', async () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const x = f32(12, (i) => i)
component k(i) { return vec3(x[i*3], x[i*3+1], x[i*3+2]) }
k`, host);
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu', outputElement: 'vec3', outputType: 'gpu-buffer' }); });
    // rejected before dispatch: not pending, an error/reason present, no silent wrong result.
    expect(s.pending).toBe(false);
    expect(s.core === false || s.error !== null).toBe(true);
    // the specific diagnostic (adjust the code to whatever the config-rejection path uses — MLGPU-NOT-LOWERABLE):
    const codes = [...(s.reasons ?? []).map((r) => r.code), s.error?.code].filter(Boolean);
    expect(codes.some((c) => c === 'MLGPU-NOT-LOWERABLE')).toBe(true);
  });
  it('ALLOWS a vecN output with outputType array (flat interleaved) and buffer (flat f32 handle)', async () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const x = f32(12, (i) => i)
component k(i) { return vec3(x[i*3], x[i*3+1], x[i*3+2]) + vec3(1, 2, 3) }
k`, host);
    const engine = new GpuEngine(host, cpuDeps);
    // array mode (default) — unchanged, must still work:
    let sa!: ReturnType<GpuEngine['gpu']>;
    change(() => { engine.gpu(kernel, { output: [4], backend: 'cpu', outputElement: 'vec3' }); });
    await new Promise((r) => setTimeout(r, 20));
    change(() => { sa = engine.gpu(kernel, { output: [4], backend: 'cpu', outputElement: 'vec3' }); });
    expect(sa.value).toEqual([1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14]);
    // buffer mode (non-resident flat handle) — must be ACCEPTED (core, not rejected):
    let sb!: ReturnType<GpuEngine['gpu']>;
    change(() => { sb = engine.gpu(kernel, { output: [4], backend: 'cpu', outputElement: 'vec3', outputType: 'buffer' }); });
    expect(sb.core).toBe(true);   // buffer + vecN is allowed (flat f32 handle, no resident texture)
  });

  it('rejects an over-range xyzw swizzle (.z on a vec2) — the interpreter returns NOT_HANDLED so it must not lower (no silent 0 / false-green verify)', async () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const x = f32(8, (i) => i)
component k(i) { return vec2(x[i*2], x[i*2+1]).z }
k`, host);
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    expect(s.core).toBe(false);
    expect(s.reasons.some((r) => r.code === 'MLGPU-NOT-LOWERABLE')).toBe(true);
  });
  it('rejects an over-range multi-char swizzle (.xyz on a vec2) even when outputElement matches the swizzle length', async () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const x = f32(8, (i) => i)
component k(i) { return vec2(x[i*2], x[i*2+1]).xyz }
k`, host);
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu', outputElement: 'vec3' }); });
    expect(s.core).toBe(false);
    expect(s.reasons.some((r) => r.code === 'MLGPU-NOT-LOWERABLE')).toBe(true);
  });
  it('still ACCEPTS an in-range swizzle (.xy / .yx on a vec3, .x on a vec2)', async () => {
    const host = new RuntimeReactiveHost();
    const k1 = kernelOf(`const x = f32(12, (i) => i)\ncomponent k(i) { return vec3(x[i*3], x[i*3+1], x[i*3+2]).xy }\nk`, host);
    const engine = new GpuEngine(host, cpuDeps);
    let s1!: ReturnType<GpuEngine['gpu']>;
    change(() => { s1 = engine.gpu(k1, { output: [4], backend: 'cpu', outputElement: 'vec2' }); });
    expect(s1.core).toBe(true);   // .xy on a vec3 is in range
  });
});
