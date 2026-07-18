import { describe, it, expect } from 'vitest';
import { RuntimeReactiveHost, change } from '@metael/runtime';
import { evaluateProgram, isUserFn, RecordingHostEnv } from '@metael/lang';
import type { UserFn } from '@metael/lang';
import { GpuEngine } from './resource.ts';
import { gateKernel } from './gate.ts';
import { emitWgsl } from './emit-wgsl.ts';
import { packF16, f16ToF32, align4 } from './f16-pack.ts';

function kernelOf(src: string, host: RuntimeReactiveHost): UserFn {
  const res = evaluateProgram(src, { host, env: new RecordingHostEnv() });
  if (!isUserFn(res.value)) throw new Error('kernel'); return res.value;
}
const cpuDeps = { tryWebGpu: async () => null, tryWebGl2: () => null, limitsHint: { maxStorageBufferBindingSize: 1 << 28, maxComputeWorkgroupsPerDimension: 65535 } };

describe('f16 precision', () => {
  it('an f16 request falls back to f32 with a note on a backend without shader-f16 (cpu floor), producing correct values', async () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const x = f32(4, (i) => i)\ncomponent k(i) { return x[i] * 2 }\nk`, host);
    const engine = new GpuEngine(host, cpuDeps);
    const cfg = { output: [4], backend: 'cpu' as const, precision: 'f16' as const };
    change(() => { engine.gpu(kernel, cfg); });
    await new Promise((r) => setTimeout(r, 20));
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, cfg); });
    expect(s.pending).toBe(false);
    expect(s.error).toBeNull();
    expect(s.value).toEqual([0, 2, 4, 6]);       // correct — ran at f32
    expect(s.note).not.toBeNull();               // a fallback note is present
    expect(s.note).toContain('f16');
  });
  it('the f16 WGSL emitter produces self-consistent f16 (enable f16; array<f16>) — structural', () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const x = f32(4, (i) => i)\ncomponent k(i) { return x[i] * 2 }\nk`, host);
    const { bindings } = gateKernel(kernel, host);
    const wgsl = emitWgsl(kernel, bindings, 'f16');
    expect(wgsl).toContain('enable f16;');
    expect(wgsl).toContain('array<f16>');
  });
  it('a default (f32) dispatch has a null note (back-compat)', async () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const x = f32(4, (i) => i)\ncomponent k(i) { return x[i] * 2 }\nk`, host);
    const engine = new GpuEngine(host, cpuDeps);
    change(() => { engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    await new Promise((r) => setTimeout(r, 20));
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    expect(s.note).toBeNull();
  });
  it('an f16 request with a scalar uniform falls back to f32 with a note (the scalar-uniform f16 path is not yet supported)', async () => {
    const host = new RuntimeReactiveHost();
    // `s` is a scalar-uniform (a bare number in the closure) → the f16 uniform-packing wrinkle → fall back.
    const kernel = kernelOf(`const x = f32(4, (i) => i)\nconst s = 3\ncomponent k(i) { return x[i] * s }\nk`, host);
    const engine = new GpuEngine(host, cpuDeps);
    const cfg = { output: [4], backend: 'cpu' as const, precision: 'f16' as const };
    change(() => { engine.gpu(kernel, cfg); });
    await new Promise((r) => setTimeout(r, 20));
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, cfg); });
    expect(s.pending).toBe(false);
    expect(s.error).toBeNull();
    expect(s.value).toEqual([0, 3, 6, 9]);       // correct — ran at f32
    expect(s.note).not.toBeNull();
    expect(s.note).toContain('f16');
  });
  it('a multi-output f16 request falls back to f32 with a note (cpu floor), producing correct named outputs', async () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const x = f32(4, (i) => i)\ncomponent k(i) { return { dbl: x[i] * 2, sq: x[i] * x[i] } }\nk`, host);
    const engine = new GpuEngine(host, cpuDeps);
    const cfg = { output: [4], backend: 'cpu' as const, precision: 'f16' as const, outputs: { dbl: {}, sq: {} } };
    change(() => { engine.gpu(kernel, cfg); });
    await new Promise((r) => setTimeout(r, 20));
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, cfg); });
    expect(s.pending).toBe(false);
    expect(s.error).toBeNull();
    expect(s.outputs).toEqual({ dbl: [0, 2, 4, 6], sq: [0, 1, 4, 9] });
    expect(s.note).not.toBeNull();
    expect(s.note).toContain('f16');
  });
  // The real f16 storage-buffer path (odd input/output element counts) is only reachable on a shader-f16
  // adapter, which this environment lacks — so these lock the 4-byte-alignment INVARIANT of the extracted
  // packing helper (they verify the fix's invariant, not a pre-fix failure; the bug only manifests on a real
  // device where writeBuffer/copyBufferToBuffer reject a 2-mod-4 byte size).
  it('f16 input packing rounds to a 4-byte-aligned (even-length) Uint16Array (WebGPU writeBuffer requires size % 4 === 0)', () => {
    const odd = packF16(Float32Array.from([1, 2, 3]));   // 3 → 4 u16 (8 bytes, %4===0)
    expect(odd.length % 2).toBe(0);
    expect(odd.byteLength % 4).toBe(0);
    // the real values round-trip (via f16ToF32 of the leading 3), the pad slot is 0:
    const back = [f16ToF32(odd[0]!), f16ToF32(odd[1]!), f16ToF32(odd[2]!)];
    expect(back).toEqual([1, 2, 3]);
    expect(odd[3]).toBe(0);   // pad
  });
  it('an even-length f16 input is already aligned (no extra pad)', () => {
    const even = packF16(Float32Array.from([1, 2, 3, 4]));
    expect(even.length).toBe(4);
    expect(even.byteLength % 4).toBe(0);
  });
  it('align4 rounds a byte count up to the next multiple of 4', () => {
    expect([align4(0), align4(1), align4(2), align4(3), align4(4), align4(6)]).toEqual([0, 4, 4, 4, 4, 8]);
  });
});
