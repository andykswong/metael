import { describe, it, expect } from 'vitest';
import { RuntimeReactiveHost, change } from '@metael/runtime';
import { evaluateProgram, isUserFn, RecordingHostEnv } from '@metael/lang';
import type { UserFn } from '@metael/lang';
import { GpuEngine } from './resource.ts';
import { gateBinMapper, cpuHistogram } from './histogram.ts';
import { emitHistogramWgsl } from './emit-wgsl.ts';

function mapperOf(src: string, host: RuntimeReactiveHost): UserFn {
  const res = evaluateProgram(src, { host, env: new RecordingHostEnv() });
  if (!isUserFn(res.value)) throw new Error('mapper: ' + JSON.stringify(res.diagnostics)); return res.value;
}
const cpuDeps = { tryWebGpu: async () => null, tryWebGl2: () => null, limitsHint: { maxStorageBufferBindingSize: 1 << 28, maxComputeWorkgroupsPerDimension: 65535 } };

describe('histogram kernel kind (atomic scatter — CPU oracle floor)', () => {
  it('counts input elements into bins via a bin-mapper', async () => {
    const host = new RuntimeReactiveHost();
    // bin = x % 4 → 4 bins; input [0..15] → each bin gets 4 elements.
    const binOf = mapperOf(`component binOf(x) { return x % 4 }\nbinOf`, host);
    const xs = evaluateProgram(`f32(16, (i) => i)`, { host, env: new RecordingHostEnv() }).value as object;
    const engine = new GpuEngine(host, cpuDeps);
    const cfg = { input: xs, bins: 4, backend: 'cpu' as const };
    change(() => { engine.gpuHistogram(binOf, cfg); });
    await new Promise((r) => setTimeout(r, 20));
    let s!: ReturnType<GpuEngine['gpuHistogram']>;
    change(() => { s = engine.gpuHistogram(binOf, cfg); });
    expect(s.value).toEqual([4, 4, 4, 4]);   // 0,4,8,12→bin0; 1,5,9,13→bin1; etc.
  });
  it('drops out-of-range bin indices (not counted, no crash)', async () => {
    const host = new RuntimeReactiveHost();
    const binOf = mapperOf(`component binOf(x) { return x }\nbinOf`, host);   // bin = x directly
    const xs = evaluateProgram(`f32([0, 1, 2, 99, -3, 1])`, { host, env: new RecordingHostEnv() }).value as object;
    const engine = new GpuEngine(host, cpuDeps);
    const cfg = { input: xs, bins: 3, backend: 'cpu' as const };
    change(() => { engine.gpuHistogram(binOf, cfg); });
    await new Promise((r) => setTimeout(r, 20));
    let s!: ReturnType<GpuEngine['gpuHistogram']>;
    change(() => { s = engine.gpuHistogram(binOf, cfg); });
    expect(s.value).toEqual([1, 2, 1]);   // bin0:{0}=1, bin1:{1,1}=2, bin2:{2}=1; 99 & -3 dropped
  });
  it('rejects a bin-mapper with the wrong arity (not exactly 1 param)', async () => {
    const host = new RuntimeReactiveHost();
    const binOf = mapperOf(`component bad(x, y) { return x }\nbad`, host);
    const xs = evaluateProgram(`f32([1, 2, 3])`, { host, env: new RecordingHostEnv() }).value as object;
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpuHistogram']>;
    change(() => { s = engine.gpuHistogram(binOf, { input: xs, bins: 3, backend: 'cpu' }); });
    expect(s.core).toBe(false);
  });
  it('a non-buffer input settles a LOCAL error, not a tree-collapsing throw (carry the Phase-4 fix)', async () => {
    const host = new RuntimeReactiveHost();
    const binOf = mapperOf(`component binOf(x) { return x }\nbinOf`, host);
    const vecInput = evaluateProgram(`vec3(1, 2, 3)`, { host, env: new RecordingHostEnv() }).value as object;
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpuHistogram']>;
    expect(() => { change(() => { s = engine.gpuHistogram(binOf, { input: vecInput, bins: 3, backend: 'cpu' }); }); }).not.toThrow();
    expect(s.core).toBe(false);
    const codes = [...(s.reasons ?? []).map((r) => r.code), s.error?.code].filter(Boolean);
    expect(codes).toContain('MLGPU-BAD-INPUT');
  });
  it('emitHistogramWgsl emits atomic bins + atomicAdd (structural)', () => {
    const host = new RuntimeReactiveHost();
    const binOf = mapperOf(`component binOf(x) { return x % 4 }\nbinOf`, host);
    const { bindings } = gateBinMapper(binOf, host);
    const wgsl = emitHistogramWgsl(binOf, bindings, 4);
    expect(wgsl).toContain('atomic<u32>');
    expect(wgsl).toContain('atomicAdd');
    expect(wgsl).toContain('@compute');
    // Pin the VALUE-CRITICAL scatter bounds guard `_b >= 0 && _b < i32(_p.bins)`: a regression deleting it
    // would still compile + pass the substring checks above, but scatter out-of-range on a real device (an
    // OOB / wrong-bin write the no-adapter test can't observe). Also pin the i32(...) trunc-toward-zero bin
    // cast that matches the CPU oracle's Math.trunc.
    expect(wgsl).toMatch(/_b\s*>=\s*0/);
    expect(wgsl).toMatch(/_b\s*<\s*i32\(_p\.bins\)/);
    expect(wgsl).toContain('i32(');
    // Pin the NON-FINITE bin-index DROP: the CPU oracle drops a NaN bin (Number.isFinite guard), but i32(NaN)
    // in WGSL is INDETERMINATE (often 0) → a NaN bin would be counted as bin 0, diverging from the oracle. The
    // scatter must guard finiteness BEFORE the i32 cast via a `_bf == _bf` self-comparison (false only for NaN).
    // A regression deleting it compiles + passes the bounds checks above but silently mis-counts a NaN bin.
    expect(wgsl).toMatch(/_bf\s*==\s*_bf/);
  });
  it('rejects a histogram whose scatter grid exceeds the device dispatch limit (MLGPU-ALLOC)', async () => {
    const host = new RuntimeReactiveHost();
    const binOf = mapperOf(`component binOf(x) { return x % 4 }\nbinOf`, host);
    // HISTOGRAM_WORKGROUP is 64. 4096 elements → ceil(4096/64)=64 scatter groups; set maxWg=8 → 64>8 → reject.
    const xs = evaluateProgram(`f32(4096, (i) => i)`, { host, env: new RecordingHostEnv() }).value as object;
    const tinyDeps = { tryWebGpu: async () => null, tryWebGl2: () => null, limitsHint: { maxStorageBufferBindingSize: 1 << 28, maxComputeWorkgroupsPerDimension: 8 } };
    const engine = new GpuEngine(host, tinyDeps);
    let s!: ReturnType<GpuEngine['gpuHistogram']>;
    change(() => { s = engine.gpuHistogram(binOf, { input: xs, bins: 4, backend: 'webgpu' }); });
    // Not pending, an MLGPU-ALLOC reason/error present (rejected at the gate, never dispatched).
    expect(s.pending).toBe(false);
    const codes = [...(s.reasons ?? []).map((r) => r.code), s.error?.code].filter(Boolean);
    expect(codes).toContain('MLGPU-ALLOC');
  });
  it('ACCEPTS a normal-sized histogram (scatter grid within the limit)', async () => {
    const host = new RuntimeReactiveHost();
    const binOf = mapperOf(`component binOf(x) { return x % 4 }\nbinOf`, host);
    // 1000 elements → ceil(1000/64)=16 groups « the default 65535 limit → accepted (no over-rejection).
    const xs = evaluateProgram(`f32(1000, (i) => i)`, { host, env: new RecordingHostEnv() }).value as object;
    const engine = new GpuEngine(host, cpuDeps);
    change(() => { engine.gpuHistogram(binOf, { input: xs, bins: 4, backend: 'cpu' }); });
    await new Promise((r) => setTimeout(r, 20));
    let s!: ReturnType<GpuEngine['gpuHistogram']>;
    change(() => { s = engine.gpuHistogram(binOf, { input: xs, bins: 4, backend: 'cpu' }); });
    expect(s.core).toBe(true);
    expect(s.value).toEqual([250, 250, 250, 250]);   // 1000 elements, x%4 → 250 per bin
  });
  it('rejects a bad bins count (0, negative, non-integer) with MLGPU-BAD-INPUT', async () => {
    const host = new RuntimeReactiveHost();
    const binOf = mapperOf(`component binOf(x) { return x }\nbinOf`, host);
    const xs = evaluateProgram(`f32([0, 1, 2])`, { host, env: new RecordingHostEnv() }).value as object;
    const engine = new GpuEngine(host, cpuDeps);
    for (const bins of [0, -3, 1.5]) {
      let s!: ReturnType<GpuEngine['gpuHistogram']>;
      change(() => { s = engine.gpuHistogram(binOf, { input: xs, bins, backend: 'cpu' }); });
      expect(s.core, `bins=${bins}`).toBe(false);
      const codes = [...(s.reasons ?? []).map((r) => r.code), s.error?.code].filter(Boolean);
      expect(codes, `bins=${bins}`).toContain('MLGPU-BAD-INPUT');
    }
  });
  it('an empty input yields all-zero counts of length bins', async () => {
    const host = new RuntimeReactiveHost();
    const binOf = mapperOf(`component binOf(x) { return x % 3 }\nbinOf`, host);
    const xs = evaluateProgram(`f32([])`, { host, env: new RecordingHostEnv() }).value as object;
    const engine = new GpuEngine(host, cpuDeps);
    change(() => { engine.gpuHistogram(binOf, { input: xs, bins: 3, backend: 'cpu' }); });
    await new Promise((r) => setTimeout(r, 20));
    let s!: ReturnType<GpuEngine['gpuHistogram']>;
    change(() => { s = engine.gpuHistogram(binOf, { input: xs, bins: 3, backend: 'cpu' }); });
    expect(s.value).toEqual([0, 0, 0]);
  });
});

// Two DISTINCT same-length buffers must NOT collide in the dispatch memo (parity with reduce/map). A fresh
// buffer reads generation 0 and the contents are not in kernelHash, so before the content-fingerprint fix the
// second histogram returned the first's cached counts (silent stale). The content fingerprint discriminates.
describe('histogram memo — content fingerprint (no same-length collision)', () => {
  it('two distinct same-length buffers do NOT collide (different counts, not stale)', async () => {
    const host = new RuntimeReactiveHost();
    const binOf = mapperOf(`component binOf(x) { return x }\nbinOf`, host);   // bin = x
    // Both length 4. `a` puts all in bins 0..3 once each; `b` puts all in bin 0.
    const a = evaluateProgram(`f32([0, 1, 2, 3])`, { host, env: new RecordingHostEnv() }).value as object;
    const b = evaluateProgram(`f32([0, 0, 0, 0])`, { host, env: new RecordingHostEnv() }).value as object;
    const engine = new GpuEngine(host, cpuDeps);
    change(() => { engine.gpuHistogram(binOf, { input: a, bins: 4, backend: 'cpu' }); });
    change(() => { engine.gpuHistogram(binOf, { input: b, bins: 4, backend: 'cpu' }); });
    await new Promise((r) => setTimeout(r, 30));
    let sa!: ReturnType<GpuEngine['gpuHistogram']>; let sb!: ReturnType<GpuEngine['gpuHistogram']>;
    change(() => { sa = engine.gpuHistogram(binOf, { input: a, bins: 4, backend: 'cpu' }); });
    change(() => { sb = engine.gpuHistogram(binOf, { input: b, bins: 4, backend: 'cpu' }); });
    expect(sa.value).toEqual([1, 1, 1, 1]);
    expect(sb.value).toEqual([4, 0, 0, 0]);   // was [1,1,1,1] pre-fix (the collision)
  });
});

// The exact linear CPU oracle is the load-bearing, node-testable deliverable — a small direct unit check that
// cpuHistogram counts + drops out-of-range independent of the reactive-resource plumbing.
describe('cpuHistogram (the oracle) — exact counts + out-of-range drop', () => {
  it('increments the mapped bin per element; drops <0, >=bins, and non-finite', () => {
    const host = new RuntimeReactiveHost();
    const modBin = mapperOf(`component b(x) { return x % 3 }\nb`, host);
    expect(cpuHistogram(modBin, [0, 1, 2, 3, 4, 5], 3, host)).toEqual([2, 2, 2]);
    const idBin = mapperOf(`component b(x) { return x }\nb`, host);
    // -1 and 5 fall outside [0,3) → dropped; 0,1,2 counted once each.
    expect(cpuHistogram(idBin, [-1, 0, 1, 2, 5], 3, host)).toEqual([1, 1, 1]);
    // NON-FINITE drop: a mapper that overflows to +Infinity (f64) for a non-zero input → Math.trunc(Infinity)
    // is not finite → dropped by the `Number.isFinite(b)` guard; the 0 input maps to bin 0 and is counted.
    const ovfBin = mapperOf(`component b(x) { return x * 1e308 * 10 }\nb`, host);
    expect(cpuHistogram(ovfBin, [0, 1, 2], 3, host)).toEqual([1, 0, 0]);
  });
});
