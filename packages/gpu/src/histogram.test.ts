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

  it('DROPS (does not clamp) the exact boundary bin === bins and every out-of-range input', () => {
    const host = new RuntimeReactiveHost();
    const idBin = mapperOf(`component b(x) { return x }\nb`, host);
    // The upper boundary is EXCLUSIVE: bin === bins (3) is out of range and dropped, NOT clamped to bin 2.
    // A clamp would have counted 3 and 4 into bin 2 → [0,0,3]; the drop yields exactly one count in bin 2.
    expect(cpuHistogram(idBin, [3, 4, 2], 3, host)).toEqual([0, 0, 1]);
    // ALL inputs out of range (all negative, all >= bins) → all-zero counts, no clamp to the end bins.
    expect(cpuHistogram(idBin, [-1, -2, 3, 4, 100], 3, host)).toEqual([0, 0, 0]);
    // A truncating map: 2.9 → trunc → bin 2 (in range, counted); -0.5 → trunc → 0 (bin 0, counted);
    // 3.0 → bin 3 (dropped). Confirms Math.trunc-toward-zero + the exclusive upper bound together.
    const idTrunc = mapperOf(`component b(x) { return x }\nb`, host);
    expect(cpuHistogram(idTrunc, [2.9, -0.5, 3.0], 3, host)).toEqual([1, 0, 1]);
  });
});

// The bin-mapper GATE is strictly NARROWER than the shared map-kernel gate: it ADDS a purity rule (no
// closed-over buffer/vec-mat/helper) and a scalar-index rule (the one param `x` is a scalar → `x[...]` is
// meaningless). These drive the mapper-specific reject paths (the shared gateKernel machinery is covered by
// gate.test.ts). Assertions pin the exact MLGPU code + message + core === false so a regression that drops a
// reason (or the whole check) fails loudly.
describe('gateBinMapper — purity: a closed-over buffer / vec-mat / helper is rejected (strictly narrower than the map gate)', () => {
  it('rejects a bin-mapper that closes over an input BUFFER (buf[x]) — MLGPU-NOT-LOWERABLE, not pure over x', () => {
    const host = new RuntimeReactiveHost();
    // `buf` is a closed-over f32 buffer; a map kernel could index it, but a bin-mapper's only input is x.
    const binOf = mapperOf(`const buf = f32(4, (i) => i)\ncomponent binOf(x) { return buf[x] }\nbinOf`, host);
    const v = gateBinMapper(binOf, host);
    expect(v.core).toBe(false);
    const r = v.reasons.find((d) => d.code === 'MLGPU-NOT-LOWERABLE' && /may not reference a buffer/.test(d.message));
    expect(r, JSON.stringify(v.reasons)).toBeTruthy();
    expect(r!.message).toContain(`buffer ('buf')`);
  });
  it("rejects a bin-mapper that closes over a VEC/MAT uniform (v.x) — MLGPU-NOT-LOWERABLE 'vec/mat'", () => {
    const host = new RuntimeReactiveHost();
    const binOf = mapperOf(`const v = vec3(1, 2, 3)\ncomponent binOf(x) { return x + v.x }\nbinOf`, host);
    const v = gateBinMapper(binOf, host);
    expect(v.core).toBe(false);
    const r = v.reasons.find((d) => d.code === 'MLGPU-NOT-LOWERABLE' && /may not reference a vec\/mat/.test(d.message));
    expect(r, JSON.stringify(v.reasons)).toBeTruthy();
    expect(r!.message).toContain(`vec/mat ('v')`);
  });
  it("rejects a bin-mapper that calls a HELPER function — MLGPU-NOT-LOWERABLE 'helper function'", () => {
    const host = new RuntimeReactiveHost();
    const binOf = mapperOf(`component helper(y) { return y * 2 }\ncomponent binOf(x) { return helper(x) }\nbinOf`, host);
    const v = gateBinMapper(binOf, host);
    expect(v.core).toBe(false);
    const r = v.reasons.find((d) => d.code === 'MLGPU-NOT-LOWERABLE' && /may not reference a helper function/.test(d.message));
    expect(r, JSON.stringify(v.reasons)).toBeTruthy();
    expect(r!.message).toContain(`helper function ('helper')`);
  });
});

// The scalar-index rule: the bin-mapper's one param `x` is a SCALAR, so any `x[...]` in the body is
// meaningless (harmless on the CPU — the interpreter returns null → 0 — but a GPU compile error). The
// detection is a FILE-LOCAL structural walk (indexesAnyParam) reached ONLY through gateBinMapper; each case
// below embeds `x[...]` inside a distinct AST construct so every branch of that walk is visited and the
// 'a bin-mapper parameter is a scalar and cannot be indexed' reason fires.
describe('gateBinMapper — the scalar param x cannot be indexed (drives the indexesAnyParam walk)', () => {
  const indexScalar = (src: string, host: RuntimeReactiveHost) => {
    const binOf = mapperOf(src, host);
    const v = gateBinMapper(binOf, host);
    const r = v.reasons.find((d) => d.code === 'MLGPU-NOT-LOWERABLE' && /scalar and cannot be indexed/.test(d.message));
    return { v, r };
  };
  // One body per construct — the `x[...]` sits inside the named AST node so the walk MUST descend through it.
  const cases: [string, string][] = [
    // ── expr-walk branches ──
    ['index (bare x[...] in a return)', `component binOf(x) { return x[0] }\nbinOf`],
    ['member (x[0].y)', `component binOf(x) { return x[0].y }\nbinOf`],
    ['unary (-x[0])', `component binOf(x) { return -x[0] }\nbinOf`],
    ['cond/ternary (t ? x[0] : e)', `component binOf(x) { return 1 > 0 ? x[0] : 2 }\nbinOf`],
    ['call (abs(x[0]))', `component binOf(x) { return abs(x[0]) }\nbinOf`],
    ['object literal ({ a: x[0] })', `component binOf(x) { const o = { a: x[0] }\nreturn 0 }\nbinOf`],
    ['array literal ([x[0]])', `component binOf(x) { const o = [x[0]]\nreturn 0 }\nbinOf`],
    // ── stmt-walk branches ──
    ['const/let init (const o = x[0])', `component binOf(x) { const o = x[0]\nreturn o }\nbinOf`],
    ['assign VALUE (y = x[0])', `component binOf(x) { let y = 0\ny = x[0]\nreturn y }\nbinOf`],
    // assign TARGET arm: `x[0]` on the LHS of an assignment (the `inExpr(s.target)` arm of the walk).
    ['assign TARGET (x[0] = 5)', `component binOf(x) { x[0] = 5\nreturn 0 }\nbinOf`],
    ['expr statement (bare x[0])', `component binOf(x) { x[0]\nreturn 0 }\nbinOf`],
    ['if test (if (x[0] > 0) ...)', `component binOf(x) { if (x[0] > 0) { return 1 }\nreturn 0 }\nbinOf`],
    ['for body (for (i of range(2)) return x[0])', `component binOf(x) { for (i of range(2)) { return x[0] }\nreturn 0 }\nbinOf`],
    ['while test (while (x[0] > 0) ...)', `component binOf(x) { while (x[0] > 0) { return 0 }\nreturn 0 }\nbinOf`],
    // while BODY arm (test has no index): `x[0]` only in the loop body (the `s.body.some(inStmt)` arm).
    ['while body (while (0 > 1) return x[0])', `component binOf(x) { while (0 > 1) { return x[0] }\nreturn 0 }\nbinOf`],
    // value-less `return` first (the `: false` arm of the return case), then a later `return x[0]` trips it.
    ['bare return then return x[0]', `component binOf(x) { if (x > 9) { return }\nreturn x[0] }\nbinOf`],
    // call: INDIRECT callee carrying the index (`(x[0])()` — the `e.callee.kind !== 'ident'` arm).
    ['call indirect callee (foo[x[0]]())', `component binOf(x) { return foo[x[0]]() }\nbinOf`],
    // call: WRAPPING-block carrying the index (the `e.block?.some(inStmt)` arm of the call case).
    ['call wrapping block (foo() { return x[0] })', `component binOf(x) { foo() { return x[0] }\nreturn 0 }\nbinOf`],
    // stmt-walk DEFAULT branch: a nested function statement (unhandled kind → falls through) is visited FIRST,
    // returns false, and the walk continues to the later `return x[0]` that trips the scalar-index reason.
    ['stmt default (nested function then return x[0])', `component binOf(x) { function h() { return 0 }\nreturn x[0] }\nbinOf`],
  ];
  for (const [tag, src] of cases) {
    it(`rejects: ${tag}`, () => {
      const { v, r } = indexScalar(src, new RuntimeReactiveHost());
      expect(v.core, `expected core=false for ${tag}: ${JSON.stringify(v.reasons)}`).toBe(false);
      expect(r, `expected the scalar-index reason for ${tag}: ${JSON.stringify(v.reasons)}`).toBeTruthy();
      expect(r!.message).toBe('a bin-mapper parameter is a scalar and cannot be indexed');
    });
  }

  it('a scalar-only body (no x[...]) does NOT trip the scalar-index reason (the walk returns false)', () => {
    // Negative control: the walk descends member/binary/cond/call/object/array/const/if without an `x[...]`
    // anywhere → indexesAnyParam is false → no scalar-index reason, and this pure arithmetic mapper is core.
    const host = new RuntimeReactiveHost();
    const binOf = mapperOf(`component binOf(x) { const k = 1 > 0 ? abs(x) : x % 2\nif (k > 0) { return k }\nreturn 0 }\nbinOf`, host);
    const v = gateBinMapper(binOf, host);
    expect(v.reasons.some((d) => /scalar and cannot be indexed/.test(d.message))).toBe(false);
    expect(v.core).toBe(true);
  });
});
