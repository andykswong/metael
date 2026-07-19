import { describe, it, expect } from 'vitest';
import { RuntimeReactiveHost, change } from '@metael/runtime';
import { evaluateProgram, isUserFn, RecordingHostEnv } from '@metael/lang';
import type { UserFn } from '@metael/lang';
import { GpuEngine } from './resource.ts';
import { cpuReduce, gateReducer } from './reduce.ts';
import { emitReduceWgsl } from './emit-wgsl.ts';
import { checkReduceMatch } from './oracle.ts';

function reducerOf(src: string, host: RuntimeReactiveHost): UserFn {
  const res = evaluateProgram(src, { host, env: new RecordingHostEnv() });
  if (!isUserFn(res.value)) throw new Error('reducer'); return res.value;
}
const cpuDeps = { tryWebGpu: async () => null, tryWebGl2: () => null, limitsHint: { maxStorageBufferBindingSize: 1 << 28, maxComputeWorkgroupsPerDimension: 65535 } };

describe('reduction kernel kind (CPU fold — the oracle floor)', () => {
  it('sums a buffer via a binary associative reducer', async () => {
    const host = new RuntimeReactiveHost();
    const reducer = reducerOf(`component add(acc, x) { return acc + x }\nadd`, host);
    const xs = evaluateProgram(`f32(100, (i) => i + 1)`, { host, env: new RecordingHostEnv() }).value as object;
    const engine = new GpuEngine(host, cpuDeps);
    const cfg = { input: xs, identity: 0, backend: 'cpu' as const };
    change(() => { engine.gpuReduce(reducer, cfg); });
    await new Promise((r) => setTimeout(r, 20));
    let s!: ReturnType<GpuEngine['gpuReduce']>;
    change(() => { s = engine.gpuReduce(reducer, cfg); });
    expect(s.value).toBe(5050);   // sum(1..100)
  });
  it('folds a max via a comparison reducer + an identity', async () => {
    const host = new RuntimeReactiveHost();
    const reducer = reducerOf(`component mx(a, b) { return a > b ? a : b }\nmx`, host);
    const xs = evaluateProgram(`f32([3, 9, 2, 7, 5])`, { host, env: new RecordingHostEnv() }).value as object;
    const engine = new GpuEngine(host, cpuDeps);
    const cfg = { input: xs, identity: -1e30, backend: 'cpu' as const };
    change(() => { engine.gpuReduce(reducer, cfg); });
    await new Promise((r) => setTimeout(r, 20));
    let s!: ReturnType<GpuEngine['gpuReduce']>;
    change(() => { s = engine.gpuReduce(reducer, cfg); });
    expect(s.value).toBe(9);
  });
  it('rejects a reduce whose first-pass grid exceeds the device dispatch limit (MLGPU-ALLOC, not a silent mis-dispatch)', async () => {
    const host = new RuntimeReactiveHost();
    const reducer = reducerOf(`component add(acc, x) { return acc + x }\nadd`, host);
    // 4096 elements, ceil(4096/256)=16 first-pass groups; set the limit to 8 so 16 > 8 → reject up front.
    const xs = evaluateProgram(`f32(4096, (i) => i)`, { host, env: new RecordingHostEnv() }).value as object;
    const tinyLimitDeps = { tryWebGpu: async () => null, tryWebGl2: () => null, limitsHint: { maxStorageBufferBindingSize: 1 << 28, maxComputeWorkgroupsPerDimension: 8 } };
    const engine = new GpuEngine(host, tinyLimitDeps);
    let s!: ReturnType<GpuEngine['gpuReduce']>;
    change(() => { s = engine.gpuReduce(reducer, { input: xs, identity: 0, backend: 'webgpu' }); });
    // Not pending, an MLGPU-ALLOC reason/error present (rejected at the gate, never dispatched).
    expect(s.pending).toBe(false);
    const codes = [...(s.reasons ?? []).map((r) => r.code), s.error?.code].filter(Boolean);
    expect(codes).toContain('MLGPU-ALLOC');
  });
  it('ACCEPTS a normal-sized reduce (first-pass grid within the limit)', async () => {
    const host = new RuntimeReactiveHost();
    const reducer = reducerOf(`component add(acc, x) { return acc + x }\nadd`, host);
    const xs = evaluateProgram(`f32(1000, (i) => i + 1)`, { host, env: new RecordingHostEnv() }).value as object;
    const engine = new GpuEngine(host, cpuDeps);   // default maxWG 65535 → ceil(1000/256)=4 groups « limit
    change(() => { engine.gpuReduce(reducer, { input: xs, identity: 0, backend: 'cpu' }); });
    await new Promise((r) => setTimeout(r, 20));
    let s!: ReturnType<GpuEngine['gpuReduce']>;
    change(() => { s = engine.gpuReduce(reducer, { input: xs, identity: 0, backend: 'cpu' }); });
    expect(s.core).toBe(true);
    expect(s.value).toBe(500500);   // sum(1..1000)
  });
  it('rejects scan:true loudly (prefix-scan is unbuilt — do not silently return the scalar fold)', async () => {
    const host = new RuntimeReactiveHost();
    const reducer = reducerOf(`component add(acc, x) { return acc + x }\nadd`, host);
    const xs = evaluateProgram(`f32([1, 2, 3])`, { host, env: new RecordingHostEnv() }).value as object;
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpuReduce']>;
    change(() => { s = engine.gpuReduce(reducer, { input: xs, identity: 0, backend: 'cpu', scan: true }); });
    expect(s.core).toBe(false);
    expect(s.pending).toBe(false);
    expect(s.reasons.some((r) => r.code === 'MLGPU-NOT-LOWERABLE' && /scan/i.test(r.message))).toBe(true);
  });
  it('rejects a reducer with the wrong arity (not exactly 2 params)', async () => {
    const host = new RuntimeReactiveHost();
    const reducer = reducerOf(`component bad(a) { return a }\nbad`, host);
    const xs = evaluateProgram(`f32([1, 2, 3])`, { host, env: new RecordingHostEnv() }).value as object;
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpuReduce']>;
    change(() => { s = engine.gpuReduce(reducer, { input: xs, identity: 0, backend: 'cpu' }); });
    expect(s.core).toBe(false);
    expect(s.reasons.length).toBeGreaterThan(0);
  });
  it('rejects a reducer that indexes a scalar parameter (acc[x]) — a scalar cannot be indexed', async () => {
    const host = new RuntimeReactiveHost();
    const reducer = reducerOf(`component r(acc, x) { return acc[x] }\nr`, host);
    const xs = evaluateProgram(`f32([1, 2, 3])`, { host, env: new RecordingHostEnv() }).value as object;
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpuReduce']>;
    change(() => { s = engine.gpuReduce(reducer, { input: xs, identity: 0, backend: 'cpu' }); });
    expect(s.core).toBe(false);
    expect(s.reasons.some((r) => r.code === 'MLGPU-NOT-LOWERABLE')).toBe(true);
  });
  it('a vec input to gpuReduce settles a LOCAL MLGPU-BAD-INPUT, not a tree-collapsing throw', async () => {
    const host = new RuntimeReactiveHost();
    const reducer = reducerOf(`component add(acc, x) { return acc + x }\nadd`, host);
    const vecInput = evaluateProgram(`vec3(1, 2, 3)`, { host, env: new RecordingHostEnv() }).value as object;
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpuReduce']>;
    expect(() => { change(() => { s = engine.gpuReduce(reducer, { input: vecInput, identity: 0, backend: 'cpu' }); }); }).not.toThrow();
    expect(s.core).toBe(false);
    expect(s.pending).toBe(false);
    const codes = [...(s.reasons ?? []).map((r) => r.code), s.error?.code].filter(Boolean);
    expect(codes).toContain('MLGPU-BAD-INPUT');
  });
  it('a plain non-buffer input (number / array / null) still settles MLGPU-BAD-INPUT cleanly (no throw)', async () => {
    const host = new RuntimeReactiveHost();
    const reducer = reducerOf(`component add(acc, x) { return acc + x }\nadd`, host);
    const engine = new GpuEngine(host, cpuDeps);
    for (const bad of [42 as unknown, [1, 2, 3] as unknown, null as unknown]) {
      let s!: ReturnType<GpuEngine['gpuReduce']>;
      expect(() => { change(() => { s = engine.gpuReduce(reducer, { input: bad, identity: 0, backend: 'cpu' }); }); }).not.toThrow();
      expect(s.core).toBe(false);
      expect(s.pending).toBe(false);
      const codes = [...(s.reasons ?? []).map((r) => r.code), s.error?.code].filter(Boolean);
      expect(codes).toContain('MLGPU-BAD-INPUT');
    }
  });
  it('accepts ±Infinity as a reduction identity (the true neutral for max/min) and keys them distinctly', async () => {
    const host = new RuntimeReactiveHost();
    const mx = reducerOf(`component mx(a, b) { return a > b ? a : b }\nmx`, host);
    const xs = evaluateProgram(`f32([-5, -9, -2])`, { host, env: new RecordingHostEnv() }).value as object;
    const engine = new GpuEngine(host, cpuDeps);
    const cfg = { input: xs, identity: -Infinity, backend: 'cpu' as const };
    change(() => { engine.gpuReduce(mx, cfg); });
    await new Promise((r) => setTimeout(r, 20));
    let s!: ReturnType<GpuEngine['gpuReduce']>;
    change(() => { s = engine.gpuReduce(mx, cfg); });
    expect(s.core).toBe(true);
    expect(s.value).toBe(-2);   // max of the negatives — -Infinity is the true neutral (no -1e30 leak)
  });
  it('still rejects NaN as an identity', async () => {
    const host = new RuntimeReactiveHost();
    const reducer = reducerOf(`component add(acc, x) { return acc + x }\nadd`, host);
    const xs = evaluateProgram(`f32([1, 2, 3])`, { host, env: new RecordingHostEnv() }).value as object;
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpuReduce']>;
    change(() => { s = engine.gpuReduce(reducer, { input: xs, identity: NaN, backend: 'cpu' }); });
    expect(s.core).toBe(false);
    const codes = [...(s.reasons ?? []).map((r) => r.code), s.error?.code].filter(Boolean);
    expect(codes).toContain('MLGPU-BAD-INPUT');
  });
});

// Two DISTINCT same-length buffers must NOT collide in the dispatch memo. A fresh buffer reads generation 0
// and the buffer CONTENTS are not in kernelHash, so before the content-fingerprint fix the second reduce
// returned the first's cached scalar (a silent stale result). A content fingerprint discriminates them AND
// stays convergent (a rebuilt identical-content buffer hashes the same → a memo hit → the pipeline fixes).
describe('reduction memo — content fingerprint (no same-length collision, still converges)', () => {
  it('two distinct same-length buffers do NOT collide in the memo (no stale result)', async () => {
    const host = new RuntimeReactiveHost();
    const add = reducerOf(`component add(acc, x) { return acc + x }\nadd`, host);
    const a = evaluateProgram(`f32([1, 2, 3])`, { host, env: new RecordingHostEnv() }).value as object;
    const b = evaluateProgram(`f32([10, 20, 30])`, { host, env: new RecordingHostEnv() }).value as object;
    const engine = new GpuEngine(host, cpuDeps);
    change(() => { engine.gpuReduce(add, { input: a, identity: 0, backend: 'cpu' }); });
    change(() => { engine.gpuReduce(add, { input: b, identity: 0, backend: 'cpu' }); });
    await new Promise((r) => setTimeout(r, 30));
    let sa!: ReturnType<GpuEngine['gpuReduce']>; let sb!: ReturnType<GpuEngine['gpuReduce']>;
    change(() => { sa = engine.gpuReduce(add, { input: a, identity: 0, backend: 'cpu' }); });
    change(() => { sb = engine.gpuReduce(add, { input: b, identity: 0, backend: 'cpu' }); });
    expect(sa.value).toBe(6);
    expect(sb.value).toBe(60);   // was 6 pre-fix (the collision)
  });
  it('the SAME buffer re-read hits the memo (no redundant re-dispatch — convergence)', async () => {
    const host = new RuntimeReactiveHost();
    const add = reducerOf(`component add(acc, x) { return acc + x }\nadd`, host);
    const a = evaluateProgram(`f32([1, 2, 3])`, { host, env: new RecordingHostEnv() }).value as object;
    const engine = new GpuEngine(host, cpuDeps);
    change(() => { engine.gpuReduce(add, { input: a, identity: 0, backend: 'cpu' }); });
    await new Promise((r) => setTimeout(r, 20));
    let s1!: ReturnType<GpuEngine['gpuReduce']>; let s2!: ReturnType<GpuEngine['gpuReduce']>;
    change(() => { s1 = engine.gpuReduce(add, { input: a, identity: 0, backend: 'cpu' }); });
    change(() => { s2 = engine.gpuReduce(add, { input: a, identity: 0, backend: 'cpu' }); });
    expect(s1).toBe(s2);   // the SAME settled resource object (memo hit) — content fingerprint is stable for the same buffer
    expect(s1.value).toBe(6);
  });
  it('a rebuilt buffer with IDENTICAL content hits the memo (fingerprint is content-deterministic → pipeline converges)', async () => {
    const host = new RuntimeReactiveHost();
    const add = reducerOf(`component add(acc, x) { return acc + x }\nadd`, host);
    const engine = new GpuEngine(host, cpuDeps);
    const a1 = evaluateProgram(`f32([1, 2, 3])`, { host, env: new RecordingHostEnv() }).value as object;
    change(() => { engine.gpuReduce(add, { input: a1, identity: 0, backend: 'cpu' }); });
    await new Promise((r) => setTimeout(r, 20));
    // a DISTINCT object with the SAME content → must be a memo hit (same fingerprint), NOT a re-dispatch.
    const a2 = evaluateProgram(`f32([1, 2, 3])`, { host, env: new RecordingHostEnv() }).value as object;
    let s!: ReturnType<GpuEngine['gpuReduce']>;
    change(() => { s = engine.gpuReduce(add, { input: a2, identity: 0, backend: 'cpu' }); });
    expect(s.value).toBe(6);   // hits the memo, correct value (no infinite loop, no stale)
  });
});

// The WGSL workgroup-shared tree reduction emit — a STRUCTURAL snapshot (there is no WebGPU adapter here, so
// the reduction VALUE path is not runtime-tested; a real-device getCompilationInfo() compile gate lives in
// webgpu.browser.test.ts and skips absent an adapter). These lock the shape a garbage/OOB fold or a missing
// barrier would break — defects a compile gate cannot catch.
describe('WGSL reduce emitter (workgroup-shared tree reduction)', () => {
  it('emits a workgroup-shared tree reduction (var<workgroup> + workgroupBarrier + the reducer op)', () => {
    const host = new RuntimeReactiveHost();
    const reducer = reducerOf(`component add(acc, x) { return acc + x }\nadd`, host);
    const { bindings } = gateReducer(reducer, host);
    const wgsl = emitReduceWgsl(reducer, bindings, 0);
    expect(wgsl).toContain('var<workgroup>');
    expect(wgsl).toContain('workgroupBarrier()');
    expect(wgsl).toContain('@compute');
    expect(wgsl).toContain('_reduce');                 // the reducer fn
    expect(wgsl).toMatch(/acc\s*\+\s*x|_scratch/);     // the fold applies the reducer op
  });

  it('loads out-of-range lanes with the IDENTITY (never a garbage/OOB read) via select', () => {
    // The partial-workgroup guard: a lane past `inLen` must hold the fold-neutral identity, the WGSL analogue
    // of the WebGL2 `if (idx < _inLen)` guard. A garbage fold would be a VALUE bug a compile gate can't catch.
    const host = new RuntimeReactiveHost();
    const reducer = reducerOf(`component add(acc, x) { return acc + x }\nadd`, host);
    const { bindings } = gateReducer(reducer, host);
    const wgsl = emitReduceWgsl(reducer, bindings, 0);
    // scratch[lane] = select(identity, _in[i], i < inLen) — identity when out of range.
    expect(wgsl).toMatch(/_scratch\[[^\]]+\]\s*=\s*select\(\s*_p\.identity\s*,\s*_in\[[^\]]+\]\s*,\s*[^)]*<\s*_p\.inLen\s*\)/);
    // A barrier AFTER the load AND once per fold step: at least two workgroupBarrier() calls.
    expect([...wgsl.matchAll(/workgroupBarrier\(\)/g)].length).toBeGreaterThanOrEqual(2);
    // Thread 0 writes this workgroup's partial to _out[wid].
    expect(wgsl).toMatch(/if\s*\(\s*lid\.x\s*==\s*0u\s*\)\s*\{\s*_out\[wid\.x\]\s*=\s*_scratch\[0\]/);
  });

  it('bakes the workgroup size G as a const (@workgroup_size(256) + array<f32, 256u> + initial stride)', () => {
    const host = new RuntimeReactiveHost();
    const reducer = reducerOf(`component add(acc, x) { return acc + x }\nadd`, host);
    const { bindings } = gateReducer(reducer, host);
    const wgsl = emitReduceWgsl(reducer, bindings, 0);
    expect(wgsl).toContain('@workgroup_size(256)');
    expect(wgsl).toContain('array<f32, 256u>');
    expect(wgsl).toContain('128u');   // initial stride = G/2
  });

  it('the fold-step workgroupBarrier is at loop scope, OUTSIDE the divergent fold-if (uniformity — a barrier in a divergent branch is UB/compile-error on a real device)', () => {
    const host = new RuntimeReactiveHost();
    const reducer = reducerOf(`component add(acc, x) { return acc + x }\nadd`, host);
    const { bindings } = gateReducer(reducer, host);
    const wgsl = emitReduceWgsl(reducer, bindings, 0);
    // The fold writes scratch inside `if (lid.x < _stride) { ... }`; the barrier must come AFTER that block
    // closes, not inside it. Find the fold-if (`< _stride)`), its opening then closing brace, then the barrier
    // — the barrier's index must be AFTER the if-close index (loop scope → reached uniformly by every lane).
    // This assertion FAILS if the barrier moves inside the if (its index would then precede the close brace).
    const foldIdx = wgsl.indexOf('_stride)');           // the `if (lid.x < _stride)` line
    const barrierAfterFold = wgsl.indexOf('workgroupBarrier()', foldIdx);
    const ifCloseAfterFold = wgsl.indexOf('}', wgsl.indexOf('{', foldIdx));   // the fold-if's closing brace
    expect(ifCloseAfterFold).toBeGreaterThan(foldIdx);
    expect(barrierAfterFold).toBeGreaterThan(ifCloseAfterFold);   // barrier is AFTER the if closes → loop scope, uniform
  });

  it('reuses emitExpr for the reducer body — a max reducer lowers its ternary (select), not a hand-rolled op', () => {
    const host = new RuntimeReactiveHost();
    const mx = reducerOf(`component mx(a, b) { return a > b ? a : b }\nmx`, host);
    const { bindings } = gateReducer(mx, host);
    const wgsl = emitReduceWgsl(mx, bindings, -1e30);
    expect(wgsl).toContain('_reduce');
    expect(wgsl).toContain('select(');   // a>b?a:b lowers to select(b, a, (a>b)) — the tested emitExpr path
  });

  it('declares a scalar-constant reducer uniform in _RParams (namespaced _u_<name>)', () => {
    // A reducer may close over a scalar CONSTANT (role:'scalar') — a uniform of the fold. It must ride the
    // _RParams block as `_u_<name>` (the emitExpr scalar lowering), mirroring the map emitter + reduce GLSL.
    const host = new RuntimeReactiveHost();
    const reducer = reducerOf(`const bias = 2\ncomponent add(acc, x) { return acc + x + bias }\nadd`, host);
    const { bindings } = gateReducer(reducer, host);
    const wgsl = emitReduceWgsl(reducer, bindings, 0);
    expect(wgsl).toContain('_u_bias');
    expect(wgsl).toContain('_p._u_bias');   // read through the params block
  });
});

// The identity a caller passes MUST be the reducer's NEUTRAL element (see cpuReduce / ReduceConfig.identity):
// the CPU linear fold applies it ONCE, the GPU tree fold re-seeds it into every tile on every pass. Neutrality
// isn't statically decidable (a property of the reducer relative to the seed), so no gate can flag it — it is
// a documented contract that verify:true catches. These tests LOCK that the linear oracle is correct for a
// neutral identity and DEMONSTRATE that a non-neutral identity is a NON-silent divergence under the oracle.
// gateReducer adds two reducer-specific rules ON TOP of the shared map-kernel gate: PURITY (a reducer's only
// inputs are its two scalar params — a closed-over buffer / vec-mat uniform / helper callee has no place in a
// binary fold) and NO SCALAR INDEX (indexing a scalar param `acc[x]` is meaningless). These lock the exact
// MLGPU-NOT-LOWERABLE reason (code + the noun / phrase) each rule pushes — the narrowing the reducer gate does
// that the map-kernel gate does not.
describe('gateReducer — purity over the two scalar params (no closed-over buffer / vec-mat / helper)', () => {
  const pure = (r: ReturnType<typeof gateReducer>) =>
    r.reasons.find((d) => d.code === 'MLGPU-NOT-LOWERABLE' && /must be pure over its two parameters/.test(d.message));
  it('rejects a reducer that reads a closed-over BUFFER (acc + buf[x]) — the noun is "buffer"', () => {
    const host = new RuntimeReactiveHost();
    const reducer = reducerOf(`const buf = f32([1, 2, 3])\ncomponent r(acc, x) { return acc + buf[x] }\nr`, host);
    const v = gateReducer(reducer, host);
    expect(v.core).toBe(false);
    const reason = pure(v);
    expect(reason).toBeDefined();
    expect(reason!.message).toContain("a buffer");
    expect(reason!.message).toContain("'buf'");
  });
  it('rejects a reducer that reads a closed-over VEC/MAT uniform (acc + u.x) — the noun is "vec/mat"', () => {
    const host = new RuntimeReactiveHost();
    const reducer = reducerOf(`const u = vec3(1, 2, 3)\ncomponent r(acc, x) { return acc + u.x }\nr`, host);
    const v = gateReducer(reducer, host);
    expect(v.core).toBe(false);
    const reason = pure(v);
    expect(reason).toBeDefined();
    expect(reason!.message).toContain("a vec/mat");
    expect(reason!.message).toContain("'u'");
  });
  it('rejects a reducer that CALLS a helper (acc + h(...)) — the noun is "helper function"', () => {
    const host = new RuntimeReactiveHost();
    const reducer = reducerOf(`component h(y) { return y + 1 }\ncomponent r(acc, x) { return h(acc) + x }\nr`, host);
    const v = gateReducer(reducer, host);
    expect(v.core).toBe(false);
    const reason = pure(v);
    expect(reason).toBeDefined();
    expect(reason!.message).toContain("a helper function");
    expect(reason!.message).toContain("'h'");
  });
});

// A reducer's two params are SCALARS, so indexing EITHER of them (`acc[x]` / `x[i]`) is meaningless. The shared
// map-kernel gate accepts a scalar-param index (it only refuses to descend into a BUFFER ident), so gateReducer
// adds `indexesAnyParam`, a structural walk that flags the scalar index REGARDLESS of which expr/stmt construct
// wraps it. These embed `acc[x]` inside every reachable construct of that walk and assert the SAME reason fires
// each time — the point being that the wrapper never hides the scalar index.
describe('gateReducer — a scalar param cannot be indexed, wherever the index is embedded', () => {
  const SCALAR_INDEX = 'a reducer parameter is a scalar and cannot be indexed';
  const flagsScalarIndex = (host: RuntimeReactiveHost, src: string) => {
    const v = gateReducer(reducerOf(src, host), host);
    expect(v.core).toBe(false);
    expect(v.reasons.some((d) => d.code === 'MLGPU-NOT-LOWERABLE' && d.message === SCALAR_INDEX)).toBe(true);
  };
  it('inside a MEMBER chain (acc[x].y)', () => { flagsScalarIndex(new RuntimeReactiveHost(), `component r(acc, x) { return acc[x].y }\nr`); });
  it('inside a UNARY negation (-acc[x])', () => { flagsScalarIndex(new RuntimeReactiveHost(), `component r(acc, x) { return -acc[x] }\nr`); });
  it('inside a BINARY (acc[x] + 1)', () => { flagsScalarIndex(new RuntimeReactiveHost(), `component r(acc, x) { return acc[x] + 1 }\nr`); });
  it('inside a TERNARY test (acc[x] > 0 ? 1 : 2)', () => { flagsScalarIndex(new RuntimeReactiveHost(), `component r(acc, x) { return acc[x] > 0 ? 1 : 2 }\nr`); });
  it('inside a CALL arg (sin(acc[x]))', () => { flagsScalarIndex(new RuntimeReactiveHost(), `component r(acc, x) { return sin(acc[x]) }\nr`); });
  it('inside an OBJECT-literal value ({ a: acc[x] })', () => { flagsScalarIndex(new RuntimeReactiveHost(), `component r(acc, x) { return { a: acc[x] } }\nr`); });
  it('inside an ARRAY-literal element ([acc[x]])', () => { flagsScalarIndex(new RuntimeReactiveHost(), `component r(acc, x) { return [acc[x]] }\nr`); });
  it('inside a CONST init (const y = acc[x])', () => { flagsScalarIndex(new RuntimeReactiveHost(), `component r(acc, x) { const y = acc[x]\n return y }\nr`); });
  it('inside an ASSIGN RHS (y = acc[x])', () => { flagsScalarIndex(new RuntimeReactiveHost(), `component r(acc, x) { let y = 0\n y = acc[x]\n return y }\nr`); });
  it('inside an EXPR statement (acc[x])', () => { flagsScalarIndex(new RuntimeReactiveHost(), `component r(acc, x) { acc[x]\n return acc }\nr`); });
  it('inside an IF test (if (acc[x] > 0))', () => { flagsScalarIndex(new RuntimeReactiveHost(), `component r(acc, x) { if (acc[x] > 0) { return 1 }\n return acc }\nr`); });
  it('inside a FOR body (for (…) { acc[x] })', () => { flagsScalarIndex(new RuntimeReactiveHost(), `component r(acc, x) { for (const i of range(2)) { acc[x] }\n return acc }\nr`); });
  it('inside a WHILE test (while (acc[x] > 0))', () => { flagsScalarIndex(new RuntimeReactiveHost(), `component r(acc, x) { while (acc[x] > 0) { return 1 }\n return acc }\nr`); });
  it('with a nested-decl statement present (the walk skips the decl but still finds the index in the return)', () => {
    // The `function h() {}` statement hits `indexesAnyParam`'s default (non-index) stmt branch and returns
    // false; the subsequent `return acc[x]` still flags the scalar index — the walk does not short-circuit.
    flagsScalarIndex(new RuntimeReactiveHost(), `component r(acc, x) { function h() { return 1 }\n return acc[x] }\nr`);
  });
  // The `indexesAnyParam` walk short-circuits on the FIRST truthy sub-branch, so the following reach the
  // OTHER side of each `||`: the index lives in a sub-position the earlier cases don't exercise.
  it('as an INDIRECT callee (acc[x]()) — the non-ident-callee side of the call branch', () => { flagsScalarIndex(new RuntimeReactiveHost(), `component r(acc, x) { return acc[x]() }\nr`); });
  it('inside a wrapping-block CALL (foo() { acc[x] }) — the call node\'s block side', () => { flagsScalarIndex(new RuntimeReactiveHost(), `component r(acc, x) { return foo() { acc[x] } }\nr`); });
  it('as an ASSIGN TARGET (acc[x] = 1) — the target side of the assign branch (value has no index)', () => { flagsScalarIndex(new RuntimeReactiveHost(), `component r(acc, x) { acc[x] = 1\n return acc }\nr`); });
  it('inside an IF THEN body (if (x>0) { return acc[x] }) — the then-branch stmt walk', () => { flagsScalarIndex(new RuntimeReactiveHost(), `component r(acc, x) { if (x > 0) { return acc[x] }\n return acc }\nr`); });
  it('inside an IF ELSE body (… else { return acc[x] }) — the else-branch stmt walk', () => { flagsScalarIndex(new RuntimeReactiveHost(), `component r(acc, x) { if (x > 0) { return 1 } else { return acc[x] }\n return acc }\nr`); });
  it('inside a WHILE body (while (x>0) { acc[x] }) — the body-stmt walk (test has no index)', () => { flagsScalarIndex(new RuntimeReactiveHost(), `component r(acc, x) { while (x > 0) { acc[x] }\n return acc }\nr`); });
  it('coexists with a value-less RETURN (the return branch\'s false arm) elsewhere in the body', () => {
    // The `return` with no value takes `indexesAnyParam`'s `s.value ? … : false` false arm; the trailing
    // `return acc[x]` still flags the scalar index.
    flagsScalarIndex(new RuntimeReactiveHost(), `component r(acc, x) { if (x > 0) { return }\n return acc[x] }\nr`);
  });
});

// cpuReduce is the correctness ORACLE floor — an exact linear left-fold through the shipped interpreter. The
// neutral-identity block below exercises the sum/max shapes; these add product/min (distinct ops) and the
// Number(null) coercion path: a reducer whose step evaluates to null (a divide-by-zero → the interpreter emits
// a diagnostic and yields null) is coerced by `Number(...)` to 0, mirroring the buffer-write coercion.
describe('cpuReduce — the linear left-fold oracle (product / min / null-coercion)', () => {
  it('folds a PRODUCT seeded by the neutral identity 1', () => {
    const host = new RuntimeReactiveHost();
    const mul = reducerOf(`component mul(a, b) { return a * b }\nmul`, host);
    expect(cpuReduce(mul, [1, 2, 3, 4], 1, host)).toBe(24);
    expect(cpuReduce(mul, [], 1, host)).toBe(1);   // empty fold → the seed itself
  });
  it('folds a MIN seeded by a very-large neutral identity', () => {
    const host = new RuntimeReactiveHost();
    const mn = reducerOf(`component mn(a, b) { return a < b ? a : b }\nmn`, host);
    expect(cpuReduce(mn, [3, 9, 2, 7, 5], 1e30, host)).toBe(2);
    expect(cpuReduce(mn, [-5, -9, -2], Infinity, host)).toBe(-9);
  });
  it('coerces a null fold step to 0 (a /0 → the interpreter yields null → Number(null) === 0)', () => {
    const host = new RuntimeReactiveHost();
    // acc / (x - x) divides by zero → the interpreter records ML-LANG-DIV-ZERO and returns null; cpuReduce's
    // Number(...) then coerces that null to 0 (the buffer-write coercion), so the fold settles at 0.
    const bad = reducerOf(`component bad(acc, x) { return acc / (x - x) }\nbad`, host);
    expect(cpuReduce(bad, [5], 10, host)).toBe(0);
  });
  it('DECLINES a host-dispatched call — cpuReduce runs with a resolveCall-declining env, so an unknown head yields null → 0', () => {
    const host = new RuntimeReactiveHost();
    // `widget(...)` is neither a param nor a lang builtin → the interpreter dispatches it to the environment's
    // resolveCall. cpuReduce supplies a declineEnv whose resolveCall answers { handled: false }, so the call
    // yields null and Number(null) coerces the fold to 0 — the fold never escapes to a host head.
    const r = reducerOf(`component r(acc, x) { return widget(acc, x) }\nr`, host);
    expect(cpuReduce(r, [1, 2, 3], 5, host)).toBe(0);
  });
});

describe('reduction — the neutral-identity contract', () => {
  it('cpuReduce (the oracle) applies a NEUTRAL identity exactly once', () => {
    const host = new RuntimeReactiveHost();
    const add = reducerOf(`component add(acc, x) { return acc + x }\nadd`, host);
    // Linear left-fold seeded by the neutral 0 → the seed contributes nothing: 0 + sum(xs).
    expect(cpuReduce(add, [1, 2, 3, 4, 5], 0, host)).toBe(15);
    expect(cpuReduce(add, Array.from({ length: 100 }, (_, i) => i + 1), 0, host)).toBe(5050);
    // max with a neutral very-negative sentinel identity → folded once, contributes nothing to the max.
    const mx = reducerOf(`component mx(a, b) { return a > b ? a : b }\nmx`, host);
    expect(cpuReduce(mx, [3, 9, 2, 7, 5], -1e30, host)).toBe(9);
  });

  it('is NON-silent for a non-neutral identity — a tree fold diverges and the oracle (verify) flags it', () => {
    const host = new RuntimeReactiveHost();
    const add = reducerOf(`component add(acc, x) { return acc + x }\nadd`, host);
    const xs = [1, 2, 3, 4, 5, 6, 7, 8];
    // A tree fold that re-seeds `identity` into every tile on every pass — exactly what the WebGL2 reduce
    // shader does (`_acc = _identity` per output texel per pass). This models the GPU numeric behavior in JS.
    const treeFold = (data: number[], identity: number, tile: number): number => {
      let cur = data;
      for (;;) {
        const out: number[] = [];
        for (let i = 0; i < cur.length; i += tile) {
          let acc = identity;   // re-seeded per tile per pass
          for (let j = i; j < Math.min(i + tile, cur.length); j++) acc = acc + cur[j]!;
          out.push(acc);
        }
        if (out.length === 1) return out[0]!;
        cur = out;
      }
    };
    // A NEUTRAL identity (0): the tree fold and the linear oracle AGREE — no divergence, verify passes.
    expect(treeFold(xs, 0, 4)).toBe(cpuReduce(add, xs, 0, host));
    expect(checkReduceMatch(treeFold(xs, 0, 4), cpuReduce(add, xs, 0, host)).ok).toBe(true);
    // A NON-neutral identity (5 for +): the linear oracle applies it once (5 + 36 = 41); the tree fold
    // applies it once per tile per pass (15, 31 → 51). The divergence is caught by checkReduceMatch — so with
    // verify:true (resource.ts calls checkReduceMatch(scalar, cpuReduce(...))) the wrong answer is NOT silent.
    const oracle = cpuReduce(add, xs, 5, host);
    expect(oracle).toBe(41);
    const treeValue = treeFold(xs, 5, 4);
    expect(treeValue).toBe(51);
    expect(checkReduceMatch(treeValue, oracle).ok).toBe(false);
  });
});
