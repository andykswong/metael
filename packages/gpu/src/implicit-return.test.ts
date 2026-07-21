import { describe, it, expect } from 'vitest';
import { MATH_BUILTINS } from '@metael/math/lang';
import { RuntimeReactiveHost, change } from '@metael/runtime';
import { evaluateProgram, isUserFn, RecordingHostEnv, parseProgram, stripSpans } from '@metael/lang';
import type { UserFn, Stmt } from '@metael/lang';
import { GpuEngine } from './resource.ts';
import { normalizeImplicitReturn } from './gate.ts';
import { gateReducer } from './reduce.ts';
import { gateBinMapper } from './histogram.ts';
import { emitReduceWgsl, emitHistogramWgsl } from './emit-wgsl.ts';
import { CPU_LIMITS } from './cost.ts';

function kernelOf(src: string, host: RuntimeReactiveHost): UserFn {
  const res = evaluateProgram(src, { host, env: new RecordingHostEnv(), builtins: [MATH_BUILTINS] });
  if (!isUserFn(res.value)) throw new Error('expected kernel');
  return res.value;
}

/** Parse a program and return the first `function`/`component` declaration's body statements. */
function bodyOf(src: string): Stmt[] {
  const { program } = parseProgram(src);
  const fn = program.stmts.find((s) => s.kind === 'component' || s.kind === 'function');
  if (!fn || !('body' in fn)) throw new Error('no function found');
  return fn.body;
}

const cpuDeps = { tryWebGpu: async () => null, tryWebGl2: () => null, limitsHint: CPU_LIMITS };

/** Build a typed-array buffer value from a metael source expression (e.g. `f32([1,2,3,4])`). */
function bufferOf(src: string, host: RuntimeReactiveHost): object {
  return evaluateProgram(src, { host, env: new RecordingHostEnv(), builtins: [MATH_BUILTINS] }).value as object;
}

// Dispatch a map kernel over `output` on the CPU backend and return its settled value, driving the
// host-driven async queue with a change()+drain settle loop (mirrors resource.test.ts / reduce.test.ts).
async function dispatchMap(src: string, output: readonly number[]): Promise<{ value: unknown; core: boolean }> {
  const host = new RuntimeReactiveHost();
  const kernel = kernelOf(src, host);
  const engine = new GpuEngine(host, cpuDeps);
  const cfg = { output: [...output], backend: 'cpu' as const };
  change(() => { engine.gpu(kernel, cfg); });
  await new Promise((r) => setTimeout(r, 20));
  let settled!: ReturnType<GpuEngine['gpu']>;
  change(() => { settled = engine.gpu(kernel, cfg); });
  return { value: settled.value, core: settled.core };
}

// Dispatch a reduction on the CPU backend. Returns the settled scalar + core + the SYNCHRONOUSLY-emitted WGSL
// (the shader `resource.wgsl` holds — the `_reduce` body carries the RED-before / GREEN-after evidence).
async function dispatchReduce(src: string, inputSrc: string, identity: number): Promise<{ value: unknown; core: boolean; wgsl: string; reasons: string[] }> {
  const host = new RuntimeReactiveHost();
  const reducer = kernelOf(src, host);
  const xs = bufferOf(inputSrc, host);
  const engine = new GpuEngine(host, cpuDeps);
  const cfg = { input: xs, identity, backend: 'cpu' as const };
  let first!: ReturnType<GpuEngine['gpuReduce']>;
  change(() => { first = engine.gpuReduce(reducer, cfg); });
  const wgsl = first.wgsl;   // emitted synchronously at the gpuReduce entry (before the async fold)
  await new Promise((r) => setTimeout(r, 20));
  let settled!: ReturnType<GpuEngine['gpuReduce']>;
  change(() => { settled = engine.gpuReduce(reducer, cfg); });
  return { value: settled.value, core: settled.core, wgsl, reasons: (settled.reasons ?? []).map((r) => r.code) };
}

// Dispatch a histogram on the CPU backend. Returns the settled per-bin counts + core + the WGSL (the `_binOf`
// body carries the RED-before / GREEN-after evidence). Histogram emits WGSL only (no GLSL — no fragment atomics).
async function dispatchHistogram(src: string, inputSrc: string, bins: number): Promise<{ value: unknown; core: boolean; wgsl: string; reasons: string[] }> {
  const host = new RuntimeReactiveHost();
  const binMapper = kernelOf(src, host);
  const xs = bufferOf(inputSrc, host);
  const engine = new GpuEngine(host, cpuDeps);
  const cfg = { input: xs, bins, backend: 'cpu' as const };
  let first!: ReturnType<GpuEngine['gpuHistogram']>;
  change(() => { first = engine.gpuHistogram(binMapper, cfg); });
  const wgsl = first.wgsl;
  await new Promise((r) => setTimeout(r, 20));
  let settled!: ReturnType<GpuEngine['gpuHistogram']>;
  change(() => { settled = engine.gpuHistogram(binMapper, cfg); });
  return { value: settled.value, core: settled.core, wgsl, reasons: (settled.reasons ?? []).map((r) => r.code) };
}

describe('map kernel — implicit last-expression return (MLGPU parity with the interpreter oracle)', () => {
  it('dispatches a trailing-bare-expression kernel correctly (was silently all-zeros)', async () => {
    // component k(i) { i + 1 } — the kernel's param IS the thread coordinate, so over output [4] the cells
    // are i ∈ {0,1,2,3} → [1,2,3,4]. Before the fix this dispatched [0,0,0,0] (the emitters lowered only an
    // explicit `return`; the trailing bare expr was evaluated-and-discarded).
    const { value, core } = await dispatchMap(`component k(i) { i + 1 }\nk`, [4]);
    expect(value).toEqual([1, 2, 3, 4]);
    expect(core).toBe(true);
  });

  it('the implicit-return form equals the explicit-return form', async () => {
    const implicit = await dispatchMap(`component k(i) { i + 1 }\nk`, [4]);
    const explicit = await dispatchMap(`component k(i) { return i + 1 }\nk`, [4]);
    expect(implicit.value).toEqual(explicit.value);
    expect(implicit.value).toEqual([1, 2, 3, 4]);
    expect(implicit.core).toBe(true);
    expect(explicit.core).toBe(true);
  });

  it('a body whose last statement is an explicit return after a let is unchanged', async () => {
    const { value, core } = await dispatchMap(`component k(i) { let x = i + 1\nreturn x }\nk`, [4]);
    expect(value).toEqual([1, 2, 3, 4]);
    expect(core).toBe(true);
  });

  it('a trailing for-loop before an explicit return is not mis-treated as the implicit return', async () => {
    // The `for` is NOT the last statement (an explicit `return acc` follows) — the last-expr rule never
    // touches it, and the loop accumulates i three times: acc = 3*i.
    const { value, core } = await dispatchMap(
      `component k(i) { let acc = 0\nfor (const j of range(3)) { acc = acc + i }\nreturn acc }\nk`,
      [4],
    );
    expect(value).toEqual([0, 3, 6, 9]);
    expect(core).toBe(true);
  });
});

describe('normalizeImplicitReturn — mirrors the interpreter execBlockValue rule exactly', () => {
  it('rewrites a trailing bare expr into an explicit return', () => {
    const got = normalizeImplicitReturn(bodyOf(`component k(i) { i + 1 }`));
    const want = bodyOf(`component k(i) { return i + 1 }`);
    expect(stripSpans(got)).toEqual(stripSpans(want));
  });

  it('rewrites only the trailing expr, leaving a preceding let intact', () => {
    const got = normalizeImplicitReturn(bodyOf(`component k(i) { let x = i + 1\nx * 2 }`));
    const want = bodyOf(`component k(i) { let x = i + 1\nreturn x * 2 }`);
    expect(stripSpans(got)).toEqual(stripSpans(want));
  });

  it('leaves a body already ending in an explicit return untouched', () => {
    const body = bodyOf(`component k(i) { return i + 1 }`);
    expect(stripSpans(normalizeImplicitReturn(body))).toEqual(stripSpans(body));
  });

  it('leaves a body whose last statement is not an expr (a trailing let) unchanged', () => {
    const body = bodyOf(`component k(i) { i + 1\nlet x = 3 }`);
    expect(stripSpans(normalizeImplicitReturn(body))).toEqual(stripSpans(body));
  });

  it('leaves a body whose last statement is a for-loop (a block statement, not an expr) unchanged', () => {
    // A trailing `for` yields no implicit value in the interpreter (its execStmt returns null) — mirroring the
    // trailing-`if` no-recursion case, a block statement as the last stmt is NOT treated as a return, so the
    // `for` stays the last stmt untouched (no rewrite to a return of its body).
    const body = bodyOf(`component k(i) { let acc = 0\nfor (const j of range(3)) { acc = acc + i } }`);
    const got = normalizeImplicitReturn(body);
    expect(got[got.length - 1]!.kind).toBe('for');
    expect(stripSpans(got)).toEqual(stripSpans(body));
  });

  it('does NOT recurse into a trailing if (a nested trailing expr is not the function return)', () => {
    const body = bodyOf(`component k(i) { if (i > 0) { i + 1 } }`);
    // The last statement is an `if`, not a bare expr — the interpreter yields no implicit value here, so the
    // body is returned unchanged (the inner `i + 1` is NOT promoted to a return).
    expect(stripSpans(normalizeImplicitReturn(body))).toEqual(stripSpans(body));
  });

  it('leaves an empty body unchanged', () => {
    expect(normalizeImplicitReturn([])).toEqual([]);
  });
});

describe('reduce kernel — implicit last-expression return (MLGPU parity: gate ↔ emitter)', () => {
  it('dispatches an implicit-return reducer correctly and equals the explicit-return form (CPU)', async () => {
    // component add(acc, x) { acc + x } — a trailing bare expr. On CPU both implicit + explicit fold to 10 (the
    // interpreter oracle honors the last-expr return); the VALUE alone can't RED. The meaningful proof is the
    // emitted WGSL below. Here we lock in that the CPU value + core are correct + match the explicit form.
    const implicit = await dispatchReduce(`component add(acc, x) { acc + x }\nadd`, `f32([1, 2, 3, 4])`, 0);
    const explicit = await dispatchReduce(`component add(acc, x) { return acc + x }\nadd`, `f32([1, 2, 3, 4])`, 0);
    expect(implicit.value).toBe(10);
    expect(implicit.core).toBe(true);
    expect(implicit.reasons).toEqual([]);
    expect(implicit.value).toBe(explicit.value);
    expect(implicit.core).toBe(explicit.core);
  });

  it('the implicit reducer emits WGSL whose _reduce body has a return (RED before the normalization fix)', () => {
    // The RED→GREEN proof: gateReducer + emitReduceWgsl on the UN-normalized implicit reducer would emit a
    // non-void `fn _reduce(acc, x) -> f32` whose body is a bare `acc + x;` with NO `return` — an invalid shader.
    // The engine's gpuReduce entry normalizes the reducer body FIRST (matching the map path), so the shader the
    // resource emits must equal the explicit reducer's shader (both normalized → identical WGSL). We assert
    // through the ENGINE's synchronously-emitted `resource.wgsl` (dispatchReduce returns it), which reflects the
    // entry-point normalization. Before the fix the two WGSL strings DIFFER (implicit lacks the `return`).
    const host = new RuntimeReactiveHost();
    const explicit = kernelOf(`component add(acc, x) { return acc + x }\nadd`, host);
    const { bindings } = gateReducer(explicit, host);
    const explicitWgsl = emitReduceWgsl(explicit, bindings, 0);
    // The engine-emitted WGSL for the IMPLICIT reducer (post entry-point normalization) must equal the explicit
    // one and contain a `return` inside the `_reduce` body.
    return dispatchReduce(`component add(acc, x) { acc + x }\nadd`, `f32([1, 2, 3, 4])`, 0).then((implicit) => {
      expect(implicit.wgsl).toContain('fn _reduce');
      // Isolate JUST the `_reduce` function body (fn header → its first closing brace at column 0), so this
      // `return` check can't be satisfied by main's `return;` — it must be the normalized `return acc + x;`.
      const start = implicit.wgsl.indexOf('fn _reduce');
      const reduceBody = implicit.wgsl.slice(start, implicit.wgsl.indexOf('\n}', start) + 2);
      expect(reduceBody).toContain('return');       // the trailing `acc + x` is now `return acc + x;`
      expect(implicit.wgsl).toBe(explicitWgsl);       // implicit ≡ explicit shader (both normalized)
    });
  });
});

describe('histogram kernel — implicit last-expression return (MLGPU parity: gate ↔ emitter)', () => {
  it('dispatches an implicit-return bin-mapper correctly and equals the explicit-return form (CPU)', async () => {
    // component bin(x) { x } — maps value→bin index directly. Over [0,1,2,3] with bins:4 each bin gets 1 count.
    const implicit = await dispatchHistogram(`component bin(x) { x }\nbin`, `f32([0, 1, 2, 3])`, 4);
    const explicit = await dispatchHistogram(`component bin(x) { return x }\nbin`, `f32([0, 1, 2, 3])`, 4);
    expect(implicit.value).toEqual([1, 1, 1, 1]);
    expect(implicit.core).toBe(true);
    expect(implicit.reasons).toEqual([]);
    expect(implicit.value).toEqual(explicit.value);
    expect(implicit.core).toBe(explicit.core);
  });

  it('the implicit bin-mapper emits WGSL whose _binOf body has a return (RED before the normalization fix)', () => {
    // Same RED→GREEN proof as the reducer: the un-normalized implicit bin-mapper would emit a non-void
    // `fn _binOf(x) -> f32` with a bare `x;` and NO `return`. The gpuHistogram entry normalizes first, so the
    // engine-emitted WGSL must equal the explicit bin-mapper's shader (and carry a `return` in `_binOf`).
    const host = new RuntimeReactiveHost();
    const explicit = kernelOf(`component bin(x) { return x }\nbin`, host);
    const { bindings } = gateBinMapper(explicit, host);
    const explicitWgsl = emitHistogramWgsl(explicit, bindings, 4);
    return dispatchHistogram(`component bin(x) { x }\nbin`, `f32([0, 1, 2, 3])`, 4).then((implicit) => {
      expect(implicit.wgsl).toContain('fn _binOf');
      // Isolate JUST the `_binOf` function body (fn header → its first column-0 closing brace) so the `return`
      // check can't be satisfied by main's early `return;` — it must be the normalized `return x;`.
      const start = implicit.wgsl.indexOf('fn _binOf');
      const binOfBody = implicit.wgsl.slice(start, implicit.wgsl.indexOf('\n}', start) + 2);
      expect(binOfBody).toContain('return');        // the trailing `x` is now `return x;`
      expect(implicit.wgsl).toBe(explicitWgsl);       // implicit ≡ explicit shader (both normalized)
    });
  });
});
