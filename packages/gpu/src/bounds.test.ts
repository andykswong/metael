import { describe, it, expect } from 'vitest';
import { MATH_BUILTINS } from '@metael/math/lang';
import { RuntimeReactiveHost, change } from '@metael/runtime';
import { evaluateProgram, isUserFn, RecordingHostEnv } from '@metael/lang';
import type { UserFn, Diagnostic, Expr } from '@metael/lang';
import { GpuEngine } from './resource.ts';
import { intervalOf, checkStaticBounds } from './bounds.ts';
import { buildBindingTable, collectFreeNames, type BindingTable } from './binding.ts';

function kernelOf(src: string, host: RuntimeReactiveHost): UserFn {
  const res = evaluateProgram(src, { host, env: new RecordingHostEnv(), builtins: [MATH_BUILTINS] });
  if (!isUserFn(res.value)) throw new Error('kernel: ' + JSON.stringify(res.diagnostics)); return res.value;
}
const cpuDeps = { tryWebGpu: async () => null, tryWebGl2: () => null, limitsHint: { maxStorageBufferBindingSize: 1 << 28, maxComputeWorkgroupsPerDimension: 65535 } };

/** Resolve a parsed kernel's binding table against its closure — the same table `gpu()` feeds to
 *  `checkStaticBounds`, built here so a unit test can drive the prover DIRECTLY (bypassing the gate) for
 *  constructs the gate would reject for an unrelated reason (a non-range `for`, a `while`, a nested
 *  `function`, an object/array literal) — isolating the bounds-prover's own reject-or-pass decision. */
function bindingsOf(kernel: UserFn, host: RuntimeReactiveHost): BindingTable {
  return buildBindingTable(kernel, collectFreeNames(kernel), host);
}
const hasIndexStatic = (reasons: readonly Diagnostic[]): boolean => reasons.some((r) => r.code === 'MLGPU-INDEX-STATIC');

describe('static out-of-bounds bounds-prover', () => {
  it('rejects a provably-OOB index a[i + N] (interval entirely >= length)', async () => {
    const host = new RuntimeReactiveHost();
    // x.length === 4, output [4] → coord i ∈ [0,3], i+4 ∈ [4,7] ≥ 4 → provably OOB.
    const kernel = kernelOf(`const x = f32(4, (i) => i)\ncomponent k(i) { return x[i + 4] }\nk`, host);
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    expect(s.core).toBe(false);
    expect(s.reasons.some((r) => r.code === 'MLGPU-INDEX-STATIC')).toBe(true);
  });
  it('rejects a literal OOB index a[99] on a length-4 buffer', async () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const x = f32(4, (i) => i)\ncomponent k(i) { return x[99] }\nk`, host);
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    expect(s.core).toBe(false);
    expect(s.reasons.some((r) => r.code === 'MLGPU-INDEX-STATIC')).toBe(true);
  });
  it('rejects a provably-OOB index on a PLAIN-array buffer (its length is statically known too)', async () => {
    const host = new RuntimeReactiveHost();
    // A plain array x has length 3; output [3] → coord i ∈ [0,2], i+3 ∈ [3,5] ≥ 3 → provably OOB. The
    // prover reads a plain array's length directly (no descriptor), so it catches this like a typed array.
    const kernel = kernelOf(`const x = [1, 2, 3]\ncomponent k(i) { return x[i + 3] }\nk`, host);
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [3], backend: 'cpu' }); });
    expect(s.core).toBe(false);
    expect(s.reasons.some((r) => r.code === 'MLGPU-INDEX-STATIC')).toBe(true);
  });
  it('ACCEPTS a safe in-range index a[i] (interval ⊂ [0,length)) — inert, no diagnostic', async () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const x = f32(4, (i) => i)\ncomponent k(i) { return x[i] }\nk`, host);
    const engine = new GpuEngine(host, cpuDeps);
    change(() => { engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    await new Promise((r) => setTimeout(r, 20));
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    expect(s.core).toBe(true);
    expect(s.value).toEqual([0, 1, 2, 3]);
  });
  it('ACCEPTS a data-dependent / partially-in-range index (a[i-1], a[i*2]) — unprovable → oracle covers it', async () => {
    const host = new RuntimeReactiveHost();
    const k1 = kernelOf(`const x = f32(4, (i) => i)\ncomponent k(i) { return x[i * 2] }\nk`, host);   // [0,6] overlaps [0,4) → pass
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(k1, { output: [4], backend: 'cpu' }); });
    expect(s.core).toBe(true);   // NOT statically rejected — partial OOB is the oracle's job
  });
  it('ACCEPTS a matmul-style index a[row * N + k] (the flagship kernel must not be falsely rejected)', async () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const N = 4
const a = f32(N * N, (i) => i)
component k(row, col) { let s = 0 for (const j of range(N)) { s = s + a[row * N + j] } return s }
k`, host);
    const engine = new GpuEngine(host, cpuDeps);
    change(() => { engine.gpu(kernel, { output: [4, 4], backend: 'cpu' }); });
    await new Promise((r) => setTimeout(r, 20));
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4, 4], backend: 'cpu' }); });
    expect(s.core).toBe(true);   // row∈[0,3], j∈[0,3], N=4 → row*4+j ∈ [0,15] ⊂ [0,16) → SAFE, accepted
  });

  it('ACCEPTS a[i - 1] (interval [-1, N-2] overlaps [0,N) at i>=1) — partial OOB is the oracle/runtime job', async () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const x = f32(4, (i) => i)\ncomponent k(i) { return x[i - 1] }\nk`, host);
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    expect(s.core).toBe(true);
    expect(s.reasons.some((r) => r.code === 'MLGPU-INDEX-STATIC')).toBe(false);
  });

  it('REJECTS a[i - 5] on a length-4 buffer (interval [-5, -2] entirely < 0 even after ceil(hi))', async () => {
    const host = new RuntimeReactiveHost();
    // i ∈ [0,3] → i-5 ∈ [-5, -2]; ceil(-2) = -2 < 0 → provably < 0 for every coord → reject.
    const kernel = kernelOf(`const x = f32(4, (i) => i)\ncomponent k(i) { return x[i - 5] }\nk`, host);
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    expect(s.core).toBe(false);
    expect(s.reasons.some((r) => r.code === 'MLGPU-INDEX-STATIC')).toBe(true);
  });

  it('ACCEPTS a[i * -1] (a negative-operand multiply: interval [-3, 0] overlaps [0,N))', async () => {
    const host = new RuntimeReactiveHost();
    // i ∈ [0,3], i*(-1) ∈ [-3, 0] — includes 0 (in range at i=0) → NOT all-OOB → pass. Proves signed `*`.
    const kernel = kernelOf(`const x = f32(4, (i) => i)\ncomponent k(i) { return x[i * -1] }\nk`, host);
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    expect(s.core).toBe(true);
  });

  it('ACCEPTS a data-dependent index a[b[i]] (an inner buffer read is unprovable → ⊤ → pass)', async () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const a = f32(4, (i) => i)\nconst b = f32(4, (i) => i)\ncomponent k(i) { return a[b[i]] }\nk`, host);
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    expect(s.core).toBe(true);   // b[i] is a runtime value → the index interval is ⊤ → never statically rejected
  });

  it('ACCEPTS a[k] with an unprovable range bound k ∈ range(m), m a data-dependent local', async () => {
    const host = new RuntimeReactiveHost();
    // m = a[0] (a buffer read → ⊤), so range(m)'s var is ⊤ → a[k] is unprovable → pass (never falsely rejected).
    const kernel = kernelOf(`const a = f32(4, (i) => i)\ncomponent k(i) { let s = 0 const m = a[0] for (const j of range(m)) { s = s + a[j] } return s }\nk`, host);
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    expect(s.core).toBe(true);
  });

  it('REJECTS a const-folded provable-OOB index a[base + j], base=8, j∈range(2), on length-4', async () => {
    const host = new RuntimeReactiveHost();
    // base = 8 (const literal), j ∈ [0,1] → base+j ∈ [8,9] ≥ 4 → provably OOB for every coord/j → reject.
    const kernel = kernelOf(`const a = f32(4, (i) => i)\nconst base = 8\ncomponent k(i) { let s = 0 for (const j of range(2)) { s = s + a[base + j] } return s }\nk`, host);
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    expect(s.core).toBe(false);
    expect(s.reasons.some((r) => r.code === 'MLGPU-INDEX-STATIC')).toBe(true);
  });

  it('ACCEPTS an all-OOB index guarded by an if (a[99] under `if`) — the guard may exclude it → not proven-for-every-coord', async () => {
    const host = new RuntimeReactiveHost();
    // a[99] is all-OOB in isolation, but it only runs when i > 100 — which never holds for i ∈ [0,3]. Rejecting
    // would be UNSOUND (the access is unreachable). Suppress the rejection inside a conditional branch.
    const kernel = kernelOf(`const a = f32(4, (i) => i)\ncomponent k(i) { let s = 0 if (i > 100) { s = a[99] } return s }\nk`, host);
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    expect(s.core).toBe(true);
    expect(s.reasons.some((r) => r.code === 'MLGPU-INDEX-STATIC')).toBe(false);
  });

  it('ACCEPTS an all-OOB index in a maybe-zero-iteration loop (a[99] in `for range(m)`, m data-dependent)', async () => {
    const host = new RuntimeReactiveHost();
    // range(m) with m = a[0] could iterate 0 times → the body is not guaranteed → no rejection (unsound to).
    const kernel = kernelOf(`const a = f32(4, (i) => i)\ncomponent k(i) { let s = 0 const m = a[0] for (const j of range(m)) { s = a[99] } return s }\nk`, host);
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    expect(s.core).toBe(true);
    expect(s.reasons.some((r) => r.code === 'MLGPU-INDEX-STATIC')).toBe(false);
  });

  it('ACCEPTS a reassigned accumulator index a[s] where s is mutated in a loop (⊤ → pass)', async () => {
    const host = new RuntimeReactiveHost();
    // s is reassigned (s = s + 1), so a[s] is ⊤ regardless of its init → never falsely rejected.
    const kernel = kernelOf(`const a = f32(4, (i) => i)\ncomponent k(i) { let s = 99 for (const j of range(2)) { s = j } return a[s] }\nk`, host);
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    expect(s.core).toBe(true);   // s is assigned in the loop → its interval is ⊤ → a[s] passes (oracle covers it)
  });

  it('does NOT reject an all-OOB index in dead code after a guard clause returns (the defensive-guard idiom)', async () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const x = f32(4, (i) => i)
component k(i) { if (i + 4 >= x.length) { return 0 } return x[i + 4] }
k`, host);
    const engine = new GpuEngine(host, cpuDeps);
    change(() => { engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    await new Promise((r) => setTimeout(r, 20));
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    expect(s.core).toBe(true);        // the OOB access is unreachable → NOT rejected
    expect(s.value).toEqual([0, 0, 0, 0]);   // matches the interpreter (the guard always fires)
  });
  it('does NOT reject an all-OOB index after an early return', async () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const x = f32(4, (i) => i)
component k(i) { if (i < 100) { return x[i] } return x[99] }
k`, host);
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    expect(s.core).toBe(true);
  });
  it('does NOT reject unconditional dead code after a return', async () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const x = f32(4, (i) => i)
component k(i) { return x[i] return x[99] }
k`, host);   // NOTE: if the parser rejects a statement after return, use a different dead-code shape or drop this case + note it
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    expect(s.core).toBe(true);
  });
  it('STILL rejects a REACHABLE trailing all-OOB index (a bare if does not definitely return)', async () => {
    const host = new RuntimeReactiveHost();
    // the `if (i<2)` does NOT cover all coords (i∈[0,3]); for i>=2 control falls through to x[i+100] → reachable + all-OOB for the falling-through coords.
    // x[i+100] with i∈[0,3] → [100,103] entirely >= 4 → provably OOB on the reachable path → MUST still reject.
    const kernel = kernelOf(`const x = f32(4, (i) => i)
component k(i) { if (i < 2) { return x[i] } return x[i + 100] }
k`, host);
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    expect(s.core).toBe(false);
    expect(s.reasons.some((r) => r.code === 'MLGPU-INDEX-STATIC')).toBe(true);
  });

  // ─── `intervalOf` directly: a buffer/uniform/callee ident (not a scalar, not in env) is ⊤ (null) ───
  it('intervalOf: a buffer ident is ⊤ (null) — a buffer is not a numeric quantity', () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const x = f32(4, (i) => i)\ncomponent k(i) { return x[i] }\nk`, host);
    const bindings = bindingsOf(kernel, host);
    expect(bindings.byName.get('x')?.role).toBe('buffer');
    const ret = kernel.body.find((s) => s.kind === 'return') as Extract<UserFn['body'][number], { kind: 'return' }>;
    const idx = ret.value as Extract<Expr, { kind: 'index' }>;
    const bufIdent = idx.object;   // the bare buffer ident `x`
    // A buffer ident is not in `env` and its binding role is 'buffer' (not 'scalar') → null (⊤). This is the
    // guard that keeps `a[b[i]]` unprovable: the inner buffer read never yields a bogus finite interval.
    expect(intervalOf(bufIdent, new Map(), bindings)).toBeNull();
    // An ident absent from BOTH env and the binding table is likewise ⊤ (same return-null path).
    expect(intervalOf({ kind: 'ident', name: 'notbound', span: bufIdent.span }, new Map(), bindings)).toBeNull();
  });

  // ─── `guardProvablyTrue` `<=` case: a provably-true `<=` guard makes the fall-through dead (via a returning if) ───
  it('does NOT reject an all-OOB index after a provably-true `<=` guard-clause return', async () => {
    const host = new RuntimeReactiveHost();
    // i ∈ [0,3]; `i <= 100` is provably true for EVERY coord (l.hi=3 <= r.lo=100) AND the then-block returns,
    // so the trailing `x[i + 100]` is unreachable dead code → NOT rejected (proving the `<=` always-true leg).
    const kernel = kernelOf(`const x = f32(4, (i) => i)\ncomponent k(i) { if (i <= 100) { return 0 } return x[i + 100] }\nk`, host);
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    expect(s.core).toBe(true);
    expect(hasIndexStatic(s.reasons)).toBe(false);
  });

  // ─── `guardProvablyTrue` default case (`==`): an equality guard is NOT range-decidable → the trailing OOB is reachable ───
  it('STILL rejects a reachable trailing OOB after an `==` guard (== is not a provable-always-true range guard)', async () => {
    const host = new RuntimeReactiveHost();
    // `if (i == 2) { return 0 }` — `==` is not decidable from a range interval, so guardProvablyTrue returns
    // false → the if does NOT definitely return → the trailing `x[i + 100]` stays on the guaranteed path.
    // i ∈ [0,3] → i+100 ∈ [100,103] entirely >= 4 → provably OOB → MUST reject (never suppressed by an == guard).
    const kernel = kernelOf(`const x = f32(4, (i) => i)\ncomponent k(i) { if (i == 2) { return 0 } return x[i + 100] }\nk`, host);
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    expect(s.core).toBe(false);
    expect(hasIndexStatic(s.reasons)).toBe(true);
  });

  // ─── `walkExpr` `object`/`array` cases: the walk descends into object-entry / array-element expressions ───
  // The single-output gate rejects object/array literals, so these drive `checkStaticBounds` DIRECTLY on the
  // parsed kernel + its bindings — proving the prover still walks INTO a literal and rejects a provable OOB there.
  it('walkExpr descends into an object-literal entry and rejects a provable OOB inside it (direct checkStaticBounds)', () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const x = f32(4, (i) => i)\ncomponent k(i) { return {v: x[i + 100]} }\nk`, host);
    const reasons: Diagnostic[] = [];
    checkStaticBounds(kernel, bindingsOf(kernel, host), [4], reasons);
    // i ∈ [0,3] → i+100 ∈ [100,103] >= 4, and the object literal is a guaranteed (top-level return) expr → reject.
    expect(hasIndexStatic(reasons)).toBe(true);
  });
  it('walkExpr descends into an array-literal element and rejects a provable OOB inside it (direct checkStaticBounds)', () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const x = f32(4, (i) => i)\ncomponent k(i) { return [x[i + 100]] }\nk`, host);
    const reasons: Diagnostic[] = [];
    checkStaticBounds(kernel, bindingsOf(kernel, host), [4], reasons);
    expect(hasIndexStatic(reasons)).toBe(true);
  });
  it('walkExpr does NOT reject a partially-in-range object-literal entry (unprovable → oracle covers it)', () => {
    const host = new RuntimeReactiveHost();
    // x[i*2] → [0,6] overlaps [0,4) → not all-OOB → the descent finds nothing to reject (the safe direction).
    const kernel = kernelOf(`const x = f32(4, (i) => i)\ncomponent k(i) { return {v: x[i * 2]} }\nk`, host);
    const reasons: Diagnostic[] = [];
    checkStaticBounds(kernel, bindingsOf(kernel, host), [4], reasons);
    expect(hasIndexStatic(reasons)).toBe(false);
  });

  // ─── `walkStmt` `expr` case: a bare expression statement is on the guaranteed path ───
  it('rejects a provable OOB in a bare expression statement (walkStmt `expr` case)', async () => {
    const host = new RuntimeReactiveHost();
    // `x[i + 100]` as a top-level expression statement (its value discarded) is still a guaranteed access →
    // i+100 ∈ [100,103] >= 4 → provably OOB → reject. Exercises walkStmt's `expr` arm.
    const kernel = kernelOf(`const x = f32(4, (i) => i)\ncomponent k(i) { x[i + 100] return 0 }\nk`, host);
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    expect(s.core).toBe(false);
    expect(hasIndexStatic(s.reasons)).toBe(true);
  });

  // ─── `for … of range(C)` with a constant C <= 0: the body is DEAD, never analyzed (dead=true) ───
  it('does NOT reject a blatantly-OOB index inside a `for … of range(0)` loop (dead body)', async () => {
    const host = new RuntimeReactiveHost();
    // range(0) provably iterates ZERO times → the body never runs → `x[9999]` is dead code → NOT rejected,
    // even though [9999,9999] is trivially all-OOB on a length-4 buffer. Proves the C<=0 dead-body suppression.
    const kernel = kernelOf(`const x = f32(4, (i) => i)\ncomponent k(i) { for (const j of range(0)) { x[9999] } return 0 }\nk`, host);
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    expect(s.core).toBe(true);
    expect(hasIndexStatic(s.reasons)).toBe(false);
  });

  // ─── `for … of <non-range>`: the else branch walks the iterable expr; the body is not-guaranteed ───
  it('does NOT reject a blatantly-OOB index inside a non-range `for … of buffer` loop (direct checkStaticBounds)', () => {
    const host = new RuntimeReactiveHost();
    // A `for (const j of x)` over a BUFFER (not range()) is not a lowerable loop (the gate rejects it), so the
    // prover is driven directly. The `for` else-branch walks the iterable and the body with guaranteed=false
    // (a non-range iterable's trip count is unknown) → the all-OOB `x[9999]` in the body is NOT rejected.
    const kernel = kernelOf(`const x = f32(4, (i) => i)\ncomponent k(i) { for (const j of x) { x[9999] } return 0 }\nk`, host);
    const reasons: Diagnostic[] = [];
    checkStaticBounds(kernel, bindingsOf(kernel, host), [4], reasons);
    expect(hasIndexStatic(reasons)).toBe(false);
  });

  // ─── `while` statement: walks test + body with guaranteed=false ───
  it('does NOT reject a blatantly-OOB index inside a `while` loop body (direct checkStaticBounds)', () => {
    const host = new RuntimeReactiveHost();
    // A `while` is data-dependent (the gate rejects it), so drive the prover directly. walkStmt's `while` arm
    // walks the body with guaranteed=false (it may iterate zero times) → the all-OOB `x[9999]` is NOT rejected.
    const kernel = kernelOf(`const x = f32(4, (i) => i)\ncomponent k(i) { while (i < 2) { x[9999] } return 0 }\nk`, host);
    const reasons: Diagnostic[] = [];
    checkStaticBounds(kernel, bindingsOf(kernel, host), [4], reasons);
    expect(hasIndexStatic(reasons)).toBe(false);
  });

  // ─── `walkExpr` `cond` case: a ternary's test always runs (guaranteed); its branches are conditional ───
  it('rejects a provable OOB in a ternary TEST (the test always runs → guaranteed path)', async () => {
    const host = new RuntimeReactiveHost();
    // The test subexpression of `x[i+100] > 0 ? 1 : 0` is evaluated for every coord → guaranteed. i+100 ∈
    // [100,103] >= 4 → provably OOB → reject. Exercises walkExpr's `cond` arm on the (guaranteed) test.
    const kernel = kernelOf(`const x = f32(4, (i) => i)\ncomponent k(i) { return x[i + 100] > 0 ? 1 : 0 }\nk`, host);
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    expect(s.core).toBe(false);
    expect(hasIndexStatic(s.reasons)).toBe(true);
  });
  it('does NOT reject a provable OOB in a ternary BRANCH (a branch is conditional → not guaranteed)', async () => {
    const host = new RuntimeReactiveHost();
    // The `then`/`else` of `i > 0 ? x[i+100] : 0` run only for some coords → walkExpr descends with
    // guaranteed=false → the all-OOB `x[i+100]` is NOT rejected (the mirror of the guaranteed-test case).
    const kernel = kernelOf(`const x = f32(4, (i) => i)\ncomponent k(i) { return i > 0 ? x[i + 100] : 0 }\nk`, host);
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    expect(s.core).toBe(true);
    expect(hasIndexStatic(s.reasons)).toBe(false);
  });

  // ─── `walkExpr` `call` case: a builtin-call argument is on the guaranteed path ───
  it('rejects a provable OOB inside a builtin-call argument (walkExpr `call` arm)', async () => {
    const host = new RuntimeReactiveHost();
    // `abs(x[i + 100])` — the argument is evaluated for every coord → guaranteed. i+100 ∈ [100,103] >= 4 →
    // provably OOB → reject. Exercises walkExpr descending into `e.args`.
    const kernel = kernelOf(`const x = f32(4, (i) => i)\ncomponent k(i) { return abs(x[i + 100]) }\nk`, host);
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    expect(s.core).toBe(false);
    expect(hasIndexStatic(s.reasons)).toBe(true);
  });

  // ─── `walkExpr` `index` else-branch: an index whose object is NOT a buffer ident is walked normally ───
  it('walks a nested index whose object is not a buffer ident without a false rejection (direct checkStaticBounds)', () => {
    const host = new RuntimeReactiveHost();
    // `x[i][0]`: the OUTER index's object is `x[i]` (an index expr, not a bare buffer ident) → the else-branch
    // walks both object and index. The inner `x[i]` is in range; no all-OOB access exists → no rejection.
    const kernel = kernelOf(`const x = f32(4, (i) => i)\ncomponent k(i) { return x[i][0] }\nk`, host);
    const reasons: Diagnostic[] = [];
    checkStaticBounds(kernel, bindingsOf(kernel, host), [4], reasons);
    expect(hasIndexStatic(reasons)).toBe(false);
  });

  // ─── `walkStmt` default case: a nested function/component statement is inert (nothing to prove) ───
  it('is inert on a nested `function` declaration statement (walkStmt default case, direct checkStaticBounds)', () => {
    const host = new RuntimeReactiveHost();
    // A nested `function` is not lowerable (the gate rejects it), so drive the prover directly. walkStmt's
    // default arm ignores it (it introduces no index to prove); the guaranteed top-level `x[i]` is in range → pass.
    const kernel = kernelOf(`const x = f32(4, (i) => i)\ncomponent k(i) { function helper() { return 0 } return x[i] }\nk`, host);
    const reasons: Diagnostic[] = [];
    checkStaticBounds(kernel, bindingsOf(kernel, host), [4], reasons);
    expect(hasIndexStatic(reasons)).toBe(false);
  });
});
