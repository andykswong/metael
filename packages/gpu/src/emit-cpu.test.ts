import { describe, it, expect } from 'vitest';
import { evaluateProgram, isUserFn, makeCallable } from '@metael/lang';
import type { UserFn } from '@metael/lang';
import { PlainStorageHost, RecordingHostEnv } from '@metael/lang';
import { gateKernel } from './gate.ts';
import { emitCpu } from './emit-cpu.ts';
import { checkMatch } from './oracle.ts';

function kernelOf(src: string): { fn: UserFn; host: PlainStorageHost } {
  const host = new PlainStorageHost();
  const res = evaluateProgram(src, { host, env: new RecordingHostEnv() });
  if (!isUserFn(res.value)) throw new Error('expected kernel');
  return { fn: res.value, host };
}

describe('CPU emitter — eval-free closure tree over descriptor handlers', () => {
  it('computes a scalar saxpy cell', () => {
    const { fn, host } = kernelOf(`
      const N = 8
      const x = f32(N, (i) => i)
      const y = f32(N, (i) => 2 * i)
      function k(i) { return 3 * x[i] + y[i] }
      k`);
    const { bindings } = gateKernel(fn, host);
    const run = emitCpu(fn, bindings, host);   // scalar output → a 1-element [value] per cell
    expect(run([3])).toEqual([15]);
    expect(run([0])).toEqual([0]);
  });
  it('computes a matmul cell with a bounded loop (kernel authored as a component so `let sum` is reactive)', () => {
    const { fn, host } = kernelOf(`
      const N = 3
      const a = f32(N * N, (i) => i)
      const b = f32(N * N, (i) => (i % (N + 1) == 0) ? 1 : 0)
      component product(row, col) {
        let sum = 0
        for (const k of range(N)) { sum = sum + a[row * N + k] * b[k * N + col] }
        return sum
      }
      product`);
    const { bindings } = gateKernel(fn, host);
    const run = emitCpu(fn, bindings, host);
    expect(run([1, 2])).toEqual([1 * 3 + 2]);
  });
  it('handles a transcendental', () => {
    const { fn, host } = kernelOf(`const a = f32(4, (i) => i)\nfunction k(i) { return sin(a[i]) }\nk`);
    const { bindings } = gateKernel(fn, host);
    const run = emitCpu(fn, bindings, host);
    expect(run([1])[0]).toBeCloseTo(Math.sin(1), 12);
  });
  it('a boolean used arithmetically coerces to 0/1 (matches the interpreter toNum), not NaN', () => {
    const { fn, host } = kernelOf(`const x = f32(4, (i) => i)\ncomponent k(i) { return (x[i] > 1) * 10 }\nk`);
    const { bindings } = gateKernel(fn, host);
    const cpu = emitCpu(fn, bindings, host);
    const output = [0, 1, 2, 3].map((i) => cpu([i])[0]!);
    expect(output).toEqual([0, 0, 10, 10]);   // x=[0,1,2,3]; >1 at i=2,3
    expect(checkMatch({ fn, host, bindings, output, dims: [4], precision: 'f32', sampleCount: 4 }).ok).toBe(true);
  });
  it('a bad-domain builtin (sqrt of a negative) fails closed to 0, matching the interpreter', () => {
    const { fn, host } = kernelOf(`const x = f32(4, (i) => i)\ncomponent k(i) { return sqrt(x[i] - 2) }\nk`);
    const { bindings } = gateKernel(fn, host);
    const cpu = emitCpu(fn, bindings, host);
    const output = [0, 1, 2, 3].map((i) => cpu([i])[0]!);   // x-2 = [-2,-1,0,1] → sqrt → [0,0,0,1]
    expect(output).toEqual([0, 0, 0, 1]);
    expect(checkMatch({ fn, host, bindings, output, dims: [4], precision: 'f32', sampleCount: 4 }).ok).toBe(true);
  });
  it('a bad-domain builtin used mid-arithmetic propagates like the interpreter ([]→NaN mid-expr, →0 as output)', () => {
    // sqrt(x[i]-2)+1 : the interpreter returns deepFreeze([]) for the bad arg; +1 coerces via toNum([])=NaN,
    // so the CELL is Number(NaN)=NaN at i=0,1 and Number(1|2)=1|2 at i=2,3. The CPU path must MATCH — this is
    // WHY the bad sentinel is the empty array, not a bare 0 (a bare 0 would give 1 at i=0,1 and diverge).
    // The sampled oracle's ulpDistance can't express NaN≡NaN, so cross-check the interpreter directly here.
    const { fn, host } = kernelOf(`const x = f32(4, (i) => i)\ncomponent k(i) { return sqrt(x[i] - 2) + 1 }\nk`);
    const { bindings } = gateKernel(fn, host);
    const cpu = emitCpu(fn, bindings, host);
    const output = [0, 1, 2, 3].map((i) => cpu([i])[0]!);
    const call = makeCallable(fn, { host, env: { resolveCall: () => ({ handled: false }) }, maxSteps: 100000 });
    const ref = [0, 1, 2, 3].map((i) => Number(call(i)));
    expect(ref).toEqual([NaN, NaN, 1, 2]);          // the interpreter's actual per-cell result ([]+1 → NaN)
    output.forEach((v, i) => expect(Object.is(v, ref[i])).toBe(true));   // CPU-emit ≡ interpreter, NaN incl.
  });
  it('a for-of range bound is floored (matches range() semantics)', () => {
    const { fn, host } = kernelOf(`const x = f32(4, (i) => i)\ncomponent k(i) { let s = 0; for (const j of range(x[i])) { s = s + 1 } return s }\nk`);
    const { bindings } = gateKernel(fn, host);
    const cpu = emitCpu(fn, bindings, host);
    const output = [0, 1, 2, 3].map((i) => cpu([i])[0]!);   // range(0),range(1),range(2),range(3) counts = [0,1,2,3]
    expect(output).toEqual([0, 1, 2, 3]);
    expect(checkMatch({ fn, host, bindings, output, dims: [4], precision: 'f32', sampleCount: 4 }).ok).toBe(true);
  });
  it('modulo (%) matches JS remainder and the interpreter oracle', () => {
    const { fn, host } = kernelOf(`const x = f32(6, (i) => i)\nfunction k(i) { return x[i] % 3 }\nk`);
    const { bindings } = gateKernel(fn, host);
    const cpu = emitCpu(fn, bindings, host);
    const output = [0, 1, 2, 3, 4, 5].map((i) => cpu([i])[0]!);   // i % 3
    expect(output).toEqual([0, 1, 2, 0, 1, 2]);
    expect(checkMatch({ fn, host, bindings, output, dims: [6], precision: 'f32', sampleCount: 6 }).ok).toBe(true);
  });
  it('modulo by zero fails closed to 0 as an output (b === 0 → null → coerced), matching the interpreter', () => {
    const { fn, host } = kernelOf(`const x = f32(4, (i) => i)\ncomponent k(i) { return x[i] % 0 }\nk`);
    const { bindings } = gateKernel(fn, host);
    const cpu = emitCpu(fn, bindings, host);
    const output = [0, 1, 2, 3].map((i) => cpu([i])[0]!);   // % 0 → null → 0
    expect(output).toEqual([0, 0, 0, 0]);
    expect(checkMatch({ fn, host, bindings, output, dims: [4], precision: 'f32', sampleCount: 4 }).ok).toBe(true);
  });
  it('equality (==) is a loose-equals boolean coerced to 0/1 arithmetically (matches the interpreter)', () => {
    const { fn, host } = kernelOf(`const x = f32(4, (i) => i)\ncomponent k(i) { return (x[i] == 2) * 10 }\nk`);
    const { bindings } = gateKernel(fn, host);
    const cpu = emitCpu(fn, bindings, host);
    const output = [0, 1, 2, 3].map((i) => cpu([i])[0]!);   // == 2 only at i=2
    expect(output).toEqual([0, 0, 10, 0]);
    expect(checkMatch({ fn, host, bindings, output, dims: [4], precision: 'f32', sampleCount: 4 }).ok).toBe(true);
  });
  it('inequality (!=) is the negation of loose-equals (matches the interpreter)', () => {
    const { fn, host } = kernelOf(`const x = f32(4, (i) => i)\ncomponent k(i) { return (x[i] != 2) * 10 }\nk`);
    const { bindings } = gateKernel(fn, host);
    const cpu = emitCpu(fn, bindings, host);
    const output = [0, 1, 2, 3].map((i) => cpu([i])[0]!);   // != 2 everywhere but i=2
    expect(output).toEqual([10, 10, 0, 10]);
    expect(checkMatch({ fn, host, bindings, output, dims: [4], precision: 'f32', sampleCount: 4 }).ok).toBe(true);
  });
  it('an out-of-bounds read is null → 0 as an output (matches the interpreter downstream coercion)', () => {
    const { fn, host } = kernelOf(`const x = f32(4, (i) => i)\nfunction k(i) { return x[i + 100] }\nk`);
    const { bindings } = gateKernel(fn, host);
    const cpu = emitCpu(fn, bindings, host);
    const output = [0, 1, 2, 3].map((i) => cpu([i])[0]!);   // every read OOB → null → 0
    expect(output).toEqual([0, 0, 0, 0]);
    expect(checkMatch({ fn, host, bindings, output, dims: [4], precision: 'f32', sampleCount: 4 }).ok).toBe(true);
  });
  it('an out-of-bounds read used mid-arithmetic is null→NaN (toNum(null)), like the interpreter', () => {
    // x[i+100] is OOB → getIndexSafe returns null; toNum(null)=NaN, so null+1 → NaN mid-expression, and the
    // cell (an output) coerces NaN→NaN (extractComps' Number(NaN)). Cross-check the interpreter directly:
    // the sampled oracle's ulpDistance can't express NaN≡NaN.
    const { fn, host } = kernelOf(`const x = f32(4, (i) => i)\nfunction k(i) { return x[i + 100] + 1 }\nk`);
    const { bindings } = gateKernel(fn, host);
    const cpu = emitCpu(fn, bindings, host);
    const output = [0, 1, 2, 3].map((i) => cpu([i])[0]!);
    const call = makeCallable(fn, { host, env: { resolveCall: () => ({ handled: false }) }, maxSteps: 100000 });
    const ref = [0, 1, 2, 3].map((i) => Number(call(i)));
    expect(ref).toEqual([NaN, NaN, NaN, NaN]);
    output.forEach((v, i) => expect(Object.is(v, ref[i])).toBe(true));   // CPU-emit ≡ interpreter, NaN incl.
  });
  it('a bare expression statement is evaluated for effect then the return is used (const/expr/return walk)', () => {
    // `x[i];` is a bare expr statement (execS `expr` case — evaluated, result discarded); the `const t` and the
    // `return` exercise the const-init + return walk. Output is t = x[i]*2.
    const { fn, host } = kernelOf(`const x = f32(4, (i) => i)\ncomponent k(i) { x[i]\n const t = x[i] * 2\n return t }\nk`);
    const { bindings } = gateKernel(fn, host);
    const cpu = emitCpu(fn, bindings, host);
    const output = [0, 1, 2, 3].map((i) => cpu([i])[0]!);
    expect(output).toEqual([0, 2, 4, 6]);
    expect(checkMatch({ fn, host, bindings, output, dims: [4], precision: 'f32', sampleCount: 4 }).ok).toBe(true);
  });
  it('an if/else with a reassigned accumulator drives the if-branch + assign paths', () => {
    // `if (x[i] > 1) { s = 10 } else { s = 1 }` exercises execS `if` (both arms) + `assign` to a reactive let.
    const { fn, host } = kernelOf(`const x = f32(4, (i) => i)\ncomponent k(i) { let s = 0\n if (x[i] > 1) { s = 10 } else { s = 1 }\n return s }\nk`);
    const { bindings } = gateKernel(fn, host);
    const cpu = emitCpu(fn, bindings, host);
    const output = [0, 1, 2, 3].map((i) => cpu([i])[0]!);   // x=[0,1,2,3]; >1 at i=2,3
    expect(output).toEqual([1, 1, 10, 10]);
    expect(checkMatch({ fn, host, bindings, output, dims: [4], precision: 'f32', sampleCount: 4 }).ok).toBe(true);
  });
});

// The scalar-math builtin switch (applyBuiltin) and the vec fallbacks (applyVecBuiltin) in emit-cpu are
// UNREACHABLE from a dispatched kernel and are documented as such in the source ("retained only
// defensively"): every builtin name the CPU hand-walk could encounter is in VEC_NAMES, so a kernel
// referencing one delegates WHOLE to the interpreter (the vec/mat-bearing fast path) and never hand-walks
// applyBuiltin; a builtin reachable only via a helper body is gate-rejected (helper calls are not lowerable)
// before emitCpu is ever built. So there is no dispatched-kernel input that reaches those branches — they
// are left uncovered by design rather than pinned by a test that constructs an impossible-in-production AST.
