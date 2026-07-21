import { describe, it, expect } from 'vitest';
import { MATH_BUILTINS } from '@metael/math/lang';
import { evaluateProgram, isUserFn, makeCallable } from '@metael/lang';
import type { UserFn } from '@metael/lang';
import { PlainStorageHost, RecordingHostEnv } from '@metael/lang';
import { gateKernel } from './gate.ts';
import { emitCpu } from './emit-cpu.ts';
import { checkMatch } from './oracle.ts';

function kernelOf(src: string): { fn: UserFn; host: PlainStorageHost } {
  const host = new PlainStorageHost();
  const res = evaluateProgram(src, { host, env: new RecordingHostEnv(), builtins: [MATH_BUILTINS] });
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
    const call = makeCallable(fn, { host, env: { resolveCall: () => ({ handled: false }) }, maxSteps: 100000, builtins: [MATH_BUILTINS] });
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
    const call = makeCallable(fn, { host, env: { resolveCall: () => ({ handled: false }) }, maxSteps: 100000, builtins: [MATH_BUILTINS] });
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

  // ─── The structured constructors + column access — CPU-emit ≡ the interpreter oracle ───
  // A vec/mat-bearing kernel delegates the WHOLE cell to the interpreter (the oracle), so the CPU path is
  // identical by construction. These pin that the composed ctors + a mat column read produce the right
  // output cells AND that checkMatch (the interpreter oracle) agrees — the CPU emit path handles all three.
  it('vecN composition: vec3(vec2(i, i*2), i+7) flattens correctly (CPU-emit == oracle)', () => {
    const { fn, host } = kernelOf('component k(i) { return vec3(vec2(i, i * 2), i + 7) } k');
    const { bindings } = gateKernel(fn, host, 3);
    const cpu = emitCpu(fn, bindings, host, 3);
    // 3 cells × 3 comps, interleaved: cell i = [i, i*2, i+7].
    const output = [0, 1, 2].flatMap((i) => cpu([i]));
    expect(output).toEqual([0, 0, 7, 1, 2, 8, 2, 4, 9]);
    expect(checkMatch({ fn, host, bindings, output, dims: [3], precision: 'f32', sampleCount: 3, comps: 3 }).ok).toBe(true);
  });
  it('matMxN from column vecs: (mat2(vec2(1,2), vec2(3,4)) * vec2(i,i+1)).x (CPU-emit == oracle)', () => {
    const { fn, host } = kernelOf('component k(i) { const m = mat2(vec2(1, 2), vec2(3, 4)) const v = vec2(i, i + 1) return (m * v).x } k');
    const { bindings } = gateKernel(fn, host, 1);
    const cpu = emitCpu(fn, bindings, host, 1);
    // Column-major mat2 [[1,2],[3,4]] · [i, i+1] → .x (row 0) = 1*i + 3*(i+1) = 4i + 3.
    const output = [0, 1, 2, 3].map((i) => cpu([i])[0]!);
    expect(output).toEqual([3, 7, 11, 15]);
    expect(checkMatch({ fn, host, bindings, output, dims: [4], precision: 'f32', sampleCount: 4 }).ok).toBe(true);
  });
  it('m[i] mat column read: mat3(...)[1].xyz reads column 1 as a vec3 (CPU-emit == oracle)', () => {
    // Column-major mat3 with an i-dependent middle column; m[1] is that column (column-major slots 3,4,5).
    const { fn, host } = kernelOf('component k(i) { const m = mat3(1, 2, 3, 4 * i, 5 * i, 6 * i, 7, 8, 9) return m[1].xyz } k');
    const { bindings } = gateKernel(fn, host, 3);
    const cpu = emitCpu(fn, bindings, host, 3);
    // cell i = column 1 = [4i, 5i, 6i].
    const output = [0, 1, 2].flatMap((i) => cpu([i]));
    expect(output).toEqual([0, 0, 0, 4, 5, 6, 8, 10, 12]);
    expect(checkMatch({ fn, host, bindings, output, dims: [3], precision: 'f32', sampleCount: 3, comps: 3 }).ok).toBe(true);
  });
});

describe('CPU emitter — newly-lowered scalar builtins match the interpreter oracle', () => {
  it('mod is FLOORED (sign follows the divisor), matching the interpreter — not JS % (was NaN before the fix)', () => {
    // BEFORE the fix, `mod` was in neither VEC_NAMES nor applyBuiltin, so the hand-walk hit applyBuiltin's
    // default → NaN (all-zero as a cell after coercion). Now it delegates WHOLE to the interpreter (in
    // VEC_NAMES, like sin/cos), so the CPU cell == the interpreter's floored mod.
    const { fn, host } = kernelOf('const a = f32(4, (i) => i)\ncomponent k(i) { return mod(a[i], 3) }\nk');
    const { bindings } = gateKernel(fn, host);
    const cpu = emitCpu(fn, bindings, host);
    const output = [0, 1, 2, 3].map((i) => cpu([i])[0]!);
    expect(output).toEqual([0, 1, 2, 0]);   // NOT [NaN, NaN, NaN, NaN]
    expect(checkMatch({ fn, host, bindings, output, dims: [4], precision: 'f32', sampleCount: 4 }).ok).toBe(true);
  });
  it('mod with a NEGATIVE dividend follows the DIVISOR sign (floored), matching the interpreter', () => {
    // a[i] - 2 = [-2,-1,0,1]; mod(_, 3) floored = [1, 2, 0, 1] (JS % would give [-2,-1,0,1]).
    const { fn, host } = kernelOf('const a = f32(4, (i) => i)\ncomponent k(i) { return mod(a[i] - 2, 3) }\nk');
    const { bindings } = gateKernel(fn, host);
    const cpu = emitCpu(fn, bindings, host);
    const output = [0, 1, 2, 3].map((i) => cpu([i])[0]!);
    expect(output).toEqual([1, 2, 0, 1]);
    expect(checkMatch({ fn, host, bindings, output, dims: [4], precision: 'f32', sampleCount: 4 }).ok).toBe(true);
  });
  it('asinh matches the interpreter (Math.asinh), CPU-emit == oracle', () => {
    const { fn, host } = kernelOf('const a = f32(4, (i) => i)\ncomponent k(i) { return asinh(a[i]) }\nk');
    const { bindings } = gateKernel(fn, host);
    const cpu = emitCpu(fn, bindings, host);
    expect(cpu([2])[0]).toBeCloseTo(Math.asinh(2), 6);
    const output = [0, 1, 2, 3].map((i) => cpu([i])[0]!);
    expect(checkMatch({ fn, host, bindings, output, dims: [4], precision: 'f32', sampleCount: 4 }).ok).toBe(true);
  });
  it('acosh/atanh return NaN out-of-domain (no fail-loud guard) — CPU-emit == interpreter', () => {
    // acosh domain is x>=1: at i=0 (x=0) the interpreter returns NaN (a raw Math.acosh), and so must the CPU.
    const { fn, host } = kernelOf('const a = f32(4, (i) => i)\ncomponent k(i) { return acosh(a[i]) }\nk');
    const { bindings } = gateKernel(fn, host);
    const cpu = emitCpu(fn, bindings, host);
    expect(Number.isNaN(cpu([0])[0]!)).toBe(true);        // acosh(0) → NaN (out of domain, no guard)
    expect(cpu([2])[0]).toBeCloseTo(Math.acosh(2), 6);    // acosh(2) is in-domain
  });
  it('countOneBits is the population count over x>>>0, CPU-emit == interpreter oracle', () => {
    // a[i] = 2*i+1 = [1, 3, 5, 7] → popcounts [1, 2, 2, 3].
    const { fn, host } = kernelOf('const a = f32(4, (i) => 2*i + 1)\ncomponent k(i) { return countOneBits(a[i]) }\nk');
    const { bindings } = gateKernel(fn, host);
    const cpu = emitCpu(fn, bindings, host);
    const output = [0, 1, 2, 3].map((i) => cpu([i])[0]!);
    expect(output).toEqual([1, 2, 2, 3]);
    expect(checkMatch({ fn, host, bindings, output, dims: [4], precision: 'f32', sampleCount: 4 }).ok).toBe(true);
  });
  it('reverseBits reverses the 32-bit pattern (unsigned), CPU-emit == interpreter oracle — incl. a large output', () => {
    // a[i] = [1, 2, 3, 4] → reversed = [2^31, 2^30, 0xC0000000, 2^29]. 0xC0000000 (3221225472) is NOT f32-exact
    // as a raw integer, but the oracle frounds BOTH sides, and reversal preserves the f32-exact bit-span, so the
    // CPU cell (== interpreter, both f64) frounds to the same f32 as the shader would — the match is exact.
    const { fn, host } = kernelOf('const a = f32(4, (i) => i + 1)\ncomponent k(i) { return reverseBits(a[i]) }\nk');
    const { bindings } = gateKernel(fn, host);
    const cpu = emitCpu(fn, bindings, host);
    const output = [0, 1, 2, 3].map((i) => cpu([i])[0]!);
    expect(output).toEqual([2 ** 31, 2 ** 30, 0xC0000000, 2 ** 29]);
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
