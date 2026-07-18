import { describe, it, expect } from 'vitest';
import { evaluateProgram, isUserFn } from '@metael/lang';
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

describe('sampled differential oracle', () => {
  it('CPU-emit ≡ interpreter (bit-identical) for an exact kernel', () => {
    const { fn, host } = kernelOf(`
      const N = 8
      const x = f32(N, (i) => i * 2)
      function k(i) { return x[i] + 1 }
      k`);
    const { bindings } = gateKernel(fn, host);
    const cpu = emitCpu(fn, bindings, host);
    const output = Array.from({ length: 8 }, (_, i) => cpu([i])[0]!);
    const verdict = checkMatch({ fn, host, bindings, output, dims: [8], precision: 'f32', sampleCount: 8 });
    expect(verdict.ok).toBe(true);
    expect(verdict.kind).toBe('exact');
    expect(verdict.maxUlp).toBe(0);
  });
  it('flags a deliberately-wrong output as a mismatch', () => {
    const { fn, host } = kernelOf(`const x = f32(8, (i) => i)\nfunction k(i) { return x[i] }\nk`);
    const { bindings } = gateKernel(fn, host);
    const wrong = Array.from({ length: 8 }, (_, i) => i + 1000);
    const verdict = checkMatch({ fn, host, bindings, output: wrong, dims: [8], precision: 'f32', sampleCount: 8 });
    expect(verdict.ok).toBe(false);
  });
  it('CPU-emit ≡ interpreter for a DIVIDING kernel (div-by-zero parity: null≡null, never Infinity)', () => {
    const { fn, host } = kernelOf(`const x = f32(8, (i) => i)\nfunction k(i) { return 10 / x[i] }\nk`);
    const { bindings } = gateKernel(fn, host);
    const cpu = emitCpu(fn, bindings, host);
    const output = Array.from({ length: 8 }, (_, i) => cpu([i])[0]!);
    expect(Number.isFinite(output[0]) || output[0] === 0 || Number.isNaN(output[0])).toBe(true);
    const verdict = checkMatch({ fn, host, bindings, output, dims: [8], precision: 'f32', sampleCount: 8 });
    expect(verdict.ok).toBe(true);
  });
  it('a let-accumulator matmul kernel (component) is CPU-emit ≡ interpreter — not null/0', () => {
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
    const cpu = emitCpu(fn, bindings, host);
    const dims = [3, 3];
    const output = Array.from({ length: 9 }, (_, f) => cpu([Math.floor(f / 3), f % 3])[0]!);
    // a * identity = a, so output[row,col] = a[row*3+col] = the flat index
    expect(output).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8]);
    const verdict = checkMatch({ fn, host, bindings, output, dims, precision: 'f32', sampleCount: 9 });
    expect(verdict.ok).toBe(true);       // interpreter agrees at every sampled cell — NOT the null→0 garbage
    expect(verdict.maxUlp).toBe(0);
  });
  it('a large-N matmul verifies without exhausting the oracle budget (a FRESH per-cell budget, not aggregate)', () => {
    // A single makeCallable shared across all ~256 samples would run out of steps partway through this heavy
    // per-cell loop and throw mid-sweep → a spurious mismatch. A fresh callable per cell keeps each cell's
    // budget independent. N=64 → each cell runs a range(64) loop; 256 samples × that would blow one budget.
    const N = 64;
    const { fn, host } = kernelOf(`
      const N = ${N}
      const a = f32(N * N, (i) => i % 7)
      const b = f32(N * N, (i) => (i % (N + 1) == 0) ? 1 : 0)
      component product(row, col) {
        let sum = 0
        for (const k of range(N)) { sum = sum + a[row * N + k] * b[k * N + col] }
        return sum
      }
      product`);
    const { bindings } = gateKernel(fn, host);
    const cpu = emitCpu(fn, bindings, host);
    const output = Array.from({ length: N * N }, (_, f) => cpu([Math.floor(f / N), f % N])[0]!);
    const verdict = checkMatch({ fn, host, bindings, output, dims: [N, N], precision: 'f32', sampleCount: 256 });
    expect(verdict.ok).toBe(true);   // every sampled cell verified — the per-cell budget did not run out
  });
  it('does NOT rubber-stamp ok:true when zero cells are sampled (a degenerate/empty output)', () => {
    // If the output shape is degenerate (total 0), the sampling loop runs zero iterations. That is NOT
    // "verified correct" — the verdict must fail rather than default to ok:true/maxUlp:0.
    const { fn, host } = kernelOf(`const x = f32(8, (i) => i)\nfunction k(i) { return x[i] }\nk`);
    const { bindings } = gateKernel(fn, host);
    const verdict = checkMatch({ fn, host, bindings, output: [], dims: [0], precision: 'f32', sampleCount: 8 });
    expect(verdict.ok).toBe(false);
  });
});
