import { describe, it, expect } from 'vitest';
import { evaluateProgram, PlainStorageHost, RecordingHostEnv, makeSeededRng } from '@metael/lang';
import { STD_BUILTINS } from './index.ts';

const run = (src: string, seed = 0) =>
  evaluateProgram(src, { host: new PlainStorageHost(), env: new RecordingHostEnv(), seed, builtins: [STD_BUILTINS] });

describe('std random builtin', () => {
  it('rand() is deterministic for a fixed seed and matches the seeded PRNG', () => {
    const a = run('rand()', 42).value;
    const b = run('rand()', 42).value;
    const expected = makeSeededRng(42)();
    expect(a).toBe(expected);
    expect(b).toBe(expected);
  });

  it('rand() advances one shared per-run sequence (not re-seeded per call)', () => {
    const res = run('[rand(), rand(), rand()]', 42);
    const rng = makeSeededRng(42);
    expect(res.value).toEqual([rng(), rng(), rng()]);   // three SUCCESSIVE draws
  });

  it('different seeds → different rand() values', () => {
    expect(run('rand()', 1).value).not.toBe(run('rand()', 2).value);
  });

  it('rand() is budget-charged: an unbounded loop trips ML-LANG-BUDGET (does not hang)', () => {
    const res = evaluateProgram('while (true) { rand() }', {
      host: new PlainStorageHost(), env: new RecordingHostEnv(), seed: 0, maxSteps: 1000, builtins: [STD_BUILTINS],
    });
    expect(res.diagnostics.some((d) => d.code === 'ML-LANG-BUDGET')).toBe(true);
  });

  it('a user function named rand shadows the builtin (unbound-head-only)', () => {
    expect(run('function rand() { 99 } rand()', 3).value).toBe(99);
  });
});
