// packages/lang/src/collections-iterable.test.ts
import { describe, it, expect } from 'vitest';
import { evaluateProgram, PlainStorageHost, RecordingHostEnv } from './index.ts';

function run(src: string) {
  return evaluateProgram(src, { host: new PlainStorageHost(), env: new RecordingHostEnv() });
}

describe('collection builtins accept a typed array (any iterate-able), not just a plain array', () => {
  it('map / slice / join / reduce / filter / includes over an f32 buffer', () => {
    expect(run('map(f32([1, 2, 3]), (x) => x * 2)').value).toEqual([2, 4, 6]);
    expect(run('slice(f32([0, 1, 2, 3]), 1, 3)').value).toEqual([1, 2]);
    expect(run('join(f32([1, 2, 3]), ",")').value).toBe('1,2,3');
    expect(run('reduce(f32([1, 2, 3]), (a, x) => a + x, 0)').value).toBe(6);
    expect(run('filter(f32([1, 2, 3, 4]), (x) => x > 2)').value).toEqual([3, 4]);
    expect(run('includes(f32([1, 2, 3]), 2)').value).toBe(true);
    expect(run('reverse(f32([1, 2, 3]))').value).toEqual([3, 2, 1]);
    expect(run('some(f32([1, 2, 3]), (x) => x > 2)').value).toBe(true);
    expect(run('every(f32([1, 2, 3]), (x) => x > 0)').value).toBe(true);
    expect(run('findIndex(f32([1, 2, 3]), (x) => x == 2)').value).toBe(1);
  });
  it('a NON-iterable arg still errors (an object has no iterate)', () => {
    const r = run('map({ a: 1 }, (x) => x)');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-BUILTIN-ARG')).toBe(true);
  });
  it('the result is a plain (frozen) array, not a typed array — element identity is preserved', () => {
    // map over an f32 yields JS numbers; the caller can rebuild an f32 with f32([...]) if desired.
    expect(run('map(f32([1, 2]), (x) => x + "!")').value).toEqual(['1!', '2!']);
  });

  it('a collection builtin over a typed array SUBSCRIBES to its generation (parity with for-of)', () => {
    // A whole-buffer read through a collection builtin must register the buffer's generation dependency
    // — exactly like for-of, index, and string-concat — so a reactive context re-runs on an in-place
    // write. We prove the subscription is even ATTEMPTED by counting readGeneration calls on the same
    // host that minted the buffer's generation. The pre-fix helper (iterate-only, no readGeneration)
    // leaves this at 0; the fixed helper mirrors the for-of path and calls it.
    const host = new PlainStorageHost();
    let reads = 0;
    const orig = host.readGeneration.bind(host);
    host.readGeneration = (g) => { reads++; return orig(g); };
    // Reads the buffer ONLY via a collection builtin (reduce) — the sole generation-registration path.
    evaluateProgram('const b = f32([1, 2, 3])\nreduce(b, (a, x) => a + x, 0)', { host, env: new RecordingHostEnv() });
    expect(reads).toBeGreaterThan(0);
    // Baseline: for-of over the same buffer is the established subscription path — it also reads.
    // A top-level for-of executes immediately (a component decl would not run until invoked).
    const host2 = new PlainStorageHost();
    let forReads = 0;
    const orig2 = host2.readGeneration.bind(host2);
    host2.readGeneration = (g) => { forReads++; return orig2(g); };
    evaluateProgram('const b = f32([1, 2, 3])\nfor (x of b) { x }', { host: host2, env: new RecordingHostEnv() });
    expect(forReads).toBeGreaterThan(0);
  });

  it('a plain string is NOT silently treated as a char array (no descriptor → errors)', () => {
    const r = evaluateProgram('map("abc", (x) => x)', { host: new PlainStorageHost(), env: new RecordingHostEnv() });
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-BUILTIN-ARG')).toBe(true);
  });
});
