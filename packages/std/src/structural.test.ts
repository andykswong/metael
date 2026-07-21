import { describe, it, expect } from 'vitest';
import { evaluateProgram, PlainStorageHost, RecordingHostEnv } from '@metael/lang';
import { STD_BUILTINS } from './index.ts';

const run = (src: string, data?: unknown) =>
  evaluateProgram(src, { data, host: new PlainStorageHost(), env: new RecordingHostEnv(), builtins: [STD_BUILTINS] });

describe('std structural builtins', () => {
  it('keys / values / entries decompose an object', () => {
    expect(run('keys({ a: 1, b: 2 })').value).toEqual(['a', 'b']);
    expect(run('values({ a: 1, b: 2 })').value).toEqual([1, 2]);
    expect(run('entries({ a: 1, b: 2 })').value).toEqual([['a', 1], ['b', 2]]);
  });
  it('keys / values / entries of a non-object are fail-loud ML-LANG-BUILTIN-ARG', () => {
    expect(run('keys(5)').diagnostics.some((d) => d.code === 'ML-LANG-BUILTIN-ARG')).toBe(true);
    expect(run('values([1, 2])').diagnostics.some((d) => d.code === 'ML-LANG-BUILTIN-ARG')).toBe(true);
  });

  // --- RENAMED: object (was fromEntries) ---
  it('object builds an object from [key, value] pairs', () => {
    expect(run('object([["a", 1], ["b", 2]])').value).toEqual({ a: 1, b: 2 });
  });
  it('object round-trips: object(entries(o)) == o', () => {
    expect(run('const o = { a: 1, b: 2 }; object(entries(o))').value).toEqual({ a: 1, b: 2 });
  });
  it('object ignores a forbidden key (FORBIDDEN_KEYS-guarded)', () => {
    const r = run('object([["__proto__", 1], ["a", 2]])');
    expect((r.value as Record<string, unknown>).a).toBe(2);
    expect(Object.getOwnPropertyNames(r.value as object)).not.toContain('__proto__');
  });
  it('object of a non-array is fail-loud ML-LANG-BUILTIN-ARG', () => {
    expect(run('object(5)').diagnostics.some((d) => d.code === 'ML-LANG-BUILTIN-ARG')).toBe(true);
  });
  it('object builds a NULL-PROTOTYPE record (no inherited toString/constructor)', () => {
    const rec = run('object([["a", 1]])').value as Record<string, unknown>;
    expect(rec.a).toBe(1);
    expect(Object.getPrototypeOf(rec)).toBe(null);
    expect('toString' in rec).toBe(false);
    expect(rec.constructor).toBe(undefined);
  });
  it('fromEntries is no longer a builtin — only `object` is registered', () => {
    const r = run('fromEntries([["a", 1]])');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-UNKNOWN-CALL')).toBe(true);
  });

  // --- NEW: has ---
  it('has reports own-property presence', () => {
    expect(run('has({ a: 1, b: 2 }, "a")').value).toBe(true);
    expect(run('has({ a: 1 }, "z")').value).toBe(false);
  });
  it('has is false for a forbidden key (never walks the prototype chain)', () => {
    expect(run('has({ a: 1 }, "__proto__")').value).toBe(false);
    expect(run('has({ a: 1 }, "constructor")').value).toBe(false);
  });
  it('has of a non-object is fail-loud ML-LANG-BUILTIN-ARG', () => {
    expect(run('has(5, "a")').diagnostics.some((d) => d.code === 'ML-LANG-BUILTIN-ARG')).toBe(true);
    expect(run('has([1, 2], "0")').diagnostics.some((d) => d.code === 'ML-LANG-BUILTIN-ARG')).toBe(true);
  });
});
