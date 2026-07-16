import { describe, it, expect } from 'vitest';
import { evaluateProgram } from './evaluate.ts';
import { PlainStorageHost, RecordingHostEnv } from './ports.ts';
import { isTypedArray, descriptorOf } from './custom-types.ts';

const run = (src: string, opts?: { insideComponent?: boolean; maxSteps?: number }) =>
  evaluateProgram(src, { host: new PlainStorageHost(), env: new RecordingHostEnv(), ...opts });

describe('typed-array constructors', () => {
  it('f32(n) allocates a zeroed buffer with .length', () => {
    expect(run('const a = f32(3); a.length').value).toBe(3);
    expect(run('const a = f32(3); a[0]').value).toBe(0);
  });
  it('f32([…]) builds from a literal', () => {
    expect(run('const a = f32([1, 2, 3]); a[1]').value).toBe(2);
  });
  it('f32(n, gen) fills via the generator, budgeted per element', () => {
    expect(run('const a = f32(4, (i) => i * 10); a[3]').value).toBe(30);
  });
  it('i32 / u32 / f64 all construct', () => {
    expect(run('i32([1, 2]).length').value).toBe(2);
    expect(run('u32([7]).length').value).toBe(1);
    expect(run('f64([1.5]).length').value).toBe(1);
  });
  it('a bad constructor arg is fail-loud ML-LANG-BUILTIN-ARG', () => {
    const r = run('f32("nope")');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-BUILTIN-ARG')).toBe(true);
  });
  it('an over-cap length is ML-LANG-BUDGET (never allocates)', () => {
    const r = run('f32(999999999)');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-BUDGET')).toBe(true);
  });
});

describe('typed-array accessors', () => {
  it('OOB read is fail-loud ML-LANG-INDEX-RANGE (fixes the silent-null bug)', () => {
    const r = run('const a = f32([1, 2]); a[5]');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-INDEX-RANGE')).toBe(true);
  });
  it('an unknown member is ML-LANG-UNKNOWN-MEMBER', () => {
    const r = run('const a = f32([1]); a.nope');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-UNKNOWN-MEMBER')).toBe(true);
  });
  it('for-of sums a typed array (component let)', () => {
    const r = run('let total = 0; for (const x of f32([1, 2, 3])) { total = total + x } total', { insideComponent: true });
    expect(r.value).toBe(6);
  });
  it('display is bounded, never the raw store; a short buffer shows no length suffix', () => {
    expect(run('"" + f32([1, 2, 3])').value).toBe('f32[1, 2, 3]');
  });
});

describe('typed-array mutation + coercion', () => {
  it('a let typed array is writable in place', () => {
    const r = run('let a = f32([1, 2, 3]); a[0] = 9; a[0]', { insideComponent: true });
    expect(r.value).toBe(9);
  });
  it('a const typed array is frozen — an index write is ML-LANG-IMMUTABLE', () => {
    const r = run('const a = f32([1, 2, 3]); a[0] = 9; a[0]');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-IMMUTABLE')).toBe(true);
    expect(r.value).toBe(1);
  });
  it('i32 coercion truncates + wraps mod 2^32', () => {
    const r = run('let a = i32([0]); a[0] = 4294967298; a[0]', { insideComponent: true });
    expect(r.value).toBe(2);
  });
  it('f32 coercion applies Math.fround', () => {
    const r = run('let a = f32([0]); a[0] = 1.1; a[0]', { insideComponent: true });
    expect(r.value).toBeCloseTo(Math.fround(1.1), 6);
  });
  it('a non-number write is ML-LANG-BUILTIN-ARG', () => {
    const r = run('let a = f32([0]); a[0] = "x"; a[0]', { insideComponent: true });
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-BUILTIN-ARG')).toBe(true);
  });
  it('buf + buf is a genuine "not defined" → ML-LANG-OP-UNSUPPORTED', () => {
    const r = run('f32([1]) + f32([2])');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-OP-UNSUPPORTED')).toBe(true);
  });
});

describe('typed-array lowering + helpers', () => {
  it('isTypedArray recognises a buffer; a plain array is not one', () => {
    expect(isTypedArray(evaluateProgram('f32([1])', { host: new PlainStorageHost(), env: new RecordingHostEnv() }).value)).toBe(true);
  });
  it('the descriptor exposes a linear-buffer lowering', () => {
    const buf = evaluateProgram('f32([1, 2])', { host: new PlainStorageHost(), env: new RecordingHostEnv() }).value;
    const low = descriptorOf(buf)?.lower;
    expect(low?.access).toBe('linear-buffer');
    expect(low?.element).toBe('f32');
    expect(low?.gpuStorable).toBe(true);
  });
  it('f64 is NOT gpu-storable', () => {
    const buf = evaluateProgram('f64([1])', { host: new PlainStorageHost(), env: new RecordingHostEnv() }).value;
    expect(descriptorOf(buf)?.lower?.gpuStorable).toBe(false);
  });
});

describe('typed-array security — the Symbol-hidden store never leaks', () => {
  it('keys/values/entries expose no own enumerable properties', () => {
    expect(run('keys(f32([1, 2, 3]))').value).toEqual([]);
    expect(run('values(f32([1, 2, 3]))').value).toEqual([]);
    expect(run('entries(f32([1, 2, 3]))').value).toEqual([]);
  });
  it('object-spread of a buffer carries no fields (store/descriptor/frozen are non-enumerable)', () => {
    expect(run('keys({ ...f32([1, 2, 3]) })').value).toEqual([]);
  });
  it('display shows a bounded head, never the backing store, and truncates a long buffer', () => {
    // > 8 elements → head of 8 + an ellipsis marker + the length (shown ONLY when abbreviated).
    const s = run('"" + f32([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])').value as string;
    expect(s.startsWith('f32[')).toBe(true);
    expect(s).toContain('… (len 10)');
  });
});

describe('typed-array constructor arg edges', () => {
  it('the MAX_BUFFER_LENGTH boundary: at the cap constructs, one over is ML-LANG-BUDGET', () => {
    // 16777216 = 2^24 = MAX_BUFFER_LENGTH (exported from evaluate.ts). A zeroed 64MB Float32Array
    // allocates fast; one-over must trip the budget guard BEFORE allocating.
    expect(run('f32(16777216).length').value).toBe(16777216);
    const over = run('f32(16777217)');
    expect(over.diagnostics.some((d) => d.code === 'ML-LANG-BUDGET')).toBe(true);
  });
  it('no-arg / negative / non-finite length is ML-LANG-BUILTIN-ARG', () => {
    expect(run('f32()').diagnostics.some((d) => d.code === 'ML-LANG-BUILTIN-ARG')).toBe(true);
    expect(run('f32(-1)').diagnostics.some((d) => d.code === 'ML-LANG-BUILTIN-ARG')).toBe(true);
  });
  it('a fractional length floors', () => {
    expect(run('f32(3.7).length').value).toBe(3);
  });
  it('a non-function generator arg is ML-LANG-BUILTIN-ARG', () => {
    expect(run('f32(2, 5)').diagnostics.some((d) => d.code === 'ML-LANG-BUILTIN-ARG')).toBe(true);
  });
  it('construction COERCES a non-number element silently (NaN for float, 0 for int) — intentional, unlike a fail-loud write', () => {
    // Distinct from an in-place write, which is fail-loud ML-LANG-BUILTIN-ARG. Construction mirrors a
    // TypedArray-from-array: a non-number becomes NaN (float) / 0 (int). This test PINS that intent.
    const f = run('const a = f32([1, "x", 3]); a[1]');
    expect(Number.isNaN(f.value as number)).toBe(true);
    expect(f.diagnostics.some((d) => d.code === 'ML-LANG-BUILTIN-ARG')).toBe(false);
    const i = run('const a = i32([1, "x", 3]); a[1]');
    expect(i.value).toBe(0);
  });
});

describe('typed-array accessor + coercion edges', () => {
  it('an OOB WRITE (>= length) is ML-LANG-INDEX-RANGE', () => {
    const r = run('let a = f32([1, 2, 3]); a[5] = 9; a[0]', { insideComponent: true });
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-INDEX-RANGE')).toBe(true);
  });
  it('a negative index read is ML-LANG-INDEX-RANGE', () => {
    const r = run('const a = f32([1, 2, 3]); a[0 - 1]');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-INDEX-RANGE')).toBe(true);
  });
  it('a non-integer index read is ML-LANG-INDEX-RANGE', () => {
    const r = run('const a = f32([1, 2, 3]); a[1.5]');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-INDEX-RANGE')).toBe(true);
  });
  it('for-of over an empty buffer runs zero iterations', () => {
    const r = run('let n = 0; for (const x of f32(0)) { n = n + 1 } n', { insideComponent: true });
    expect(r.value).toBe(0);
  });
  it('u32 coercion wraps a negative value to unsigned', () => {
    const r = run('let a = u32([0]); a[0] = 0 - 1; a[0]', { insideComponent: true });
    expect(r.value).toBe(4294967295);
  });
  it('f64 stores an exact double (no fround)', () => {
    const r = run('let a = f64([0]); a[0] = 1.1; a[0]', { insideComponent: true });
    expect(r.value).toBe(1.1);
  });
});

describe('const deep-immutability reaches a nested typed array (the frozen box, never Object.freeze)', () => {
  it('a buffer nested in a const object literal is frozen — an index write is ML-LANG-IMMUTABLE', () => {
    const r = run('const o = { buf: f32([1, 2, 3]) }; o.buf[0] = 9; o.buf[0]');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-IMMUTABLE')).toBe(true);
    expect(r.value).toBe(1);
  });
  it('a buffer nested in a const array literal is frozen — an index write is ML-LANG-IMMUTABLE', () => {
    const r = run('const xs = [f32([1, 2, 3])]; xs[0][0] = 9; xs[0][0]');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-IMMUTABLE')).toBe(true);
    expect(r.value).toBe(1);
  });
  it('reading a nested buffer still works + never Object.freeze-throws (no ML-LANG-INTERNAL)', () => {
    const r = run('const o = { buf: f32([1, 2, 3]) }; o.buf.length');
    expect(r.value).toBe(3);
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-INTERNAL')).toBe(false);
  });
  it('a TOP-LEVEL let typed array is STILL mutable in place (the intrinsic path bypasses deepFreeze)', () => {
    const r = run('let a = f32([1, 2, 3]); a[0] = 9; a[0]', { insideComponent: true });
    expect(r.value).toBe(9);   // guards against the fix over-freezing the real mutable path
  });
});
