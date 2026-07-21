import { describe, it, expect } from 'vitest';
import { evaluateProgram } from './evaluate.ts';
import { PlainStorageHost, RecordingHostEnv } from './ports.ts';
import { MATH_BUILTINS } from '@metael/math/lang';
import { STD_BUILTINS } from '@metael/std';

const run = (src: string, opts?: object) => evaluateProgram(src, { host: new PlainStorageHost(), env: new RecordingHostEnv(), builtins: [MATH_BUILTINS], ...opts });

describe('the protocol preserves the load-bearing invariants', () => {
  it('FAST PATH: scalar arithmetic is unaffected (no descriptor path taken)', () => {
    expect(run('1 + 2 * 3 - 4 / 2').value).toBe(5);
    expect(run('5 < 9 && 3 == 3').value).toBe(true);
    expect(run('!(0) && "" == ""').value).toBe(true);
  });
  it('FAST PATH microbench: a scalar-heavy loop stays within budget with headroom', () => {
    const r = run('let acc = 0; for (const i of range(10000)) { acc = acc + i } acc', { insideComponent: true });
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-BUDGET')).toBe(false);
    expect(r.value).toBe(49995000);   // range(10000) → sum 0..9999 = 10000*9999/2, well under the default step budget
  });
  it('DETERMINISM: same source + seed → identical result for a buffer-generator mix', () => {
    const src = 'const a = f32(3, (i) => rand() + i); a[0] + a[1] + a[2]';   // rand ← std, f32 ← math
    const a = evaluateProgram(src, { host: new PlainStorageHost(), env: new RecordingHostEnv(), seed: 42, builtins: [MATH_BUILTINS, STD_BUILTINS] });
    const b = evaluateProgram(src, { host: new PlainStorageHost(), env: new RecordingHostEnv(), seed: 42, builtins: [MATH_BUILTINS, STD_BUILTINS] });
    expect(a.value).toEqual(b.value);
    const c = evaluateProgram(src, { host: new PlainStorageHost(), env: new RecordingHostEnv(), seed: 43, builtins: [MATH_BUILTINS, STD_BUILTINS] });
    expect(a.value).not.toEqual(c.value);
  });
  it('NEVER-THROWS: an OOB typed-array read is a diagnostic, not a host throw', () => {
    const r = run('const a = f32([1]); a[99]');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-INDEX-RANGE')).toBe(true);
    expect(r.value).toBe(null);
  });
  it('NEVER-THROWS: a generator that recurses infinitely fails closed (ML-LANG-BUDGET)', () => {
    const src = 'function loop(x) { loop(x) } f32(2, (i) => loop(i))';
    const r = evaluateProgram(src, { host: new PlainStorageHost(), env: new RecordingHostEnv(), maxDepth: 20, builtins: [MATH_BUILTINS] });
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-BUDGET')).toBe(true);
  });
  it('deepFreeze does NOT throw on a typed array nested in an object/array literal', () => {
    const r = run('const o = { buf: f32([1, 2, 3]), xs: [f32([4])] }; o.buf.length');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-INTERNAL')).toBe(false);
    expect(r.value).toBe(3);
  });
  it('a vec nested in an object literal is fine (deepFreeze opaque-leaf exemption)', () => {
    const r = run('const o = { v: vec3(1,2,3) }; o.v.x');
    expect(r.value).toBe(1);
  });
});
