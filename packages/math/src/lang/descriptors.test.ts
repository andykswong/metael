import { describe, it, expect } from 'vitest';
import { makeVec, makeMat, vecStoreOf, MATH_BUILTINS } from './index.ts';
import { descriptorOf, evaluateProgram, PlainStorageHost, RecordingHostEnv } from '@metael/lang';
const run = (src: string) => evaluateProgram(src, { host: new PlainStorageHost(), env: new RecordingHostEnv(), builtins: [MATH_BUILTINS] }).value;
const runFull = (src: string) => evaluateProgram(src, { host: new PlainStorageHost(), env: new RecordingHostEnv(), builtins: [MATH_BUILTINS] });
const hasBadArg = (src: string) => runFull(src).diagnostics.some((d) => d.code === 'ML-LANG-BUILTIN-ARG');
describe('vec/mat instances (moved) + f64 propagation', () => {
  it('makeVec f32 rounds; f64 is exact', () => {
    expect(vecStoreOf(makeVec([0.1, 0.2])).c[0]).toBe(Math.fround(0.1));         // default f32
    expect(vecStoreOf(makeVec([0.1, 0.2], 'f64')).c[0]).toBe(0.1);              // exact
  });
  it('f64 SURVIVES an operator chain (no silent downcast)', () => {
    const a = makeVec([0.1, 0.2], 'f64'); const b = makeVec([0.1, 0.2], 'f64');
    // the vec descriptor's binary '+' must produce an f64 result, not silently fround
    const sum = descriptorOf(a)!.binary!('+', a, b);
    expect(vecStoreOf(sum).c[0]).toBe(0.2);   // exact — NOT Math.fround(0.2)
  });
  it('vec componentwise + - * / stay byte-identical after dropping the redundant Array.from copies', () => {
    // Fix: the vec `binary` handler no longer copies its operand stores (core add/sub/mul/div never mutate +
    // return a fresh array). Behavior must be unchanged, and the inputs must not be mutated in place.
    const a = makeVec([1, 2, 3]); const b = makeVec([4, 5, 6]);
    const add = descriptorOf(a)!.binary!('+', a, b);
    const sub = descriptorOf(a)!.binary!('-', b, a);
    const mul = descriptorOf(a)!.binary!('*', a, b);
    const div = descriptorOf(a)!.binary!('/', b, a);
    expect(Array.from(vecStoreOf(add).c)).toEqual([5, 7, 9]);
    expect(Array.from(vecStoreOf(sub).c)).toEqual([3, 3, 3]);
    expect(Array.from(vecStoreOf(mul).c)).toEqual([4, 10, 18]);
    expect(Array.from(vecStoreOf(div).c)).toEqual([4, 2.5, 2]);
    // the operands are untouched (no in-place mutation from passing the stores straight through)
    expect(Array.from(vecStoreOf(a).c)).toEqual([1, 2, 3]);
    expect(Array.from(vecStoreOf(b).c)).toEqual([4, 5, 6]);
  });
  it('vec ops evaluated through the interpreter still fold correctly (end-to-end)', () => {
    expect(run('(vec3(1, 2, 3) + vec3(4, 5, 6)).y')).toBe(7);
    expect(run('(vec3(4, 5, 6) - vec3(1, 2, 3)).z')).toBe(3);
    expect(run('(vec2(2, 3) * vec2(4, 5)).x')).toBe(8);
  });
  it('a numeric builtin resolves via MATH_BUILTINS injection', () => { expect(run('dot(vec2(1,2), vec2(3,4))')).toBe(11); });
  it('descriptorOf reports matMxN for a mat', () => { expect(descriptorOf(makeMat([1,0,0,1], 2, 2))?.lower?.shape).toBe('matMxN'); });

  // GLSL/WGSL structured constructors + column access:
  it('vecN composition flattens vec + scalar args in order', () => {
    expect(run('vec3(vec2(1,2), 3).z')).toBe(3);
    expect(run('vec4(vec2(1,2), vec2(3,4)).w')).toBe(4);
    expect(run('vec4(vec3(1,2,3), 4).x')).toBe(1);
    expect(run('vec3(5).y')).toBe(5);   // splat retained
  });
  it('vecN composition rejects a wrong total width', () => {
    const res = evaluateProgram('vec3(vec2(1,2))', { host: new PlainStorageHost(), env: new RecordingHostEnv(), builtins: [MATH_BUILTINS] });
    expect(res.diagnostics.some((d) => d.code === 'ML-LANG-BUILTIN-ARG')).toBe(true);
  });
  it('matMxN builds from column vectors (column-major)', () => {
    // mat2(col0, col1) with col0=(1,2), col1=(3,4) → flat [1,2,3,4]; m[0] is column 0
    expect(run('mat2(vec2(1,2), vec2(3,4))[0].y')).toBe(2);
  });
  it('m[i] returns the i-th column as a vec; composes with swizzle', () => {
    // mat4 with translation column (10,11,12) → m[3].xyz == vec3(10,11,12)
    expect(run('mat4(1,0,0,0, 0,1,0,0, 0,0,1,0, 10,11,12,1)[3].xyz.x')).toBe(10);
    // the mat3-of-mat4 idiom that replaces mat3FromMat4:
    expect(run('mat3(mat4(1,0,0,0, 0,1,0,0, 0,0,1,0, 10,11,12,1)[0].xyz, mat4(1,0,0,0, 0,1,0,0, 0,0,1,0, 10,11,12,1)[1].xyz, mat4(1,0,0,0, 0,1,0,0, 0,0,1,0, 10,11,12,1)[2].xyz)[0].x')).toBe(1);
  });

  // The geometric/quaternion scalar slots (refract eta, qaxisangle angle, qslerp t) reject a
  // non-number STRICTLY — a string is NOT coerced; it fails loud with ML-LANG-BUILTIN-ARG and a frozen [].
  it('geometric/quat scalar slots reject a non-number (strict, not coerced)', () => {
    expect(hasBadArg('refract(vec3(1,0,0), vec3(0,1,0), "0.5")')).toBe(true);
    expect(hasBadArg('qaxisangle(vec3(0,0,1), "1.5")')).toBe(true);
    expect(hasBadArg('qslerp(vec4(0,0,0,1), vec4(0,0,0,1), "0.5")')).toBe(true);
  });
  it('geometric/quat scalar slots accept a real number (no over-tightening)', () => {
    expect(hasBadArg('refract(vec3(1,0,0), vec3(0,1,0), 0.5)')).toBe(false);
    expect(descriptorOf(run('refract(vec3(1,0,0), vec3(0,1,0), 0.5)'))?.lower?.shape).toBe('vecN');
  });
});
