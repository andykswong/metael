// packages/lang/src/buffer-view.test.ts
import { describe, it, expect } from 'vitest';
import { evaluateProgram, PlainStorageHost, RecordingHostEnv, descriptorOf } from './index.ts';
import { makeTypedArray, BUFFER_KINDS, MATH_BUILTINS } from '@metael/math/lang';

function buf(src: string): unknown {
  return evaluateProgram(src, { host: new PlainStorageHost(), env: new RecordingHostEnv(), builtins: [MATH_BUILTINS] }).value;
}

describe('typed-array bufferView — the zero-copy raw-store seam', () => {
  it('exposes the backing TypedArray directly (f32 → the SAME Float32Array, no copy)', () => {
    const v = buf('f32([1, 2, 3])');
    const view = descriptorOf(v)!.bufferView!(v);
    expect(view.element).toBe('f32');
    expect(view.data).toBeInstanceOf(Float32Array);
    expect(Array.from(view.data as Float32Array)).toEqual([1, 2, 3]);
  });
  it('reports the true element kind for i32/u32/f64', () => {
    expect(descriptorOf(buf('i32([1, 2])'))!.bufferView!(buf('i32([1, 2])')).element).toBe('i32');
    expect(descriptorOf(buf('f64([1, 2])'))!.bufferView!(buf('f64([1, 2])')).element).toBe('f64');
  });
  it('BUFFER_KINDS + makeTypedArray are exported (a consumer can build an f32 handle)', () => {
    const host = new PlainStorageHost();
    const store = Float32Array.from([5, 6, 7]);
    const handle = makeTypedArray('f32', store, host.allocateGeneration());
    const view = descriptorOf(handle)!.bufferView!(handle);
    expect(view.data).toBe(store);   // zero-copy: the SAME Float32Array is the store
    expect(BUFFER_KINDS.f32.element).toBe('f32');
  });
});
