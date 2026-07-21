// packages/gpu/src/buffer.test.ts
import { describe, it, expect } from 'vitest';
import { RuntimeReactiveHost } from '@metael/runtime';
import { descriptorOf } from '@metael/lang';
import { gpuBuffer } from './buffer.ts';

describe('gpuBuffer', () => {
  it('wraps a plain array into an iterable f32 typed-array custom value', () => {
    const buf = gpuBuffer([1, 2, 3, 4], new RuntimeReactiveHost());
    const desc = descriptorOf(buf);
    expect(desc?.iterate).toBeTypeOf('function');                 // satisfies gpuReduce/gpuHistogram's input gate
    expect(Array.from(desc!.iterate!(buf), (v) => Number(v))).toEqual([1, 2, 3, 4]);
  });

  it('accepts a Float32Array too', () => {
    const buf = gpuBuffer(new Float32Array([5, 6]), new RuntimeReactiveHost());
    expect(Array.from(descriptorOf(buf)!.iterate!(buf), Number)).toEqual([5, 6]);
  });
});
