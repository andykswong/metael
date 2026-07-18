// packages/gpu/src/handle.test.ts
import { describe, it, expect } from 'vitest';
import { descriptorOf, isTypedArray } from '@metael/lang';
import { makeGpuBufferHandle, residentInfo, disposeHandle } from './handle.ts';
import { evaluateProgram, PlainStorageHost, type HostEnvironment, type Arg, type HostValue, type SourceSpan } from '@metael/lang';

describe('GpuBufferHandle — a resident-buffer custom value', () => {
  it('presents as a linear-buffer custom value (length + lazy readback index + iterate)', () => {
    let reads = 0;
    const handle = makeGpuBufferHandle({
      backendKind: 'webgpu', length: 4,
      readback: () => { reads++; return new Float32Array([10, 20, 30, 40]); },
    });
    expect(isTypedArray(handle)).toBe(true);              // classifies as a buffer input via descriptorOf.lower
    const d = descriptorOf(handle)!;
    expect(d.getMember!(handle, 'length')).toBe(4);
    expect(d.getIndex!(handle, 2)).toBe(30);              // triggers a lazy readback
    expect(d.getIndex!(handle, 0)).toBe(10);              // cached — no second readback
    expect(reads).toBe(1);
    expect(Array.from(d.iterate!(handle) as number[])).toEqual([10, 20, 30, 40]);
  });
  it('exposes its resident identity (backendKind + a device-buffer accessor) for same-backend binding', () => {
    const gpuBuf = { __fake: true };
    const handle = makeGpuBufferHandle({ backendKind: 'webgpu', length: 2, gpuBuffer: gpuBuf, readback: () => new Float32Array([1, 2]) });
    const info = residentInfo(handle);
    expect(info?.backendKind).toBe('webgpu');
    expect(info?.gpuBuffer).toBe(gpuBuf);
    expect(typeof info?.nonce).toBe('number');   // the memo change-signal (a per-handle monotonic nonce)
  });
  it('reading a disposed handle whose cache was never materialized surfaces a diagnostic, not a freed-buffer crash', () => {
    let freed = false;
    const handle = makeGpuBufferHandle({ backendKind: 'webgpu', length: 3, readback: () => new Float32Array([1, 2, 3]), dispose: () => { freed = true; } });
    disposeHandle(handle);
    expect(freed).toBe(true);
    const d = descriptorOf(handle)!;
    // Cache was never filled → a read after dispose must THROW BufferError (caught+mapped by the interpreter), not call readback on a freed buffer.
    expect(() => d.getIndex!(handle, 0)).toThrow();  // throws BufferError('MLGPU-USE-AFTER-DISPOSE')
  });
  it('a handle whose cache was materialized BEFORE dispose still reads (dispose frees only the GPU buffer, not the CPU cache)', () => {
    const handle = makeGpuBufferHandle({ backendKind: 'webgpu', length: 3, readback: () => new Float32Array([1, 2, 3]) });
    const d = descriptorOf(handle)!;
    expect(d.getIndex!(handle, 0)).toBe(1);  // materializes the cache
    disposeHandle(handle);
    expect(d.getIndex!(handle, 2)).toBe(3);  // still reads from the cached array — no readback, no throw
  });
  it('a readback shorter than the declared length surfaces a diagnostic (not silent undefined-as-number)', () => {
    const handle = makeGpuBufferHandle({ backendKind: 'webgpu', length: 4, readback: () => new Float32Array([1, 2]) });  // short!
    const d = descriptorOf(handle)!;
    expect(() => d.getIndex!(handle, 3)).toThrow();  // MLGPU-READBACK-SHORT, not undefined
  });
  it('a disposed-handle read, through the interpreter, becomes a diagnostic (not an uncaught throw)', () => {
    const handle = makeGpuBufferHandle({ backendKind: 'cpu', length: 3, readback: () => new Float32Array([1, 2, 3]) });
    disposeHandle(handle);
    const env: HostEnvironment = {
      resolveCall(head: string, _key: string, _args: Arg[], _children: HostValue[], _span: SourceSpan) {
        return head === 'resident' ? { handled: true as const, value: handle, kind: 'value' as const } : { handled: false as const };
      },
    };
    const res = evaluateProgram('const h = resident()\nh[0]', { host: new PlainStorageHost(), env });
    expect(res.diagnostics.some((d) => d.code === 'MLGPU-USE-AFTER-DISPOSE')).toBe(true);
  });
  it('an OOB index on a handle, through the interpreter, becomes ML-LANG-INDEX-RANGE (not an uncaught throw)', () => {
    const handle = makeGpuBufferHandle({ backendKind: 'cpu', length: 3, readback: () => new Float32Array([1, 2, 3]) });
    // A minimal env: the `resident` head returns the handle as a pure value; everything else declines.
    const env: HostEnvironment = {
      resolveCall(head: string, _key: string, _args: Arg[], _children: HostValue[], _span: SourceSpan) {
        return head === 'resident' ? { handled: true as const, value: handle, kind: 'value' as const } : { handled: false as const };
      },
    };
    // `const h = resident()` binds the handle; `h[9]` is OOB (length 3) → the descriptor throws BufferError,
    // the interpreter's `instanceof BufferError` catch maps it to a diagnostic + a null value (never a throw).
    const res = evaluateProgram('const h = resident()\nh[9]', { host: new PlainStorageHost(), env });
    expect(res.diagnostics.some((d) => d.code === 'ML-LANG-INDEX-RANGE')).toBe(true);
  });
});
