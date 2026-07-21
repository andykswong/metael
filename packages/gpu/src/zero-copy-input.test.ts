// packages/gpu/src/zero-copy-input.test.ts
import { describe, it, expect } from 'vitest';
import { RuntimeReactiveHost, change } from '@metael/runtime';
import { evaluateProgram, isUserFn, RecordingHostEnv } from '@metael/lang';
import { makeTypedArray, MATH_BUILTINS } from '@metael/math/lang';
import type { UserFn, HostEnvironment, Arg, HostValue, SourceSpan } from '@metael/lang';
import { GpuEngine } from './resource.ts';
import type { Backend, DispatchInput } from './device/index.ts';

function kernelOf(src: string, host: RuntimeReactiveHost): UserFn {
  const res = evaluateProgram(src, { host, env: new RecordingHostEnv(), builtins: [MATH_BUILTINS] });
  if (!isUserFn(res.value)) throw new Error('kernel'); return res.value;
}

describe('zero-copy input resolution', () => {
  it('an f32 input is handed to the backend as the SAME Float32Array (no iterate/copy)', async () => {
    let captured: DispatchInput | null = null;
    const spy: Backend = {
      kind: 'cpu', limits: { maxStorageBufferBindingSize: 1 << 28, maxComputeWorkgroupsPerDimension: 65535 },
      async dispatch(input) { captured = input; return { output: new Float32Array([0, 2, 4, 6]), ms: 1 }; },
      [Symbol.dispose]() {},
    };
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const x = f32([0, 1, 2, 3])\ncomponent k(i) { return x[i] * 2 }\nk`, host);
    const engine = new GpuEngine(host, { tryWebGpu: async () => spy, tryWebGl2: () => null, limitsHint: spy.limits });
    change(() => { engine.gpu(kernel, { output: [4] }); });
    await new Promise((r) => setTimeout(r, 20));
    expect(captured).not.toBeNull();
    const xInput = captured!.inputs.find((i) => i.name === 'x')!;
    expect(xInput.data).toBeInstanceOf(Float32Array);
    expect(Array.from(xInput.data)).toEqual([0, 1, 2, 3]);
  });

  it('an i32 input is converted ONCE to a Float32Array for the backend (the non-f32 lowering)', async () => {
    // An i32/u32/f64 input has no Float32Array store, so it is converted once to f32 (the GPU storage type)
    // before dispatch. A CPU-backend test can't cover this: the CPU path computes via descriptors and never
    // reads input.inputs. Assert what a REAL (webgpu) backend RECEIVES via a spy — a Float32Array with the
    // converted values.
    let captured: DispatchInput | null = null;
    const spy: Backend = {
      kind: 'webgpu', limits: { maxStorageBufferBindingSize: 1 << 28, maxComputeWorkgroupsPerDimension: 65535 },
      async dispatch(input) { captured = input; return { output: new Float32Array([0, 0, 0, 0]), ms: 1 }; }, [Symbol.dispose]() {},
    };
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const x = i32([10, 20, 30, 40])\ncomponent k(i) { return x[i] * 2 }\nk`, host);
    const engine = new GpuEngine(host, { tryWebGpu: async () => spy, tryWebGl2: () => null, limitsHint: spy.limits });
    change(() => { engine.gpu(kernel, { output: [4] }); });
    await new Promise((r) => setTimeout(r, 20));
    expect(captured).not.toBeNull();
    const xInput = captured!.inputs.find((i) => i.name === 'x')!;
    expect(xInput.data).toBeInstanceOf(Float32Array);
    expect(Array.from(xInput.data)).toEqual([10, 20, 30, 40]);   // i32 → f32, values preserved
  });

  it('the f32 store is passed by REFERENCE (zero-copy), verified by OBJECT IDENTITY', async () => {
    let captured: DispatchInput | null = null;
    const spy: Backend = {
      kind: 'cpu', limits: { maxStorageBufferBindingSize: 1 << 28, maxComputeWorkgroupsPerDimension: 65535 },
      async dispatch(input) { captured = input; return { output: new Float32Array([0, 0, 0]), ms: 1 }; }, [Symbol.dispose]() {},
    };
    const host = new RuntimeReactiveHost();
    // Own the backing store, wrap it in an f32 handle, inject it as `x` via a `data` head the kernel closes over.
    const store = Float32Array.from([9, 8, 7]);
    const xHandle = makeTypedArray('f32', store, host.allocateGeneration());
    const env: HostEnvironment = {
      resolveCall(head: string, _k: string, _a: Arg[], _c: HostValue[], _s: SourceSpan) {
        return head === 'data' ? { handled: true as const, value: xHandle, kind: 'value' as const } : { handled: false as const };
      },
    };
    const parsed = evaluateProgram(`const x = data()\ncomponent k(i) { return x[i] }\nk`, { host, env });
    const kernel = parsed.value as UserFn;
    const engine = new GpuEngine(host, { tryWebGpu: async () => spy, tryWebGl2: () => null, limitsHint: spy.limits });
    change(() => { engine.gpu(kernel, { output: [3] }); });
    await new Promise((r) => setTimeout(r, 20));
    // Zero-copy: the backend's input data IS the same Float32Array we built (bufferView returned the live store).
    expect(captured!.inputs[0]!.data).toBe(store);
  });
});
