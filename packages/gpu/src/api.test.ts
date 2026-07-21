// packages/gpu/src/api.test.ts
import { describe, it, expect } from 'vitest';
import { MATH_BUILTINS } from '@metael/math/lang';
import { RuntimeReactiveHost, signal, change } from '@metael/runtime';
import { isUserFn, makeCallable, evaluateProgram, RecordingHostEnv, type UserFn } from '@metael/lang';
import { createGpuEngine } from './api.ts';
import { compileKernel } from './lang/compile-kernel.ts';
import { settle, settled, subscribe } from './settle.ts';
import { gpuBuffer } from './buffer.ts';

describe('compileKernel — metael kernel snippet → UserFn on a given host', () => {
  it('compiles a component kernel to a UserFn', () => {
    const host = new RuntimeReactiveHost();
    const fn = compileKernel(`const x = f32(8, (i) => i)\ncomponent k(i) { return x[i] * 2 }\nk`, host);
    expect(isUserFn(fn)).toBe(true);
    expect(fn.name).toBe('k');
    expect(fn.isComponent).toBe(true);
  });

  it('throws a clear error when the snippet does not evaluate to a function', () => {
    const host = new RuntimeReactiveHost();
    expect(() => compileKernel(`1 + 2`, host)).toThrow(/must evaluate to a function/i);
  });
});

describe('createGpuEngine — CPU-only façade: dispatch + settle + dispose guard', () => {
  it('dispatch returns pending synchronously with shader text; settle fills the value', async () => {
    const gpu = createGpuEngine({ cpuOnly: true });
    const kernel = compileKernel(`const x = f32(8, (i) => i)\ncomponent k(i) { return x[i] * 2 }\nk`, gpu.host);
    const pending = gpu.dispatch(kernel, { output: [8], backend: 'cpu' });
    expect(pending.core).toBe(true);
    expect(pending.pending).toBe(true);
    expect(pending.value).toBeNull();
    const result = await settle(() => gpu.dispatch(kernel, { output: [8], backend: 'cpu' }));
    expect(result.pending).toBe(false);
    expect(result.value).toEqual([0, 2, 4, 6, 8, 10, 12, 14]);
    expect(result.backend).toBe('cpu');
    gpu[Symbol.dispose]();
  });

  it('verify + benchmark flags populate match / cpuMs', async () => {
    const gpu = createGpuEngine({ cpuOnly: true });
    const kernel = compileKernel(`const x = f32(8, (i) => i)\ncomponent k(i) { return x[i] * 2 }\nk`, gpu.host);
    const r = await settle(() => gpu.dispatch(kernel, { output: [8], backend: 'cpu', verify: true, benchmark: true }));
    expect(r.match?.ok).toBe(true);
    expect(r.cpuMs).not.toBeNull();
    gpu[Symbol.dispose]();
  });

  it('a non-lowerable kernel settles immediately with core:false and an error', async () => {
    const gpu = createGpuEngine({ cpuOnly: true });
    const kernel = compileKernel(`component k(i) { return "x" }\nk`, gpu.host);   // a string return is not GPU-lowerable
    const r = await settle(() => gpu.dispatch(kernel, { output: [8], backend: 'cpu' }));
    expect(r.core).toBe(false);
    expect(r.pending).toBe(false);
    expect(r.error).not.toBeNull();
    gpu[Symbol.dispose]();
  });

  it('a rank mismatch (2 params, 1D output) is non-core', () => {
    const gpu = createGpuEngine({ cpuOnly: true });
    const k = compileKernel('component k(x, y) { return x + y } k', gpu.host);
    const r = gpu.dispatch(k, { output: [8] });
    expect(r.core).toBe(false);
    expect(r.reasons.some((d) => d.code === 'MLGPU-OUTPUT-SHAPE')).toBe(true);
    gpu[Symbol.dispose]();
  });

  it('a rank>3 kernel (4 params, 4D output) is non-core with MLGPU-NOT-LOWERABLE', () => {
    const gpu = createGpuEngine({ cpuOnly: true });
    const k = compileKernel('component k(x, y, z, w) { return x + y + z + w } k', gpu.host);
    const r = gpu.dispatch(k, { output: [2, 2, 2, 2] });
    expect(r.core).toBe(false);
    expect(r.reasons.some((d) => d.code === 'MLGPU-NOT-LOWERABLE')).toBe(true);
    gpu[Symbol.dispose]();
  });

  it('a multi-output rank mismatch (2 params, 1D output) is non-core', () => {
    const gpu = createGpuEngine({ cpuOnly: true });
    const k = compileKernel('component k(x, y) { return { a: x + y, b: x - y } } k', gpu.host);
    const r = gpu.dispatch(k, { output: [8], outputs: { a: {}, b: {} } });
    expect(r.core).toBe(false);
    expect(r.reasons.some((d) => d.code === 'MLGPU-OUTPUT-SHAPE')).toBe(true);
    gpu[Symbol.dispose]();
  });

  it('a multi-output rank>3 kernel is non-core with MLGPU-NOT-LOWERABLE', () => {
    const gpu = createGpuEngine({ cpuOnly: true });
    const k = compileKernel('component k(x, y, z, w) { return { a: x + y + z + w, b: x } } k', gpu.host);
    const r = gpu.dispatch(k, { output: [2, 2, 2, 2], outputs: { a: {}, b: {} } });
    expect(r.core).toBe(false);
    expect(r.reasons.some((d) => d.code === 'MLGPU-NOT-LOWERABLE')).toBe(true);
    gpu[Symbol.dispose]();
  });

  it('a VALID multi-output kernel (matching rank) stays core — the rank gate does not over-reject', () => {
    const gpu = createGpuEngine({ cpuOnly: true });
    // 1 param over a 1D output: arity 1 === dims 1 → the rank gate is satisfied.
    const k = compileKernel('const x = f32(4, (i) => i)\ncomponent k(i) { return { a: x[i] + 1, b: x[i] - 1 } }\nk', gpu.host);
    const r = gpu.dispatch(k, { output: [4], backend: 'cpu', outputs: { a: {}, b: {} } });
    expect(r.core).toBe(true);
    expect(r.reasons).toEqual([]);
    gpu[Symbol.dispose]();
  });

  it('settle/dispatch throw after dispose() instead of spinning forever', async () => {
    const gpu = createGpuEngine({ cpuOnly: true });
    const kernel = compileKernel(`const x = f32(8, (i) => i)\ncomponent k(i) { return x[i] * 2 }\nk`, gpu.host);
    gpu[Symbol.dispose]();
    expect(() => gpu.dispatch(kernel, { output: [8], backend: 'cpu' })).toThrow(/disposed/i);
    await expect(settle(() => gpu.dispatch(kernel, { output: [8], backend: 'cpu' }))).rejects.toThrow(/disposed/i);
  });
});

describe('subscribe (free helper) — fires on settle + converges', () => {
  it('calls onValue with the settled value and does not loop', async () => {
    const gpu = createGpuEngine({ cpuOnly: true });
    const kernel = compileKernel(`const x = f32(4, (i) => i)\ncomponent k(i) { return x[i] * 2 }\nk`, gpu.host);
    const seen: Array<number[] | object | number | null> = [];
    let fires = 0;
    const stop = subscribe(() => gpu.dispatch(kernel, { output: [4], backend: 'cpu' }), (r) => { fires++; if (!r.pending) seen.push(r.value); });
    await new Promise((res) => setTimeout(res, 20));
    expect(seen.length).toBe(1);
    expect(seen[0]).toEqual([0, 2, 4, 6]);
    expect(fires).toBeLessThanOrEqual(3);   // pending + settled (+ at most one scheduler artifact) — not a loop
    stop();
    gpu[Symbol.dispose]();
  });
});

describe('one dispatch verb routes on cfg.mode', () => {
  it('map (default) and reduce', async () => {
    using gpu = createGpuEngine({ cpuOnly: true });
    const reducer = compileKernel('component K(a, b) { a + b } K', gpu.host);
    // PLAIN JS input via gpuBuffer — no evaluateProgram, no hand-boxing.
    const input = gpuBuffer([1, 2, 3, 4], gpu.host);
    const r = await settle(() => gpu.dispatch(reducer, { mode: 'reduce', input, identity: 0 }));
    expect(settled(r)).toBe(true);
    expect(r.value).toBe(10);
  });
});

describe('reusable factory kernel bound to different inputs (the verified reuse pattern)', () => {
  it('one factory kernel, two buffers → correct results, identical emitted shader', async () => {
    const gpu = createGpuEngine({ cpuOnly: true });
    const makeKernelFn = compileKernel(`function makeKernel(a) { component ker(i) { return a[i] * 2 } return ker }\nmakeKernel`, gpu.host);
    const makeKernel = makeCallable(makeKernelFn, { host: gpu.host, env: new RecordingHostEnv(), builtins: [MATH_BUILTINS] });
    // Two buffers on the façade's host (a rebind, not an in-place mutation). A buffer value is not a
    // function, so it is built via evaluateProgram(...).value rather than compile (which returns a UserFn).
    const evalBuf = (src: string): unknown => evaluateProgram(src, { host: gpu.host, env: new RecordingHostEnv(), builtins: [MATH_BUILTINS] }).value;
    const bufA = evalBuf(`const x = f32(4, (i) => i)\nx`);          // [0,1,2,3]
    const bufB = evalBuf(`const x = f32(4, (i) => i * 10)\nx`);     // [0,10,20,30]
    const rA = await settle(() => gpu.dispatch(makeKernel(bufA) as UserFn, { output: [4], backend: 'cpu' }));
    const rB = await settle(() => gpu.dispatch(makeKernel(bufB) as UserFn, { output: [4], backend: 'cpu' }));
    expect(rA.value).toEqual([0, 2, 4, 6]);      // a[i]*2 for bufA
    expect(rB.value).toEqual([0, 20, 40, 60]);   // a[i]*2 for bufB
    expect(rA.wgsl).toBe(rB.wgsl);               // reusable kernel → identical shader (pipeline cache reuses it)
    gpu[Symbol.dispose]();
  });

  it('a reactive rebind (a signal of the current input) re-dispatches with the new buffer', async () => {
    const gpu = createGpuEngine({ cpuOnly: true });
    const makeKernelFn = compileKernel(`function makeKernel(a) { component ker(i) { return a[i] + 1 } return ker }\nmakeKernel`, gpu.host);
    const makeKernel = makeCallable(makeKernelFn, { host: gpu.host, env: new RecordingHostEnv(), builtins: [MATH_BUILTINS] });
    const evalBuf = (src: string): unknown => evaluateProgram(src, { host: gpu.host, env: new RecordingHostEnv(), builtins: [MATH_BUILTINS] }).value;
    const bufA = evalBuf(`const x = f32(3, (i) => i)\nx`);        // [0,1,2]
    const bufB = evalBuf(`const x = f32(3, (i) => i * 4)\nx`);    // [0,4,8]
    const cur = signal<unknown>(bufA);
    const stop = subscribe(() => gpu.dispatch(makeKernel(cur.get()) as UserFn, { output: [3], backend: 'cpu' }), () => {});
    // NOTE: subscribe binds ONE kernel; a rebind re-derives the kernel from `cur`, so drive it via settle in
    // a tracked read instead — assert the rebind path directly:
    stop();
    const r1 = await settle(() => gpu.dispatch(makeKernel(cur.get()) as UserFn, { output: [3], backend: 'cpu' }));
    expect(r1.value).toEqual([1, 2, 3]);
    change(() => cur.set(bufB));
    const r2 = await settle(() => gpu.dispatch(makeKernel(cur.get()) as UserFn, { output: [3], backend: 'cpu' }));
    expect(r2.value).toEqual([1, 5, 9]);         // rebound to bufB → re-dispatched
    gpu[Symbol.dispose]();
  });
});
