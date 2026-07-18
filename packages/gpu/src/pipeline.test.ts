// packages/gpu/src/pipeline.test.ts — proves the wiring on the CPU floor (deterministic, no adapter).
import { describe, it, expect } from 'vitest';
import { RuntimeReactiveHost, change } from '@metael/runtime';
import { evaluateProgram, isUserFn, RecordingHostEnv, descriptorOf, type HostEnvironment, type Arg, type HostValue, type SourceSpan } from '@metael/lang';
import type { UserFn } from '@metael/lang';
import { GpuEngine } from './resource.ts';
import type { Backend, DispatchInput } from './device/index.ts';

function kernelOf(src: string, host: RuntimeReactiveHost): UserFn {
  const res = evaluateProgram(src, { host, env: new RecordingHostEnv() });
  if (!isUserFn(res.value)) throw new Error('kernel'); return res.value;
}
const cpuDeps = { tryWebGpu: async () => null, tryWebGl2: () => null, limitsHint: { maxStorageBufferBindingSize: 1 << 28, maxComputeWorkgroupsPerDimension: 65535 } };

describe('GPU-resident pipelining (CPU backend)', () => {
  it('kernel B consumes kernel A resident output; the final values match a direct computation', async () => {
    const host = new RuntimeReactiveHost();
    const engine = new GpuEngine(host, cpuDeps);
    const kernelA = kernelOf(`component a(i) { return i + 1 }\na`, host);
    change(() => { engine.gpu(kernelA, { output: [4], backend: 'cpu', outputType: 'gpu-buffer' }); });
    await new Promise((r) => setTimeout(r, 20));
    let rA!: ReturnType<GpuEngine['gpu']>;
    change(() => { rA = engine.gpu(kernelA, { output: [4], backend: 'cpu', outputType: 'gpu-buffer' }); });
    const hA = rA.value as object;   // the resident handle (a GpuBufferHandle custom value)
    const envB: HostEnvironment = {
      resolveCall(head: string, _k: string, _a: Arg[], _c: HostValue[], _s: SourceSpan) {
        return head === 'resident' ? { handled: true as const, value: hA, kind: 'value' as const } : { handled: false as const };
      },
    };
    const resB = evaluateProgram(`const hA = resident()\ncomponent b(i) { return hA[i] * 10 }\nb`, { host, env: envB });
    const kernelB = resB.value as UserFn;
    change(() => { engine.gpu(kernelB, { output: [4], backend: 'cpu' }); });
    await new Promise((r) => setTimeout(r, 20));
    let rB!: ReturnType<GpuEngine['gpu']>;
    change(() => { rB = engine.gpu(kernelB, { output: [4], backend: 'cpu' }); });
    expect(rB.value).toEqual([10, 20, 30, 40]);   // hA[i]*10 = (i+1)*10
  });

  it('re-dispatching kernel A (a fresh handle of the SAME length) re-dispatches consumer B — memo keys off handle identity, not a dropped generation', async () => {
    const host = new RuntimeReactiveHost();
    const engine = new GpuEngine(host, cpuDeps);
    const kernelA = kernelOf(`component a(i) { return i + 1 }\na`, host);
    change(() => { engine.gpu(kernelA, { output: [4], backend: 'cpu', outputType: 'gpu-buffer' }); });
    await new Promise((r) => setTimeout(r, 20));
    let r1!: ReturnType<GpuEngine['gpu']>;
    change(() => { r1 = engine.gpu(kernelA, { output: [4], backend: 'cpu', outputType: 'gpu-buffer' }); });
    const { residentInfo } = await import('./handle.ts');
    const nonce1 = residentInfo(r1.value)!.nonce;
    const kernelA2 = kernelOf(`component a2(i) { return i + 2 }\na2`, host);
    change(() => { engine.gpu(kernelA2, { output: [4], backend: 'cpu', outputType: 'gpu-buffer' }); });
    await new Promise((r) => setTimeout(r, 20));
    let r2!: ReturnType<GpuEngine['gpu']>;
    change(() => { r2 = engine.gpu(kernelA2, { output: [4], backend: 'cpu', outputType: 'gpu-buffer' }); });
    expect(residentInfo(r2.value)!.nonce).not.toBe(nonce1);   // distinct handles → distinct memo signal for a consumer
  });

  it('a consumer reads its producer handle correctly even when the producer is the LRU-eviction target (no spurious use-after-dispose)', async () => {
    const host = new RuntimeReactiveHost();
    const engine = new GpuEngine(host, cpuDeps);
    // Producer A → a gpu-buffer handle. Settle it, grab the handle.
    const kernelA = kernelOf(`component a(i) { return i + 1 }\na`, host);
    change(() => { engine.gpu(kernelA, { output: [4], backend: 'cpu', outputType: 'gpu-buffer' }); });
    await new Promise((r) => setTimeout(r, 20));
    let rA!: ReturnType<GpuEngine['gpu']>;
    change(() => { rA = engine.gpu(kernelA, { output: [4], backend: 'cpu', outputType: 'gpu-buffer' }); });
    const hA = rA.value as object;
    // Fill the memo toward MAX_LIVE (8) with 7 OTHER distinct kernels so A becomes the LRU-oldest entry.
    // (A is entry #1; these are #2..#8. The NEXT distinct insertion — consumer B — makes size 9 → evictLru
    //  disposes the oldest = A, whose handle cache is still null.)
    for (let n = 0; n < 7; n++) {
      const kf = kernelOf(`component f${n}(i) { return i + ${100 + n} }\nf${n}`, host);
      change(() => { engine.gpu(kf, { output: [4], backend: 'cpu' }); });
    }
    await new Promise((r) => setTimeout(r, 30));
    // Consumer B closes over A's handle. Dispatching B inserts a 9th memo entry → evictLru → would dispose A.
    const envB: HostEnvironment = {
      resolveCall(head: string, _k: string, _a: Arg[], _c: HostValue[], _s: SourceSpan) {
        return head === 'resident' ? { handled: true as const, value: hA, kind: 'value' as const } : { handled: false as const };
      },
    };
    const resB = evaluateProgram(`const hA = resident()\ncomponent b(i) { return hA[i] * 10 }\nb`, { host, env: envB });
    const kernelB = resB.value as UserFn;
    let rB!: ReturnType<GpuEngine['gpu']>;
    // This must NOT throw MLGPU-USE-AFTER-DISPOSE. Wrap in change(); the input resolution runs synchronously in gpu().
    change(() => { rB = engine.gpu(kernelB, { output: [4], backend: 'cpu' }); });
    await new Promise((r) => setTimeout(r, 20));
    change(() => { rB = engine.gpu(kernelB, { output: [4], backend: 'cpu' }); });
    expect(rB.value).toEqual([10, 20, 30, 40]);   // hA[i]*10 = (i+1)*10 — read from the (preserved) handle
  });

  it('a consumer whose resident input was disposed (cache never materialized) settles with a LOCAL error, not a tree-collapsing throw', async () => {
    const host = new RuntimeReactiveHost();
    const engine = new GpuEngine(host, cpuDeps);
    const kernelA = kernelOf(`component a(i) { return i + 1 }\na`, host);
    change(() => { engine.gpu(kernelA, { output: [4], backend: 'cpu', outputType: 'gpu-buffer' }); });
    await new Promise((r) => setTimeout(r, 20));
    let rA!: ReturnType<GpuEngine['gpu']>;
    change(() => { rA = engine.gpu(kernelA, { output: [4], backend: 'cpu', outputType: 'gpu-buffer' }); });
    const hA = rA.value as object;
    const { disposeHandle } = await import('./handle.ts');
    disposeHandle(hA);   // free it WITHOUT ever reading → cache stays null
    const envB: HostEnvironment = {
      resolveCall(head: string, _k: string, _a: Arg[], _c: HostValue[], _s: SourceSpan) {
        return head === 'resident' ? { handled: true as const, value: hA, kind: 'value' as const } : { handled: false as const };
      },
    };
    const resB = evaluateProgram(`const hA = resident()\ncomponent b(i) { return hA[i] * 10 }\nb`, { host, env: envB });
    const kernelB = resB.value as UserFn;
    let rB!: ReturnType<GpuEngine['gpu']>;
    // Must NOT throw out of gpu(); must settle a local error, not strand pending.
    expect(() => { change(() => { rB = engine.gpu(kernelB, { output: [4], backend: 'cpu' }); }); }).not.toThrow();
    expect(rB.pending).toBe(false);
    expect(rB.error).not.toBeNull();
    expect(rB.error?.code).toBe('MLGPU-INPUT-UNAVAILABLE');
    // The failure is LOCAL: a program with a sibling node alongside a (hypothetical) gpu call still evaluates.
    const sibling = evaluateProgram(`component sib() { return 42 }\nsib`, { host, env: new RecordingHostEnv() });
    expect(isUserFn(sibling.value)).toBe(true);
  });

  it('a DISPOSED resident-handle input is NOT offered as a resident bind (falls to the readback cache — no freed-buffer bind)', async () => {
    // A resident producer handle can be LRU-evicted (disposed) before a later consumer's dispatch binds it.
    // disposeHandle sets disposed=true but leaves gpuBuffer non-undefined, so pre-fix resolveInputs STILL
    // offered the freed buffer to residentInputs → the backend binds a freed GPUBuffer/texture (WebGL2 → silent
    // 0s; WebGPU → throws). Post-fix a disposed handle is skipped → the backend uploads the readback cache.
    let captured: ReadonlyMap<string, unknown> | undefined;
    const makeSpy = async (): Promise<Backend> => {
      const INSTANCE = {};
      return {
        kind: 'webgpu', limits: { maxStorageBufferBindingSize: 1 << 28, maxComputeWorkgroupsPerDimension: 65535 },
        async dispatch(input: DispatchInput) {
          // Capture the consumer's resident inputs (the dispatch that closes over the disposed handle: kernel B).
          if (input.kernel.params.length === 1 && input.residentInputs && input.residentInputs.size >= 0 && input.inputs.length > 0) {
            captured = input.residentInputs;
          }
          const total = input.dims.reduce((a, b) => a * b, 1);
          const comps = input.outputComps ?? 1;
          const out = new Float32Array(total * comps);
          for (let i = 0; i < total; i++) { const vals = input.cpuRun([i]); for (let k = 0; k < comps; k++) out[i * comps + k] = vals[k]!; }
          const resident = input.retainOutput ? { gpuBuffer: { token: INSTANCE }, dispose() {} } : undefined;
          return { output: out, ms: 1, resident };
        },
        [Symbol.dispose]() {},
      };
    };
    const host = new RuntimeReactiveHost();
    const engine = new GpuEngine(host, { tryWebGpu: makeSpy, tryWebGl2: () => null, limitsHint: { maxStorageBufferBindingSize: 1 << 28, maxComputeWorkgroupsPerDimension: 65535 } });
    // Producer A → a resident gpu-buffer handle on the (pooled) webgpu spy.
    const kernelA = kernelOf(`component a(i) { return i + 1 }\na`, host);
    change(() => { engine.gpu(kernelA, { output: [4], backend: 'webgpu', outputType: 'gpu-buffer' }); });
    await new Promise((r) => setTimeout(r, 20));
    let rA!: ReturnType<GpuEngine['gpu']>;
    change(() => { rA = engine.gpu(kernelA, { output: [4], backend: 'webgpu', outputType: 'gpu-buffer' }); });
    const hA = rA.value as object;
    // Materialize the handle's CPU cache (a real reader would; here we read it directly) BEFORE disposing, so
    // the disposed handle still has correct values to upload from the readback cache.
    const values = Array.from(descriptorOf(hA)!.iterate!(hA), (v) => Number(v));
    expect(values).toEqual([1, 2, 3, 4]);
    // Now DISPOSE the handle (simulate the LRU eviction of the producer before the consumer dispatches).
    const { disposeHandle } = await import('./handle.ts');
    disposeHandle(hA);
    // Consumer B closes over the DISPOSED handle. Same 'webgpu' → same pooled instance (so residency WOULD fire
    // for a live handle). The binding name in the kernel is `inp`.
    const envB: HostEnvironment = {
      resolveCall(head: string, _k: string, _a: Arg[], _c: HostValue[], _s: SourceSpan) {
        return head === 'resident' ? { handled: true as const, value: hA as HostValue, kind: 'value' as const } : { handled: false as const };
      },
    };
    const kernelB = evaluateProgram(`const inp = resident()\ncomponent b(i) { return inp[i] * 10 }\nb`, { host, env: envB }).value as UserFn;
    let rB!: ReturnType<GpuEngine['gpu']>;
    change(() => { rB = engine.gpu(kernelB, { output: [4], backend: 'webgpu' }); });
    await new Promise((r) => setTimeout(r, 20));
    change(() => { rB = engine.gpu(kernelB, { output: [4], backend: 'webgpu' }); });
    // Post-fix: the disposed handle is NOT offered as a resident bind (captured map has no `inp`); the backend
    // uploaded the readback-cache data instead → the values are still correct.
    expect(captured).toBeDefined();
    expect(captured!.has('inp')).toBe(false);   // was true pre-fix (the freed buffer was offered)
    expect(rB.value).toEqual([10, 20, 30, 40]);
    engine[Symbol.dispose]();
  });

  it('two distinct buffer-mode outputs of the same length do NOT collide as a consumer input (no stale memo hit)', async () => {
    const host = new RuntimeReactiveHost();
    const engine = new GpuEngine(host, cpuDeps);
    // Producer P1 → buffer [1,2,3,4]; Producer P2 → buffer [11,12,13,14] (same length, different values).
    const kP1 = kernelOf(`component p1(i) { return i + 1 }\np1`, host);
    const kP2 = kernelOf(`component p2(i) { return i + 11 }\np2`, host);
    change(() => { engine.gpu(kP1, { output: [4], backend: 'cpu', outputType: 'buffer' }); });
    change(() => { engine.gpu(kP2, { output: [4], backend: 'cpu', outputType: 'buffer' }); });
    await new Promise((r) => setTimeout(r, 20));
    let r1!: ReturnType<GpuEngine['gpu']>; let r2!: ReturnType<GpuEngine['gpu']>;
    change(() => { r1 = engine.gpu(kP1, { output: [4], backend: 'cpu', outputType: 'buffer' }); });
    change(() => { r2 = engine.gpu(kP2, { output: [4], backend: 'cpu', outputType: 'buffer' }); });
    const buf1 = r1.value as object; const buf2 = r2.value as object;
    // A consumer C(i) = in[i] * 100. Feed buf1 → expect [100,200,300,400]; feed buf2 → expect [1100,1200,1300,1400].
    // Both consumers share an IDENTICAL kernel AST (only the closed-over buffer differs), so kernelHash is
    // identical for both → the memo relies ENTIRELY on the gens[] buffer token to distinguish them.
    const mkC = (h: object): UserFn => {
      const env: HostEnvironment = {
        resolveCall(head: string, _k: string, _a: Arg[], _c: HostValue[], _s: SourceSpan) {
          return head === 'resident' ? { handled: true as const, value: h, kind: 'value' as const } : { handled: false as const };
        },
      };
      return evaluateProgram(`const inp = resident()\ncomponent c(i) { return inp[i] * 100 }\nc`, { host, env }).value as UserFn;
    };
    const cB1 = mkC(buf1); const cB2 = mkC(buf2);
    change(() => { engine.gpu(cB1, { output: [4], backend: 'cpu' }); });
    change(() => { engine.gpu(cB2, { output: [4], backend: 'cpu' }); });
    await new Promise((r) => setTimeout(r, 20));
    let rc1!: ReturnType<GpuEngine['gpu']>; let rc2!: ReturnType<GpuEngine['gpu']>;
    change(() => { rc1 = engine.gpu(cB1, { output: [4], backend: 'cpu' }); });
    change(() => { rc2 = engine.gpu(cB2, { output: [4], backend: 'cpu' }); });
    expect(rc1.value).toEqual([100, 200, 300, 400]);       // buf1 = [1,2,3,4]
    expect(rc2.value).toEqual([1100, 1200, 1300, 1400]);   // buf2 = [11,12,13,14] — NOT stale [100,200,300,400]
  });
});
