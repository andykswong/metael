import { describe, it, expect } from 'vitest';
import { MATH_BUILTINS } from '@metael/math/lang';
import { RuntimeReactiveHost, change } from '@metael/runtime';
import { evaluateProgram, isUserFn } from '@metael/lang';
import type { UserFn, HostEnvironment, Arg, HostValue, SourceSpan } from '@metael/lang';
import { RecordingHostEnv } from '@metael/lang';
import { GpuEngine } from './resource.ts';
import type { Backend, DispatchInput } from './device/index.ts';

function kernelOf(src: string, host: RuntimeReactiveHost): UserFn {
  const res = evaluateProgram(src, { host, env: new RecordingHostEnv(), builtins: [MATH_BUILTINS] });
  if (!isUserFn(res.value)) throw new Error('expected kernel');
  return res.value;
}
const cpuOnlyDeps = { tryWebGpu: async () => null, tryWebGl2: () => null, limitsHint: { maxStorageBufferBindingSize: 1 << 28, maxComputeWorkgroupsPerDimension: 65535 } };

describe('gpu reactive resource (CPU backend, node)', () => {
  it('classify/emit are synchronous; value fills after the async drain', async () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const x = f32(8, (i) => i)\ncomponent k(i) { return x[i] * 2 }\nk`, host);
    const engine = new GpuEngine(host, cpuOnlyDeps);
    let resource!: ReturnType<GpuEngine['gpu']>;
    change(() => { resource = engine.gpu(kernel, { output: [8], backend: 'cpu' }); });
    expect(resource.core).toBe(true);
    expect(resource.pending).toBe(true);
    expect(resource.value).toBeNull();
    await new Promise((r) => setTimeout(r, 20));
    let settled!: ReturnType<GpuEngine['gpu']>;
    change(() => { settled = engine.gpu(kernel, { output: [8], backend: 'cpu' }); });
    expect(settled.pending).toBe(false);
    expect(settled.value).toEqual([0, 2, 4, 6, 8, 10, 12, 14]);
    expect(settled.backend).toBe('cpu');
    // Default dispatch (no verify/benchmark flags): the GPU RESULT only. The oracle + the CPU race are
    // OPT-IN — a CPU-side interpreter sweep + a second full dispatch on every run defeat the point of the
    // GPU — so match/cpuMs/speedup stay null unless explicitly requested.
    expect(settled.match).toBeNull();
    expect(settled.cpuMs).toBeNull();
    expect(settled.speedup).toBeNull();
  });
  it('verify + benchmark are opt-in: they populate match / cpuMs / speedup', async () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const x = f32(8, (i) => i)\ncomponent k(i) { return x[i] * 2 }\nk`, host);
    const engine = new GpuEngine(host, cpuOnlyDeps);
    const cfg = { output: [8], backend: 'cpu' as const, verify: true, benchmark: true };
    change(() => { engine.gpu(kernel, cfg); });
    await new Promise((r) => setTimeout(r, 20));
    let settled!: ReturnType<GpuEngine['gpu']>;
    change(() => { settled = engine.gpu(kernel, cfg); });
    expect(settled.pending).toBe(false);
    expect(settled.match?.ok).toBe(true);      // verify → the interpreter oracle ran
    expect(settled.cpuMs).not.toBeNull();      // benchmark → a CPU baseline was timed
    // On the CPU floor the dispatch already IS the CPU run, so there is no GPU to race → speedup stays null.
    expect(settled.speedup).toBeNull();
    expect(settled.backend).toBe('cpu');
  });
  it('the flags are part of the memo key: verify:false and verify:true get DISTINCT resources on one engine', async () => {
    // Same kernel/output/backend but different flags must NOT collide in the dispatch memo. Without the flag
    // component in the key, the second (verify:true) call would return the first's cached verify:false
    // resource (match=null) instead of re-dispatching + running the oracle.
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const x = f32(8, (i) => i)\ncomponent k(i) { return x[i] * 2 }\nk`, host);
    const engine = new GpuEngine(host, cpuOnlyDeps);
    change(() => { engine.gpu(kernel, { output: [8], backend: 'cpu' }); });                 // no flags
    change(() => { engine.gpu(kernel, { output: [8], backend: 'cpu', verify: true }); });   // verify on
    await new Promise((r) => setTimeout(r, 30));
    let plain!: ReturnType<GpuEngine['gpu']>; let verified!: ReturnType<GpuEngine['gpu']>;
    change(() => { plain = engine.gpu(kernel, { output: [8], backend: 'cpu' }); });
    change(() => { verified = engine.gpu(kernel, { output: [8], backend: 'cpu', verify: true }); });
    expect(plain).not.toBe(verified);        // distinct cached resources, not one collided entry
    expect(plain.match).toBeNull();          // the no-flags dispatch never ran the oracle
    expect(verified.match?.ok).toBe(true);   // the verify:true dispatch re-ran + the oracle populated match
  });
  it('a rank-3 kernel on the CPU floor computes per-(x,y,z) cells + verifies (row-major flat decomposition)', async () => {
    // The CPU backend + the verify oracle both reconstruct (x,y,z) from a flat index via the shared row-major
    // flatten (flat = (x*H + y)*D + z). A distinct value per cell (x*100 + y*10 + z) with all-distinct dims
    // [W=2, H=3, D=4] means a swapped/collapsed axis would scramble the output → verify would fail.
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf('component k(x, y, z) { return x * 100 + y * 10 + z }\nk', host);
    const engine = new GpuEngine(host, cpuOnlyDeps);
    const cfg = { output: [2, 3, 4], backend: 'cpu' as const, verify: true };
    change(() => { engine.gpu(kernel, cfg); });
    await new Promise((r) => setTimeout(r, 30));
    let settled!: ReturnType<GpuEngine['gpu']>;
    change(() => { settled = engine.gpu(kernel, cfg); });
    expect(settled.pending).toBe(false);
    expect(settled.backend).toBe('cpu');
    expect(settled.match?.ok).toBe(true);   // the CPU coords ≡ the oracle's decomposed coords
    const out = settled.value as number[];
    // flat = (x*H + y)*D + z with H=3, D=4. (1,2,3) → (1*3+2)*4+3 = 23 → 123; (0,1,0) → 4 → 10.
    expect(out[23]).toBe(123);
    expect(out[0]).toBe(0);
    expect(out[4]).toBe(10);
  });
  it('a non-lowerable kernel is terminal (not pending, has reasons)', () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`component k(i) { return "x" + i }\nk`, host);
    const engine = new GpuEngine(host, cpuOnlyDeps);
    let r!: ReturnType<GpuEngine['gpu']>;
    change(() => { r = engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    expect(r.core).toBe(false);
    expect(r.pending).toBe(false);
    expect(r.reasons.length).toBeGreaterThan(0);
  });
  it('the memo hit prevents re-dispatch (same key returns the same object pre-settle)', () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const x = f32(4, (i) => i)\ncomponent k(i) { return x[i] }\nk`, host);
    const engine = new GpuEngine(host, cpuOnlyDeps);
    let a!: ReturnType<GpuEngine['gpu']>; let b!: ReturnType<GpuEngine['gpu']>;
    change(() => { a = engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    change(() => { b = engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    expect(a).toBe(b);
  });

  it('dispose() cancels an enqueued dispatch → no device acquired/leaked after teardown', async () => {
    // The playground mounts a HEADLESS diagnostics probe per keystroke, then unmounts it (→ dispose) BEFORE
    // the enqueued auto-backend dispatch drains on its microtask. dispose() must cancel that pending task so
    // it never acquires a GPUDevice into the cleared memo (a real per-keystroke device leak on a live adapter,
    // invisible on the CPU floor). Prove it with a spying fake WebGPU backend.
    let acquired = 0; let disposed = 0;
    const spyGpu = async (): Promise<Backend> => {
      acquired++;
      return {
        kind: 'webgpu', limits: { maxStorageBufferBindingSize: 1 << 28, maxComputeWorkgroupsPerDimension: 65535 },
        async dispatch() { return { output: new Float32Array([0, 2, 4, 6, 8, 10, 12, 14]), ms: 1 }; },
        [Symbol.dispose]() { disposed++; },
      };
    };
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const x = f32(8, (i) => i)\ncomponent k(i) { return x[i] * 2 }\nk`, host);
    const engine = new GpuEngine(host, { tryWebGpu: spyGpu, tryWebGl2: () => null, limitsHint: { maxStorageBufferBindingSize: 1 << 28, maxComputeWorkgroupsPerDimension: 65535 } });
    change(() => { engine.gpu(kernel, { output: [8] }); });   // auto backend → enqueues a dispatch
    engine[Symbol.dispose]();                                          // tear down BEFORE the microtask drains
    await new Promise((r) => setTimeout(r, 30));               // give the (cancelled) task a chance to run
    expect(acquired).toBe(0);                                  // never touched WebGPU → no device acquired
    expect(disposed).toBe(0);                                  // nothing acquired → nothing to leak
  });

  it('dispose() during device acquisition frees the just-acquired backend (async teardown race)', async () => {
    // dispose() can land DURING the await of device acquisition. The task must then free the backend it just
    // got rather than store it in the cleared memo. Simulate a slow acquisition, dispose mid-flight.
    let disposed = 0;
    const slowGpu = async (): Promise<Backend> => {
      await new Promise((r) => setTimeout(r, 20));   // acquisition takes a beat
      return {
        kind: 'webgpu', limits: { maxStorageBufferBindingSize: 1 << 28, maxComputeWorkgroupsPerDimension: 65535 },
        async dispatch() { return { output: new Float32Array([0]), ms: 1 }; },
        [Symbol.dispose]() { disposed++; },
      };
    };
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const x = f32(8, (i) => i)\ncomponent k(i) { return x[i] * 2 }\nk`, host);
    const engine = new GpuEngine(host, { tryWebGpu: slowGpu, tryWebGl2: () => null, limitsHint: { maxStorageBufferBindingSize: 1 << 28, maxComputeWorkgroupsPerDimension: 65535 } });
    change(() => { engine.gpu(kernel, { output: [8] }); });
    await new Promise((r) => setTimeout(r, 5));   // let the task start + enter the acquisition await
    engine[Symbol.dispose]();                             // dispose WHILE acquisition is in flight
    await new Promise((r) => setTimeout(r, 40));  // let acquisition resolve → the task must free the backend
    expect(disposed).toBe(1);                     // the just-acquired device was freed, not orphaned
  });

  it('dispose() during the DISPATCH await frees the backend + writes no stale cell (the post-dispatch guard)', async () => {
    // Acquisition is fast but dispatch is slow; dispose() lands DURING dispatch. The task must re-check
    // disposed AFTER the dispatch await, free the backend, and NOT store it or write the cell — otherwise a
    // real GPUDevice leaks into the cleared memo (invisible on the CPU floor, a per-frame leak on an adapter).
    let disposed = 0; let acquired = 0;
    const slowDispatchGpu = async (): Promise<Backend> => {
      acquired++;
      return {
        kind: 'webgpu', limits: { maxStorageBufferBindingSize: 1 << 28, maxComputeWorkgroupsPerDimension: 65535 },
        async dispatch() { await new Promise((r) => setTimeout(r, 25)); return { output: new Float32Array([0, 2, 4, 6, 8, 10, 12, 14]), ms: 1 }; },
        [Symbol.dispose]() { disposed++; },
      };
    };
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const x = f32(8, (i) => i)\ncomponent k(i) { return x[i] * 2 }\nk`, host);
    const engine = new GpuEngine(host, { tryWebGpu: slowDispatchGpu, tryWebGl2: () => null, limitsHint: { maxStorageBufferBindingSize: 1 << 28, maxComputeWorkgroupsPerDimension: 65535 } });
    change(() => { engine.gpu(kernel, { output: [8] }); });
    await new Promise((r) => setTimeout(r, 8));   // let acquisition finish + dispatch start
    engine[Symbol.dispose]();                             // dispose WHILE dispatch is in flight
    await new Promise((r) => setTimeout(r, 40));  // let dispatch resolve → the post-dispatch guard must fire
    expect(acquired).toBe(1);                     // the device was acquired (dispatch had started)
    expect(disposed).toBe(1);                     // …and freed by the post-dispatch disposed re-check, not orphaned
  });

  it('pools the backend across dispatches — two dispatches acquire ONE device (not one per dispatch)', async () => {
    let acquired = 0;
    const spyGpu = async (): Promise<Backend> => {
      acquired++;
      return { kind: 'webgpu', limits: { maxStorageBufferBindingSize: 1 << 28, maxComputeWorkgroupsPerDimension: 65535 },
        async dispatch() { return { output: new Float32Array([0, 2, 4, 6]), ms: 1 }; }, [Symbol.dispose]() {} };
    };
    const host = new RuntimeReactiveHost();
    const k1 = kernelOf(`const x = f32(4, (i) => i)\ncomponent k(i) { return x[i] * 2 }\nk`, host);
    const k2 = kernelOf(`const y = f32(4, (i) => i)\ncomponent k2(i) { return y[i] + 1 }\nk2`, host);
    const engine = new GpuEngine(host, { tryWebGpu: spyGpu, tryWebGl2: () => null, limitsHint: { maxStorageBufferBindingSize: 1 << 28, maxComputeWorkgroupsPerDimension: 65535 } });
    change(() => { engine.gpu(k1, { output: [4], backend: 'webgpu' }); });
    change(() => { engine.gpu(k2, { output: [4], backend: 'webgpu' }); });
    await new Promise((r) => setTimeout(r, 30));
    expect(acquired).toBe(1);   // ONE pooled device for both dispatches — the fix; pre-fix this was 2
    engine[Symbol.dispose]();
  });

  it('a pooled backend binds a producer resident output as a consumer input (residency fires through the engine)', async () => {
    let acquired = 0; let residentBinds = 0;
    const makeSpy = async (): Promise<Backend> => {
      acquired++;
      const INSTANCE = {};   // this backend instance's identity token
      return {
        kind: 'webgpu', limits: { maxStorageBufferBindingSize: 1 << 28, maxComputeWorkgroupsPerDimension: 65535 },
        async dispatch(input: DispatchInput) {
          // Count resident inputs bound to THIS instance (same-instance token match) — the residency fast-path.
          if (input.residentInputs) for (const [, v] of input.residentInputs) {
            if (v && typeof v === 'object' && (v as { token?: unknown }).token === INSTANCE) residentBinds++;
          }
          // Compute the correct output via the emitted CPU closure (the fallback path — always correct).
          // Mirror the interleaved layout: `outputComps` values per cell (default 1).
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
    // Producer A → gpu-buffer. Settle, grab the handle.
    const kernelA = kernelOf(`component a(i) { return i + 1 }\na`, host);
    change(() => { engine.gpu(kernelA, { output: [4], backend: 'webgpu', outputType: 'gpu-buffer' }); });
    await new Promise((r) => setTimeout(r, 20));
    let rA!: ReturnType<GpuEngine['gpu']>;
    change(() => { rA = engine.gpu(kernelA, { output: [4], backend: 'webgpu', outputType: 'gpu-buffer' }); });
    const hA = rA.value as object;
    // Consumer B closes over A's handle. Same 'webgpu' requested → SAME pooled instance → token matches.
    const envB: HostEnvironment = {
      resolveCall(head: string, _k: string, _a: Arg[], _c: HostValue[], _s: SourceSpan) {
        return head === 'resident' ? { handled: true as const, value: hA as HostValue, kind: 'value' as const } : { handled: false as const };
      },
    };
    const kernelB = evaluateProgram(`const hA = resident()\ncomponent b(i) { return hA[i] * 10 }\nb`, { host, env: envB }).value as UserFn;
    change(() => { engine.gpu(kernelB, { output: [4], backend: 'webgpu' }); });
    await new Promise((r) => setTimeout(r, 20));
    let rB!: ReturnType<GpuEngine['gpu']>;
    change(() => { rB = engine.gpu(kernelB, { output: [4], backend: 'webgpu' }); });
    expect(acquired).toBe(1);         // ONE pooled instance for A + B
    expect(residentBinds).toBe(1);    // B bound A's resident output on the SAME instance — residency FIRED
    expect(rB.value).toEqual([10, 20, 30, 40]);   // and the values are correct
    engine[Symbol.dispose]();
  });

  it('two distinct same-length user buffers under the SAME kernel body do NOT collide (map path content fingerprint)', async () => {
    // Both kernels have an IDENTICAL AST (only the closed-over buffer differs), so kernelHash is identical.
    // A fresh f32 buffer reads generation 0 and its contents aren't in kernelHash, so pre-fix the second
    // dispatch returned the first's cached output (a silent stale result). The content fingerprint splits them.
    const host = new RuntimeReactiveHost();
    const engine = new GpuEngine(host, cpuOnlyDeps);
    const bufA = evaluateProgram(`f32([1, 2, 3, 4])`, { host, env: new RecordingHostEnv(), builtins: [MATH_BUILTINS] }).value as object;
    const bufB = evaluateProgram(`f32([10, 20, 30, 40])`, { host, env: new RecordingHostEnv(), builtins: [MATH_BUILTINS] }).value as object;
    const mkK = (buf: object): UserFn => {
      const env: HostEnvironment = {
        resolveCall(head: string, _k: string, _a: Arg[], _c: HostValue[], _s: SourceSpan) {
          return head === 'src' ? { handled: true as const, value: buf as HostValue, kind: 'value' as const } : { handled: false as const };
        },
      };
      return evaluateProgram(`const src = src()\ncomponent k(i) { return src[i] * 100 }\nk`, { host, env }).value as UserFn;
    };
    const kA = mkK(bufA); const kB = mkK(bufB);
    change(() => { engine.gpu(kA, { output: [4], backend: 'cpu' }); });
    change(() => { engine.gpu(kB, { output: [4], backend: 'cpu' }); });
    await new Promise((r) => setTimeout(r, 30));
    let rA!: ReturnType<GpuEngine['gpu']>; let rB!: ReturnType<GpuEngine['gpu']>;
    change(() => { rA = engine.gpu(kA, { output: [4], backend: 'cpu' }); });
    change(() => { rB = engine.gpu(kB, { output: [4], backend: 'cpu' }); });
    expect(rA.value).toEqual([100, 200, 300, 400]);
    expect(rB.value).toEqual([1000, 2000, 3000, 4000]);   // was [100,200,300,400] pre-fix (the collision)
  });

  it('re-ladders to the next backend when a runtime dispatch throws (cpu is the true floor at dispatch time)', async () => {
    let webgpuDispatched = 0; let webgpuDisposed = 0;
    const throwingWebgpu = async (): Promise<Backend> => ({
      kind: 'webgpu', limits: { maxStorageBufferBindingSize: 1 << 28, maxComputeWorkgroupsPerDimension: 65535 },
      async dispatch() { webgpuDispatched++; throw new Error('device lost mid-dispatch'); },
      [Symbol.dispose]() { webgpuDisposed++; },
    });
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const x = f32(4, (i) => i)\ncomponent k(i) { return x[i] * 2 }\nk`, host);
    // requested 'auto' → webgpu (the throwing spy) → its dispatch throws → re-ladder to webgl2 (null here) → cpu.
    const engine = new GpuEngine(host, { tryWebGpu: throwingWebgpu, tryWebGl2: () => null, limitsHint: { maxStorageBufferBindingSize: 1 << 28, maxComputeWorkgroupsPerDimension: 65535 } });
    const cfg = { output: [4], verify: true };   // auto backend
    change(() => { engine.gpu(kernel, cfg); });
    await new Promise((r) => setTimeout(r, 40));
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, cfg); });
    expect(webgpuDispatched).toBe(1);        // the webgpu dispatch was attempted (and threw)
    expect(s.pending).toBe(false);
    expect(s.error).toBeNull();              // NOT a terminal MLGPU-DISPATCH — it re-laddered
    expect(s.backend).toBe('cpu');           // settled on the cpu floor
    expect(s.value).toEqual([0, 2, 4, 6]);   // correct values from the cpu dispatch
    expect(s.match?.ok).toBe(true);          // verify passed on the cpu result
    expect(webgpuDisposed).toBe(0);          // the pooled webgpu backend was NOT disposed by the retry
    engine[Symbol.dispose]();
  });

  it('re-ladders through TWO failing rungs (webgpu throws → webgl2 throws → cpu succeeds)', async () => {
    // The single-throw test above resolves webgl2 to a cpu backend (tryWebGl2: () => null), so the
    // webgl2-dispatch-throws → cpu fall never runs. Pin the genuine 3-rung fall: BOTH a throwing webgpu
    // AND a throwing webgl2, asserting it falls all the way to the cpu floor.
    let webgpuTried = 0; let webgl2Tried = 0;
    const throwingWebgpu = async (): Promise<Backend> => ({
      kind: 'webgpu', limits: { maxStorageBufferBindingSize: 1 << 28, maxComputeWorkgroupsPerDimension: 65535 },
      async dispatch() { webgpuTried++; throw new Error('webgpu device lost'); }, [Symbol.dispose]() {},
    });
    const throwingWebgl2 = (): Backend => ({
      kind: 'webgl2', limits: { maxStorageBufferBindingSize: 1 << 28, maxComputeWorkgroupsPerDimension: 65535 },
      async dispatch() { webgl2Tried++; throw new Error('webgl2 context lost'); }, [Symbol.dispose]() {},
    });
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const x = f32(4, (i) => i)\ncomponent k(i) { return x[i] * 2 }\nk`, host);
    const engine = new GpuEngine(host, { tryWebGpu: throwingWebgpu, tryWebGl2: throwingWebgl2, limitsHint: { maxStorageBufferBindingSize: 1 << 28, maxComputeWorkgroupsPerDimension: 65535 } });
    const cfg = { output: [4], verify: true };   // auto → webgpu(throws) → webgl2(throws) → cpu
    change(() => { engine.gpu(kernel, cfg); });
    await new Promise((r) => setTimeout(r, 40));
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, cfg); });
    expect(webgpuTried).toBe(1);              // webgpu attempted + threw
    expect(webgl2Tried).toBe(1);             // webgl2 attempted + threw (the middle rung)
    expect(s.pending).toBe(false);
    expect(s.error).toBeNull();              // NOT terminal — fell through to cpu
    expect(s.backend).toBe('cpu');
    expect(s.value).toEqual([0, 2, 4, 6]);
    engine[Symbol.dispose]();
  });

  it('a gate-accepted-but-un-emittable head fails closed to a local MLGPU-EMIT diagnostic (no tree collapse)', () => {
    // `perspective` builds a mat4 and has NO shader lowering, yet passes the (shape-only) gate — so the
    // synchronous emit throws the loud no-lowering error. The engine's safeEmit must catch it into a local
    // MLGPU-EMIT diagnostic (non-core-like: non-pending + an error) rather than letting the throw escape the
    // reader's derive and collapse the whole component tree. This is the defense-in-depth for a gate↔emitter
    // drift: a wrong-but-quiet 0 (the old silent placeholder) is upgraded to a caught, visible error.
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf('component k(i) { return (perspective(1, 1, 1, 2) * vec4(1, 0, 0, 0)).x } k', host);
    const engine = new GpuEngine(host, cpuOnlyDeps);
    let r!: ReturnType<GpuEngine['gpu']>;
    // gpu() must NOT throw (would escape into the reader's derive); it returns a settled-with-error resource.
    expect(() => { change(() => { r = engine.gpu(kernel, { output: [4], backend: 'cpu' }); }); }).not.toThrow();
    expect(r.pending).toBe(false);
    expect(r.error?.code).toBe('MLGPU-EMIT');
    engine[Symbol.dispose]();
  });
});
