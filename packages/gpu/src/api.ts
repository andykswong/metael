// packages/gpu/src/api.ts
// The host-API façade: batteries-included helpers so host TypeScript drives the compute engine without
// hand-wiring a host + engine + the change()/drain/re-read settle dance. The KERNEL is authored in metael
// (its AST is what the emitters lower — there is no JS-closure kernel path), so compileKernel is the one
// point that touches the language, and only as the kernel-authoring format. A kernel body has no head
// calls, so RecordingHostEnv's resolveCall is never exercised (a stray head-in-expression fails closed).
import { RuntimeReactiveHost, change, effect } from '@metael/runtime';   // value use (new RuntimeReactiveHost) → plain import
import { evaluateProgram, isUserFn, RecordingHostEnv, type UserFn } from '@metael/lang';
import { GpuEngine, type GpuConfig, type GpuResource, type GpuEngineDeps } from './resource.ts';
// The backends live in their own modules — device/index.ts exports only selectBackend + types.
import { tryWebGpuBackend } from './device/webgpu.ts';
import { tryWebGl2Backend } from './device/webgl2.ts';
import { CPU_LIMITS } from './cost.ts';

/** Compile a metael kernel snippet into the UserFn the engine lowers. Evaluate against `host` so the
 *  kernel's closure (its `const a = f32(...)` inputs, or a factory's captured params) lives on the same
 *  host the engine reads. Throws if the program's value is not a function/component. */
export function compileKernel(src: string, host: RuntimeReactiveHost): UserFn {
  const res = evaluateProgram(src, { host, env: new RecordingHostEnv() });
  if (!isUserFn(res.value)) throw new Error('kernel source must evaluate to a function or component');
  return res.value;
}

export interface CreateGpuEngineOptions {
  /** CPU-only (headless/test) deps: no WebGPU/WebGL2 acquisition. Default false → the real device ladder. */
  readonly cpuOnly?: boolean;
  /** Fully custom deps (overrides cpuOnly). */
  readonly deps?: GpuEngineDeps;
}

export interface GpuEngineFacade extends Disposable {
  readonly host: RuntimeReactiveHost;
  readonly engine: GpuEngine;
  /** Compile a kernel snippet against this façade's host (so its closure shares the engine's host). */
  compile(src: string): UserFn;
  /** Dispatch inside the change() boundary; returns the (pending) resource synchronously. Throws after dispose. */
  dispatch(kernel: UserFn, cfg: GpuConfig): GpuResource;
  /** Await the settled resource (dispatch → drain macrotasks → re-read the memo until settled). Throws after dispose. */
  settle(kernel: UserFn, cfg: GpuConfig): Promise<GpuResource>;
  /** Subscribe to a resource's lifecycle for a fixed (kernel, cfg). `onValue` fires once with the PENDING
   *  resource, then once with the SETTLED resource — guard `if (!r.pending)` for the value. `onValue` runs
   *  inside the effect's reactive tracking scope, so it must sink to NON-reactive targets (a reactive read
   *  inside onValue would subscribe and cause spurious re-fires). To rebind a REUSABLE kernel to a new
   *  input, re-derive the kernel from a reactive signal and dispatch/settle inside your own effect — a
   *  rebind is a new kernel value, not a re-fire of this subscription. Returns a stop() disposer. Throws
   *  after dispose(). */
  subscribe(kernel: UserFn, cfg: GpuConfig, onValue: (r: GpuResource) => void): () => void;
  // [Symbol.dispose](): void — from Disposable: frees the engine (device + memo); dispatch/settle/subscribe throw afterward.
}

// A macrotask-drain backstop against a never-settling dispatch (far above any real settle latency).
const MAX_SETTLE_ITERS = 10_000;
const cpuOnlyDeps: GpuEngineDeps = { tryWebGpu: async () => null, tryWebGl2: () => null, limitsHint: CPU_LIMITS };
const realDeps: GpuEngineDeps = { tryWebGpu: tryWebGpuBackend, tryWebGl2: tryWebGl2Backend, limitsHint: CPU_LIMITS };

export function createGpuEngine(opts: CreateGpuEngineOptions = {}): GpuEngineFacade {
  const host = new RuntimeReactiveHost();
  const deps = opts.deps ?? (opts.cpuOnly ? cpuOnlyDeps : realDeps);
  const engine = new GpuEngine(host, deps);
  let disposed = false;   // guards settle from spinning: after dispose a fresh gpu() stays pending forever

  const dispatch = (kernel: UserFn, cfg: GpuConfig): GpuResource => {
    if (disposed) throw new Error('gpu engine facade has been disposed');
    let r!: GpuResource;
    change(() => { r = engine.gpu(kernel, cfg); });
    return r;
  };

  const settle = async (kernel: UserFn, cfg: GpuConfig): Promise<GpuResource> => {
    let r = dispatch(kernel, cfg);
    let iters = 0;
    while (r.pending && !r.error && !disposed) {
      if (++iters > MAX_SETTLE_ITERS) throw new Error('gpu dispatch did not settle within the iteration bound');
      await new Promise<void>((res) => setTimeout(res, 0));
      if (disposed) break;
      r = dispatch(kernel, cfg);
    }
    return r;
  };

  const subscribe = (kernel: UserFn, cfg: GpuConfig, onValue: (r: GpuResource) => void): () => void => {
    if (disposed) throw new Error('gpu engine facade has been disposed');
    // Reading engine.gpu inside a tracked effect subscribes to the resource cell (the engine reads the
    // settled value through its cell so a reader re-runs on writeCell). change() is only a batch boundary,
    // not a tracking boundary, so the cell read is still tracked. On the settle writeCell the effect
    // re-fires; gpu() then hits the memo (now holding the settled resource) and early-returns BEFORE the
    // pending branch → no re-enqueue → converges after one re-fire.
    return effect(() => {
      let r!: GpuResource;
      change(() => { r = engine.gpu(kernel, cfg); });
      onValue(r);
    });
  };

  return {
    host, engine,
    compile: (src) => compileKernel(src, host),
    dispatch, settle, subscribe,
    [Symbol.dispose]: () => { disposed = true; engine[Symbol.dispose](); },
  };
}
