// packages/gpu/src/api.ts
// The host-API façade: batteries-included plumbing so host TypeScript drives the compute engine without
// hand-wiring a host + engine + the change()/drain/re-read settle dance. This is the API-first CORE: it
// carries no interpreter dependency. The metael-DSL binding — compiling a kernel from a source string
// (compileKernel) + the head vocabulary (GpuHostEnv) — lives behind the ./lang subpath.
import { RuntimeReactiveHost, change } from '@metael/runtime';   // value use (new RuntimeReactiveHost) → plain import
import { type UserFn } from '@metael/lang';   // the kernel value shape the engine lowers (built via @metael/gpu/lang)
import { GpuEngine, type GpuConfig, type ReduceConfig, type HistogramConfig, type GpuResource, type GpuEngineDeps } from './resource.ts';
// The backends live in their own modules — device/index.ts exports only selectBackend + types.
import { tryWebGpuBackend } from './device/webgpu.ts';
import { tryWebGl2Backend } from './device/webgl2.ts';
import { CPU_LIMITS } from './cost.ts';

/** Options for {@link createGpuEngine}: run CPU-only (headless/test), or inject fully custom device deps. */
export interface CreateGpuEngineOptions {
  /** CPU-only (headless/test) deps: no WebGPU/WebGL2 acquisition. Default false → the real device ladder. */
  readonly cpuOnly?: boolean;
  /** Fully custom deps (overrides cpuOnly). */
  readonly deps?: GpuEngineDeps;
}

/** The discriminated dispatch config: `cfg.mode` selects the kernel kind. `'map'` (default, or omitted)
 *  folds a {@link GpuConfig} through `engine.gpu`; `'reduce'` a {@link ReduceConfig} through `engine.gpuReduce`;
 *  `'histogram'` a {@link HistogramConfig} through `engine.gpuHistogram`. The mode makes the caller's intent
 *  explicit and picks the arity the engine's gate expects (N thread coords vs a 2-arg reducer vs a 1-arg
 *  bin-mapper). */
export type DispatchConfig =
  | ({ mode?: 'map' } & GpuConfig)
  | ({ mode: 'reduce' } & ReduceConfig)
  | ({ mode: 'histogram' } & HistogramConfig);

/** A thin façade over a {@link GpuEngine} + its reactive host: `dispatch` wires up the change() boundary so
 *  host TypeScript can drive a compute kernel without hand-rolling the reactive plumbing. Await or subscribe
 *  to a dispatch with the FREE `settle`/`subscribe` helpers over a `() => facade.dispatch(k, cfg)` thunk.
 *  Dispose it to free the engine. */
export interface GpuEngineFacade extends Disposable {
  /** The reactive host the engine + every compiled kernel's closure live on. */
  readonly host: RuntimeReactiveHost;
  /** The underlying engine, for direct access beyond the façade's helpers. */
  readonly engine: GpuEngine;
  /** Dispatch inside the change() boundary; returns the (pending) resource synchronously. `cfg.mode` selects
   *  the kind: `'map'` (default) folds to `engine.gpu`, `'reduce'` to `engine.gpuReduce`, `'histogram'` to
   *  `engine.gpuHistogram`. Await it with the free `settle(() => facade.dispatch(k, cfg))`. Throws after
   *  dispose. */
  dispatch(kernel: UserFn, cfg: DispatchConfig): GpuResource;
  // [Symbol.dispose](): void — from Disposable: frees the engine (device + memo); dispatch throws afterward.
}

const cpuOnlyDeps: GpuEngineDeps = { tryWebGpu: async () => null, tryWebGl2: () => null, limitsHint: CPU_LIMITS };
const realDeps: GpuEngineDeps = { tryWebGpu: tryWebGpuBackend, tryWebGl2: tryWebGl2Backend, limitsHint: CPU_LIMITS };

/** Create a ready-to-use {@link GpuEngineFacade}: a fresh reactive host + a {@link GpuEngine} over the real
 *  device ladder (or CPU-only / custom deps per `opts`), with a mode-routing `dispatch`. Await or subscribe
 *  to a dispatch with the free `settle`/`subscribe` helpers. The one-call entry point for driving compute
 *  from host TypeScript. */
export function createGpuEngine(opts: CreateGpuEngineOptions = {}): GpuEngineFacade {
  const host = new RuntimeReactiveHost();
  const deps = opts.deps ?? (opts.cpuOnly ? cpuOnlyDeps : realDeps);
  const engine = new GpuEngine(host, deps);
  let disposed = false;   // after dispose a fresh dispatch throws — this breaks the free settle's re-dispatch loop

  const dispatch = (kernel: UserFn, cfg: DispatchConfig): GpuResource => {
    if (disposed) throw new Error('gpu engine facade has been disposed');
    let r!: GpuResource;
    change(() => {
      switch (cfg.mode) {
        case 'reduce':    r = engine.gpuReduce(kernel, cfg); break;
        case 'histogram': r = engine.gpuHistogram(kernel, cfg); break;
        default:          r = engine.gpu(kernel, cfg); break;   // 'map' | undefined
      }
    });
    return r;
  };

  return {
    host, engine,
    dispatch,
    [Symbol.dispose]: () => { disposed = true; engine[Symbol.dispose](); },
  };
}
