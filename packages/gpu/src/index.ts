// @metael/gpu public barrel — the API-first compute core: the batteries-included host-TS engine facade
// (createGpuEngine → dispatch) + the free settle/subscribe/settled helpers + gpuBuffer, the engine, and the
// device-acquisition seam. The metael-DSL binding (GpuHostEnv + compileKernel) lives in the ./lang subpath
// (@metael/gpu/lang) — importing it, not this barrel, is what pulls the interpreter (evaluateProgram). The
// gate/bounds/binding/emitter/oracle/hash pieces are IMPLEMENTATION DETAIL — the tests reach them by
// relative path (./gate.ts, ./emit-wgsl.ts, …), not through this barrel, so they carry no public stability
// contract and are deliberately NOT re-exported here.

// The host-TS façade — the one-call front door for driving compute from TypeScript.
export { createGpuEngine } from './api.ts';
export type { CreateGpuEngineOptions, GpuEngineFacade, DispatchConfig } from './api.ts';
// The free helpers over a `() => facade.dispatch(k, cfg)` thunk: await/subscribe/narrow a dispatch.
export { settle, subscribe, settled } from './settle.ts';
// Wrap plain host data (Float32Array | number[]) into a reduce/histogram input buffer without hand-boxing.
export { gpuBuffer } from './buffer.ts';

// The engine the vocabulary drives (a host may construct + drive it directly, beyond the façade).
export { GpuEngine } from './resource.ts';
export type { GpuConfig, ReduceConfig, HistogramConfig, GpuResource, GpuEngineDeps } from './resource.ts';

// The device-acquisition seam: an embedder building custom GpuEngineDeps needs the backend probes, the
// pre-acquisition device-limits hint, and the Backend/BackendKind/DeviceLimits types the deps reference.
export { CPU_LIMITS } from './cost.ts';
export type { DeviceLimits } from './cost.ts';
export { tryWebGpuBackend } from './device/webgpu.ts';
export { tryWebGl2Backend } from './device/webgl2.ts';
export type { Backend, BackendKind } from './device/index.ts';
