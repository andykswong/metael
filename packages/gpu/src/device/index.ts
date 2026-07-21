// The backend seam: buffers, one pipeline, dispatch, readback, device-limit reporting. Three impls behind
// it (CPU always works; WebGPU + WebGL2 slot in later). Compute-only — no depth buffer / render pass.
import type { UserFn } from '@metael/lang';
import type { BindingTable } from '../binding.ts';
import type { DeviceLimits } from '../cost.ts';

/** Which compute backend ran (or is requested): the WebGPU device, the WebGL2 compute-via-fragment fallback,
 *  or the always-available CPU interpreter floor. */
export type BackendKind = 'webgpu' | 'webgl2' | 'cpu';

/** The payload a backend consumes to run one MAP dispatch: the kernel + its bindings, the output grid, the
 *  precision, both shaders, the CPU-fallback per-cell function, the resolved input buffers + scalar uniforms,
 *  and the optional resident-buffer plumbing (retain the output on-device / bind a prior output directly). */
export interface DispatchInput {
  /** The kernel being dispatched (carried for the CPU backend + oracle; the GPU legs run the shaders). */
  readonly kernel: UserFn;
  /** The kernel's resolved bindings (buffer/scalar/coord roles + shapes). */
  readonly bindings: BindingTable;
  /** The output grid dimensions — one thread per cell. */
  readonly dims: readonly number[];
  /** The numeric precision this dispatch runs at (`'f16'` only reaches a shader-f16 WebGPU device). */
  readonly precision: 'f16' | 'f32';
  /** The WGSL compute shader (consumed by the WebGPU leg). */
  readonly wgsl: string;
  /** The GLSL-ES compute-via-fragment shader (consumed by the WebGL2 leg). */
  readonly glsl: string;
  /** Compute ONE output cell's components (length `outputComps`, default 1). For a scalar (`f32`) output
   *  the array is a single-element `[value]`, byte-identical to the pre-vecN scalar cell; for a vecN output
   *  it is the cell's N components in x,y,z,w order — the FLAT-INTERLEAVED layout every backend produces. */
  readonly cpuRun: (coords: readonly number[]) => number[];
  /** The output element's component width (f32→1, vec2→2, vec3→3, vec4→4). Absent → 1 (scalar, back-compat
   *  with any caller not setting it). Every backend writes `outputComps` values per cell, interleaved. */
  readonly outputComps?: number;
  /** The input buffers by name — each an f32 CPU-side copy every backend can upload (a resident fast-path is
   *  offered separately via `residentInputs`). */
  readonly inputs: readonly { readonly name: string; readonly data: Float32Array }[];
  /** The kernel's closed-over scalar-constant uniforms by name, set as `_u_<name>` in the shaders. */
  readonly scalars: readonly { readonly name: string; readonly value: number }[];
  /** When true, the backend RETAINS the output buffer on-device and returns a `resident` handle to it in the
   *  result (in addition to reading it back for the oracle / CPU-fallback output) — the plumbing that lets a
   *  later dispatch bind this output directly as an input with no readback + re-upload. */
  readonly retainOutput?: boolean;
  /** Resident inputs by name: a buffer produced by a PRIOR dispatch, bound directly (no re-upload) IF it
   *  belongs to THIS backend instance (a GPUBuffer is bound to its GPUDevice, a texture to its context — a
   *  foreign one is unusable). `inputs` still carries the CPU-fallback data for a foreign/cross-backend/CPU
   *  dispatch. The value is a backend-native resident object (opaque). */
  readonly residentInputs?: ReadonlyMap<string, unknown>;
}
/** A REDUCTION dispatch: fold `inputValues` → 1 scalar, seeded by `identity`. Each GPU leg consumes its OWN
 *  shader: `glsl` is the WebGL2 reducer-fold-over-a-tile fragment shader (from `emitReduceGlsl`); `wgsl` is the
 *  WebGPU workgroup-shared tree reduction (from `emitReduceWgsl`). `tile` is the WebGL2 baked TILE constant
 *  (the ping-pong driver sizes each pass's output as ceil(currentLen/tile)); the WebGPU driver uses its own
 *  baked workgroup size G. `scalars` are the reducer's closed-over scalar-constant uniforms (set as `_u_<name>`
 *  by both legs). A DISTINCT payload from the map `DispatchInput` — a reduce is not a per-cell map. */
export interface ReduceDispatchInput {
  readonly glsl: string;
  readonly wgsl: string;
  readonly inputValues: Float32Array;
  readonly identity: number;
  readonly tile: number;
  readonly scalars: readonly { readonly name: string; readonly value: number }[];
}
export interface ReduceDispatchResult { readonly value: number; readonly ms: number }
/** A HISTOGRAM dispatch: an ATOMIC SCATTER of `inputValues` into `bins` counts. `wgsl` is the WebGPU
 *  atomic-scatter compute shader (from `emitHistogramWgsl`) — one thread per input element maps it to a bin
 *  index and `atomicAdd`s that bin. `bins` is the number of buckets (the output count-array length); `scalars`
 *  are the bin-mapper's closed-over scalar-constant uniforms (set as `_u_<name>`). NO `glsl` field: WebGL2's
 *  fragment stage has no atomics, so the histogram FALLS TO the CPU oracle on WebGL2 (the engine routes it
 *  there directly, never through a WebGL2 dispatchHistogram). A DISTINCT payload from the map/reduce dispatches. */
export interface HistogramDispatchInput {
  readonly wgsl: string;
  readonly inputValues: Float32Array;
  readonly bins: number;
  readonly scalars: readonly { readonly name: string; readonly value: number }[];
}
export interface HistogramDispatchResult { readonly counts: number[]; readonly ms: number }
/** The result of a MAP dispatch: the readback output (always present) + the elapsed time, plus the optional
 *  resident-buffer handle when `retainOutput` was requested. */
export interface DispatchResult {
  /** The output values in the flat-interleaved layout (`output[cell * comps + k]`). Always present — it is
   *  the oracle source + the CPU-fallback / foreign-backend readback. */
  readonly output: Float32Array;   // always present (CPU-fallback / oracle source)
  /** The dispatch time in milliseconds. */
  readonly ms: number;
  /** Present when `retainOutput` was set: the backend-native resident buffer + a disposer to free it. The
   *  `gpuBuffer` is opaque (a GPUBuffer / a WebGL2 texture wrapper / a Float32Array for CPU); only the SAME
   *  backend instance that produced it can bind it as a resident input. */
  readonly resident?: { readonly gpuBuffer: unknown; readonly dispose: () => void };
}

/** A compute backend: a device (or the CPU interpreter) that runs a map dispatch, and optionally a reduction
 *  or a histogram scatter. Three impls slot behind it — CPU (always works), WebGPU, and WebGL2. */
export interface Backend {
  /** Which backend this is — used by the engine's re-ladder + backend-honesty reporting. */
  readonly kind: BackendKind;
  /** This device's limits, reported to the cost gate. */
  readonly limits: DeviceLimits;
  /** Optional device-capability flags. `f16` is true when the backend can run a `precision: 'f16'` dispatch
   *  correctly (a WebGPU device with the `shader-f16` feature). Absent/undefined ⇒ NO f16 shader path (the
   *  cpu + webgl2 backends, and a WebGPU device without the feature) → the engine falls an f16 request back
   *  to f32 with a note. */
  readonly features?: { readonly f16: boolean };
  /** Run one MAP dispatch: compute every output cell and read the result back. */
  dispatch(input: DispatchInput): Promise<DispatchResult>;
  /** Run a REDUCTION (fold N inputs → 1 scalar). Optional — a backend that can't reduce (a WGSL leg still
   *  scaffolded) omits it, so the engine re-ladders to a rung that can (webgl2's ping-pong / the cpu floor).
   *  webgl2 realizes a multi-pass ping-pong tree reduction; cpu delegates to the linear-fold oracle. */
  dispatchReduce?(input: ReduceDispatchInput): Promise<ReduceDispatchResult>;
  /** Run a HISTOGRAM (atomic-scatter N inputs → `bins` counts). Optional — only the WebGPU backend implements
   *  it (real storage atomics). The cpu + webgl2 backends OMIT it: cpu because the engine routes a cpu
   *  histogram straight to the `cpuHistogram` oracle, and webgl2 because its fragment stage has NO atomics —
   *  so the engine routes a webgl2 histogram to the cpu oracle too (a documented fallback, settled 'cpu'). */
  dispatchHistogram?(input: HistogramDispatchInput): Promise<HistogramDispatchResult>;
  /** Release the device + any GPU resources this backend holds. */
  [Symbol.dispose](): void;
}

/** Probe for the best available backend, verifying a REAL adapter/device (navigator.gpu truthy ≠ working).
 *  Falls WebGPU → WebGL2 → CPU (the floor). Returns the CPU backend when no GPU is present (headless node). */
export async function selectBackend(prefer: 'auto' | BackendKind, makeCpu: () => Backend,
  tryWebGpu: () => Promise<Backend | null>, tryWebGl2: () => Backend | null): Promise<Backend> {
  if (prefer === 'cpu') return makeCpu();
  if (prefer === 'webgpu' || prefer === 'auto') { const b = await tryWebGpu(); if (b) return b; if (prefer === 'webgpu') return makeCpu(); }
  if (prefer === 'webgl2' || prefer === 'auto') { const b = tryWebGl2(); if (b) return b; }
  return makeCpu();
}
