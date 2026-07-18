// A GpuBufferHandle: a custom value wrapping a buffer that lives on the GPU (resident). It classifies as a
// linear-buffer input (so the binding table + memo treat it like any typed array), binds directly when the
// next dispatch is on the SAME backend (no readback), and lazily reads back to a cached Float32Array when a
// CPU reader (an index / iterate / a different backend) needs the values. The handle owns its GPU buffer;
// its dispose frees it (called by the engine's memo LRU + teardown).
import { tagCustom, NOT_HANDLED, BufferError, type Lowering, type TypeDescriptor } from '@metael/lang';
import type { BackendKind } from './device/index.ts';

const RESIDENT: unique symbol = Symbol('mlgpu.resident');
// A per-handle monotonic nonce. The custom-type generation signal is NOT usable here: tagCustom attaches it
// only for a MUTABLE descriptor (setIndex/setMember present) — this descriptor is immutable, so generationOf
// would return undefined. The memo (resource.ts) reads THIS nonce for a resident-handle input instead.
let NONCE = 0;
interface ResidentState {
  readonly backendKind: BackendKind;
  readonly length: number;
  readonly nonce: number;                     // memo-key change-signal (NOT the custom-type generation)
  readonly gpuBuffer?: unknown;               // the backend-native resident buffer (GPUBuffer / WebGLTexture); undefined for cpu
  readonly readback: () => Float32Array;      // lazily materialize the CPU values
  cache: Float32Array | null;                 // filled on first CPU read
  disposed: boolean;
  dispose?: () => void;                        // free the resident GPU buffer
}
interface ResidentBox { [RESIDENT]: ResidentState }

const LOWER: Lowering = { element: 'f32', shape: 'scalar', gpuStorable: true, access: 'linear-buffer' };

const readCache = (s: ResidentState): Float32Array => {
  if (s.cache !== null) return s.cache;
  // Dispose frees only the GPU-side buffer, not an already-materialized CPU cache. A read AFTER dispose is
  // only unsafe if the cache was never filled — that would call readback() against a freed buffer (a native
  // crash / garbage). Surface it as a catchable diagnostic instead. (A cached read above already returned.)
  if (s.disposed) throw new BufferError('MLGPU-USE-AFTER-DISPOSE', 'read of a disposed resident buffer whose values were never materialized');
  const data = s.readback();
  // The declared length gates every bounds check (getIndex); if the actual readback is shorter, an in-range
  // index would launder `undefined` through `as number`. Validate once, on the single fill. A LONGER readback
  // is harmless (extra trailing cells are never indexed) — only a SHORT one is corrupting.
  if (data.length < s.length) throw new BufferError('MLGPU-READBACK-SHORT', `resident readback returned ${data.length} elements but the handle declares length ${s.length}`);
  return (s.cache = data);
};

const DESCRIPTOR: TypeDescriptor = {
  name: 'gpubuffer',
  lower: LOWER,
  frozen: () => true,   // a resident output is immutable (a pipeline stage, not a mutable let)
  getMember: (v, prop) => (prop === 'length' ? (v as ResidentBox)[RESIDENT].length : NOT_HANDLED),
  getIndex: (v, key) => {
    if (typeof key !== 'number') return NOT_HANDLED;
    const s = (v as ResidentBox)[RESIDENT];
    if (!Number.isInteger(key) || key < 0 || key >= s.length) throw new BufferError('ML-LANG-INDEX-RANGE', `index ${String(key)} is out of range (length ${s.length})`);
    // The preceding bounds check guarantees `key` is in range, so the read is defined (the `as number`
    // narrows away noUncheckedIndexedAccess's `number | undefined`).
    return readCache(s)[key] as number;
  },
  iterate: (v) => Array.from(readCache((v as ResidentBox)[RESIDENT])),
  bufferView: (v) => ({ data: readCache((v as ResidentBox)[RESIDENT]), element: 'f32' as const }),
  display: (v) => { const s = (v as ResidentBox)[RESIDENT]; return `gpubuffer[${s.backendKind}, len ${s.length}]`; },
};

export interface HandleSpec {
  readonly backendKind: BackendKind;
  readonly length: number;
  readonly gpuBuffer?: unknown;
  readonly readback: () => Float32Array;
  readonly dispose?: () => void;
}
export function makeGpuBufferHandle(spec: HandleSpec): object {
  const box: ResidentBox = { [RESIDENT]: { backendKind: spec.backendKind, length: spec.length, nonce: ++NONCE, gpuBuffer: spec.gpuBuffer, readback: spec.readback, cache: null, disposed: false, dispose: spec.dispose } };
  // tagCustom with NO gen — the descriptor is immutable, so a gen would be dropped anyway; the nonce is the
  // memo change-signal. tagCustom still records the descriptor so descriptorOf/isTypedArray classify it.
  return tagCustom(box as object, DESCRIPTOR);
}
export function residentInfo(v: unknown): { backendKind: BackendKind; gpuBuffer: unknown; length: number; nonce: number; disposed: boolean } | null {
  const s = (v as Partial<ResidentBox>)?.[RESIDENT];
  // Report `disposed` so a consumer's resolveInputs never offers a FREED buffer as a resident bind (a
  // use-after-free: WebGL2 binds a deleted texture → silent 0s; WebGPU → a validation throw). A disposed
  // handle falls to the readback-cache upload path instead (its cache was pre-filled before eviction).
  return s ? { backendKind: s.backendKind, gpuBuffer: s.gpuBuffer, length: s.length, nonce: s.nonce, disposed: s.disposed } : null;
}
export function disposeHandle(v: unknown): void {
  const s = (v as Partial<ResidentBox>)?.[RESIDENT];
  if (s && !s.disposed) { s.disposed = true; s.dispose?.(); }
}
