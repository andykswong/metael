// packages/gpu/src/buffer.ts
// A free helper to wrap plain host data into a metael typed-array custom value, so host TypeScript can
// supply a reduce/histogram input without hand-boxing a metael value or running evaluateProgram. Imports
// the value protocol (@metael/math/lang) + the custom-type tags (@metael/lang) — NOT the interpreter.
import type { ReactiveHost } from '@metael/lang';
import { markFrozen } from '@metael/lang';
import { makeTypedArray } from '@metael/math/lang';

/** Wrap a plain `Float32Array | number[]` into a frozen f32 typed-array custom value — the host-friendly way
 *  to supply a `dispatch(..., { mode:'reduce'|'histogram', input })` buffer WITHOUT hand-boxing a metael
 *  value or running `evaluateProgram`. Takes `host` because the typed-array descriptor needs a generation
 *  ref for reactive in-place mutation (only a `ReactiveHost` mints one); the façade exposes `engine.host`.
 *  A resident `GpuBufferHandle` from a prior `outputType:'gpu-buffer'` dispatch is already a valid input, so
 *  this is only the PLAIN-data entry point. */
export function gpuBuffer(data: Float32Array | readonly number[], host: ReactiveHost): object {
  const store = data instanceof Float32Array ? data : Float32Array.from(data);
  const buf = makeTypedArray('f32', store, host.allocateGeneration());
  markFrozen(buf);   // a const buffer — immutable, like a `const x = f32([...])` in source
  return buf;
}
