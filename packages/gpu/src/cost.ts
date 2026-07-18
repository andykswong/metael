// Interpreter budgets bound the INTERPRETER, not the GPU. Before any dispatch, bound the allocation:
// total input bytes are summed into the package-cap (MAX_GPU_ALLOC) check; the OUTPUT buffer is checked
// against the per-binding storage limit; each INPUT buffer's per-binding limit is validated by the
// device backend at dispatch time (this gate holds only the lumped input total, not per-binding lengths).
import type { Diagnostic } from '@metael/lang';
import { makeDiagnostic } from '@metael/lang';

export const MAX_GPU_ALLOC = 512 * 1024 * 1024;   // 512 MiB total-allocation ceiling (independent of device)

export interface DeviceLimits { readonly maxStorageBufferBindingSize: number; readonly maxComputeWorkgroupsPerDimension: number }
export const CPU_LIMITS: DeviceLimits = { maxStorageBufferBindingSize: MAX_GPU_ALLOC, maxComputeWorkgroupsPerDimension: 65535 };

/** Bound the allocation for a dispatch. Returns null if within limits, else an MLGPU-ALLOC diagnostic.
 *  `outputBytes` = output element count × element bytes (checked against the per-binding storage limit);
 *  `inputBytes` = Σ input-buffer bytes (summed with outputBytes into the MAX_GPU_ALLOC package cap only —
 *  each input's per-binding limit is the device backend's check at dispatch). The caller, which holds the
 *  live buffer lengths, computes both totals. */
export function checkCost(outputBytes: number, inputBytes: number, output: readonly number[], limits: DeviceLimits): Diagnostic | null {
  const total = outputBytes + inputBytes;
  // Every output dimension must be a positive integer. A negative dim yields negative outputBytes that
  // slips past every ceiling check below (then crashes the Float32Array allocation); a fractional/NaN dim
  // silently truncates the dispatch grid. Reject up front with the same MLGPU-ALLOC contract.
  if (output.length === 0) return makeDiagnostic('MLGPU-ALLOC', 'output shape must have at least one dimension');
  for (const dim of output) if (!Number.isInteger(dim) || dim < 1) return makeDiagnostic('MLGPU-ALLOC', `output dimension ${dim} must be a positive integer`);
  for (const dim of output) if (dim > limits.maxComputeWorkgroupsPerDimension * 256) return makeDiagnostic('MLGPU-ALLOC', `output dimension ${dim} exceeds the device dispatch limit`);
  if (outputBytes > limits.maxStorageBufferBindingSize) return makeDiagnostic('MLGPU-ALLOC', `output buffer (${outputBytes} bytes) exceeds the device storage-buffer limit (${limits.maxStorageBufferBindingSize})`);
  if (total > MAX_GPU_ALLOC) return makeDiagnostic('MLGPU-ALLOC', `total allocation (${total} bytes) exceeds the ${MAX_GPU_ALLOC}-byte ceiling`);
  return null;
}
