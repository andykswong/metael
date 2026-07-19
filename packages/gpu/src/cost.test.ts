import { describe, it, expect } from 'vitest';
import { checkCost, CPU_LIMITS, MAX_GPU_ALLOC } from './cost.ts';

describe('resource-cost gate', () => {
  it('accepts a modest dispatch', () => {
    expect(checkCost(512 * 512 * 4, 2 * 512 * 512 * 4, [512, 512], CPU_LIMITS)).toBeNull();
  });
  it('rejects an over-ceiling total allocation with MLGPU-ALLOC', () => {
    const huge = MAX_GPU_ALLOC + 1;
    expect(checkCost(huge, 0, [huge / 4], CPU_LIMITS)?.code).toBe('MLGPU-ALLOC');
  });
  it('rejects an over-limit single output buffer', () => {
    expect(checkCost(CPU_LIMITS.maxStorageBufferBindingSize + 4, 0, [1], CPU_LIMITS)?.code).toBe('MLGPU-ALLOC');
  });
  it('rejects a non-positive-integer output dimension (negative / zero / fractional / NaN) with MLGPU-ALLOC', () => {
    // A negative dim yields negative outputBytes that slips past the ceiling checks then crashes the
    // Float32Array alloc; a fractional/NaN dim silently truncates the grid. All must fail loud up front.
    expect(checkCost(-20, 0, [-5], CPU_LIMITS)?.code).toBe('MLGPU-ALLOC');
    expect(checkCost(0, 0, [0], CPU_LIMITS)?.code).toBe('MLGPU-ALLOC');
    expect(checkCost(10, 0, [2.5], CPU_LIMITS)?.code).toBe('MLGPU-ALLOC');
    expect(checkCost(NaN, 0, [NaN], CPU_LIMITS)?.code).toBe('MLGPU-ALLOC');
    expect(checkCost(0, 0, [], CPU_LIMITS)?.code).toBe('MLGPU-ALLOC');   // no dimensions
  });
  it('still accepts a valid 2-D output shape', () => {
    expect(checkCost(8 * 8 * 4, 0, [8, 8], CPU_LIMITS)).toBeNull();
  });
  it('accepts a valid 3-D output shape and bounds every one of its three dimensions', () => {
    // The per-dim guards iterate the whole `output` array, so a 3-D shape is covered with no extra code.
    expect(checkCost(4 * 4 * 4 * 4, 0, [4, 4, 4], CPU_LIMITS)).toBeNull();
    // A non-positive 3rd dimension is rejected (proves the loop reaches dims[2]).
    expect(checkCost(4 * 4 * 4, 0, [4, 4, 0], CPU_LIMITS)?.code).toBe('MLGPU-ALLOC');
    expect(checkCost(4 * 4 * 4, 0, [4, 4, -1], CPU_LIMITS)?.code).toBe('MLGPU-ALLOC');
    // A 3rd dimension over the per-dimension dispatch ceiling is rejected.
    const overDim = CPU_LIMITS.maxComputeWorkgroupsPerDimension * 256 + 1;
    expect(checkCost(4, 0, [1, 1, overDim], CPU_LIMITS)?.code).toBe('MLGPU-ALLOC');
  });
});
