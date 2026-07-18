import { describe, it, expect } from 'vitest';
import * as gpu from './index.ts';

describe('@metael/gpu public API surface', () => {
  it('exports the host-API facade alongside the engine', () => {
    expect(typeof gpu.createGpuEngine).toBe('function');
    expect(typeof gpu.compileKernel).toBe('function');
    expect(typeof gpu.GpuEngine).toBe('function');   // engine still exported
  });
});
