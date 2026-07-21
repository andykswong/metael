import { describe, it, expect } from 'vitest';
import * as gpu from './index.ts';
import * as gpuLang from './lang/index.ts';

describe('@metael/gpu public API surface', () => {
  it('exports the API-first facade + free helpers + the engine (no interpreter)', () => {
    expect(typeof gpu.createGpuEngine).toBe('function');
    expect(typeof gpu.settle).toBe('function');
    expect(typeof gpu.subscribe).toBe('function');
    expect(typeof gpu.settled).toBe('function');
    expect(typeof gpu.gpuBuffer).toBe('function');
    expect(typeof gpu.GpuEngine).toBe('function');   // engine still exported
  });

  it('does NOT export the DSL binding from core — it moved to @metael/gpu/lang', () => {
    // compileKernel runs evaluateProgram + GpuHostEnv is the head vocabulary — both live behind ./lang so
    // the core barrel carries no interpreter dependency. Their absence here is the surface proof of the split.
    expect(gpu).not.toHaveProperty('compileKernel');
    expect(gpu).not.toHaveProperty('GpuHostEnv');
  });

  it('the ./lang subpath exports the DSL binding (compileKernel + GpuHostEnv)', () => {
    expect(typeof gpuLang.compileKernel).toBe('function');
    expect(typeof gpuLang.GpuHostEnv).toBe('function');
  });

  it('exports the device-acquisition seam an embedder needs for custom GpuEngineDeps', () => {
    expect(typeof gpu.tryWebGpuBackend).toBe('function');
    expect(typeof gpu.tryWebGl2Backend).toBe('function');
    expect(gpu.CPU_LIMITS).toBeTypeOf('object');
  });

  it('does NOT re-export the compiler/oracle/device internals (they are reached by relative path, not the barrel)', () => {
    // These carry no public stability contract; keeping them out of the barrel keeps the support surface
    // minimal. If a host genuinely needs one, that is a signal to design a real public seam for it.
    for (const internal of [
      'gateKernel', 'normalizeImplicitReturn', 'checkStaticBounds', 'intervalOf', 'buildBindingTable', 'collectFreeNames',
      'checkCost', 'MAX_GPU_ALLOC', 'emitCpu', 'emitWgsl', 'emitGlsl', 'checkMatch',
      'gateReducer', 'cpuReduce', 'gateBinMapper', 'cpuHistogram', 'selectBackend', 'makeCpuBackend',
      'compsOf', 'kernelHash',
    ]) {
      expect(gpu).not.toHaveProperty(internal);
    }
  });
});
