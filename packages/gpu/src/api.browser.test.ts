import { describe, it, expect } from 'vitest';
import { createGpuEngine } from './api.ts';
import { settle } from './settle.ts';
import { compileKernel } from './lang/compile-kernel.ts';

describe('createGpuEngine — real adapter (Chromium)', () => {
  it('settles on a real backend with correct values + shader text', async () => {
    const gpu = createGpuEngine();   // real device ladder (WebGPU→WebGL2→CPU)
    const kernel = compileKernel(`const a = f32(16, (i) => i)\nconst b = f32(16, (i) => i * 2)\ncomponent add(i) { return a[i] + b[i] }\nadd`, gpu.host);
    const r = await settle(() => gpu.dispatch(kernel, { output: [16] }));
    expect(['webgpu', 'webgl2', 'cpu']).toContain(r.backend);
    expect(r.value).toEqual(Array.from({ length: 16 }, (_, i) => i + i * 2));
    expect(r.wgsl.length).toBeGreaterThan(0);
    gpu[Symbol.dispose]();
  });

  it('verify:true confirms the GPU output matches the interpreter oracle on a real adapter', async () => {
    const gpu = createGpuEngine();
    const kernel = compileKernel(`const x = f32(16, (i) => i)\ncomponent k(i) { return x[i] * 3 }\nk`, gpu.host);
    const r = await settle(() => gpu.dispatch(kernel, { output: [16], verify: true }));
    expect(r.match?.ok).toBe(true);
    gpu[Symbol.dispose]();
  });
});
