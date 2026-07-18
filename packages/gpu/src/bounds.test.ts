import { describe, it, expect } from 'vitest';
import { RuntimeReactiveHost, change } from '@metael/runtime';
import { evaluateProgram, isUserFn, RecordingHostEnv } from '@metael/lang';
import type { UserFn } from '@metael/lang';
import { GpuEngine } from './resource.ts';

function kernelOf(src: string, host: RuntimeReactiveHost): UserFn {
  const res = evaluateProgram(src, { host, env: new RecordingHostEnv() });
  if (!isUserFn(res.value)) throw new Error('kernel: ' + JSON.stringify(res.diagnostics)); return res.value;
}
const cpuDeps = { tryWebGpu: async () => null, tryWebGl2: () => null, limitsHint: { maxStorageBufferBindingSize: 1 << 28, maxComputeWorkgroupsPerDimension: 65535 } };

describe('static out-of-bounds bounds-prover', () => {
  it('rejects a provably-OOB index a[i + N] (interval entirely >= length)', async () => {
    const host = new RuntimeReactiveHost();
    // x.length === 4, output [4] → coord i ∈ [0,3], i+4 ∈ [4,7] ≥ 4 → provably OOB.
    const kernel = kernelOf(`const x = f32(4, (i) => i)\ncomponent k(i) { return x[i + 4] }\nk`, host);
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    expect(s.core).toBe(false);
    expect(s.reasons.some((r) => r.code === 'MLGPU-INDEX-STATIC')).toBe(true);
  });
  it('rejects a literal OOB index a[99] on a length-4 buffer', async () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const x = f32(4, (i) => i)\ncomponent k(i) { return x[99] }\nk`, host);
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    expect(s.core).toBe(false);
    expect(s.reasons.some((r) => r.code === 'MLGPU-INDEX-STATIC')).toBe(true);
  });
  it('ACCEPTS a safe in-range index a[i] (interval ⊂ [0,length)) — inert, no diagnostic', async () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const x = f32(4, (i) => i)\ncomponent k(i) { return x[i] }\nk`, host);
    const engine = new GpuEngine(host, cpuDeps);
    change(() => { engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    await new Promise((r) => setTimeout(r, 20));
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    expect(s.core).toBe(true);
    expect(s.value).toEqual([0, 1, 2, 3]);
  });
  it('ACCEPTS a data-dependent / partially-in-range index (a[i-1], a[i*2]) — unprovable → oracle covers it', async () => {
    const host = new RuntimeReactiveHost();
    const k1 = kernelOf(`const x = f32(4, (i) => i)\ncomponent k(i) { return x[i * 2] }\nk`, host);   // [0,6] overlaps [0,4) → pass
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(k1, { output: [4], backend: 'cpu' }); });
    expect(s.core).toBe(true);   // NOT statically rejected — partial OOB is the oracle's job
  });
  it('ACCEPTS a matmul-style index a[row * N + k] (the flagship kernel must not be falsely rejected)', async () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const N = 4
const a = f32(N * N, (i) => i)
component k(row, col) { let s = 0 for (const j of range(N)) { s = s + a[row * N + j] } return s }
k`, host);
    const engine = new GpuEngine(host, cpuDeps);
    change(() => { engine.gpu(kernel, { output: [4, 4], backend: 'cpu' }); });
    await new Promise((r) => setTimeout(r, 20));
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4, 4], backend: 'cpu' }); });
    expect(s.core).toBe(true);   // row∈[0,3], j∈[0,3], N=4 → row*4+j ∈ [0,15] ⊂ [0,16) → SAFE, accepted
  });

  it('ACCEPTS a[i - 1] (interval [-1, N-2] overlaps [0,N) at i>=1) — partial OOB is the oracle/runtime job', async () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const x = f32(4, (i) => i)\ncomponent k(i) { return x[i - 1] }\nk`, host);
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    expect(s.core).toBe(true);
    expect(s.reasons.some((r) => r.code === 'MLGPU-INDEX-STATIC')).toBe(false);
  });

  it('REJECTS a[i - 5] on a length-4 buffer (interval [-5, -2] entirely < 0 even after ceil(hi))', async () => {
    const host = new RuntimeReactiveHost();
    // i ∈ [0,3] → i-5 ∈ [-5, -2]; ceil(-2) = -2 < 0 → provably < 0 for every coord → reject.
    const kernel = kernelOf(`const x = f32(4, (i) => i)\ncomponent k(i) { return x[i - 5] }\nk`, host);
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    expect(s.core).toBe(false);
    expect(s.reasons.some((r) => r.code === 'MLGPU-INDEX-STATIC')).toBe(true);
  });

  it('ACCEPTS a[i * -1] (a negative-operand multiply: interval [-3, 0] overlaps [0,N))', async () => {
    const host = new RuntimeReactiveHost();
    // i ∈ [0,3], i*(-1) ∈ [-3, 0] — includes 0 (in range at i=0) → NOT all-OOB → pass. Proves signed `*`.
    const kernel = kernelOf(`const x = f32(4, (i) => i)\ncomponent k(i) { return x[i * -1] }\nk`, host);
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    expect(s.core).toBe(true);
  });

  it('ACCEPTS a data-dependent index a[b[i]] (an inner buffer read is unprovable → ⊤ → pass)', async () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const a = f32(4, (i) => i)\nconst b = f32(4, (i) => i)\ncomponent k(i) { return a[b[i]] }\nk`, host);
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    expect(s.core).toBe(true);   // b[i] is a runtime value → the index interval is ⊤ → never statically rejected
  });

  it('ACCEPTS a[k] with an unprovable range bound k ∈ range(m), m a data-dependent local', async () => {
    const host = new RuntimeReactiveHost();
    // m = a[0] (a buffer read → ⊤), so range(m)'s var is ⊤ → a[k] is unprovable → pass (never falsely rejected).
    const kernel = kernelOf(`const a = f32(4, (i) => i)\ncomponent k(i) { let s = 0 const m = a[0] for (const j of range(m)) { s = s + a[j] } return s }\nk`, host);
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    expect(s.core).toBe(true);
  });

  it('REJECTS a const-folded provable-OOB index a[base + j], base=8, j∈range(2), on length-4', async () => {
    const host = new RuntimeReactiveHost();
    // base = 8 (const literal), j ∈ [0,1] → base+j ∈ [8,9] ≥ 4 → provably OOB for every coord/j → reject.
    const kernel = kernelOf(`const a = f32(4, (i) => i)\nconst base = 8\ncomponent k(i) { let s = 0 for (const j of range(2)) { s = s + a[base + j] } return s }\nk`, host);
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    expect(s.core).toBe(false);
    expect(s.reasons.some((r) => r.code === 'MLGPU-INDEX-STATIC')).toBe(true);
  });

  it('ACCEPTS an all-OOB index guarded by an if (a[99] under `if`) — the guard may exclude it → not proven-for-every-coord', async () => {
    const host = new RuntimeReactiveHost();
    // a[99] is all-OOB in isolation, but it only runs when i > 100 — which never holds for i ∈ [0,3]. Rejecting
    // would be UNSOUND (the access is unreachable). Suppress the rejection inside a conditional branch.
    const kernel = kernelOf(`const a = f32(4, (i) => i)\ncomponent k(i) { let s = 0 if (i > 100) { s = a[99] } return s }\nk`, host);
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    expect(s.core).toBe(true);
    expect(s.reasons.some((r) => r.code === 'MLGPU-INDEX-STATIC')).toBe(false);
  });

  it('ACCEPTS an all-OOB index in a maybe-zero-iteration loop (a[99] in `for range(m)`, m data-dependent)', async () => {
    const host = new RuntimeReactiveHost();
    // range(m) with m = a[0] could iterate 0 times → the body is not guaranteed → no rejection (unsound to).
    const kernel = kernelOf(`const a = f32(4, (i) => i)\ncomponent k(i) { let s = 0 const m = a[0] for (const j of range(m)) { s = a[99] } return s }\nk`, host);
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    expect(s.core).toBe(true);
    expect(s.reasons.some((r) => r.code === 'MLGPU-INDEX-STATIC')).toBe(false);
  });

  it('ACCEPTS a reassigned accumulator index a[s] where s is mutated in a loop (⊤ → pass)', async () => {
    const host = new RuntimeReactiveHost();
    // s is reassigned (s = s + 1), so a[s] is ⊤ regardless of its init → never falsely rejected.
    const kernel = kernelOf(`const a = f32(4, (i) => i)\ncomponent k(i) { let s = 99 for (const j of range(2)) { s = j } return a[s] }\nk`, host);
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    expect(s.core).toBe(true);   // s is assigned in the loop → its interval is ⊤ → a[s] passes (oracle covers it)
  });

  it('does NOT reject an all-OOB index in dead code after a guard clause returns (the defensive-guard idiom)', async () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const x = f32(4, (i) => i)
component k(i) { if (i + 4 >= x.length) { return 0 } return x[i + 4] }
k`, host);
    const engine = new GpuEngine(host, cpuDeps);
    change(() => { engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    await new Promise((r) => setTimeout(r, 20));
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    expect(s.core).toBe(true);        // the OOB access is unreachable → NOT rejected
    expect(s.value).toEqual([0, 0, 0, 0]);   // matches the interpreter (the guard always fires)
  });
  it('does NOT reject an all-OOB index after an early return', async () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const x = f32(4, (i) => i)
component k(i) { if (i < 100) { return x[i] } return x[99] }
k`, host);
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    expect(s.core).toBe(true);
  });
  it('does NOT reject unconditional dead code after a return', async () => {
    const host = new RuntimeReactiveHost();
    const kernel = kernelOf(`const x = f32(4, (i) => i)
component k(i) { return x[i] return x[99] }
k`, host);   // NOTE: if the parser rejects a statement after return, use a different dead-code shape or drop this case + note it
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    expect(s.core).toBe(true);
  });
  it('STILL rejects a REACHABLE trailing all-OOB index (a bare if does not definitely return)', async () => {
    const host = new RuntimeReactiveHost();
    // the `if (i<2)` does NOT cover all coords (i∈[0,3]); for i>=2 control falls through to x[i+100] → reachable + all-OOB for the falling-through coords.
    // x[i+100] with i∈[0,3] → [100,103] entirely >= 4 → provably OOB on the reachable path → MUST still reject.
    const kernel = kernelOf(`const x = f32(4, (i) => i)
component k(i) { if (i < 2) { return x[i] } return x[i + 100] }
k`, host);
    const engine = new GpuEngine(host, cpuDeps);
    let s!: ReturnType<GpuEngine['gpu']>;
    change(() => { s = engine.gpu(kernel, { output: [4], backend: 'cpu' }); });
    expect(s.core).toBe(false);
    expect(s.reasons.some((r) => r.code === 'MLGPU-INDEX-STATIC')).toBe(true);
  });
});
