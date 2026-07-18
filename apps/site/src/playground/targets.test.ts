import { describe, it, expect, vi } from 'vitest';
import { GpuHostEnv } from '@metael/gpu';
import { runTarget, runComputeSettled } from './targets.ts';

describe('runTarget', () => {
  it('ui target mounts headless and reports a tree + no diagnostics', () => {
    const run = runTarget('ui', 'component Story() { span("hi") }', undefined, {});
    expect(run.kind).toBe('ui');
    if (run.kind === 'ui') {
      expect(run.diagnostics).toEqual([]);
      expect(run.handle.tree()).not.toBeNull();
      run.handle.unmount();
    }
  });

  it('compute target evaluates to a pretty string + the raw value', () => {
    const run = runTarget('compute', 'map(range(4), (i) => i * i)', undefined, {});
    expect(run.kind).toBe('compute');
    if (run.kind === 'compute') {
      expect(run.value).toEqual([0, 1, 4, 9]);
      expect(run.text).toBe('[0, 1, 4, 9]');
      expect(run.diagnostics).toEqual([]);
    }
  });

  it('compute target injects data', () => {
    const run = runTarget('compute', 'map(data, (r) => r.n)', undefined, { data: [{ n: 1 }, { n: 2 }] });
    if (run.kind === 'compute') expect(run.value).toEqual([1, 2]);
  });

  it('surfaces diagnostics from a bad source (ui)', () => {
    const run = runTarget('ui', 'component Story( {', undefined, {});
    expect(run.diagnostics.length).toBeGreaterThan(0);
  });

  it('a gpu run mounts headless, derives clean, and unmount() disposes its engine without throwing', () => {
    const src = `const x = f32(8, (i) => i)
component k(i) { return x[i] * 2 }
component Story() {
  const r = gpu(k, { output: [8], backend: "cpu" })
  div { p(r.core ? "core" : "not core") pre({ class: "shader" }, r.wgsl) }
}`;
    const run = runTarget('gpu', src, undefined, {});
    expect(run.kind).toBe('ui');
    if (run.kind !== 'ui') return;
    expect(run.diagnostics).toEqual([]);   // the composite env resolves the `gpu` head → derives clean
    // unmount() disposes the composite GpuUiEnv (→ the GpuEngine → any WebGPU device). Must not throw, and
    // a second unmount must be idempotent (no double-dispose crash).
    expect(() => { run.handle.unmount(); run.handle.unmount(); }).not.toThrow();
  });

  it('surfaces a non-lowerable gpu kernel reason as a run diagnostic (a pasted kernel with no authored badge)', () => {
    // A valid-metael program whose kernel is NOT GPU-lowerable: the body indexes a resource object member
    // (`r.value[i]`), so the free name `r` has no buffer lowering → the gate rejects it (core=false). The
    // source parses + derives clean (zero lang diagnostics), so WITHOUT the bridge the failure is invisible.
    // The gpu resource's gate reason must reach the run's diagnostics so the playground's diagnostics panel
    // shows WHY nothing computed.
    const src = `const x = f32(4, (i) => i)
component a(i) { return x[i] + 1 }
component Story() {
  const r = gpu(a, { output: [4], outputType: "gpu-buffer", backend: "cpu" })
  component b(i) { return r.value[i] * 2 }
  const rb = gpu(b, { output: [4], backend: "cpu" })
  div { pre({ class: "shader" }, rb.wgsl) }
}`;
    const run = runTarget('gpu', src, undefined, {});
    expect(run.kind).toBe('ui');
    if (run.kind !== 'ui') return;
    // The gate reason for the non-lowerable stage-B kernel is surfaced (MLGPU-*), not swallowed.
    expect(run.diagnostics.some((d) => d.code.startsWith('MLGPU-'))).toBe(true);
    run.handle.unmount();
  });
});

describe('runComputeSettled — headless gpu compute (no DOM)', () => {
  it('evaluates a gpu program to a settled value with no container/mount', async () => {
    const src = `
      const x = f32(4, (i) => i)
      component k(i) { return x[i] * 2 }
      const r = gpu(k, { output: [4], backend: "cpu" })
      r`;
    const out = await runComputeSettled(src, {});
    expect(out.diagnostics).toEqual([]);
    // r is a GpuResource; after settle its value is the computed array
    expect((out.value as { value: number[] }).value).toEqual([0, 2, 4, 6]);
  });

  it('disposes the compute env on EVERY exit path — including when evaluation throws', async () => {
    // The dispose lives in a finally so a throwing run (the runtime can throw on arbitrary source) still
    // frees the engine (→ any acquired GPU device). It is not directly observable (env is internal), so we
    // observe its side effect: GpuComputeEnv[Symbol.dispose] calls the engine env's dispose exactly once.
    // Force a throw by feeding `data` whose getter throws when the program reads it — evaluateProgram (inside
    // change()) then propagates, unwinding the settle loop before the return.
    const disposeSpy = vi.spyOn(GpuHostEnv.prototype, Symbol.dispose);
    try {
      await expect(
        runComputeSettled('data.x', { data: { get x() { throw new Error('boom'); } } }),
      ).rejects.toThrow('boom');
      expect(disposeSpy).toHaveBeenCalledTimes(1);   // disposed despite the throw (the finally fired)
    } finally {
      disposeSpy.mockRestore();
    }
  });

  it('disposes the compute env on the NORMAL return path (regression guard)', async () => {
    const disposeSpy = vi.spyOn(GpuHostEnv.prototype, Symbol.dispose);
    try {
      const src = `
        const x = f32(4, (i) => i)
        component k(i) { return x[i] * 2 }
        const r = gpu(k, { output: [4], backend: "cpu" })
        r`;
      const out = await runComputeSettled(src, {});
      expect((out.value as { value: number[] }).value).toEqual([0, 2, 4, 6]);
      expect(disposeSpy).toHaveBeenCalledTimes(1);
    } finally {
      disposeSpy.mockRestore();
    }
  });

  it('settles a gpu resource nested two levels deep (walks to any depth, not just one)', async () => {
    // `{ out: { r } }` puts the GpuResource two levels below the top value. A one-level pending check would
    // exit the settle loop on the first (still-pending) eval → the nested resource stays pending/value:null.
    const src = `const x = f32(4, (i) => i)
component k(i) { return x[i] * 2 }
const r = gpu(k, { output: [4], backend: "cpu" })
{ out: { r: r } }`;
    const out = await runComputeSettled(src, {});
    expect(out.diagnostics).toEqual([]);
    expect((out.value as { out: { r: { value: number[] } } }).out.r.value).toEqual([0, 2, 4, 6]);
  });

  it('settles a gpu resource nested one level deep (`{ result: r }`)', async () => {
    const src = `const x = f32(4, (i) => i)
component k(i) { return x[i] * 2 }
const r = gpu(k, { output: [4], backend: "cpu" })
{ result: r }`;
    const out = await runComputeSettled(src, {});
    expect(out.diagnostics).toEqual([]);
    expect((out.value as { result: { value: number[] } }).result.value).toEqual([0, 2, 4, 6]);
  });

  it('settles when the program returns only a PROJECTION of the resource (`{ value: r.value }`)', async () => {
    // The footgun: the returned value carries no resource (no `pending` field), so a value-only pending check
    // exits on pass 1 while `r` is still pending → r.value was null. Gating the settle loop on the ENGINE's
    // declared-resource pending state (not just the returned value) lets it settle, so the projection is real.
    const src = `const x = f32(4, (i) => i)
component k(i) { return x[i] * 2 }
const r = gpu(k, { output: [4], backend: "cpu" })
{ value: r.value }`;
    const out = await runComputeSettled(src, {});
    expect(out.diagnostics).toEqual([]);
    expect((out.value as { value: number[] }).value).toEqual([0, 2, 4, 6]);   // NOT null
  });
});
