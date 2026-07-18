import { describe, it, expect } from 'vitest';
import { runTarget } from './targets.ts';
import { exampleById } from './examples.ts';

// The GPU playground target end-to-end in a real browser: mount a gpu example (the composite env resolves
// the `gpu` head), let the async dispatch settle, and assert the DOM shows the settled resource — a backend
// label + the generated compute shader panel. In a headless browser the backend ladder falls to the CPU
// floor, still correct (the resource matches the interpreter oracle). This proves the async re-render path:
// the engine's microtask drain writes the resource cell → the mount's tracked walk-effect re-derives → the
// panel updates from "computing…" to the settled backend + timing.
describe('the GPU playground target settles + renders a shader (Chromium)', () => {
  it('mounts a gpu matmul example, races, and shows the backend + generated WGSL', async () => {
    const ex = exampleById('gpu-matmul')!;
    const c = document.createElement('div');
    document.body.appendChild(c);
    const run = runTarget('gpu', ex.source, c, {});
    expect(run.kind).toBe('ui');
    if (run.kind !== 'ui') return;
    expect(run.diagnostics).toEqual([]);   // derives clean — the gpu head resolves via the composite env

    // The synchronous first frame already emits the WGSL (classify/emit is synchronous).
    expect(c.querySelector('.shader')?.textContent ?? '').toContain('@compute');

    // Let the async dispatch settle + re-render (microtask drain → writeCell → re-derive).
    await new Promise((r) => setTimeout(r, 500));

    const text = c.textContent ?? '';
    // After settle the status line names the actual backend (headless → likely cpu).
    expect(text).toMatch(/webgpu|webgl2|cpu/);
    // The generated compute shader panel is present + non-empty.
    expect(c.querySelector('.shader')?.textContent ?? '').toContain('@compute');
    // The status line no longer says "computing…" — the race resolved and the panel re-rendered.
    expect(text).not.toContain('computing on');
    // The panel shows the actual COMPUTED cells (not just a pass/fail flag) — the settled resource's value
    // rendered through the display heads (this example opts into verify + benchmark).
    expect(c.querySelector('.result')?.textContent ?? '').toMatch(/first cells: \[.+, \.\.\.\]/);
    expect(text).toContain('match=true');

    run.handle.unmount();
  });

  it('mounts the A->B pipeline example: stage A settles a resident buffer, then stage B derives + settles', async () => {
    const ex = exampleById('gpu-pipeline')!;
    const c = document.createElement('div');
    document.body.appendChild(c);
    const run = runTarget('gpu', ex.source, c, {});
    expect(run.kind).toBe('ui');
    if (run.kind !== 'ui') return;
    expect(run.diagnostics).toEqual([]);   // derives clean under the composite env
    // Frame 1: stage A is pending (rA.value == null), so stage B is not yet built — the panel shows A's state.
    expect(c.textContent ?? '').toContain('stage A on');

    // Let BOTH stages settle: A's dispatch drains → walk-effect re-derives → stage B builds + dispatches →
    // B's dispatch drains → re-derives again. This is the nested async re-dispatch the pipeline wiring enables.
    await new Promise((r) => setTimeout(r, 800));

    const text = c.textContent ?? '';
    expect(text).not.toContain('stage A on');    // A resolved → B was built
    expect(text).not.toContain('stage B on');    // B resolved too
    expect(c.querySelector('.result')?.textContent ?? '').toMatch(/B = A\*2: \[.+, \.\.\.\]/);
    // B's first cells are (i+1)*2 → 2, 4, 6, 8, 10, 12.
    expect(c.querySelector('.result')?.textContent ?? '').toContain('[2, 4, 6, 8, 10, 12, ...]');
    expect(c.querySelector('.shader')?.textContent ?? '').toContain('@compute');   // B's shader rendered
    run.handle.unmount();
  });
});
