import { describe, it, expect, beforeEach } from 'vitest';
import { EditorView } from '@codemirror/view';
import { createPlayground } from './create.ts';

let container: HTMLElement;
beforeEach(() => { container = document.createElement('div'); document.body.appendChild(container); });

// Drive the editor through CodeMirror's own transaction API rather than synthesizing contenteditable
// input: locate the view from its DOM and dispatch a full-document replace. A (non-programmatic) dispatch
// fires the updateListener exactly as real typing would, so onChange → schedule → render all run.
function type(root: HTMLElement, source: string): void {
  const view = EditorView.findFromDOM(root.querySelector('.cm-editor') as HTMLElement)!;
  view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: source } });
}

describe('createPlayground (real DOM)', () => {
  // Engine tests pin defaultExampleId: 'counter' so they are independent of which example is the gallery
  // default (the counter is the simplest deterministic UI preview: a single span showing "0").
  it('renders a UI example into a live preview', () => {
    const pg = createPlayground(container, { defaultExampleId: 'counter' });
    pg.runNow();
    const preview = container.querySelector('.pg-preview-host')!;
    expect(preview.querySelector('.counter')).not.toBeNull();
    expect(preview.querySelector('.count')!.textContent).toBe('0');
    pg[Symbol.dispose]();
  });

  it('a click inside the UI preview mutates it (real @metael/vdom delegation)', () => {
    const pg = createPlayground(container, { defaultExampleId: 'counter' });
    pg.runNow();
    const preview = container.querySelector('.pg-preview-host')!;
    // the counter's "+" is the second button (order: "-", span, "+")
    const plus = Array.from(preview.querySelectorAll('button')).find((b) => b.textContent === '+') as HTMLButtonElement;
    plus.click();
    expect(preview.querySelector('.count')!.textContent).toBe('1');
    pg[Symbol.dispose]();
  });

  it('switching to compute renders a JSON value', () => {
    const pg = createPlayground(container, { defaultExampleId: 'fib' });
    pg.runNow();
    const compute = container.querySelector('.pg-compute')!;
    expect(compute.textContent).toContain('[');
    expect((compute as HTMLElement).hidden).toBe(false);
    pg[Symbol.dispose]();
  });

  it('a bad edit keeps the last-good preview + shows a banner', () => {
    const pg = createPlayground(container, { defaultExampleId: 'counter' });
    pg.runNow();
    const preview = container.querySelector('.pg-preview-host')!;
    expect(preview.querySelector('.counter')).not.toBeNull();
    type(pg.root, 'component Story( {');   // broken
    pg.runNow();
    expect(preview.querySelector('.counter')).not.toBeNull();       // last-good retained
    expect((container.querySelector('.pg-banner') as HTMLElement).hidden).toBe(false);
    expect(container.querySelector('.pg-diags')!.textContent).toContain('ML-LANG');
    pg[Symbol.dispose]();
  });

  it('getState reflects the current editor + target', () => {
    const pg = createPlayground(container, { defaultExampleId: 'fib' });
    const state = pg.getState();
    expect(state.target).toBe('compute');
    expect(state.source).toContain('fib');
    pg[Symbol.dispose]();
  });

  it('switching the target loads that target\'s default example (never a stale null)', () => {
    const pg = createPlayground(container, { defaultExampleId: 'counter' });
    pg.runNow();
    expect(pg.getState().target).toBe('ui');
    // flip the target selector UI → compute
    const sel = container.querySelector('.pg-target') as HTMLSelectElement;
    sel.value = 'compute';
    sel.dispatchEvent(new Event('change', { bubbles: true }));
    // it should have LOADED a compute example (not kept the counter source → which would eval to null)
    const st = pg.getState();
    expect(st.target).toBe('compute');
    expect(st.source).not.toContain('component');   // no longer the UI counter component
    const compute = container.querySelector('.pg-compute') as HTMLElement;
    expect(compute.hidden).toBe(false);
    expect(compute.textContent!.trim()).not.toBe('null');   // a real value, not null
    expect(compute.textContent).toContain('[');             // fib → an array
    // the example picker stays in sync with the loaded example
    expect((container.querySelector('.pg-examples') as HTMLSelectElement).value).toBe('fib');
    // now switch BACK to UI: it should restore the counter (the example we were on), not the gallery default (todo)
    sel.value = 'ui';
    sel.dispatchEvent(new Event('change', { bubbles: true }));
    expect(pg.getState().target).toBe('ui');
    expect(pg.getState().source).toContain('counter');   // restored, not todo
    expect(container.querySelector('.pg-preview-host .counter')).not.toBeNull();
    pg[Symbol.dispose]();
  });

  it('surfaces a late (async-guarded) non-lowerable gpu kernel in the diagnostics panel', async () => {
    // The reported bug: a two-stage pipeline where stage B is GUARDED behind stage A's async settle
    // (`if (rA.value == null) … else gpu(b, …)`), and B's body indexes a resource member (`rA.value[i]`)
    // → B is NOT GPU-lowerable. B is only DERIVED on a re-derive AFTER A settles (async), so a mount-time
    // diagnostics snapshot misses it: the preview shows an empty shader + "B on cpu" and NOTHING surfaces.
    // The panel must show the gate reason once the late re-derive produces the non-core resource.
    const pg = createPlayground(container, { defaultExampleId: 'gpu-matmul' });
    const badPipeline = `component Story() {
  const N = 8
  const seed = f32(N, (i) => i)
  component a(i) { return seed[i] + 1 }
  const rA = gpu(a, { output: [N], outputType: "gpu-buffer" })
  div({ class: "gpu-demo" }) {
    if (rA.value == null) {
      p({ class: "status" }, "stage A...")
    } else {
      component b(i) { return rA.value[i] * 2 }
      const rB = gpu(b, { output: [N] })
      pre({ class: "shader" }, rB.wgsl)
    }
  }
}`;
    // Ensure the gpu target is active, then load the bad source + run.
    const sel = container.querySelector('.pg-target') as HTMLSelectElement;
    sel.value = 'gpu'; sel.dispatchEvent(new Event('change', { bubbles: true }));
    type(container.querySelector('.pg-root') as HTMLElement, badPipeline);
    pg.runNow();
    // Stage A settles on a later microtask/timeout → stage B re-derives non-core → the late gpu callback
    // repaints the panel. Poll (not a fixed sleep) so the assertion isn't flaky under full-suite load.
    const diags = container.querySelector('.pg-diags')!;
    const deadline = Date.now() + 3000;
    while (!(diags.textContent ?? '').includes('MLGPU-') && Date.now() < deadline) {
      await new Promise((r) => setTimeout(r, 20));
    }
    expect(diags.textContent ?? '').toContain('MLGPU-');   // the gate reason reached the panel (was silent)
    pg[Symbol.dispose]();
  });

  it('exposes accessible names + live regions for screen readers', () => {
    const pg = createPlayground(container, { defaultExampleId: 'counter' });
    pg.runNow();
    // both selects are named (the picker only exists in the non-compact shell)
    expect(container.querySelector('.pg-target')!.getAttribute('aria-label')).toBe('Run target');
    expect(container.querySelector('.pg-examples')!.getAttribute('aria-label')).toBe('Load example');
    // the error surface is announced without a focus move
    expect(container.querySelector('.pg-banner')!.getAttribute('role')).toBe('alert');
    expect(container.querySelector('.pg-diags')!.getAttribute('aria-live')).toBe('polite');
    // the transient "Copied!" label change is announced
    expect(container.querySelector('.pg-share')!.getAttribute('aria-live')).toBe('polite');
    pg[Symbol.dispose]();
  });
});
