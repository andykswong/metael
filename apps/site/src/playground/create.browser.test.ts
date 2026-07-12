import { describe, it, expect, beforeEach } from 'vitest';
import { createPlayground } from './create.ts';

let container: HTMLElement;
beforeEach(() => { container = document.createElement('div'); document.body.appendChild(container); });

function type(root: HTMLElement, source: string): void {
  const ta = root.querySelector('textarea') as HTMLTextAreaElement;
  ta.value = source;
  ta.dispatchEvent(new Event('input', { bubbles: true }));
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
    pg.destroy();
  });

  it('a click inside the UI preview mutates it (real @metael/vdom delegation)', () => {
    const pg = createPlayground(container, { defaultExampleId: 'counter' });
    pg.runNow();
    const preview = container.querySelector('.pg-preview-host')!;
    // the counter's "+" is the second button (order: "-", span, "+")
    const plus = Array.from(preview.querySelectorAll('button')).find((b) => b.textContent === '+') as HTMLButtonElement;
    plus.click();
    expect(preview.querySelector('.count')!.textContent).toBe('1');
    pg.destroy();
  });

  it('switching to compute renders a JSON value', () => {
    const pg = createPlayground(container, { defaultExampleId: 'fib' });
    pg.runNow();
    const compute = container.querySelector('.pg-compute')!;
    expect(compute.textContent).toContain('[');
    expect((compute as HTMLElement).hidden).toBe(false);
    pg.destroy();
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
    pg.destroy();
  });

  it('getState reflects the current editor + target', () => {
    const pg = createPlayground(container, { defaultExampleId: 'fib' });
    const state = pg.getState();
    expect(state.target).toBe('compute');
    expect(state.source).toContain('fib');
    pg.destroy();
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
    pg.destroy();
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
    pg.destroy();
  });
});
