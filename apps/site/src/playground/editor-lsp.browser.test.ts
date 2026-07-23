import { describe, it, expect, beforeEach } from 'vitest';
import { EditorView } from '@codemirror/view';
import { createPlayground } from './create.ts';

// The LSP-backed editor extensions end-to-end in a real browser: mount the playground (which spawns the LSP
// server as a Web Worker and fills the editor's extension compartment with the CM6 set), then drive the
// editor through CodeMirror's own transaction API and poll the DOM for the worker's async responses. We
// assert the two LOAD-BEARING language-intelligence paths — inline diagnostics (squiggles) and semantic-
// token colouring — since those are deterministic in headless Chromium; completion/hover/signature/lens need
// popup triggers or pointer hovers that are flaky headlessly and are exercised in the client's own suite.

let container: HTMLElement;
beforeEach(() => { container = document.createElement('div'); document.body.appendChild(container); });

/** Replace the whole document via CodeMirror's transaction API — a non-programmatic dispatch fires the
 *  updateListener exactly as real typing would, so onChange → schedule → render → client.didChange all run. */
function type(root: HTMLElement, source: string): void {
  const view = EditorView.findFromDOM(root.querySelector('.cm-editor') as HTMLElement)!;
  view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: source } });
}

async function pollFor(root: HTMLElement, selector: string, ms = 4000): Promise<Element | null> {
  const deadline = Date.now() + ms;
  let found = root.querySelector(selector);
  while (!found && Date.now() < deadline) {
    await new Promise((r) => setTimeout(r, 20));
    found = root.querySelector(selector);
  }
  return found;
}

describe('the LSP-backed editor extensions (Chromium)', () => {
  it('paints the cold-start highlighter synchronously — before any worker round-trip', () => {
    const pg = createPlayground(container, { defaultExampleId: 'counter' });
    pg.runNow();
    // SYNCHRONOUS, no await: the semantic-token path needs an async worker round-trip + a debounce, so
    // nothing from it can have resolved yet. Any colouring present now is the cold StreamLanguage +
    // HighlightStyle. The counter source opens with the `component` keyword → an amber-painted span, and
    // NO `.cmt-*` semantic-token class exists yet.
    const content = pg.root.querySelector('.cm-content') as HTMLElement;
    const spans = Array.from(content.querySelectorAll('span'));
    const colors = spans.map((s) => getComputedStyle(s).color);
    expect(colors).toContain('rgb(232, 163, 61)');                       // keyword amber (#e8a33d) is painted
    expect(spans.some((s) => s.className.includes('cmt-'))).toBe(false); // not the semantic-token layer yet
    pg[Symbol.dispose]();
  });

  it('colours the source with semantic tokens once the worker analyses it', async () => {
    const pg = createPlayground(container, { defaultExampleId: 'counter' });
    pg.runNow();
    // The counter source has keywords (component/const/return), so a `.cmt-keyword` mark must appear after
    // the worker's first semantic-token response reaches the ViewPlugin.
    const kw = await pollFor(pg.root, '.cmt-keyword');
    expect(kw).not.toBeNull();
    expect((kw as HTMLElement).textContent).toBeTruthy();
    pg[Symbol.dispose]();
  });

  it('shows an inline error squiggle for an undeclared identifier under a known target', async () => {
    const pg = createPlayground(container, { defaultExampleId: 'fib' });
    pg.runNow();
    // `bar` is read but never declared → ML-LANG-UNKNOWN-VAR under the compute profile. The worker publishes
    // diagnostics; the linter re-runs on the push and marks the range with `.cm-lintRange-error`.
    type(pg.root, 'const y = bar');
    pg.runNow();
    const squiggle = await pollFor(pg.root, '.cm-lintRange-error');
    expect(squiggle).not.toBeNull();
    pg[Symbol.dispose]();
  });

  it('clears the squiggle again once the source is fixed', async () => {
    const pg = createPlayground(container, { defaultExampleId: 'fib' });
    pg.runNow();
    type(pg.root, 'const y = bar');
    pg.runNow();
    expect(await pollFor(pg.root, '.cm-lintRange-error')).not.toBeNull();
    // Fix it: `bar` now declared. The next publish carries no diagnostics → the squiggle disappears.
    type(pg.root, 'const bar = 1\nconst y = bar');
    pg.runNow();
    const deadline = Date.now() + 4000;
    while (pg.root.querySelector('.cm-lintRange-error') && Date.now() < deadline) {
      await new Promise((r) => setTimeout(r, 20));
    }
    expect(pg.root.querySelector('.cm-lintRange-error')).toBeNull();
    pg[Symbol.dispose]();
  });

  it('retargets the worker vocabulary when the run target switches (metael/setProfile)', async () => {
    // End-to-end proof that flipping the target selector re-resolves the worker's vocabulary Profile.
    // The probe: a VALUE-READ of `div` (`const x = div` — not a call, which scope-check never flags). `div`
    // is a head in the `ui` (vdom) profile → no diagnostic; it is NOT in the `compute` (math+std) profile →
    // an ML-LANG-UNKNOWN-VAR squiggle. Switching the target fires create.ts's `render()` setProfile guard;
    // the worker re-resolves the profile and re-publishes, so the SAME source flips from clean to squiggled.
    // A target switch also loads that target's default example, so we re-type our probe source after each
    // switch to hold the source constant and isolate the profile change.
    const PROBE = 'component Story() { const x = div\n span("hi") }';

    const pg = createPlayground(container, { defaultExampleId: 'counter' });   // starts on the `ui` target
    pg.runNow();
    expect(pg.getState().target).toBe('ui');
    type(pg.root, PROBE);
    pg.runNow();
    // Under `ui`, `div` is a known head → after the worker analyses, there must be NO error squiggle. Poll a
    // beat to let any (wrong) publish arrive, then assert none — a semantic-token mark proves the worker ran.
    await pollFor(pg.root, '.cmt-keyword');   // worker analysed the buffer at least once under `ui`
    expect(pg.root.querySelector('.cm-lintRange-error')).toBeNull();

    // Flip the selector to `compute` (fires switchTarget → loadExampleObj → render → setProfile+didChange).
    const sel = pg.root.querySelector('.pg-target') as HTMLSelectElement;
    sel.value = 'compute';
    sel.dispatchEvent(new Event('change', { bubbles: true }));
    // Re-type the SAME probe source under the now-`compute` profile.
    type(pg.root, PROBE);
    pg.runNow();
    // `div` is unknown under `compute` → the worker publishes ML-LANG-UNKNOWN-VAR → an error squiggle appears.
    const squiggle = await pollFor(pg.root, '.cm-lintRange-error');
    expect(squiggle).not.toBeNull();

    // Flip back to `ui` and re-type: the head is known again → the squiggle clears. Proves the retarget is
    // bidirectional (not a one-way latch) and driven purely by the active profile.
    sel.value = 'ui';
    sel.dispatchEvent(new Event('change', { bubbles: true }));
    type(pg.root, PROBE);
    pg.runNow();
    const deadline = Date.now() + 4000;
    while (pg.root.querySelector('.cm-lintRange-error') && Date.now() < deadline) {
      await new Promise((r) => setTimeout(r, 20));
    }
    expect(pg.root.querySelector('.cm-lintRange-error')).toBeNull();
    pg[Symbol.dispose]();
  });
});
