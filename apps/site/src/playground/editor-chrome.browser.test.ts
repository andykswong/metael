import { describe, it, expect, beforeEach } from 'vitest';
import { createPlayground } from './create.ts';
import '../styles.css';   // the scoped `.pg-editor .cm-*` chrome under test lives here (the app loads it too)

// The editor's VISUAL CHROME — a regression guard for the CodeMirror theming, since the CM6 swap once
// shipped with the library's default light theme (a grey line-number/lint gutter bar, a 16px system
// monospace, and an unbounded height that grew the panel to the whole document instead of scrolling).
// These assert the load-bearing chrome the design requires; they read computed styles, so they only hold
// with `styles.css` loaded (imported above). Colours come from the warm-phosphor palette in that file.
let container: HTMLElement;
beforeEach(() => { container = document.createElement('div'); document.body.appendChild(container); });

describe('editor visual chrome (Chromium)', () => {
  it('uses the mono editor font, not the page sans/default monospace at the default size', () => {
    const pg = createPlayground(container, { defaultExampleId: 'counter' });
    pg.runNow();
    const scroller = container.querySelector('.pg-editor .cm-scroller') as HTMLElement;
    const cs = getComputedStyle(scroller);
    expect(cs.fontFamily).toContain('JetBrains Mono');
    expect(cs.fontSize).toBe('13.5px');   // the design size, not CM's default 16px
    pg[Symbol.dispose]();
  });

  it('blends the gutters into the dark well (no default light-grey gutter bar)', () => {
    const pg = createPlayground(container, { defaultExampleId: 'counter' });
    pg.runNow();
    const gutters = container.querySelector('.pg-editor .cm-gutters') as HTMLElement;
    const cs = getComputedStyle(gutters);
    // CM's baseTheme paints a solid light bar (rgb(245,245,245)) with a light border; ours is transparent.
    expect(cs.backgroundColor).toBe('rgba(0, 0, 0, 0)');
    expect(cs.borderRightWidth).toBe('0px');
    pg[Symbol.dispose]();
  });

  it('bounds the editor to the panel so it scrolls internally (never grows the panel to the doc height)', () => {
    // The full-featured Todo example is many lines — taller than the 380px panel — so the scroller must
    // overflow internally rather than the editor driving the grid row to the whole document's height.
    const pg = createPlayground(container, { defaultExampleId: 'todo' });
    pg.runNow();
    const pgEditor = container.querySelector('.pg-editor') as HTMLElement;
    const scroller = container.querySelector('.pg-editor .cm-scroller') as HTMLElement;
    expect(pgEditor.getBoundingClientRect().height).toBeLessThan(600);   // ~380px panel, not ~2000px doc
    expect(scroller.scrollHeight).toBeGreaterThan(scroller.clientHeight + 2);   // scrolls internally
    pg[Symbol.dispose]();
  });

  it('themes the editable content with the amber caret (visible on the dark well)', () => {
    const pg = createPlayground(container, { defaultExampleId: 'counter' });
    pg.runNow();
    const content = container.querySelector('.pg-editor .cm-content') as HTMLElement;
    // rgb(232, 163, 61) === --amber; CM's default caret-color is the (invisible-on-dark) black.
    expect(getComputedStyle(content).caretColor).toBe('rgb(232, 163, 61)');
    pg[Symbol.dispose]();
  });
});
