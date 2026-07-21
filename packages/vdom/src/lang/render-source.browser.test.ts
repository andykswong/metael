import { describe, it, expect, beforeEach } from 'vitest';
import { renderSource } from './render-source.ts';

// Real-DOM (Chromium) proofs for the DSL front door `renderSource`. The headless render-source.test.ts
// proves the tree + registry via `hasHandler`/`invokeHandler`; this file proves the DOM path that only a
// real `document` exercises: createDom stamps each element's `data-key`, event delegation resolves a real
// click through that `data-key` to the captured handler, and a value-only write patches the live DOM node
// in place (no walk re-run) while a structural write re-derives. It also nails the "pre-keyed nodes"
// landmine: with `preKeyed:true` the render core must NOT re-key the tree, so the element's `data-key` and
// the handler registry stay keyed to the SAME PathKeyMinter key — a desync would break real clicks.

const COUNTER = `
component Story() {
  let n = 0
  div({ class: "c" }) {
    button({ onClick: () => { n = n + 1 } }, "+")
    span(n)
  }
}`;

// `n` is read by an `if` CONDITION (structural) — a click re-runs the walk (a structural re-derive).
const STRUCTURAL = `
component Story() {
  let n = 0
  div() {
    if (n < 3) { span("under") } else { span("over") }
    button({ onClick: () => { n = n + 1 } }, "inc")
  }
}`;

// The PathKeyMinter key for the button under `Story > div > button` — createDom stamps this on `data-key`
// and the handler registry is keyed to match. This is the exact key the headless BUDGET tests also assert.
const BUTTON_KEY = 'Story#0/div#0/button#0';

let container: HTMLElement;
beforeEach(() => { container = document.createElement('div'); document.body.appendChild(container); });

describe('@metael/vdom renderSource — real DOM (data-key survives pre-keying + delegation)', () => {
  it('the element carries the PathKeyMinter data-key AND the handler registry matches (no desync)', () => {
    const h = renderSource(COUNTER, container, {});
    expect(h.diagnostics).toEqual([]);
    // THE LANDMINE PROOF: the button element carries the minter key on `data-key` — the render core did NOT
    // re-key the pre-keyed tree (a re-key would stamp a different key and break delegation).
    const keyed = container.querySelector(`[data-key="${BUTTON_KEY}"]`);
    expect(keyed).not.toBeNull();
    expect((keyed as HTMLElement).tagName).toBe('BUTTON');
    // And the handler registry is keyed to the SAME key — no desync between DOM attr and captured handler.
    expect(h.hasHandler(BUTTON_KEY, 'onClick')).toBe(true);
    // A REAL click resolves through delegation (event target → data-key → registry) and increments; since
    // `n` is read only by `span(n)` (a leaf), the span text patches in place with NO walk re-run.
    const span = container.querySelector('span')!;
    expect(span.textContent).toBe('0');
    const passes = h.passCount();
    (keyed as HTMLButtonElement).click();
    expect(span.textContent).toBe('1');
    expect(h.passCount()).toBe(passes);   // fine-grained: the click did NOT re-run the walk
    h.unmount();
  });

  it('a value-only click patches the SAME span text node in place with NO walk-effect re-run', () => {
    const h = renderSource(COUNTER, container, {});
    expect(h.diagnostics).toEqual([]);
    const span = container.querySelector('span')!;
    expect(span.textContent).toBe('0');
    const passesAfterBuild = h.passCount();               // the initial build pass
    (container.querySelector('button') as HTMLButtonElement).click();
    expect(container.querySelector('span')).toBe(span);   // identity preserved (no re-render / replace)
    expect(span.textContent).toBe('1');                   // DOM text updated in place (fine-grained)
    expect(h.passCount()).toBe(passesAfterBuild);         // walk-effect did NOT re-run — the fast path is real
    (container.querySelector('button') as HTMLButtonElement).click();
    expect(span.textContent).toBe('2');
    expect(h.passCount()).toBe(passesAfterBuild);         // still no re-walk across repeated leaf updates
    h.unmount();
  });

  it('a STRUCTURAL change (an `if` cell) DOES re-run the walk-effect (the diff path)', () => {
    const h = renderSource(STRUCTURAL, container, {});
    expect(h.diagnostics).toEqual([]);
    expect(container.querySelector('span')!.textContent).toBe('under');
    const passesAfterBuild = h.passCount();
    for (let i = 0; i < 3; i++) (container.querySelector('button') as HTMLButtonElement).click();  // n: 0→3 flips the if
    expect(container.querySelector('span')!.textContent).toBe('over');
    expect(h.passCount()).toBeGreaterThan(passesAfterBuild);   // structural writes re-ran the walk
    h.unmount();
  });
});
