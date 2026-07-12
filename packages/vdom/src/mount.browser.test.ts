import { describe, it, expect, beforeEach } from 'vitest';
import { mount } from './mount.ts';

// Fine-grained (value-only) path proofs at the mount level: a reactive `let` read only by a leaf position
// (span(n) / a reactive attribute) patches ONLY that DOM position with NO walk-effect re-run — the
// preact-signals fast path. A structural change (a `for`/`if` cell) is the ONLY thing that re-runs the walk.

const COUNTER = `
component Story() {
  let n = 0
  div({ class: "counter" }) {
    button({ onClick: () => { n = n + 1 } }, "+")
    span(n)
  }
}`;

// `cls` is read ONLY by a reactive attribute (leaf), never by a for/if — so a click is fine-grained.
const REACTIVE_ATTR = `
component Story() {
  let cls = "off"
  div() {
    button({ onClick: () => { cls = "on" } }, "toggle")
    span({ class: cls }, "label")
  }
}`;

// `n` is read by an `if` CONDITION (structural) — so a click re-runs the walk (a structural re-derive).
const STRUCTURAL = `
component Story() {
  let n = 0
  div() {
    if (n < 3) { span("under") } else { span("over") }
    button({ onClick: () => { n = n + 1 } }, "inc")
  }
}`;

let container: HTMLElement;
beforeEach(() => { container = document.createElement('div'); document.body.appendChild(container); });

describe('@metael/vdom mount — fine-grained value path (real DOM)', () => {
  it('a counter click patches the SAME span text node with NO walk-effect re-run (fine-grained)', () => {
    const h = mount(COUNTER, container, {});
    expect(h.diagnostics).toEqual([]);
    const span = container.querySelector('span')!;
    expect(span.textContent).toBe('0');
    const passesAfterBuild = h.passCount();          // 1 (the initial build pass)
    (container.querySelector('button') as HTMLButtonElement).click();
    expect(container.querySelector('span')).toBe(span);   // identity preserved (no re-render / replace)
    expect(span.textContent).toBe('1');                    // DOM text updated in place (fine-grained)
    expect(h.passCount()).toBe(passesAfterBuild);          // walk-effect did NOT re-run — the fast path is real
    (container.querySelector('button') as HTMLButtonElement).click();
    expect(span.textContent).toBe('2');
    expect(h.passCount()).toBe(passesAfterBuild);          // still no re-walk across repeated leaf updates
    h.unmount();
  });

  it('a reactive ATTRIBUTE change patches the live element attr in place with NO walk-effect re-run', () => {
    const h = mount(REACTIVE_ATTR, container, {});
    expect(h.diagnostics).toEqual([]);
    const span = container.querySelector('span')!;
    expect(span.getAttribute('class')).toBe('off');
    const passes = h.passCount();
    (container.querySelector('button') as HTMLButtonElement).click();
    expect(container.querySelector('span')).toBe(span);    // same element
    expect(span.getAttribute('class')).toBe('on');         // attribute patched in place
    expect(h.passCount()).toBe(passes);                    // fine-grained: no re-walk
    h.unmount();
  });

  it('a STRUCTURAL change (an `if` cell) DOES re-run the walk-effect (the diff path)', () => {
    const h = mount(STRUCTURAL, container, {});
    expect(h.diagnostics).toEqual([]);
    expect(container.querySelector('span')!.textContent).toBe('under');
    const passesAfterBuild = h.passCount();
    for (let i = 0; i < 3; i++) (container.querySelector('button') as HTMLButtonElement).click();  // n: 0→3 flips the if
    expect(container.querySelector('span')!.textContent).toBe('over');
    expect(h.passCount()).toBeGreaterThan(passesAfterBuild);   // structural writes re-ran the walk
    h.unmount();
  });
});
