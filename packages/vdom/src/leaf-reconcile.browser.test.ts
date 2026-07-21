import { describe, it, expect } from 'vitest';
import { renderSource } from './lang/render-source.ts';

describe('a reactive-text leaf effect survives a structural re-derive (reconcile)', () => {
  it('a button whose text reads a `let` keeps updating after a sibling structural change', () => {
    // `tick` drives a structural for-of (list length); the `n=` button text reads a value-only `let`.
    // A structural re-derive reconciles the preserved button node — its text leaf effect must be
    // re-registered so a later value-only `n` write still patches it.
    const src = `component Story() {
  let n = 0
  let tick = 0
  div {
    ul { for (const i of range(tick + 1)) { li("row") } }
    button({ class: "bump", onClick: () => { tick = tick + 1 } }, "bump")
    button({ class: "n", onClick: () => { n = n + 1 } }, "n=" + n)
  }
}`;
    const c = document.createElement('div'); document.body.appendChild(c);
    const h = renderSource(src, c, {});
    const N = () => c.querySelector('.n')?.textContent;
    (c.querySelector('.n') as HTMLButtonElement).click();       // n -> 1 (leaf, pre-structural)
    expect(N()).toBe('n=1');
    (c.querySelector('.bump') as HTMLButtonElement).click();    // structural re-derive (list grows)
    const passesAfterStructural = h.passCount();
    (c.querySelector('.n') as HTMLButtonElement).click();       // n -> 2 : must still patch the text
    const afterStructuralThenLeaf = N();
    // The value-only write after the structural re-derive must STILL patch the preserved node...
    expect(afterStructuralThenLeaf).toBe('n=2');
    // ...and it must remain FINE-GRAINED — a leaf patch, NOT another full re-derive (passCount unchanged).
    expect(h.passCount()).toBe(passesAfterStructural);
    h.unmount();
    expect(h.diagnostics).toEqual([]);
  });

  it('an element attribute leaf effect also survives a structural re-derive', () => {
    // The `.n` button's `class` reads `n` (a value-only attribute leaf); a sibling structural for-of grows.
    const src = `component Story() {
  let n = 0
  let tick = 0
  div {
    ul { for (const i of range(tick + 1)) { li("row") } }
    button({ class: "bump", onClick: () => { tick = tick + 1 } }, "bump")
    button({ class: "n-" + n, onClick: () => { n = n + 1 } }, "click")
  }
}`;
    const c = document.createElement('div'); document.body.appendChild(c);
    const h = renderSource(src, c, {});
    const clickBtn = () => Array.from(c.querySelectorAll('button')).find((b) => b.textContent === 'click') as HTMLButtonElement;
    clickBtn().click();                                          // n -> 1 (attr leaf, pre-structural)
    expect(clickBtn().className).toBe('n-1');
    (Array.from(c.querySelectorAll('button')).find((b) => b.textContent === 'bump') as HTMLButtonElement).click();  // structural
    clickBtn().click();                                          // n -> 2 : attribute must still patch
    const cls = clickBtn().className;
    h.unmount();
    expect(h.diagnostics).toEqual([]);
    expect(cls).toBe('n-2');
  });
});
