import { describe, it, expect } from 'vitest';
import { renderSource } from '@metael/vdom/lang';
import { MATH_BUILTINS } from '@metael/math/lang';

const mk = (src: string) => { const c = document.createElement('div'); document.body.appendChild(c); return { c, h: renderSource(src, c, { builtins: [MATH_BUILTINS] }) }; };

describe('in-place typed-array mutation is reactive through the UI', () => {
  it('element read buf[0] updates on an in-place write', () => {
    const { c, h } = mk(`component Story() {
  let buf = f32([0, 0, 0])
  div {
    span({ class: "v" }, buf[0])
    button({ class: "b", onClick: () => { buf[0] = buf[0] + 1 } }, "inc")
  }
}`);
    const before = c.querySelector('.v')?.textContent;
    (c.querySelector('.b') as HTMLButtonElement).click();
    const after = c.querySelector('.v')?.textContent;
    h.unmount();
    expect(h.diagnostics).toEqual([]);
    expect([before, after]).toEqual(['0', '1']);
  });
  it('whole-buffer display "" + buf updates on an in-place write (generation subscribed via strOf/concat)', () => {
    const { c, h } = mk(`component Story() {
  let buf = f32([0, 0, 0])
  div {
    span({ class: "v" }, "" + buf)
    button({ class: "b", onClick: () => { buf[0] = buf[0] + 1 } }, "inc")
  }
}`);
    const before = c.querySelector('.v')?.textContent;
    (c.querySelector('.b') as HTMLButtonElement).click();
    const after = c.querySelector('.v')?.textContent;
    h.unmount();
    expect(h.diagnostics).toEqual([]);
    expect(before).not.toBe(after);        // the displayed buffer text changed
    expect(after).toContain('1');          // buf[0] is now 1
  });
  it('for-of over a buffer re-renders on an in-place write (iterate subscribes)', () => {
    const { c, h } = mk(`component Story() {
  let buf = f32([0, 0, 0])
  div {
    ul { for (const x of buf) { li({ class: "x" }, x) } }
    button({ class: "b", onClick: () => { buf[0] = buf[0] + 5 } }, "inc")
  }
}`);
    const first = () => c.querySelector('.x')?.textContent;
    const before = first();
    (c.querySelector('.b') as HTMLButtonElement).click();
    const after = first();
    h.unmount();
    expect(h.diagnostics).toEqual([]);
    expect([before, after]).toEqual(['0', '5']);
  });
});
