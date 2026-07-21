import { describe, it, expect } from 'vitest';
import { renderSource } from '@metael/vdom/lang';
import { MATH_BUILTINS } from '@metael/math/lang';
import { exampleById } from './examples.ts';

describe('the interactive buffer example is reactive in the browser', () => {
  it('clicking +1 mutates cell 0 in place; both the cells and the whole-buffer display update', () => {
    const ex = exampleById('buffer')!;
    const c = document.createElement('div'); document.body.appendChild(c);
    const h = renderSource(ex.source, c, { builtins: [MATH_BUILTINS] });
    const cell0 = () => c.querySelectorAll('.cell')[0]?.textContent;
    const disp = () => c.querySelector('.disp')?.textContent;
    const beforeCell = cell0();
    const beforeDisp = disp();
    const plus = Array.from(c.querySelectorAll('button')).find((b) => b.textContent === '+1') as HTMLButtonElement;
    plus.click();
    // CAPTURE BEFORE UNMOUNT — unmount() clears the container, which would make these reads vacuous.
    const afterCell = cell0();
    const afterDisp = disp();
    h.unmount();
    expect(h.diagnostics).toEqual([]);
    expect(beforeCell).toBe('0');       // initial cell 0
    expect(afterCell).toBe('1');        // for-of iterate re-rendered cell 0 → 1 (specific, not just changed)
    expect(afterDisp).not.toBe(beforeDisp);   // the whole-buffer display re-rendered
    expect(afterDisp).toContain('1');   // strOf display now shows buf[0] = 1
  });
});
