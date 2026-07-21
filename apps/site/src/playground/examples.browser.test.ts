import { describe, it, expect, beforeEach } from 'vitest';
import { renderSource } from '@metael/vdom/lang';
import { MATH_BUILTINS } from '@metael/math/lang';
import { STD_BUILTINS } from '@metael/std';
import { exampleById } from './examples.ts';

let container: HTMLElement;
beforeEach(() => { container = document.createElement('div'); document.body.appendChild(container); });

const TODO = exampleById('todo')!.source;
const rows = () => Array.from(container.querySelectorAll('li'));
const footerBtn = (t: string) => Array.from(container.querySelectorAll('.footer button')).find((b) => b.textContent === t) as HTMLButtonElement;

describe('flagship TodoMVC example — interactive', () => {
  it('mounts clean with two rows and a live count', () => {
    const h = renderSource(TODO, container, { builtins: [MATH_BUILTINS, STD_BUILTINS] });
    expect(h.diagnostics).toEqual([]);
    expect(rows().length).toBe(2);
    expect(container.querySelector('.footer .count')!.textContent).toBe('2 left');
    h.unmount();
  });

  it('ADD via Enter appends a row', () => {
    const h = renderSource(TODO, container, { builtins: [MATH_BUILTINS, STD_BUILTINS] });
    const input = container.querySelector('input.new') as HTMLInputElement;
    input.value = 'third'; input.dispatchEvent(new Event('input', { bubbles: true }));
    input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    expect(rows().length).toBe(3);
    expect(container.textContent).toContain('third');
    h.unmount();
  });

  it('TOGGLE marks done + updates the count + the row class', () => {
    const h = renderSource(TODO, container, { builtins: [MATH_BUILTINS, STD_BUILTINS] });
    (rows()[0]!.querySelector('button.toggle') as HTMLButtonElement).click();
    expect(container.querySelector('.footer .count')!.textContent).toBe('1 left');
    expect(rows()[0]!.className).toBe('done');
    h.unmount();
  });

  it('DELETE removes a row', () => {
    const h = renderSource(TODO, container, { builtins: [MATH_BUILTINS, STD_BUILTINS] });
    (rows()[1]!.querySelector('button.del') as HTMLButtonElement).click();
    expect(rows().length).toBe(1);
    h.unmount();
  });

  it('FILTER tabs narrow the visible list', () => {
    const h = renderSource(TODO, container, { builtins: [MATH_BUILTINS, STD_BUILTINS] });
    (rows()[0]!.querySelector('button.toggle') as HTMLButtonElement).click();
    footerBtn('done').click();
    expect(rows().length).toBe(1);
    expect(rows()[0]!.className).toBe('done');
    footerBtn('active').click();
    expect(rows().length).toBe(1);
    expect(rows()[0]!.className).toBe('active');
    footerBtn('all').click();
    expect(rows().length).toBe(2);
    h.unmount();
  });

  it('EDIT-IN-PLACE: click label → input → Enter renames', () => {
    const h = renderSource(TODO, container, { builtins: [MATH_BUILTINS, STD_BUILTINS] });
    (rows()[0]!.querySelector('span.label') as HTMLElement).click();
    const input = rows()[0]!.querySelector('input.edit') as HTMLInputElement;
    expect(input).not.toBeNull();
    input.value = 'renamed'; input.dispatchEvent(new Event('input', { bubbles: true }));
    input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
    expect(container.textContent).toContain('renamed');
    h.unmount();
  });

  it('CLEAR DONE removes completed rows', () => {
    const h = renderSource(TODO, container, { builtins: [MATH_BUILTINS, STD_BUILTINS] });
    (rows()[0]!.querySelector('button.toggle') as HTMLButtonElement).click();
    footerBtn('clear done').click();
    expect(rows().length).toBe(1);
    expect(rows()[0]!.className).toBe('active');
    h.unmount();
  });
});
