import { describe, it, expect, beforeEach } from 'vitest';
import { mount } from './mount.ts';
import { COUNTER, TODO, FORM } from './examples.ts';

let container: HTMLElement;
beforeEach(() => { container = document.createElement('div'); document.body.appendChild(container); });

// A reorder swap can't be expressed as a single DSL click, so the reorder test drives it via updateData
// over a reactiveData variant of the list; add/remove use the DSL handlers (real clicks — the showcase).
const REORDERABLE = `component Story() { ul { for (const it of data) { li({ key: it.id }) { span(it.label) } } } }`;

describe('@metael/vdom — real DOM', () => {
  it('mounts with zero diagnostics', () => {
    const h = mount(COUNTER, container, {});
    expect(h.diagnostics).toEqual([]);
    expect(container.querySelector('.counter')).not.toBeNull();
    expect(container.querySelector('span')!.textContent).toBe('0');
    h.unmount();
  });

  it('FINE-GRAINED: a counter click patches only the text node (SAME span element; walk-effect not re-run)', () => {
    const h = mount(COUNTER, container, {});
    const span = container.querySelector('span')!;
    (container.querySelector('button') as HTMLButtonElement).click();
    expect(container.querySelector('span')).toBe(span);   // identity preserved — no re-render/replace
    expect(span.textContent).toBe('1');
    h.unmount();
  });

  it('STRUCTURAL: removing a row via a DSL click keeps the OTHER row as the SAME DOM instance', () => {
    const h = mount(TODO, container, {});          // items lives in the component; remove = filter() reassign
    const rows = () => Array.from(container.querySelectorAll('li'));
    expect(rows().length).toBe(2);
    const firstBefore = rows()[0]!;                 // the id:0 "first" row
    // Click the SECOND row's "x" button → items = filter(items, r => r.id != 1); id:0 row must survive.
    (rows()[1]!.querySelector('button') as HTMLButtonElement).click();
    expect(rows().length).toBe(1);
    expect(rows()[0]!).toBe(firstBefore);           // identity preserved across the structural reconcile
    h.unmount();
  });

  it('STRUCTURAL: adding a row via a DSL click (spread append) inserts a NEW node, existing identity kept', () => {
    const h = mount(TODO, container, {});
    const li = () => Array.from(container.querySelectorAll('li'));
    const a0 = li()[0]!; const b0 = li()[1]!;
    // The "add" button → items = [...items, { id: nextId, label: "new" }].
    (Array.from(container.querySelectorAll('button')).find((btn) => btn.textContent === 'add') as HTMLButtonElement).click();
    expect(li().length).toBe(3);
    expect(li()[0]!).toBe(a0); expect(li()[1]!).toBe(b0);   // existing rows kept
    h.unmount();
  });

  it('STRUCTURAL: reorder preserves node identity (keyed move, no re-create)', () => {
    const h = mount(REORDERABLE, container, { data: [{ id: 0, label: 'a' }, { id: 1, label: 'b' }], reactiveData: true });
    const byText = (t: string) => Array.from(container.querySelectorAll('li')).find((li) => li.textContent === t)!;
    const a0 = byText('a'); const b0 = byText('b');
    h.updateData([{ id: 1, label: 'b' }, { id: 0, label: 'a' }]);   // swap (not expressible as one click)
    expect(byText('a')).toBe(a0); expect(byText('b')).toBe(b0);      // same instances, moved
    expect(Array.from(container.querySelectorAll('li')).map((li) => li.textContent)).toEqual(['b', 'a']);
    h.unmount();
  });

  it('FOCUS SURVIVES: focus + caret persist across a reactive update', () => {
    const h = mount(FORM, container, {});
    const input = container.querySelector('#name-input') as HTMLInputElement;
    input.focus(); input.value = 'ab'; input.setSelectionRange(2, 2);
    input.dispatchEvent(new Event('input', { bubbles: true }));
    expect(document.activeElement).toBe(input);
    expect(input.selectionStart).toBe(2);
    expect(container.querySelector('span')!.textContent).toBe('ab');
    h.unmount();
  });

  it('DELEGATION: one root listener dispatches to the right node handler', () => {
    const h = mount(COUNTER, container, {});
    (container.querySelector('button') as HTMLButtonElement).click();
    expect(container.querySelector('span')!.textContent).toBe('1');
    h.unmount();
  });

  it('SANITIZE: a javascript: href is dropped at the DOM boundary', () => {
    const h = mount('component Story() { a({ href: "javascript:alert(1)" }, "x") }', container, {});
    expect(container.querySelector('a')!.hasAttribute('href')).toBe(false);
    h.unmount();
  });

  it('SANITIZE: text with < & is rendered literally (not double-escaped)', () => {
    const h = mount('component Story() { span("a < b & c") }', container, {});
    expect(container.querySelector('span')!.textContent).toBe('a < b & c');   // not "a &lt; b &amp; c"
    h.unmount();
  });
});
