// packages/vdom/src/render.browser.test.ts
import { describe, it, expect } from 'vitest';
import { signal } from '@metael/runtime';
import { h } from './h.ts';
import { render } from './render.ts';

function container(): HTMLElement { const el = document.createElement('div'); document.body.appendChild(el); return el; }

describe('render() — real DOM (Chromium)', () => {
  it('builds the DOM and reflects a value-only signal write in place (no rebuild)', () => {
    const c = container();
    const label = signal('one');
    const handle = render(() => h('span', {}, () => label.get()), c, {});
    expect(c.querySelector('span')!.textContent).toBe('one');
    const spanBefore = c.querySelector('span');
    handle.setState(() => label.set('two'));
    expect(c.querySelector('span')!.textContent).toBe('two');
    expect(c.querySelector('span')).toBe(spanBefore);   // same element — patched, not rebuilt
    expect(handle.passCount()).toBe(1);                 // fine-grained: no re-derive
    handle.unmount();
  });

  it('reconciles a structural change (element swap)', () => {
    const c = container();
    const show = signal(true);
    const handle = render(() => (show.get() ? h('div', {}, 'A') : h('p', {}, 'B')), c, {});
    expect(c.querySelector('div')!.textContent).toBe('A');
    handle.setState(() => show.set(false));
    expect(c.querySelector('div')).toBeNull();
    expect(c.querySelector('p')!.textContent).toBe('B');
    expect(handle.passCount()).toBe(2);
    handle.unmount();
  });

  it('a keyed-list reorder preserves element identity by key', () => {
    const c = container();
    const items = signal([{ id: 'a', n: 1 }, { id: 'b', n: 2 }, { id: 'c', n: 3 }]);
    const handle = render(
      () => h('ul', {}, ...items.get().map((it) => h('li', { key: it.id }, String(it.n)))),
      c, {},
    );
    const first = c.querySelectorAll('li')[0]! as HTMLElement;
    first.setAttribute('data-probe', 'yes');           // tag the DOM node for "a"
    handle.setState(() => items.set([{ id: 'c', n: 3 }, { id: 'a', n: 1 }, { id: 'b', n: 2 }]));
    const lis = [...c.querySelectorAll('li')] as HTMLElement[];
    expect(lis.map((l) => l.textContent)).toEqual(['3', '1', '2']);
    // "a" moved to index 1 but is the SAME element (identity preserved by key):
    expect(lis[1]!.getAttribute('data-probe')).toBe('yes');
    handle.unmount();
  });

  it('delegates an onClick handler that mutates a signal and updates the DOM', () => {
    const c = container();
    const count = signal(0);
    const handle = render(
      () => h('button', { onClick: () => count.set(count.get() + 1) }, () => `count: ${count.get()}`),
      c, {},
    );
    const btn = c.querySelector('button')! as HTMLButtonElement;
    expect(btn.textContent).toBe('count: 0');
    btn.click();
    expect(btn.textContent).toBe('count: 1');
    expect(handle.passCount()).toBe(1);                // value-only: leaf effect only
    handle.unmount();
  });

  it('disposes leaf effects for removed nodes (the leaf effect stops running)', () => {
    const c = container();
    const show = signal(true);
    const inner = signal('live');
    // Count how often the reactive-text leaf effect re-runs. If the removed node's effect is still live,
    // a post-removal write to `inner` re-runs the thunk and bumps this counter — that is the disposal bug.
    let runs = 0;
    const handle = render(
      () => (show.get()
        ? h('div', {}, h('span', {}, () => { runs++; return inner.get(); }))
        : h('div', {}, 'gone')),
      c, {},
    );
    expect(c.querySelector('span')!.textContent).toBe('live');
    const runsAfterBuild = runs;
    handle.setState(() => show.set(false));            // removes the span (+ its leaf effect)
    expect(c.querySelector('span')).toBeNull();
    // writing the now-orphaned signal must not re-run the disposed effect, must not throw, and must not
    // resurrect a node:
    handle.setState(() => inner.set('zombie'));
    expect(runs).toBe(runsAfterBuild);                 // leaf effect was disposed — thunk did not re-run
    expect(c.querySelector('span')).toBeNull();
    handle.unmount();
  });

  // --- kind-namespaced keying prevents cross-kind wrong-tag reuse ---

  it('an unkeyed conditional sibling dropping out does NOT corrupt the surviving sibling', () => {
    const c = container();
    const show = signal(true);
    const label = signal('hi');
    // Pass 1: div has [p banner, span label]; pass 2: the p is dropped (null). With a flat positional
    // keyer the span would inherit the p's key/slot and render as a <p>; with kind-namespacing the span
    // keeps '/div#0/span#0' across both passes and is reused correctly.
    const handle = render(
      () => h('div', {},
        show.get() ? h('p', {}, 'banner') : null,
        h('span', {}, () => label.get()),
      ),
      c, {},
    );
    expect(c.querySelector('p')!.textContent).toBe('banner');
    expect(c.querySelector('span')!.textContent).toBe('hi');
    const spanBefore = c.querySelector('span');
    handle.setState(() => show.set(false));
    expect(c.querySelector('p')).toBeNull();                 // banner removed
    expect(c.querySelector('span')!.textContent).toBe('hi'); // span survives as a SPAN, not a <p>
    expect(c.querySelector('span')).toBe(spanBefore);        // same element — reused, not rebuilt-as-p
    // and the span's reactive text still works after the shift:
    handle.setState(() => label.set('bye'));
    expect(c.querySelector('span')!.textContent).toBe('bye');
    handle.unmount();
  });

  it('a text↔element swap in the same slot reconciles to the right node kind', () => {
    const c = container();
    const asText = signal(true);
    // Pass 1: div's only child is text "x"; pass 2: it is an <span>. A flat keyer would give both key
    // '/div#0/0' → patchNode(prevTEXT, nextElement) takes the text branch and never builds the span.
    const handle = render(
      () => h('div', {}, asText.get() ? 'x' : h('span', {}, 'y')),
      c, {},
    );
    expect(c.querySelector('div')!.textContent).toBe('x');
    expect(c.querySelector('span')).toBeNull();
    handle.setState(() => asText.set(false));
    expect(c.querySelector('span')!.textContent).toBe('y');  // the span was actually created
    handle.unmount();
  });

  it('a structural re-derive triggered by a signal also read in a surviving reactive thunk shows the NEW value', () => {
    // The trigger signal (mode) is read BOTH at the top level (structural) AND inside the child's reactive
    // text thunk. On the structural pass the prior pass's leaf effect is disposed, but it was already
    // scheduled into the same reactive batch by the same write — so it must NOT run again and clobber the
    // reconciled node with the old value. The two branches use DISTINCT thunks on a same-key node so a stale
    // run would be visible.
    const c = container();
    const mode = signal('view');
    const handle = render(
      () => mode.get() === 'view'
        ? h('div', { id: 'x', class: 'v' }, () => 'V=' + mode.get())
        : h('div', { id: 'x', class: 'e' }, () => 'E=' + mode.get()),
      c, {},
    );
    expect(c.querySelector('#x')!.textContent).toBe('V=view');
    expect(c.querySelector('#x')!.getAttribute('class')).toBe('v');
    handle.setState(() => mode.set('edit'));
    // Both the structural swap (class → 'e') AND the fresh thunk's value must win — no stale 'V=view'.
    expect(c.querySelector('#x')!.getAttribute('class')).toBe('e');
    expect(c.querySelector('#x')!.textContent).toBe('E=edit');
    handle.unmount();
  });

  it('a structural re-derive shows the NEW reactive ATTRIBUTE value (no stale clobber)', () => {
    // Same hazard for a reactive PROP: the disposed prior-pass setAttr effect must not re-run and restore
    // the old attribute value after the reconcile applied the new one.
    const c = container();
    const mode = signal('view');
    const handle = render(
      () => mode.get() === 'view'
        ? h('div', { id: 'x', title: () => 'V=' + mode.get() }, 'body')
        : h('div', { id: 'x', title: () => 'E=' + mode.get() }, 'body'),
      c, {},
    );
    expect(c.querySelector('#x')!.getAttribute('title')).toBe('V=view');
    handle.setState(() => mode.set('edit'));
    expect(c.querySelector('#x')!.getAttribute('title')).toBe('E=edit');   // not the stale 'V=view'
    handle.unmount();
  });

  it('detaches removed TEXT-node siblings from the DOM on reconcile', () => {
    // A shrinking list of static text children must actually remove the dropped Text nodes — text vnodes
    // are not indexed as elements, so teardown must resolve their live Text node to detach it.
    const c = container();
    const items = signal(['a', 'b', 'c']);
    const handle = render(() => h('div', {}, ...items.get()), c, {});
    expect(c.querySelector('div')!.textContent).toBe('abc');
    handle.setState(() => items.set(['a']));
    expect(c.querySelector('div')!.textContent).toBe('a');    // 'b' and 'c' Text nodes detached, not left as 'abc'
    handle.setState(() => items.set(['a', 'x']));
    expect(c.querySelector('div')!.textContent).toBe('ax');
    handle.unmount();
  });

  it('detaches a conditionally-removed reactive text sibling', () => {
    const c = container();
    const show = signal(true);
    const label = signal('hi');
    const handle = render(
      () => h('div', {}, () => label.get(), show.get() ? h('span', {}, 'tail') : null),
      c, {},
    );
    expect(c.querySelector('div')!.textContent).toBe('hitail');
    handle.setState(() => show.set(false));
    expect(c.querySelector('div')!.textContent).toBe('hi');   // the span removed; reactive text remains live
    handle.setState(() => label.set('bye'));
    expect(c.querySelector('div')!.textContent).toBe('bye');  // and still reactive after the structural change
    handle.unmount();
  });
});
