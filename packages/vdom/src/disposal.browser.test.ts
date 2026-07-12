import { describe, it, expect, beforeEach } from 'vitest';
import { mount } from './mount.ts';
import { reconcile, type ReconcileHooks } from './reconcile.ts';
import { createDom } from './patch.ts';
import { type VNode } from './vnode.ts';
import { TODO } from './examples.ts';

let container: HTMLElement;
beforeEach(() => { container = document.createElement('div'); document.body.appendChild(container); });

describe('@metael/vdom disposal — a removed subtree is torn down (no leak, no resurrection)', () => {
  it('removing a row via a DSL click detaches it + it never returns on a later add', () => {
    const h = mount(TODO, container, {});          // items in-component; remove = filter() reassign, add = spread
    const rows = () => Array.from(container.querySelectorAll('li'));
    const removed = rows()[1]!;                      // the id:1 "second" row
    (removed.querySelector('button') as HTMLButtonElement).click();   // remove id:1
    expect(removed.isConnected).toBe(false);
    expect(rows().length).toBe(1);
    (Array.from(container.querySelectorAll('button')).find((b) => b.textContent === 'add') as HTMLButtonElement).click();
    expect(rows().some((li) => li === removed)).toBe(false);   // torn-down instance not resurrected
    h.unmount();
  });
});

// The teardown-by-identity contract at the reconcile level: the keyed diff drives which subtrees are gone,
// and the reconcile MUST fire the teardown hook exactly once per removed vnode (root + every descendant),
// by identity — this is the disposal obligation the whole package exists to exercise, isolated from mount.
describe('@metael/vdom reconcile teardown — the keyed diff drives teardown-by-identity, once per subtree', () => {
  const el = (tag: string, key: string, children: VNode[] = []): VNode => ({ tag, props: {}, children, key });

  it('fires onRemove once for a removed row AND once for each of its descendants', () => {
    const index = new Map<string, Element>();
    // A list of two rows, each an <li> wrapping a <span> — so a removed row is a 2-node subtree.
    const prev = [el('li', 'a', [el('span', 'a/s')]), el('li', 'b', [el('span', 'b/s')])];
    for (const c of prev) container.appendChild(createDom(c, document, index));

    const removed: string[] = [];
    const hooks: ReconcileHooks = { onRemove: (v) => removed.push(v.key) };
    // Drop row 'b' → the keyed diff yields a `remove` for 'b'; teardown must walk b AND b/s.
    reconcile(container, prev, [el('li', 'a', [el('span', 'a/s')])], document, index, hooks);

    expect(removed).toEqual(['b', 'b/s']);                       // root then descendant, once each
    expect(index.has('b')).toBe(false);                          // index freed (no leak)
    expect(index.has('b/s')).toBe(false);
    expect(index.has('a')).toBe(true);                           // survivor kept
    expect(Array.from(container.querySelectorAll('li')).map((li) => li.getAttribute('data-key'))).toEqual(['a']);
  });
});
