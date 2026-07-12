import { describe, it, expect } from 'vitest';
import { planLevel } from './patch.ts';
import { type VNode } from './vnode.ts';

const el = (tag: string, key: string): VNode => ({ tag, props: {}, children: [], key });

describe('planLevel — resolves diffKeyed ops into matched/created/removed instances', () => {
  it('no change → no ops, all matched', () => {
    const prev = [el('li', 'a'), el('li', 'b')];
    const plan = planLevel(prev, [el('li', 'a'), el('li', 'b')]);
    expect(plan.ops).toEqual([]);
    expect(plan.matched.get('a')).toBe(prev[0]);
    expect(plan.removed).toEqual([]);
  });
  it('append → one add op', () => {
    const plan = planLevel([el('li', 'a')], [el('li', 'a'), el('li', 'b')]);
    expect(plan.ops).toEqual([{ type: 'add', key: 'b', index: 1 }]);
  });
  it('remove → a remove op + the removed instance reported for teardown', () => {
    const removed = el('li', 'b');
    const plan = planLevel([el('li', 'a'), removed], [el('li', 'a')]);
    expect(plan.ops).toEqual([{ type: 'remove', key: 'b' }]);
    expect(plan.removed).toEqual([removed]);
  });
  it('reorder → move ops; matched instances reused by identity (no add/remove)', () => {
    const a = el('li', 'a'); const b = el('li', 'b'); const c = el('li', 'c');
    const plan = planLevel([a, b, c], [el('li', 'c'), el('li', 'a'), el('li', 'b')]);
    expect(plan.ops.some((o) => o.type === 'add' || o.type === 'remove')).toBe(false);
    expect(plan.ops.filter((o) => o.type === 'move').length).toBeGreaterThan(0);
    expect(plan.matched.get('a')).toBe(a);
    expect(plan.matched.get('c')).toBe(c);
  });
  it('mixed: b removed, d added', () => {
    const plan = planLevel([el('li', 'a'), el('li', 'b'), el('li', 'c')], [el('li', 'c'), el('li', 'a'), el('li', 'd')]);
    expect(plan.ops.map((o) => o.type)).toContain('remove');
    expect(plan.ops.map((o) => o.type)).toContain('add');
    expect(plan.removed.map((v) => v.key)).toEqual(['b']);
  });
});
