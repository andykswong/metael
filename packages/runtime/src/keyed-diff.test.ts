import { describe, it, expect } from 'vitest';
import { diffKeyed, applyKeyedDiff } from './keyed-diff.ts';

// A tiny keyed item: a key + a disposer spy, so we can assert `remove` runs teardown.
type Item = { key: string; disposed: boolean };
const item = (key: string): Item => ({ key, disposed: false });
const keyOf = (i: Item) => i.key;

describe('diffKeyed — pure add/remove/move op generation', () => {
  it('no change → no ops', () => {
    const a = [item('x'), item('y')];
    expect(diffKeyed(a.map(keyOf), a.map(keyOf))).toEqual([]);
  });
  it('append → one add at the tail', () => {
    const ops = diffKeyed(['x'], ['x', 'y']);
    expect(ops).toEqual([{ type: 'add', key: 'y', index: 1 }]);
  });
  it('remove → one remove op for the gone key', () => {
    const ops = diffKeyed(['x', 'y'], ['x']);
    expect(ops).toEqual([{ type: 'remove', key: 'y' }]);
  });
  it('reorder → move ops preserving identity (no add/remove)', () => {
    const ops = diffKeyed(['x', 'y', 'z'], ['z', 'x', 'y']);
    expect(ops.some((o) => o.type === 'add' || o.type === 'remove')).toBe(false);
    expect(ops.filter((o) => o.type === 'move').length).toBeGreaterThan(0);
  });
  it('a move op carries exact from/to indices', () => {
    // ['a','b'] → ['b','a']: b moves 1→0, a moves 0→1
    const ops = diffKeyed(['a', 'b'], ['b', 'a']);
    expect(ops).toEqual([
      { type: 'move', key: 'b', from: 1, to: 0 },
      { type: 'move', key: 'a', from: 0, to: 1 },
    ]);
  });
  it('mixed add + remove + move', () => {
    const ops = diffKeyed(['a', 'b', 'c'], ['c', 'a', 'd']);
    const types = new Set(ops.map((o) => o.type));
    expect(types.has('remove')).toBe(true);   // b gone
    expect(types.has('add')).toBe(true);       // d new
  });
  it('duplicate keys in `next` → the second occurrence is treated as an add, not an alias', () => {
    const ops = diffKeyed(['x'], ['x', 'x']);
    expect(ops.filter((o) => o.type === 'add').length).toBe(1);
  });
  it('empty prev → every next key is an add (first render)', () => {
    expect(diffKeyed([], ['a', 'b'])).toEqual([
      { type: 'add', key: 'a', index: 0 },
      { type: 'add', key: 'b', index: 1 },
    ]);
  });
  it('empty next → every prev key is a remove (all-removed)', () => {
    expect(diffKeyed(['a', 'b'], [])).toEqual([
      { type: 'remove', key: 'a' },
      { type: 'remove', key: 'b' },
    ]);
  });
  it('single element unchanged → no ops', () => {
    expect(diffKeyed(['solo'], ['solo'])).toEqual([]);
  });
});

describe('applyKeyedDiff — reconciliation with teardown on remove', () => {
  it('produces the next-ordered list reusing matched instances', () => {
    const prev = [item('x'), item('y')];
    const next = applyKeyedDiff(prev, ['y', 'x'], keyOf, {
      create: (key) => item(key),
      dispose: (it) => { it.disposed = true; },
    });
    expect(next.map(keyOf)).toEqual(['y', 'x']);
    expect(next[0]).toBe(prev[1]);   // 'y' instance reused (identity preserved)
    expect(next[1]).toBe(prev[0]);   // 'x' instance reused
  });

  it('a removed key runs dispose exactly once', () => {
    const prev = [item('x'), item('y')];
    const removed = prev[1]!;
    let disposeCount = 0;
    const next = applyKeyedDiff(prev, ['x'], keyOf, {
      create: (key) => item(key),
      dispose: (it) => { it.disposed = true; if (it === removed) disposeCount++; },
    });
    expect(next.map(keyOf)).toEqual(['x']);
    expect(removed.disposed).toBe(true);
    expect(disposeCount).toBe(1);   // exactly once, not just "at least once"
  });

  it('a new key is created via `create`', () => {
    const prev = [item('x')];
    const next = applyKeyedDiff(prev, ['x', 'z'], keyOf, {
      create: (key) => item(key),
      dispose: () => {},
    });
    expect(next.map(keyOf)).toEqual(['x', 'z']);
    expect(next[0]).toBe(prev[0]);         // existing reused
    expect(next[1]!.key).toBe('z');         // new created
  });

  it('empty next → all prev items are disposed (the teardown-critical branch)', () => {
    const prev = [item('a'), item('b')];
    const next = applyKeyedDiff(prev, [], keyOf, { create: (k) => item(k), dispose: (i) => { i.disposed = true; } });
    expect(next).toEqual([]);
    expect(prev.every((i) => i.disposed)).toBe(true);   // everything torn down
  });

  it('empty prev → all next keys are created', () => {
    const next = applyKeyedDiff([], ['a', 'b'], keyOf, { create: (k) => item(k), dispose: () => {} });
    expect(next.map(keyOf)).toEqual(['a', 'b']);
  });

  it('collapsing duplicate-key instances disposes the orphaned instance (dispose by identity, not key)', () => {
    const x1 = item('x'); const x2 = item('x');   // two DISTINCT instances sharing key 'x'
    const out = applyKeyedDiff([x1, x2], ['x'], keyOf, { create: (k) => item(k), dispose: (i) => { i.disposed = true; } });
    expect(out.length).toBe(1);
    expect(out[0]).toBe(x1);            // first-of-key reused
    expect(x2.disposed).toBe(true);    // the orphaned duplicate MUST be torn down (was leaking)
  });

  it('reusing an instance does not spuriously dispose it when a duplicate next-key creates a sibling', () => {
    const x = item('x');
    const out = applyKeyedDiff([x], ['x', 'x'], keyOf, { create: (k) => item(k), dispose: (i) => { i.disposed = true; } });
    expect(out.length).toBe(2);
    expect(out[0]).toBe(x);            // original reused
    expect(out[1]).not.toBe(x);        // second 'x' is a fresh instance (consume-once)
    expect(x.disposed).toBe(false);    // the reused original must NOT be disposed
  });
});

describe('diffKeyed ↔ applyKeyedDiff consistency', () => {
  it('applyKeyedDiff output order matches the next-key order diffKeyed is computed against', () => {
    const prevKeys = ['a', 'b', 'c'];
    const nextKeys = ['c', 'a', 'd'];
    const prev = prevKeys.map(item);
    const out = applyKeyedDiff(prev, nextKeys, keyOf, { create: (k) => item(k), dispose: () => {} });
    expect(out.map(keyOf)).toEqual(nextKeys);   // order is exactly the next sequence
  });
});
