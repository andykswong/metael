import { describe, it, expect } from 'vitest';
import { RuntimeReactiveHost } from './reactive-host.ts';
import { change } from './reactive.ts';

describe('RuntimeReactiveHost', () => {
  it('allocateCell + readCell round-trips the initial value', () => {
    const h = new RuntimeReactiveHost();
    const c = h.allocateCell(5);
    expect(h.readCell(c)).toBe(5);
  });

  it('runLeafEffect pipes the initial value synchronously, then re-pipes on change', () => {
    const h = new RuntimeReactiveHost();
    const c = h.allocateCell(1);
    const sunk: number[] = [];
    h.runLeafEffect(() => (h.readCell(c) as number) * 10, (v) => sunk.push(v as number));
    expect(sunk).toEqual([10]);                 // initial pipe at subscription
    change(() => h.writeCell(c, 2));
    expect(sunk).toEqual([10, 20]);             // re-ran in place on dependent write
  });

  it('disposing a leaf effect stops further re-runs', () => {
    const h = new RuntimeReactiveHost();
    const c = h.allocateCell(1);
    const sunk: number[] = [];
    const d = h.runLeafEffect(() => h.readCell(c), (v) => sunk.push(v as number));
    d[Symbol.dispose]();
    change(() => h.writeCell(c, 2));
    expect(sunk).toEqual([1]);                  // no re-pipe after dispose
  });

  it('scope() disposal tears down every leaf effect allocated inside it', () => {
    const h = new RuntimeReactiveHost();
    const c = h.allocateCell(1);
    const sunk: number[] = [];
    const s = h.scope(() => {
      h.runLeafEffect(() => h.readCell(c), (v) => sunk.push(v as number));
      return 'built';
    });
    expect(s.value).toBe('built');
    expect(sunk).toEqual([1]);
    s[Symbol.dispose]();
    change(() => h.writeCell(c, 9));
    expect(sunk).toEqual([1]);                  // scope teardown stopped the in-scope effect
  });

  it('scope() disposal frees the cells allocated inside it (no cellKey retention)', () => {
    const h = new RuntimeReactiveHost();
    const s = h.scope(() => {
      h.allocateCell(0, 'row::n#0');            // a keyed cell allocated inside the scope
      return null;
    });
    expect([...h.exportState().keys()]).toEqual(['row::n#0']);   // present while the scope lives
    s[Symbol.dispose]();
    expect([...h.exportState().keys()]).toEqual([]);             // freed on disposal — no leak
  });

  it('cellKey latch: a surviving keyed cell restores prior state; a new/unkeyed cell does not', () => {
    const first = new RuntimeReactiveHost();
    const kc = first.allocateCell(0, 'Story#0::n#0');
    change(() => first.writeCell(kc, 42));      // a handler mutated state to 42
    const priorState = first.exportState();
    expect(priorState.get('Story#0::n#0')).toBe(42);

    const second = new RuntimeReactiveHost(priorState);
    const kc2 = second.allocateCell(0, 'Story#0::n#0');
    expect(second.readCell(kc2)).toBe(42);      // surviving instance kept its state
    const freshKeyed = second.allocateCell(0, 'Story#0::m#0');   // a NEW key → resets to initializer
    expect(second.readCell(freshKeyed)).toBe(0);
    const unkeyed = second.allocateCell(7);      // an UNKEYED cell → always its initializer (never latched)
    expect(second.readCell(unkeyed)).toBe(7);
  });

  it('scope() whose run throws disposes the partial subtree and rethrows', () => {
    const h = new RuntimeReactiveHost();
    const c = h.allocateCell(1);
    const sunk: number[] = [];
    expect(() => h.scope(() => {
      h.runLeafEffect(() => h.readCell(c), (v) => sunk.push(v as number));
      throw new Error('boom');
    })).toThrow('boom');
    expect(sunk).toEqual([1]);                   // the in-scope effect ran once (initial pipe)
    change(() => h.writeCell(c, 5));
    expect(sunk).toEqual([1]);                   // ...but was torn down on the throw — no re-pipe
  });

  it('nested scopes restore the owner boundary (inner disposal does not affect the outer)', () => {
    const h = new RuntimeReactiveHost();
    const a = h.allocateCell(1); const b = h.allocateCell(1);
    const outerSunk: number[] = []; const innerSunk: number[] = [];
    const outer = h.scope(() => {
      h.runLeafEffect(() => h.readCell(a), (v) => outerSunk.push(v as number));
      const inner = h.scope(() => {
        h.runLeafEffect(() => h.readCell(b), (v) => innerSunk.push(v as number));
        return null;
      });
      inner[Symbol.dispose]();                   // dispose inner only
      return null;
    });
    change(() => h.writeCell(b, 2));             // inner effect gone → no re-pipe
    expect(innerSunk).toEqual([1]);
    change(() => h.writeCell(a, 3));             // outer effect still live
    expect(outerSunk).toEqual([1, 3]);
    outer[Symbol.dispose]();
    change(() => h.writeCell(a, 4));
    expect(outerSunk).toEqual([1, 3]);           // now torn down
  });

  it('latch preserves a genuinely-undefined settled value (has=true, get=undefined) over the initializer', () => {
    const priorState = new Map<string, unknown>([['k::u#0', undefined]]);
    const h = new RuntimeReactiveHost(priorState);
    const c = h.allocateCell(99, 'k::u#0');       // key present with value undefined → latch to undefined, NOT 99
    expect(h.readCell(c)).toBe(undefined);
  });

  it('exportState snapshots only keyed cells', () => {
    const h = new RuntimeReactiveHost();
    h.allocateCell(1);                          // unkeyed → not exported
    h.allocateCell(2, 'k::a#0');
    const s = h.exportState();
    expect([...s.keys()]).toEqual(['k::a#0']);
    expect(s.get('k::a#0')).toBe(2);
  });
});
