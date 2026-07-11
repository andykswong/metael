import { describe, it, expect } from 'vitest';
import type { Scope } from '@metael/lang';
import { RuntimeReactiveHost } from './reactive-host.ts';
import { applyKeyedDiff } from './keyed-diff.ts';
import { change } from './reactive.ts';

// Disposal conformance: a keyed `remove` disposes the removed subtree's cells + leaf effects — no
// post-removal sink calls, no cellKey retention. Modeled with the runtime's own host + keyed diff:
// each "row" is a scope() holding a leaf effect; removing the row's key must dispose that scope.

type Row = { key: string; scope: Scope<unknown>; sunk: number[] };

describe('disposal conformance', () => {
  it('a keyed remove disposes the removed row scope → no post-removal sink calls', () => {
    const host = new RuntimeReactiveHost();
    const cellByKey = new Map<string, unknown>();

    const makeRow = (key: string): Row => {
      const sunk: number[] = [];
      const scope = host.scope(() => {
        const cell = host.allocateCell(0, `${key}::n#0`);
        cellByKey.set(key, cell);
        host.runLeafEffect(() => host.readCell(cell), (v) => sunk.push(v as number));
        return null;
      });
      return { key, scope, sunk };
    };

    let rows: Row[] = [makeRow('a'), makeRow('b')];
    // Both leaf effects piped their initial value.
    expect(rows.map((r) => r.sunk)).toEqual([[0], [0]]);

    const removedB = rows[1]!;
    // Reconcile to drop 'b': applyKeyedDiff runs dispose() on the gone row → disposes its scope.
    rows = applyKeyedDiff(rows, ['a'], (r) => r.key, {
      create: (k) => makeRow(k),
      dispose: (r) => r.scope[Symbol.dispose](),
    });
    expect(rows.map((r) => r.key)).toEqual(['a']);

    // Write to B's (now-disposed) cell: its leaf effect must NOT re-pipe (it was torn down).
    const bCell = cellByKey.get('b');
    change(() => host.writeCell(bCell, 99));
    expect(removedB.sunk).toEqual([0]);   // NO post-removal sink call — disposal held

    // NO cellKey retention: B's cell is gone from the keyed store, so exportState() won't resurrect
    // its state on a subsequent re-derive (the leak the disposal contract exists to prevent).
    expect(host.exportState().has('b::n#0')).toBe(false);
    expect(host.exportState().has('a::n#0')).toBe(true);

    // A's effect still lives (control): writing A's cell re-pipes.
    const aCell = cellByKey.get('a');
    change(() => host.writeCell(aCell, 7));
    expect(rows[0]!.sunk).toEqual([0, 7]);
  });

  it('disposing a scope is idempotent (double dispose is safe)', () => {
    const host = new RuntimeReactiveHost();
    const s = host.scope(() => { host.runLeafEffect(() => 1, () => {}); return null; });
    s[Symbol.dispose]();
    expect(() => s[Symbol.dispose]()).not.toThrow();
  });
});
