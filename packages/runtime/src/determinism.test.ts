import { describe, it, expect } from 'vitest';
import { derive } from './derive.ts';
import { RecordingHostEnv, PathKeyMinter } from '@metael/lang';

// Determinism conformance: the AST + injected (data, seed) + settled state fully determine the
// host-value trace. Same source → same trace, INCLUDING seeded rand(). Exercised under the test
// doubles (no domain present).

describe('determinism conformance', () => {
  const run = (source: string, seed: number, data?: unknown) => {
    const env = new RecordingHostEnv();
    derive(source, { env, minter: new PathKeyMinter(), seed, ...(data !== undefined ? { data } : {}) });
    // The host-value trace = the ordered (head, key, resolved-arg-values) the host observed.
    return env.calls.map((c) => ({ head: c.head, key: c.key, args: c.args.map((a) => a.value) }));
  };

  it('same source + same seed → identical host-value trace', () => {
    const src = 'component Story() { for (const i of range(3)) { text("row", { r: rand() }) } }';
    expect(run(src, 42)).toEqual(run(src, 42));
  });

  it('same source + different seed → the rand()-bearing trace differs', () => {
    const src = 'component Story() { text("x", { r: rand() }) }';
    const a = run(src, 1);
    const b = run(src, 2);
    expect(a).not.toEqual(b);   // the rand() arg differs by seed
  });

  it('same source + same data → identical trace, and data genuinely drives it', () => {
    const src = 'component Story() { for (const row of data) { text(row.label, { key: row.id }) } }';
    const data = [{ id: 'a', label: 'A' }, { id: 'b', label: 'B' }];
    expect(run(src, 0, data)).toEqual(run(src, 0, data));      // repeatable
    expect(run(src, 0, data)).not.toEqual(run(src, 0, []));    // and NON-empty data changes the trace
  });

  it('keys are stable across identical runs (reconciliation identity is deterministic)', () => {
    const src = 'component Story() { text("a"); text("b") }';
    const keys1 = run(src, 0).map((c) => c.key);
    const keys2 = run(src, 0).map((c) => c.key);
    expect(keys1).toEqual(keys2);
    expect(keys1).toContain('Story#0/text#0');
    expect(keys1).toContain('Story#0/text#1');
  });
});
