import { describe, it, expect } from 'vitest';
import { evaluateProgram } from './evaluate.ts';
import { lowerEntry } from './lower.ts';
import { PlainStorageHost, RecordingHostEnv, PathKeyMinter } from './ports.ts';
import type { KeyMinter } from './ports.ts';
import { STD_BUILTINS } from '@metael/std';

// A metael-visible record is built with a NULL prototype (Object.create(null)), so a program can never
// reach an inherited `toString`/`constructor`/`__proto__` through any object it constructs. This is
// DEFENSE IN DEPTH on top of the FORBIDDEN_KEYS guard: every legitimate operation over a record
// (keys/values/entries/spread/JSON/member-read/index-read) must behave exactly as before.
const run = (src: string, data?: unknown) =>
  evaluateProgram(src, { data, host: new PlainStorageHost(), env: new RecordingHostEnv(), builtins: [STD_BUILTINS] });

describe('null-prototype records — legitimate operations are unchanged', () => {
  it('keys / values / entries still decompose an object literal', () => {
    expect(run('keys({ a: 1, b: 2 })').value).toEqual(['a', 'b']);
    expect(run('values({ a: 1, b: 2 })').value).toEqual([1, 2]);
    expect(run('entries({ a: 1, b: 2 })').value).toEqual([['a', 1], ['b', 2]]);
  });
  it('spread still copies own keys (order preserved, last-wins)', () => {
    expect(run('const m = { ...{ a: 1 }, b: 2 }; keys(m)').value).toEqual(['a', 'b']);
    expect(run('const m = { ...{ a: 1 }, b: 2 }; values(m)').value).toEqual([1, 2]);
    expect(run('const m = { ...{ a: 1 }, a: 9 }; values(m)').value).toEqual([9]);   // last-wins
  });
  it('reading a PRESENT key returns its value; a MISSING key returns null', () => {
    expect(run('const o = { a: 1 }; o.a').value).toBe(1);
    expect(run('const o = { a: 1 }; o["a"]').value).toBe(1);
    expect(run('const o = { a: 1 }; o.zzz').value).toBeNull();
    expect(run('const o = { a: 1 }; o["zzz"]').value).toBeNull();
  });
  it('a record string-coerces via JSON (own enumerable props only)', () => {
    expect(run('"" + { a: 1, b: 2 }').value).toBe('{"a":1,"b":2}');
    expect(run('"" + { ...{ a: 1 }, b: 2 }').value).toBe('{"a":1,"b":2}');
  });
});

describe('null-prototype records — no inherited proto/constructor/toString', () => {
  it('the constructed record has a null prototype', () => {
    const rec = run('const o = { a: 1, b: 2 }; o').value as object;
    expect(Object.getPrototypeOf(rec)).toBe(null);
    const spread = run('const m = { ...{ a: 1 }, b: 2 }; m').value as object;
    expect(Object.getPrototypeOf(spread)).toBe(null);
  });
  it('no inherited toString/constructor/__proto__ on the host value', () => {
    const rec = run('const o = { a: 1 }; o').value as Record<string, unknown>;
    expect('toString' in rec).toBe(false);
    expect(rec.toString).toBe(undefined);
    expect(rec.constructor).toBe(undefined);
    expect((rec as { __proto__?: unknown }).__proto__).toBe(undefined);
  });
  it('reading toString / constructor from the LANGUAGE surface is null (never a function)', () => {
    // `toString` is not a FORBIDDEN key, so the read is a plain member miss → null (no diagnostic).
    const t = run('const o = { a: 1 }; o.toString');
    expect(t.value).toBeNull();
    // `constructor`/`__proto__` are FORBIDDEN keys → blocked before the read, null with a diagnostic.
    const c = run('const o = { a: 1 }; o.constructor');
    expect(c.value).toBeNull();
    expect(c.diagnostics.some((d) => d.code === 'ML-LANG-FORBIDDEN')).toBe(true);
  });
  it('a __proto__ / constructor / prototype object-literal KEY still fails ML-LANG-FORBIDDEN', () => {
    for (const k of ['__proto__', 'constructor', 'prototype']) {
      expect(run(`const o = { ${k}: 1 };`).diagnostics.some((d) => d.code === 'ML-LANG-FORBIDDEN')).toBe(true);
    }
  });
});

// Indexing with an OBJECT (or any non-string/non-number) key must fail CLOSED to a localized diagnostic +
// null — NOT abort the whole program. A metael record is null-prototype, so `String(record)` throws
// ('Cannot convert object to primitive value'); coercing the key via String() there would escape to the
// top-level catch → ML-LANG-INTERNAL, losing every sibling statement's output. This pins the fail-closed
// contract every other readMember path honors.
describe('null-prototype records — indexing with a record key fails closed (no whole-program abort)', () => {
  it('m[o] with a record key returns null + a localized ML-LANG-BAD-KEY, and sibling output SURVIVES', () => {
    const r = run('const o = { a: 1 }\nconst m = { x: 2 }\nconst bad = m[o]\nconst good = 999\ngood');
    expect(r.value).toBe(999);                                                  // the sibling `good` still evaluates
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-BAD-KEY')).toBe(true); // localized, not fatal
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-INTERNAL')).toBe(false); // NOT a whole-program abort
  });
  it('indexing with an ARRAY key is also fail-closed (arrays are objects too)', () => {
    const r = run('const arr = [1, 2]\nconst m = { x: 2 }\nconst bad = m[arr]\nconst good = 42\ngood');
    expect(r.value).toBe(42);
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-BAD-KEY')).toBe(true);
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-INTERNAL')).toBe(false);
  });
  it('a boolean index key still coerces to a string key (String(true)) — behavior preserved', () => {
    // `m[true]` reads member 'true' → missing → null (no diagnostic). Booleans are NOT objects, so the
    // object-key guard does not catch them; the prior String()-coercion path is unchanged.
    const r = run('const m = { x: 2 }\nm[true]');
    expect(r.value).toBeNull();
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-BAD-KEY')).toBe(false);
  });
});

// A capturing minter records the `content` record that lowering merges from a list item's object args
// (lower.ts's argContent), so the test can assert that record is also null-proto and round-trips.
class CapturingMinter implements KeyMinter {
  private readonly base = new PathKeyMinter();
  readonly contents: Record<string, unknown>[] = [];
  structural(parentKey: string, kind: string, lexicalOrdinal: number): string {
    return this.base.structural(parentKey, kind, lexicalOrdinal);
  }
  listItem(parentKey: string, kind: string, authorKey: unknown, ordinal: number, content: Record<string, unknown>): string {
    this.contents.push(content);
    return this.base.listItem(parentKey, kind, authorKey, ordinal);
  }
}

describe('null-prototype records — lowering argContent round-trips (no inherited props)', () => {
  it('a list item\'s merged arg-content record is null-proto and carries the expected keys', () => {
    const minter = new CapturingMinter();
    const src = `component Story() { for (const row of data) { text(row.label, { key: row.id, tag: row.label }) } }`;
    lowerEntry(src, { host: new PlainStorageHost(), env: new RecordingHostEnv(), minter, data: [{ id: 'x', label: 'X' }] });
    expect(minter.contents.length).toBe(1);
    const content = minter.contents[0]!;
    expect(content.key).toBe('x');
    expect(content.tag).toBe('X');
    expect(Object.getPrototypeOf(content)).toBe(null);
    expect('toString' in content).toBe(false);
  });
});
