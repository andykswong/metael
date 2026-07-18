import { describe, it, expect } from 'vitest';
import { PlainStorageHost, RecordingHostEnv } from './ports.ts';
import { makeCallable, readClosureValue, isUserFn, evaluateProgram } from './evaluate.ts';
import type { UserFn } from './evaluate.ts';

function getUserFn(src: string, name: string): UserFn {
  const res = evaluateProgram(`${src}\n${name}`, { host: new PlainStorageHost(), env: new RecordingHostEnv() });
  const fn = res.value;
  if (!isUserFn(fn)) throw new Error('expected a UserFn');
  return fn;
}

describe('makeCallable — invoke a UserFn from the host with a fresh Runner', () => {
  it('calls a pure function with args', () => {
    const host = new PlainStorageHost();
    const fn = getUserFn('function sq(x) { x * x }', 'sq');
    const call = makeCallable(fn, { host, env: new RecordingHostEnv() });
    expect(call(5)).toBe(25);
  });
  it('honors a raised maxSteps (the oracle raises it for a large loop)', () => {
    const host = new PlainStorageHost();
    // A `let` inside a plain `function` would trip ML-LANG-LET-SCOPE (a function body runs pure,
    // insideComponent=false), so the large computation is expressed with reduce over range instead.
    const fn = getUserFn('function sum(n) { reduce(range(n), (a, i) => a + i, 0) }', 'sum');
    const call = makeCallable(fn, { host, env: new RecordingHostEnv(), maxSteps: 1_000_000 });
    expect(call(1000)).toBe(499500);
  });
});

describe('readClosureValue — read a free name from a UserFn closure', () => {
  it('reads a const captured in the closure', () => {
    const host = new PlainStorageHost();
    const fn = getUserFn('const N = 512; function k(i) { i + N }', 'k');
    expect(readClosureValue(fn, 'N', host)).toBe(512);
  });
  it('returns undefined for an unbound name', () => {
    const host = new PlainStorageHost();
    const fn = getUserFn('function k(i) { i }', 'k');
    expect(readClosureValue(fn, 'nope', host)).toBeUndefined();
  });
});
