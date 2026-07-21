import { describe, it, expect } from 'vitest';
import { evaluateProgram, PlainStorageHost, RecordingHostEnv, frozenClock } from '@metael/lang';
import type { ReactiveHost } from '@metael/lang';
import { STD_BUILTINS } from './index.ts';

const withClock = (src: string, t: number) =>
  evaluateProgram(src, { host: new PlainStorageHost(() => frozenClock(t)), env: new RecordingHostEnv(), builtins: [STD_BUILTINS] });

// A minimal reactive host that injects NO clock capability — exercises the fail-closed datetime path.
// Its cell/generation methods are unused by a bare now()/monotonic() eval; they satisfy the interface only.
const clocklessHost = (): ReactiveHost => ({
  allocateCell: () => 0,
  readCell: () => undefined,
  writeCell: () => {},
  runLeafEffect: () => ({ [Symbol.dispose]() {} }),
  scope: <T>(run: () => T) => ({ value: run(), [Symbol.dispose]() {} }),
  allocateGeneration: () => 0,
  readGeneration: () => 0,
  touchGeneration: () => {},
});

describe('std datetime builtins', () => {
  it('now() with a frozen-clock host is deterministic (returns the frozen wall-clock)', () => {
    expect(withClock('now()', 1_700_000_000_000).value).toBe(1_700_000_000_000);
    expect(withClock('now()', 42).value).toBe(42);
  });

  it('monotonic() with a frozen-clock host returns the frozen monotonic reading', () => {
    expect(withClock('monotonic()', 5_000).value).toBe(5_000);
  });

  it('now() with a host that injects NO clock is fail-closed: ML-LANG-NO-CLOCK + null (never a fake 0)', () => {
    const res = evaluateProgram('now()', { host: clocklessHost(), env: new RecordingHostEnv(), builtins: [STD_BUILTINS] });
    expect(res.diagnostics.some((d) => d.code === 'ML-LANG-NO-CLOCK')).toBe(true);
    expect(res.value).toBe(null);
  });

  it('monotonic() with a host that injects NO clock is fail-closed: ML-LANG-NO-CLOCK + null', () => {
    const res = evaluateProgram('monotonic()', { host: clocklessHost(), env: new RecordingHostEnv(), builtins: [STD_BUILTINS] });
    expect(res.diagnostics.some((d) => d.code === 'ML-LANG-NO-CLOCK')).toBe(true);
    expect(res.value).toBe(null);
  });

  it('now() is budget-charged (ticks per call)', () => {
    const res = evaluateProgram('while (true) { now() }', {
      host: new PlainStorageHost(() => frozenClock(1)), env: new RecordingHostEnv(), maxSteps: 1000, builtins: [STD_BUILTINS],
    });
    expect(res.diagnostics.some((d) => d.code === 'ML-LANG-BUDGET')).toBe(true);
  });

  it('a user function named now shadows the builtin (unbound-head-only)', () => {
    expect(withClock('function now() { 7 } now()', 999).value).toBe(7);
  });
});
