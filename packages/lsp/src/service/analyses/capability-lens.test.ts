import { describe, it, expect } from 'vitest';
import { composeProfiles } from '@metael/lang/profile';
import { mathProfile } from '@metael/math/lang';
import { stdProfile } from '@metael/std';
import { LanguageService } from '../index.ts';

const p = composeProfiles(mathProfile, stdProfile);

/** Open a document under the composed math+std profile and return its capability lenses. */
function lensesOf(src: string) {
  const svc = new LanguageService();
  svc.openDocument('a', src, 1);
  svc.setProfile('a', p);
  return svc.capabilityLens('a');
}

describe('capabilityLens', () => {
  it('marks a pure-arithmetic function lowerable with no reasons', () => {
    const lenses = lensesOf('function f(x) { return x + 2 * 3 }');
    expect(lenses.length).toBe(1);
    const lens = lenses[0]!;
    expect(lens.lowerable).toBe(true);
    expect(lens.label).toBe('GPU/WASM-lowerable');
    expect(lens.reasons).toEqual([]);
  });

  it('marks a function calling a host builtin (map) not lowerable, with reasons', () => {
    const lenses = lensesOf('function g(a, b) { return map(a, b) }');
    expect(lenses.length).toBe(1);
    const lens = lenses[0]!;
    expect(lens.lowerable).toBe(false);
    expect(lens.label).toBe('not lowerable');
    expect(lens.reasons!.length).toBeGreaterThan(0);
  });

  it('lenses both functions and components, and nothing else', () => {
    const lenses = lensesOf('const k = 1\nfunction f() { return 1 }\ncomponent C() { let n = 0 }');
    expect(lenses.length).toBe(2);
    // The component's reactive `let` disqualifies it.
    const comp = lenses[1]!;
    expect(comp.lowerable).toBe(false);
  });
});
