import { describe, it, expect } from 'vitest';
import { parseProgram } from '@metael/lang';
import type { Stmt } from '@metael/lang';
import { classifyProfile } from './index.ts';
import type { Profile } from './index.ts';

const withBuiltins = (names: [string, 'core' | 'host'][]): Profile => ({
  id: 't', heads: new Map(), types: new Map(),
  builtins: new Map(names.map(([name, profile]) => [name, { name, profile, portability: 'exact', takesClosure: false, arity: [0, 9] }])),
});
const fnBody = (src: string): { body: readonly Stmt[] } => {
  const fn = parseProgram(`function f() { ${src} }`).program.stmts.find((s) => s.kind === 'function');
  if (!fn || fn.kind !== 'function') throw new Error('expected a function declaration');
  return fn;
};
// A non-core body must produce at least one why-not diagnostic, all coded ML-LANG-PROFILE.
const expectNonCore = (src: string, profile = withBuiltins([])): void => {
  const r = classifyProfile(fnBody(src), profile);
  expect(r.core).toBe(false);
  expect(r.reasons.length).toBeGreaterThan(0);
  expect(r.reasons.every((d) => d.code === 'ML-LANG-PROFILE')).toBe(true);
};

describe('classifyProfile(fn, profile)', () => {
  it('a scalar arithmetic body is core', () => {
    const r = classifyProfile(fnBody('return 1 + 2 * 3;'), withBuiltins([]));
    expect(r.core).toBe(true);
    expect(r.reasons).toEqual([]);
  });
  it('a call to a host builtin from the profile is non-core with a reason', () => {
    const r = classifyProfile(fnBody('return map(a, b);'), withBuiltins([['map', 'host']]));
    expect(r.core).toBe(false);
    expect(r.reasons.some((d) => d.code === 'ML-LANG-PROFILE')).toBe(true);
  });
  it('a core builtin from the profile is fine', () => {
    const r = classifyProfile(fnBody('return sqrt(x);'), withBuiltins([['sqrt', 'core']]));
    expect(r.core).toBe(true);
  });
  it('an unknown call is conservatively non-core', () => {
    const r = classifyProfile(fnBody('return mystery(1);'), withBuiltins([]));
    expect(r.core).toBe(false);
  });

  // --- Leaf disqualifiers (each expression kind flagged directly). ---
  describe('heap/reference leaf disqualifiers', () => {
    it('a null literal is non-core', () => expectNonCore('return null;'));
    it('a string literal is non-core', () => expectNonCore('return "x";'));
    it('an object literal is non-core', () => expectNonCore('return { a: 1 };'));
    it('an array literal is non-core', () => expectNonCore('return [1, 2];'));
    it('an arrow (closure) is non-core', () => expectNonCore('const g = (x) => x; return 1;'));
    it('member access is non-core', () => expectNonCore('return a.b;'));
    it('index access is non-core', () => expectNonCore('return a[0];'));
  });

  // --- Recursion branches: in each case the ONLY disqualifier sits inside the targeted child, so a
  //     refactor that stopped walking that child would flip the case green-wrongly. Every sibling
  //     position holds a scalar-safe value, which is what pins the branch. ---
  describe('recursion into every child branch', () => {
    it('walks a unary operand (disqualifier only in the operand)', () => expectNonCore('return -null;'));
    it('walks a binary left (safe right, dirty left)', () => expectNonCore('return null + 1;'));
    it('walks a binary right (safe left, dirty right)', () => expectNonCore('return 1 + null;'));
    it('walks a ternary then-branch (safe test/else, dirty then)', () => expectNonCore('return c ? null : 1;'));
    it('walks a ternary else-branch (safe test/then, dirty else)', () => expectNonCore('return c ? 1 : null;'));
    it('walks an if THEN block (safe test, dirty then)', () => expectNonCore('if (c) { return null; }'));
    // The dirty value is ONLY in the else block; the then block and test are clean. If the walker stopped
    // recursing into `s.else`, reasons would be empty and this would wrongly report core=true.
    it('walks an if ELSE block (clean then, dirty else)', () => expectNonCore('if (c) { return 1; } else { return null; }'));
    it('walks a while body (safe test, dirty body)', () => expectNonCore('while (c) { return null; }'));
    it('walks an assign value (safe target, dirty value)', () => expectNonCore('a = null;'));
    // target `a.b` is a member access (flagged); the value `1` is clean, so only the target recursion reaches it.
    it('walks an assign target (dirty target, safe value)', () => expectNonCore('a.b = 1;'));
    it('walks a const initializer (dirty init)', () => expectNonCore('const x = null;'));
  });

  // --- Statement-kind disqualifiers (flagged at the statement, not a child). ---
  describe('statement disqualifiers', () => {
    it('a reactive let is non-core', () => expectNonCore('let x = 1;'));
    it('a for-of iterates a heap collection and is non-core', () => expectNonCore('for (const x of a) { return 1; }'));
    it('a nested function is non-core', () => expectNonCore('function g() { return 1; }'));
  });

  // --- Call classification against the active profile. ---
  describe('call resolution against the profile builtins', () => {
    it('a host builtin call is non-core', () => expectNonCore('return trace(x);', withBuiltins([['trace', 'host']])));
    it('a core builtin call is core', () => {
      expect(classifyProfile(fnBody('return abs(x);'), withBuiltins([['abs', 'core']])).core).toBe(true);
    });
    it('an unknown (unresolvable) call is conservatively non-core', () => expectNonCore('return whoKnows(1);'));
  });

  // --- Positive cases: clean recursion through control flow must NOT flag. ---
  describe('positive (core-compliant) bodies', () => {
    it('nested control flow over scalars stays core', () => {
      const r = classifyProfile(fnBody('if (a > b) { return a; } else { return b; }'), withBuiltins([]));
      expect(r.core).toBe(true);
      expect(r.reasons).toEqual([]);
    });
    it('a while over scalars with a core builtin stays core', () => {
      const r = classifyProfile(fnBody('while (n > 0) { const y = sqrt(n); }'), withBuiltins([['sqrt', 'core']]));
      expect(r.core).toBe(true);
    });
  });
});
