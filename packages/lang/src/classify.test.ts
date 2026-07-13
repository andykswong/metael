import { describe, it, expect } from 'vitest';
import { parseProgram } from './parser.ts';
import { classifyProfile } from './classify.ts';
import type { Stmt } from './ast.ts';

/** Parse a single top-level `function` decl and classify it. */
function classifyFn(src: string) {
  const { program } = parseProgram(src);
  const fn = program.stmts.find((s: Stmt) => s.kind === 'function');
  if (!fn || fn.kind !== 'function') throw new Error('no function decl in source');
  return classifyProfile(fn);
}

describe('classifyProfile', () => {
  it('a pure scalar-numeric function is core-compliant', () => {
    const r = classifyFn('function f(x) { min(abs(x), 10) * 2 }');
    expect(r.core).toBe(true);
    expect(r.reasons).toEqual([]);
  });
  it('calling a host builtin (map) makes it non-core with a reason', () => {
    const r = classifyFn('function f(xs) { map(xs, (x) => x + 1) }');
    expect(r.core).toBe(false);
    expect(r.reasons.some((d) => /host builtin 'map'/.test(d.message))).toBe(true);
  });
  it('a string literal makes it non-core', () => {
    const r = classifyFn('function f(x) { x + "hi" }');
    expect(r.core).toBe(false);
    expect(r.reasons.some((d) => /string/.test(d.message))).toBe(true);
  });
  it('an object literal makes it non-core (heap type)', () => {
    const r = classifyFn('function f(x) { { a: x } }');
    expect(r.core).toBe(false);
    expect(r.reasons.some((d) => /object|heap/.test(d.message))).toBe(true);
  });
  it('an array literal makes it non-core (heap type)', () => {
    const r = classifyFn('function f(x) { [x, x] }');
    expect(r.core).toBe(false);
  });
  it('a gpu-tolerant core builtin (sqrt) is still core-compliant', () => {
    const r = classifyFn('function f(x) { sqrt(x) }');
    expect(r.core).toBe(true);
  });
  it('an unknown call is non-core (cannot prove core)', () => {
    const r = classifyFn('function f(x) { mystery(x) }');
    expect(r.core).toBe(false);
    expect(r.reasons.some((d) => /unknown|cannot/.test(d.message))).toBe(true);
  });
});
