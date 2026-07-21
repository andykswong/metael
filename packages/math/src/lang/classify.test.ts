import { describe, it, expect } from 'vitest';
import { parseProgram } from '@metael/lang';
import type { Stmt } from '@metael/lang';
import { classifyProfile } from './classify.ts';

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

  it('a null literal makes it non-core (heap/reference value)', () => {
    const r = classifyFn('function f() { null }');
    expect(r.core).toBe(false);
    expect(r.reasons.some((d) => /null is a heap\/reference value/.test(d.message))).toBe(true);
  });

  it('member access makes it non-core (heap value)', () => {
    const r = classifyFn('function f(x) { x.field }');
    expect(r.core).toBe(false);
    expect(r.reasons.some((d) => /member access implies a heap value/.test(d.message))).toBe(true);
  });

  it('indexing makes it non-core (heap value)', () => {
    const r = classifyFn('function f(x) { x[0] }');
    expect(r.core).toBe(false);
    expect(r.reasons.some((d) => /indexing implies a heap value/.test(d.message))).toBe(true);
  });

  it('a unary operator walks its operand (finds the nested member)', () => {
    // `-x.a`: the only heap construct is the member inside the unary operand, so a member reason
    // can only appear if the unary walk recursed into its operand.
    const r = classifyFn('function f(x) { -x.a }');
    expect(r.core).toBe(false);
    expect(r.reasons.some((d) => /member access implies a heap value/.test(d.message))).toBe(true);
  });

  it('a ternary walks test, then, and else branches', () => {
    // then = member, else = index: both reasons only surface if the cond walk descends into
    // both branches.
    const r = classifyFn('function f(x) { x ? x.a : x[0] }');
    expect(r.core).toBe(false);
    expect(r.reasons.some((d) => /member access implies a heap value/.test(d.message))).toBe(true);
    expect(r.reasons.some((d) => /indexing implies a heap value/.test(d.message))).toBe(true);
  });

  it('an indirect call target (obj.m()) is non-core and walks the callee', () => {
    const r = classifyFn('function f(obj) { obj.m() }');
    expect(r.core).toBe(false);
    expect(r.reasons.some((d) => /indirect call target cannot be proven core-compliant/.test(d.message))).toBe(true);
    // walking the callee (a member) also flags the member access.
    expect(r.reasons.some((d) => /member access implies a heap value/.test(d.message))).toBe(true);
  });

  it('a const initializer is walked (member in the init makes it non-core)', () => {
    const r = classifyFn('function f(x) { const y = x.field }');
    expect(r.core).toBe(false);
    expect(r.reasons.some((d) => /member access implies a heap value/.test(d.message))).toBe(true);
  });

  it('a reactive let is non-core', () => {
    const r = classifyFn('function f(x) { let y = x }');
    expect(r.core).toBe(false);
    expect(r.reasons.some((d) => /reactive let is not core-compliant/.test(d.message))).toBe(true);
  });

  it('an assignment walks both its value and its target', () => {
    // value = string, target = member: both reasons prove both walks ran.
    const r = classifyFn('function f(x) { x.a = "hi" }');
    expect(r.core).toBe(false);
    expect(r.reasons.some((d) => /string/.test(d.message))).toBe(true);
    expect(r.reasons.some((d) => /member access implies a heap value/.test(d.message))).toBe(true);
  });

  it('a return value is walked (member in the returned expr makes it non-core)', () => {
    const r = classifyFn('function f(x) { return x.a }');
    expect(r.core).toBe(false);
    expect(r.reasons.some((d) => /member access implies a heap value/.test(d.message))).toBe(true);
  });

  it('an if walks its test, then, and else branches', () => {
    // test/then = member, else = index: all surface only if each branch is walked.
    const r = classifyFn('function f(x) { if (x.a) { x.b } else { x[0] } }');
    expect(r.core).toBe(false);
    expect(r.reasons.some((d) => /member access implies a heap value/.test(d.message))).toBe(true);
    expect(r.reasons.some((d) => /indexing implies a heap value/.test(d.message))).toBe(true);
  });

  it('a for-of loop is non-core and walks its iterable and body', () => {
    const r = classifyFn('function f(obj) { for (const x of obj.rows) { x.a } }');
    expect(r.core).toBe(false);
    expect(r.reasons.some((d) => /for-of iterates a heap collection/.test(d.message))).toBe(true);
    // the iterable (obj.rows) and body (x.a) are both members, so the walk into them flags them.
    expect(r.reasons.some((d) => /member access implies a heap value/.test(d.message))).toBe(true);
  });

  it('a while loop walks its test and body', () => {
    // while itself is not flagged; the non-core reason comes from members reached by walking
    // the test and the body.
    const r = classifyFn('function f(x) { while (x.a) { x.b } }');
    expect(r.core).toBe(false);
    expect(r.reasons.some((d) => /member access implies a heap value/.test(d.message))).toBe(true);
  });

  it('a nested function declaration is non-core', () => {
    const r = classifyFn('function f() { function g() {} }');
    expect(r.core).toBe(false);
    expect(r.reasons.some((d) => /nested function\/component is not core-compliant/.test(d.message))).toBe(true);
  });

  it('a nested component declaration is non-core', () => {
    const r = classifyFn('function f() { component G() {} }');
    expect(r.core).toBe(false);
    expect(r.reasons.some((d) => /nested function\/component is not core-compliant/.test(d.message))).toBe(true);
  });
});
