import { describe, it, expect } from 'vitest';
import { kernel, kernelAst, letVar, set, forRange, ifThen, ret } from './index.ts';
import { call, lit } from './node.ts';
import { stripSpans, parseProgram, isUserFn } from '@metael/lang';

// The equivalent metael source for a kernel, parsed to its decl AST (spans normalized).
function parsedKernelAst(src: string): unknown {
  const { program } = parseProgram(src);
  const decl = program.stmts.find((s) => s.kind === 'component' || s.kind === 'function');
  return stripSpans(decl);
}

describe('kernel(...) builder', () => {
  it('a scalar map kernel matches the parsed component AST', () => {
    const k = kernel((row, col) => call('f32', row.add(col))); // (row, col) => f32(row + col)
    expect(isUserFn(k)).toBe(true);
    // NOTE: kernel() returns a UserFn (closure value), whereas parseProgram yields the DECL node. The
    // full equivalence corpus (a later task) compares the assembled decl AST against the parsed decl AST;
    // here just assert structural sanity plus one genuine decl-AST equivalence check below.
    expect(k.params.length).toBe(2);
  });

  it('kernelAst of a scalar map equals the parsed decl AST (arrow-return → return stmt)', () => {
    // The arrow-returned KNode becomes a `return` statement — the dispatchable form the emitters write —
    // matching the parser's `component K(p0, p1) { return f32(p0 + p1) }`. Positional param names p0/p1 +
    // fixed decl name K. (Mapping it to a trailing `expr` stmt would dispatch all-zeros; the emitters
    // lower only `return` to the output write.)
    expect(stripSpans(kernelAst((p0, p1) => call('f32', p0.add(p1))))).toEqual(
      parsedKernelAst('component K(p0, p1) { return f32(p0 + p1) }'),
    );
  });

  it('kernelAst captures let / for-of range / assign / arrow-return', () => {
    const decl = kernelAst((_p0) => {
      const acc = letVar('acc', lit(0));
      forRange(4, (i) => set(acc, acc.add(i)));
      return acc; // arrow-return → a `return` stmt, matching the source's `return acc`
    });
    expect(stripSpans(decl)).toEqual(
      parsedKernelAst('component K(p0) { let acc = 0\nfor (const i of range(4)) { acc = acc + i }\nreturn acc }'),
    );
  });

  it('kernelAst captures an explicit ret() as a return stmt', () => {
    const decl = kernelAst((p0) => {
      ret(p0);
    });
    expect(stripSpans(decl)).toEqual(parsedKernelAst('component K(p0) { return p0 }'));
  });

  it('ifThen omits the else key when no else branch is given (matches the parser)', () => {
    const decl = kernelAst((p0) => {
      const acc = letVar('acc', lit(0));
      ifThen(p0.gt(lit(0)), () => set(acc, lit(1)));
      return acc;
    });
    expect(stripSpans(decl)).toEqual(
      parsedKernelAst('component K(p0) { let acc = 0\nif (p0 > 0) { acc = 1 }\nreturn acc }'),
    );
    // No `else` key at all — the parser omits it, so an `else: undefined` would break the compare.
    const body = (stripSpans(decl) as { body: Array<Record<string, unknown>> }).body;
    const ifStmt = body[1] ?? {};
    expect(ifStmt.kind).toBe('if');
    expect('else' in ifStmt).toBe(false);
  });

  it('nested forRange loops get DISTINCT loop variables (no shadow)', () => {
    // A nested forRange must NOT reuse the outer loop var: the CPU emitter runs the body over one flat
    // scope with no per-loop child scope, so a shadowed `i` would clobber the outer loop's value. Depth 0 →
    // `i`, depth 1 → `i1` — the source below uses exactly those names, so the AST equivalence proves it.
    const decl = kernelAst((_p0) => {
      const acc = letVar('acc', lit(0));
      forRange(2, () => {
        forRange(2, (inner) => set(acc, acc.add(inner)));
      });
      return acc;
    });
    expect(stripSpans(decl)).toEqual(
      parsedKernelAst('component K(p0) { let acc = 0\nfor (const i of range(2)) { for (const i1 of range(2)) { acc = acc + i1 } }\nreturn acc }'),
    );
  });
});
