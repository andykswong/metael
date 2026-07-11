/* eslint-disable @typescript-eslint/no-explicit-any -- narrowed-Stmt access in test scaffolding */
import { describe, it, expect } from 'vitest';
import { parseProgram } from './parser.ts';

const prog = (src: string) => parseProgram(src).program.stmts;

describe('statement + wrapping parser', () => {
  it('parses const and reactive let', () => {
    expect(prog('const a = 1;')[0]).toMatchObject({ kind: 'const', name: 'a' });
    expect(prog('let b = 2;')[0]).toMatchObject({ kind: 'let', name: 'b' });
  });
  it('parses a component with a destructured object param', () => {
    expect(prog('component KPI({ label, value }) { }')[0]).toMatchObject({
      kind: 'component', name: 'KPI', params: [{ kind: 'objectPattern', fields: ['label', 'value'] }],
    });
  });
  it('parses an array-destructuring param ([a, b])', () => {
    expect(prog('function pt([x, y]) { }')[0]).toMatchObject({
      kind: 'function', name: 'pt', params: [{ kind: 'arrayPattern', elements: ['x', 'y'] }],
    });
  });
  it('parses a wrapping element: call + child block (head-agnostic — layout is just a call head)', () => {
    const s = prog('layout({ mode: "flex" }) { text("hi"); }');
    expect(s[0]).toMatchObject({ kind: 'expr', expr: { kind: 'call', callee: { name: 'layout' } } });
    expect((s[0] as any).expr.block).toHaveLength(1);
  });
  it('parses a single-trailing-statement wrap: translate([...]) chart(...)', () => {
    const s = prog('translate([0,0,-400]) chart({ type: "bar" });');
    expect((s[0] as any).expr.block).toHaveLength(1);
  });
  it('newline-separated wrapping calls stay SIBLINGS, not nested (sibling guard)', () => {
    // The brace-less single-trailing-statement wrap fires ONLY when the trailing statement is on
    // the SAME logical line (no `;` and no newline between the `)` and the next token). A newline
    // (or `;`) makes them siblings — else `KPI(a)` newline `KPI(b)` would mis-nest b inside a.
    // (Parsed here as the body of a component — there is no `story` block.)
    const body = (prog('component Story() {\n  KPI(a)\n  KPI(b)\n}')[0] as any).body;
    expect(body).toHaveLength(2);                          // two siblings
    expect(body.every((st: any) => st.expr.callee.name === 'KPI')).toBe(true);
    expect(body[0].expr.block).toBeUndefined();            // KPI(a) has NO child block
  });
  it('same-line brace-less wrap still nests (translate([...]) chart(...))', () => {
    const body = (prog('component Story() { translate([0,0,-400]) chart({ type: "bar" }) }')[0] as any).body;
    expect(body).toHaveLength(1);                          // one child: the translate wrap
    expect(body[0].expr.callee.name).toBe('translate');
    expect(body[0].expr.block).toHaveLength(1);            // chart is its child
  });
  it('parses for-of, if/else, while', () => {
    expect(prog('for (const k of data.kpis) { KPI(k); }')[0]).toMatchObject({ kind: 'for', binding: 'k' });
    expect(prog('if (a) { x(); } else { y(); }')[0]).toMatchObject({ kind: 'if' });
    expect(prog('while (a) { x(); }')[0]).toMatchObject({ kind: 'while' });
  });
  it('parses brace-less single-statement control-flow bodies (Body ::= Block | Stmt)', () => {
    // JS-like `for (…) KPI(k)` / `if (…) x()` — the body is a single brace-less statement.
    const f = prog('for (const k of data.kpis) KPI(k)')[0] as any;
    expect(f).toMatchObject({ kind: 'for', binding: 'k' });
    expect(f.body).toHaveLength(1);
    expect(f.body[0]).toMatchObject({ kind: 'expr' });
    expect((prog('if (a) x()')[0] as any).then).toHaveLength(1);
    expect((prog('while (a) x()')[0] as any).body).toHaveLength(1);
  });
  it('parses a block-bodied arrow whose body is a Stmt[] (state-mutating handler)', () => {
    // `(h) => { hover = h }` — the arrow body is a statement block (an assignment), not an object.
    const arrow = (prog('const f = (h) => { hover = h };')[0] as any).init;
    expect(arrow.kind).toBe('arrow');
    expect(Array.isArray(arrow.body)).toBe(true);              // Stmt[] block body
    expect(arrow.body[0]).toMatchObject({ kind: 'assign' });
    // an expression-bodied arrow still yields an Expr body (not an array)
    expect(Array.isArray((prog('const g = (h) => h;')[0] as any).init.body)).toBe(false);
  });
  it('flags a redeclaration in one scope (single JS namespace)', () => {
    const r = parseProgram('const a = 1; const a = 2;');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-REDECL')).toBe(true);
  });
  it('emits ML-LANG-PARSE on malformed source (missing rparen)', () => {
    const r = parseProgram('component Story( { }');   // unclosed param list
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-PARSE')).toBe(true);
  });
  it('never throws on truncated input — returns diagnostics (cursor clamped at eof)', () => {
    // Regression: parsePrimary's error-recovery `next()` must not advance past the trailing eof
    // and then read `undefined.type`. Each of these previously threw a TypeError.
    for (const src of ['const a =', 'let b = 1 +', 'x =', 'foo().', '(((']) {
      expect(() => parseProgram(src), src).not.toThrow();
    }
    // …and a truncated expression still surfaces a parse diagnostic (never silently swallowed).
    expect(parseProgram('const a =').diagnostics.some((d) => d.code === 'ML-LANG-PARSE')).toBe(true);
  });
});
