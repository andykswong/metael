import { describe, it, expect } from 'vitest';
import { parseExpr } from './parser.ts';

const ast = (src: string) => parseExpr(src).expr;

describe('expression parser', () => {
  it('parses literals', () => {
    expect(ast('42')).toMatchObject({ kind: 'number', value: 42 });
    expect(ast('"hi"')).toMatchObject({ kind: 'string', value: 'hi' });
    expect(ast('true')).toMatchObject({ kind: 'bool', value: true });
    expect(ast('null')).toMatchObject({ kind: 'null' });
  });
  it('parses an object literal with a handler arrow', () => {
    expect(ast('{ a: 1, onHover: (h) => h }')).toMatchObject({
      kind: 'object',
      entries: [{ key: 'a' }, { key: 'onHover', value: { kind: 'arrow' } }],
    });
  });
  it('parses an array literal', () => {
    expect(ast('[1, 2, 3]')).toMatchObject({ kind: 'array', elements: [{ value: { kind: 'number', value: 1 } }, { value: { kind: 'number', value: 2 } }, { value: { kind: 'number', value: 3 } }] });
  });
  it('parses member access chains', () => {
    expect(ast('data.kpis')).toMatchObject({ kind: 'member', object: { kind: 'ident', name: 'data' }, property: 'kpis' });
  });
  it('respects binary precedence (* before +)', () => {
    expect(ast('1 + 2 * 3')).toMatchObject({
      kind: 'binary', op: '+', right: { kind: 'binary', op: '*' },
    });
  });
  it('parses a ternary looser than binary ops, right-associative', () => {
    // `a > 0 ? 1 : 0.5` → the whole comparison is the test; branches are 1 / 0.5.
    expect(ast('a > 0 ? 1 : 0.5')).toMatchObject({
      kind: 'cond', test: { kind: 'binary', op: '>' }, then: { kind: 'number', value: 1 }, else: { kind: 'number', value: 0.5 },
    });
    // right-assoc: `a ? b : c ? d : e` = `a ? b : (c ? d : e)`
    expect(ast('a ? b : c ? d : e')).toMatchObject({ kind: 'cond', else: { kind: 'cond' } });
  });
  it('parses a call with args', () => {
    expect(ast('KPI(kpi, { key: 1 })')).toMatchObject({ kind: 'call', callee: { name: 'KPI' }, args: [{}, { kind: 'object' }] });
  });
  it('reports member access to a forbidden key as a diagnostic', () => {
    const r = parseExpr('a.__proto__');
    expect(r.diagnostics[0]?.code).toBe('ML-LANG-FORBIDDEN');
  });
});

describe('spread in literals', () => {
  it('array spread: [...a, x] parses with a spread element', () => {
    const { expr, diagnostics } = parseExpr('[...a, 1]');
    expect(diagnostics).toEqual([]);
    expect(expr.kind).toBe('array');
    const arr = expr as Extract<typeof expr, { kind: 'array' }>;
    expect(arr.elements[0]).toMatchObject({ spread: true });
    expect(arr.elements[1]).toMatchObject({ spread: false });
  });
  it('object spread: {...o, k: 1} parses with a spread entry', () => {
    const { expr, diagnostics } = parseExpr('{ ...o, k: 1 }');
    expect(diagnostics).toEqual([]);
    expect(expr.kind).toBe('object');
    const obj = expr as Extract<typeof expr, { kind: 'object' }>;
    expect(obj.entries[0]).toMatchObject({ spread: true });
    expect(obj.entries[1]).toMatchObject({ key: 'k', spread: false });
  });
  it('a non-spread array/object is unchanged (spread false)', () => {
    const arr = parseExpr('[1, 2]').expr as Extract<ReturnType<typeof parseExpr>['expr'], { kind: 'array' }>;
    expect(arr.elements.every((e) => e.spread === false)).toBe(true);
  });
});
