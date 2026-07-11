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
    expect(ast('[1, 2, 3]')).toMatchObject({ kind: 'array', elements: [{ value: 1 }, { value: 2 }, { value: 3 }] });
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
