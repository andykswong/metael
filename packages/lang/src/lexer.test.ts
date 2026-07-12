import { describe, it, expect } from 'vitest';
import { lex } from './lexer.ts';

const kinds = (src: string) => lex(src).tokens.map((t) => t.type);

describe('lexer', () => {
  it('lexes bare identifiers (no $) and keywords', () => {
    expect(kinds('component KPI const let function')).toEqual(
      ['component', 'ident', 'const', 'let', 'function', 'eof']);
  });
  it('lexes numbers, strings, booleans, null', () => {
    expect(kinds('1 2.5 "hi" true false null')).toEqual(
      ['number', 'number', 'string', 'true', 'false', 'null', 'eof']);
  });
  it('lexes punctuation, the arrow, and the ternary question mark', () => {
    expect(kinds('{ } [ ] ( ) . , : ; => ?')).toEqual(
      ['lbrace', 'rbrace', 'lbracket', 'rbracket', 'lparen', 'rparen', 'dot', 'comma', 'colon', 'semi', 'arrow', 'question', 'eof']);
  });
  it('lexes operators with distinct assign vs equals', () => {
    expect(kinds('= == != < <= > >= + - * / % && || !')).toEqual(
      ['assign', 'eq', 'neq', 'lt', 'le', 'gt', 'ge', 'plus', 'minus', 'star', 'slash', 'percent', 'and', 'or', 'not', 'eof']);
  });
  it('reports an unterminated string as a diagnostic, never throws', () => {
    const r = lex('"abc');
    expect(r.diagnostics.length).toBeGreaterThan(0);
    expect(r.diagnostics[0]?.code).toBe('ML-LANG-LEX');
  });
});

describe('ellipsis token', () => {
  it('lexes `...` as a single ellipsis token, not three dots', () => {
    const { tokens } = lex('[...a]');
    const types = tokens.map((t) => t.type);
    expect(types).toContain('ellipsis');
    expect(types.filter((t) => t === 'dot')).toEqual([]);   // no stray dots
    expect(types).toEqual(['lbracket', 'ellipsis', 'ident', 'rbracket', 'eof']);
  });
  it('a single dot is still a dot (member access)', () => {
    expect(lex('a.b').tokens.map((t) => t.type)).toEqual(['ident', 'dot', 'ident', 'eof']);
  });
  it('two dots lex as two dots (no ellipsis false-match)', () => {
    expect(lex('a..b').tokens.filter((t) => t.type === 'ellipsis')).toEqual([]);
  });
});
