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
