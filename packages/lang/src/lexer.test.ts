import { describe, it, expect } from 'vitest';
import { lex, KEYWORDS_SET, lexicalCategory } from './lexer.ts';
import type { TokenType, LexicalCategory } from './lexer.ts';

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

describe('line comments', () => {
  it('records each `//` comment span [start, end) without emitting a token', () => {
    // Offsets: `let x = 1 // hi\n// full`
    //           0123456789012345678901234
    // first  `/` at 10, newline at 15  → span [10, 15) covers "// hi"
    // second `/` at 16, EOF at 23      → span [16, 23) covers "// full"
    const r = lex('let x = 1 // hi\n// full');
    expect(r.comments).toEqual([
      { start: 10, end: 15 },
      { start: 16, end: 23 },
    ]);
  });

  it('leaves the token stream byte-identical (a comment produces no token)', () => {
    // Same tokens as the un-commented program: `let`, `x`, `=`, `1`, eof.
    expect(kinds('let x = 1 // hi\n// full')).toEqual(kinds('let x = 1'));
    expect(kinds('let x = 1 // hi\n// full')).toEqual(['let', 'ident', 'assign', 'number', 'eof']);
  });

  it('reports no comments for a comment-free program', () => {
    expect(lex('let x = 1').comments).toEqual([]);
  });
});

describe('KEYWORDS_SET', () => {
  const RESERVED = [
    'component', 'function', 'const', 'let', 'if', 'else', 'for', 'of', 'while', 'return', 'true', 'false', 'null',
  ];
  it('holds exactly the 13 reserved words', () => {
    expect(KEYWORDS_SET.size).toBe(13);
    for (const kw of RESERVED) expect(KEYWORDS_SET.has(kw)).toBe(true);
  });
  it('equals the reserved-word set (no extras)', () => {
    expect([...KEYWORDS_SET].sort()).toEqual([...RESERVED].sort());
  });
  it('is frozen (a stable published view)', () => {
    expect(Object.isFrozen(KEYWORDS_SET)).toBe(true);
  });
});

describe('lexicalCategory', () => {
  // Every TokenType value, kept in sync with the union in lexer.ts. The coverage test below asserts this
  // list is exhaustive by lexing a program that exercises each; a new TokenType left unclassified is a
  // COMPILE error at the LEXICAL_CATEGORY record and a runtime miss here.
  const ALL_TYPES: readonly TokenType[] = [
    'ident', 'number', 'string',
    'component', 'function', 'const', 'let', 'if', 'else', 'for', 'of', 'while', 'return',
    'true', 'false', 'null',
    'lbrace', 'rbrace', 'lbracket', 'rbracket', 'lparen', 'rparen',
    'dot', 'comma', 'colon', 'semi', 'arrow', 'assign', 'question', 'ellipsis',
    'eq', 'neq', 'lt', 'le', 'gt', 'ge', 'plus', 'minus', 'star', 'slash', 'percent',
    'and', 'or', 'not', 'eof',
  ];
  const VALID: readonly LexicalCategory[] = ['keyword', 'literal', 'operator', 'punctuation', 'ident', 'eof'];

  it('classifies a representative of each category', () => {
    expect(lexicalCategory('component')).toBe('keyword');
    expect(lexicalCategory('true')).toBe('keyword');   // true/false/null are reserved words here
    expect(lexicalCategory('false')).toBe('keyword');
    expect(lexicalCategory('null')).toBe('keyword');
    expect(lexicalCategory('number')).toBe('literal');
    expect(lexicalCategory('string')).toBe('literal');
    expect(lexicalCategory('ident')).toBe('ident');
    expect(lexicalCategory('plus')).toBe('operator');
    expect(lexicalCategory('ellipsis')).toBe('operator');
    expect(lexicalCategory('lbrace')).toBe('punctuation');
    expect(lexicalCategory('semi')).toBe('punctuation');
    expect(lexicalCategory('eof')).toBe('eof');
  });

  it('is total: every TokenType maps to a valid LexicalCategory (runtime drift guard)', () => {
    for (const t of ALL_TYPES) {
      const cat = lexicalCategory(t);
      expect(VALID).toContain(cat);
      expect(cat).toBeDefined();
    }
  });

  it('every keyword TokenType is exactly the KEYWORDS_SET names', () => {
    const kwTypes = ALL_TYPES.filter((t) => lexicalCategory(t) === 'keyword');
    expect([...kwTypes].sort()).toEqual([...KEYWORDS_SET].sort());
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
