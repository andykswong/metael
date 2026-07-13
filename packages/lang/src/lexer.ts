import type { Diagnostic, SourceSpan } from './diagnostics.ts';
import { makeDiagnostic } from './diagnostics.ts';

export type TokenType =
  | 'ident' | 'number' | 'string'
  | 'component' | 'function' | 'const' | 'let' | 'if' | 'else' | 'for' | 'of' | 'while' | 'return'
  | 'true' | 'false' | 'null'
  | 'lbrace' | 'rbrace' | 'lbracket' | 'rbracket' | 'lparen' | 'rparen'
  | 'dot' | 'comma' | 'colon' | 'semi' | 'arrow' | 'assign' | 'question' | 'ellipsis'
  | 'eq' | 'neq' | 'lt' | 'le' | 'gt' | 'ge' | 'plus' | 'minus' | 'star' | 'slash' | 'percent'
  | 'and' | 'or' | 'not' | 'eof';

// `newlineBefore` = a newline occurred in the whitespace/comments preceding this token. The
// parser uses it to make the brace-less single-trailing-statement wrap fire ONLY on the same
// logical line (the sibling-vs-nest guard).
export interface Token { readonly type: TokenType; readonly value: string; readonly span: SourceSpan; readonly newlineBefore: boolean }
export interface LexResult { readonly tokens: Token[]; readonly diagnostics: Diagnostic[] }

const KEYWORDS: Record<string, TokenType> = {
  component: 'component', function: 'function', const: 'const', let: 'let', if: 'if', else: 'else',
  for: 'for', of: 'of', while: 'while', return: 'return', true: 'true', false: 'false', null: 'null',
};

export function lex(src: string): LexResult {
  const tokens: Token[] = [];
  const diagnostics: Diagnostic[] = [];
  let i = 0;
  let pendingNewline = false;   // set when skipped whitespace/comment contains a '\n'
  const push = (type: TokenType, value: string, start: number) => {
    tokens.push({ type, value, span: { start, end: i }, newlineBefore: pendingNewline });
    pendingNewline = false;
  };
  const isIdStart = (c: string) => /[A-Za-z_]/.test(c);
  const isId = (c: string) => /[A-Za-z0-9_]/.test(c);

  while (i < src.length) {
    const c = src[i]!;
    if (/\s/.test(c)) { if (c === '\n') pendingNewline = true; i++; continue; }
    // line comments (a comment always ends at a newline → also a boundary)
    if (c === '/' && src[i + 1] === '/') { while (i < src.length && src[i] !== '\n') i++; pendingNewline = true; continue; }
    const start = i;
    if (isIdStart(c)) {
      while (i < src.length && isId(src[i]!)) i++;
      const word = src.slice(start, i);
      // `Object.hasOwn`, not `KEYWORDS[word] ?? 'ident'`: a plain-object lookup resolves inherited
      // members (`toString`/`valueOf`/`constructor`/…) to native functions via the prototype chain,
      // so those very common identifier names would otherwise lex to a bogus function-valued type.
      push(Object.hasOwn(KEYWORDS, word) ? KEYWORDS[word]! : 'ident', word, start);
      continue;
    }
    if (/[0-9]/.test(c)) {
      // Numbers are `[0-9][0-9.]*` with AT MOST one `.`; a second dot or an identifier-start char
      // immediately abutting the digits is malformed (`1.2.3`, `0xFF`, `1e3`, `1_000`, `10n`). Emit a
      // fail-loud ML-LANG-LEX rather than silently producing NaN (multi-dot) or splitting into a bare
      // number + an orphan identifier (adjacency). We consume the whole run so recovery resumes cleanly.
      let dots = 0;
      while (i < src.length && /[0-9.]/.test(src[i]!)) { if (src[i] === '.') dots++; i++; }
      const adjacentId = i < src.length && isIdStart(src[i]!);
      while (i < src.length && isId(src[i]!)) i++;   // absorb the abutting identifier chars into the bad token
      if (dots > 1 || adjacentId) {
        diagnostics.push(makeDiagnostic('ML-LANG-LEX', `malformed number '${src.slice(start, i)}'`, { start, end: i }));
      }
      push('number', src.slice(start, i), start);
      continue;
    }
    if (c === '"' || c === "'") {
      const quote = c; i++;
      let str = '';
      while (i < src.length && src[i] !== quote) {
        if (src[i] === '\\') { i++; const e = src[i]; str += e === 'n' ? '\n' : e === 't' ? '\t' : e ?? ''; i++; }
        else str += src[i++];
      }
      if (i >= src.length) { diagnostics.push(makeDiagnostic('ML-LANG-LEX', 'unterminated string', { start, end: i })); break; }
      i++; // closing quote
      push('string', str, start);
      continue;
    }
    // three-char operator first (only `...`), before two/one-char so `...` never lexes as three dots.
    if (src.slice(i, i + 3) === '...') { i += 3; push('ellipsis', '...', start); continue; }
    // two-char operators first
    const two = src.slice(i, i + 2);
    const twoMap: Record<string, TokenType> = { '=>': 'arrow', '==': 'eq', '!=': 'neq', '<=': 'le', '>=': 'ge', '&&': 'and', '||': 'or' };
    if (twoMap[two]) { i += 2; push(twoMap[two]!, two, start); continue; }
    const oneMap: Record<string, TokenType> = {
      '{': 'lbrace', '}': 'rbrace', '[': 'lbracket', ']': 'rbracket', '(': 'lparen', ')': 'rparen',
      '.': 'dot', ',': 'comma', ':': 'colon', ';': 'semi', '=': 'assign', '?': 'question',
      '<': 'lt', '>': 'gt', '+': 'plus', '-': 'minus', '*': 'star', '/': 'slash', '%': 'percent', '!': 'not',
    };
    if (oneMap[c]) { i++; push(oneMap[c]!, c, start); continue; }
    diagnostics.push(makeDiagnostic('ML-LANG-LEX', `unexpected character '${c}'`, { start, end: i + 1 }));
    i++;
  }
  push('eof', '', i);
  return { tokens, diagnostics };
}
