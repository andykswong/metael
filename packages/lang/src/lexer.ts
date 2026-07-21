import type { Diagnostic, SourceSpan } from './diagnostics.ts';
import { makeDiagnostic } from './diagnostics.ts';

/**
 * The discriminant tag of a {@link Token}: the closed set of lexical categories the lexer emits —
 * the literal/word classes (`ident`/`number`/`string`), the reserved keywords, the
 * bracket/punctuation delimiters, the operators, and the terminal `eof`.
 *
 * @remarks A keyword tag (`component`/`if`/`return`/…) is produced only when a word exactly matches a
 * reserved word; every other identifier-shaped word lexes as `ident`.
 */
export type TokenType =
  | 'ident' | 'number' | 'string'
  | 'component' | 'function' | 'const' | 'let' | 'if' | 'else' | 'for' | 'of' | 'while' | 'return'
  | 'true' | 'false' | 'null'
  | 'lbrace' | 'rbrace' | 'lbracket' | 'rbracket' | 'lparen' | 'rparen'
  | 'dot' | 'comma' | 'colon' | 'semi' | 'arrow' | 'assign' | 'question' | 'ellipsis'
  | 'eq' | 'neq' | 'lt' | 'le' | 'gt' | 'ge' | 'plus' | 'minus' | 'star' | 'slash' | 'percent'
  | 'and' | 'or' | 'not' | 'eof';

/** One lexical token: its category, its source text, its source span, and whether a line break
 *  preceded it. */
export interface Token {
  /** The lexical category of the token — see {@link TokenType}. */
  readonly type: TokenType;
  /** The token's source text: the identifier/keyword word, the decoded string body (escape sequences
   *  resolved), the raw digit run for a number, or the operator/delimiter characters. Empty for `eof`. */
  readonly value: string;
  /** The half-open `[start, end)` character-offset range of the token within the source, used to
   *  anchor diagnostics. */
  readonly span: SourceSpan;
  /** Whether a newline occurred in the whitespace/comments immediately preceding this token. The
   *  parser uses it to make the brace-less single-trailing-statement wrap fire ONLY on the same
   *  logical line (the sibling-vs-nest guard). */
  readonly newlineBefore: boolean;
}
/** The outcome of {@link lex}: the emitted token stream plus any diagnostics collected while scanning. */
export interface LexResult {
  /** The tokens in source order, always terminated by a single `eof` token. */
  readonly tokens: Token[];
  /** Every diagnostic collected while scanning — a malformed number, an unterminated string, or an
   *  unexpected character, each an `ML-LANG-LEX`. Empty on a clean scan. */
  readonly diagnostics: Diagnostic[];
}

const KEYWORDS: Record<string, TokenType> = {
  component: 'component', function: 'function', const: 'const', let: 'let', if: 'if', else: 'else',
  for: 'for', of: 'of', while: 'while', return: 'return', true: 'true', false: 'false', null: 'null',
};

/**
 * Scan source text into a token stream for the parser.
 *
 * Total and non-throwing: a malformed number, an unterminated string, or an unexpected character each
 * become an `ML-LANG-LEX` diagnostic and scanning recovers, so the returned {@link LexResult} always
 * carries a complete token list (terminated by `eof`) alongside any diagnostics.
 *
 * @param src - the program source text.
 * @returns the tokens plus the diagnostics collected during scanning ({@link LexResult}).
 * @remarks Whitespace and `//` line comments are skipped but tracked: a skipped run containing a
 *          newline sets {@link Token.newlineBefore} on the following token. Numbers are `[0-9][0-9.]*`
 *          with at most one `.`; a second dot or an abutting identifier char makes the run a fail-loud
 *          malformed number rather than silently producing `NaN` or an orphan identifier. A word is
 *          classified as a keyword only on an exact reserved-word match (via an own-property lookup, so
 *          inherited names like `toString` stay plain identifiers).
 */
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
