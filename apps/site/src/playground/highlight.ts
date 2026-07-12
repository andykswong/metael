// Turn source into a list of segments that COVER THE WHOLE SOURCE (no dropped characters): each lex() token
// becomes a class-tagged segment, and the gaps between tokens (whitespace + // comments, which the lexer
// does not emit) become 'plain' segments reconstructed from span offsets. The editor overlay renders these
// as <span class="tok-*">. Highlighting the language with the language's own lexer.
import { lex } from '@metael/lang';
import type { Token } from '@metael/lang';

export type TokKind = 'keyword' | 'string' | 'number' | 'ident' | 'operator' | 'punct' | 'plain';

export interface Segment { readonly text: string; readonly kind: TokKind }

const KEYWORDS = new Set([
  'component', 'function', 'const', 'let', 'if', 'else', 'for', 'of', 'while', 'return', 'true', 'false', 'null',
]);
const OPERATORS = new Set([
  'arrow', 'assign', 'eq', 'neq', 'lt', 'le', 'gt', 'ge', 'plus', 'minus', 'star', 'slash', 'percent',
  'and', 'or', 'not', 'question', 'ellipsis',
]);
const PUNCT = new Set([
  'lbrace', 'rbrace', 'lbracket', 'rbracket', 'lparen', 'rparen', 'dot', 'comma', 'colon', 'semi',
]);

function kindOf(tok: Token): TokKind {
  if (tok.type === 'eof') return 'plain';
  if (KEYWORDS.has(tok.type)) return 'keyword';
  if (tok.type === 'string') return 'string';
  if (tok.type === 'number') return 'number';
  if (tok.type === 'ident') return 'ident';
  if (OPERATORS.has(tok.type)) return 'operator';
  if (PUNCT.has(tok.type)) return 'punct';
  return 'plain';
}

/** Segment the full source. Concatenating segment.text reproduces `src` exactly. */
export function tokensToSegments(src: string): Segment[] {
  const { tokens } = lex(src);
  const segments: Segment[] = [];
  let cursor = 0;
  for (const tok of tokens) {
    if (tok.type === 'eof') break;
    const { start, end } = tok.span;
    if (start > cursor) segments.push({ text: src.slice(cursor, start), kind: 'plain' });
    if (end > start) segments.push({ text: src.slice(start, end), kind: kindOf(tok) });
    cursor = end;
  }
  if (cursor < src.length) segments.push({ text: src.slice(cursor), kind: 'plain' });
  return segments;
}
