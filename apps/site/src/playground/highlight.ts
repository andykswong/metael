// Turn source into a list of segments that COVER THE WHOLE SOURCE (no dropped characters): each lex() token
// becomes a class-tagged segment, and the gaps between tokens (whitespace + // comments, which the lexer
// does not emit) become 'plain' segments reconstructed from span offsets. The editor overlay renders these
// as <span class="tok-*">. Highlighting the language with the language's own lexer.
import { lex, lexicalCategory } from '@metael/lang';
import type { Token } from '@metael/lang';

export type TokKind = 'keyword' | 'string' | 'number' | 'ident' | 'operator' | 'punct' | 'plain';

export interface Segment { readonly text: string; readonly kind: TokKind }

// The token → highlight-class mapping is derived from the language's own `lexicalCategory` so this
// cold-start highlighter shares the lexer's single source of truth — no local copy of the
// keyword/operator/punctuation partition to drift when the grammar changes.
function kindOf(tok: Token): TokKind {
  const cat = lexicalCategory(tok.type);
  switch (cat) {
    case 'eof': return 'plain';
    case 'keyword': return 'keyword';
    case 'literal': return tok.type === 'string' ? 'string' : 'number';
    case 'ident': return 'ident';
    case 'operator': return 'operator';
    case 'punctuation': return 'punct';
  }
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
