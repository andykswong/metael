// An instant, offline syntax highlighter for the editor: a CodeMirror `StreamLanguage` whose tokenizer is
// the language's OWN lexer (`lex` + `lexicalCategory`), so coloring appears on the very first frame —
// before the LSP worker has spawned and answered its first semantic-token request. The richer semantic
// tokens (which distinguish heads/builtins/parameters the lexer cannot) replace this once they arrive; this
// only has to be correct and immediate, sharing the lexer's single classification source of truth.
import { StreamLanguage, LanguageSupport, HighlightStyle, syntaxHighlighting } from '@codemirror/language';
import type { StringStream } from '@codemirror/language';
import { tags as t } from '@lezer/highlight';
import type { Extension } from '@codemirror/state';
import { lex, lexicalCategory } from '@metael/lang';
import type { TokenType } from '@metael/lang';

/** Map a lexed token to a CodeMirror highlight-tag name (the strings `defaultHighlightStyle` understands).
 *  Mirrors `highlight.ts`'s `kindOf` switch over `lexicalCategory`, so the cold-start palette matches the
 *  overlay used elsewhere: keyword→keyword, string/number literals split by token type, ident→variableName,
 *  operator→operator, punctuation→(bracket for the delimiters CM styles as such, else punctuation). */
function tagFor(type: TokenType): string | null {
  switch (lexicalCategory(type)) {
    case 'eof':
      return null;
    case 'keyword':
      return 'keyword';
    case 'literal':
      return type === 'string' ? 'string' : 'number';
    case 'ident':
      return 'variableName';
    case 'operator':
      return 'operator';
    case 'punctuation':
      return type === 'lbrace' || type === 'rbrace'
        || type === 'lbracket' || type === 'rbracket'
        || type === 'lparen' || type === 'rparen'
        ? 'bracket'
        : 'punctuation';
  }
}

// A line-scoped token cursor: each line is lexed once (the lexer is line-agnostic and cheap) and its tokens
// are replayed against the StringStream. Lexing per line keeps the StreamLanguage incremental — only edited
// lines re-tokenize — without threading multi-line lexer state (the grammar has no multi-line constructs
// beyond `//` line comments, which always run to end-of-line).
interface LineState {
  tokens: readonly { readonly start: number; readonly end: number; readonly type: TokenType }[];
  next: number; // index of the next token to emit on this line
  // The start offset (line-relative) of a `//` comment on this line, or -1 if none. A comment always runs
  // to end-of-line, so a single start is enough — no end is needed.
  commentStart: number;
}

/** The metael cold-start `StreamParser`: a line-scoped, lexer-driven tokenizer for the CodeMirror
 *  `StreamLanguage`. Exported for unit testing the per-line token/comment classification directly. */
export const metaelStreamParser = {
  name: 'metael',
  startState(): LineState {
    return { tokens: [], next: 0, commentStart: -1 };
  },
  token(stream: StringStream, state: LineState): string | null {
    // A fresh line: (re)lex it. StreamLanguage resets `stream.pos` to 0 at the start of each line, so this
    // fires exactly once per line and the token/comment offsets below are line-relative — matching `stream.pos`.
    if (stream.pos === 0) {
      const { tokens, comments } = lex(stream.string);
      state.tokens = tokens
        .filter((t) => t.type !== 'eof' && t.span.end > t.span.start)
        .map((t) => ({ start: t.span.start, end: t.span.end, type: t.type }));
      state.next = 0;
      // A `//` comment always runs to end-of-line, so at most one starts on any line: take the first.
      state.commentStart = comments.length > 0 ? comments[0]!.start : -1;
    }
    const tok = state.tokens[state.next];
    if (!tok) {
      // No more tokens on this line. A `//` comment (if any) always starts after every token, so any comment
      // on this line lies in this trailing region. Skip the plain gap up to it first, then colour it.
      if (state.commentStart >= 0 && stream.pos < state.commentStart) {
        stream.pos = state.commentStart;   // leading whitespace before the comment — plain
        return null;
      }
      if (state.commentStart >= 0) {        // stream.pos is now at the comment — it runs to end-of-line
        stream.skipToEnd();
        return 'comment';
      }
      // No comment: the trailing gap (whitespace) is plain.
      stream.skipToEnd();
      return null;
    }
    if (stream.pos < tok.start) {
      // A gap before the next token — leading/inter-token whitespace (a `//` comment always follows the last
      // token, so it is never reached here while a token remains): skip it plain.
      stream.pos = tok.start;
      return null;
    }
    // At the token: consume exactly its span and colour it.
    stream.pos = tok.end;
    state.next += 1;
    return tagFor(tok.type);
  },
  copyState(state: LineState): LineState {
    return { tokens: state.tokens, next: state.next, commentStart: state.commentStart };
  },
} as const;

// The colours the cold-start tags paint in. Matched to `lsp-extensions.ts`'s `.cmt-*` semantic-token
// palette so the frame does NOT visibly recolour when the worker's richer semantic tokens arrive and
// replace this layer — only the ident→head/builtin/parameter refinements differ, which are additive.
const coldStyle = HighlightStyle.define([
  { tag: t.keyword, color: '#e8a33d', fontWeight: '500' },   // matches .cmt-keyword
  { tag: t.string, color: '#9dd6a8' },                        // .cmt-string
  { tag: t.number, color: '#7fb7e6' },                        // .cmt-number
  { tag: t.variableName, color: '#e9e4d8' },                  // .cmt-variable
  { tag: t.operator, color: '#e08fb4' },                      // .cmt-operator
  { tag: t.punctuation, color: '#7b8398' },                   // .cmt-punctuation
  { tag: t.bracket, color: '#7b8398' },                       // brackets share the punctuation colour
  { tag: t.comment, color: '#5b6377', fontStyle: 'italic' },  // .cmt-comment
]);

/** The metael cold-start highlighter: instant lexer-driven colouring PLUS the `HighlightStyle` that actually
 *  paints its tags. Spread into the LSP extension set so the editor is never an unstyled textarea while the
 *  worker warms up; the semantic-token decoration layer then supersedes it (palette-matched, so seamless). */
export function coldHighlight(): Extension[] {
  return [
    new LanguageSupport(StreamLanguage.define<LineState>(metaelStreamParser)),
    syntaxHighlighting(coldStyle),
  ];
}
