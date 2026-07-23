import { lexicalCategory } from '@metael/lang';
import type { Token } from '@metael/lang';
import type { Profile } from '@metael/lang/profile';
import type { Document } from '../document.ts';
import type { ScopeModel, Binding } from '../scope-model.ts';
import type { SvcToken, SvcTokenKind } from '../results.ts';

// The token-type → colour-kind classification is derived from the language's own {@link lexicalCategory}
// so the analysis engine's colouring stays in lock-step with the lexer grammar and shares its single
// source of truth — no editor-layer copy of the keyword/operator/punctuation partition to drift.

/**
 * Compute the semantic tokens for a document — one classified span per lexed token, for syntax
 * colouring.
 *
 * @remarks
 * Every token from {@link Document.lex} is mapped to an {@link SvcTokenKind} by its type: keywords,
 * strings, and numbers directly; operators and punctuation by set membership; the terminal `eof` and
 * any type with no colour category are skipped. An `'ident'` token is refined against the scope and
 * profile in a fixed precedence that mirrors the evaluator (a local binding shadows a same-named head/
 * builtin):
 *
 * 1. the innermost {@link Binding} visible at the token's start (a `function`/`component` → `'function'`,
 *    a `param` → `'parameter'`, otherwise → `'variable'`);
 * 2. else a {@link Profile.builtins} name → `'builtin'`;
 * 3. else a {@link Profile.heads} name → `'head'`;
 * 4. else an unresolved identifier defaults to `'variable'`.
 *
 * Each `//` line comment — recorded separately from the token stream in {@link Document.lex} — is also
 * emitted as a `'comment'` token over its span.
 *
 * Pure and total: it reads the document, scope, and profile but mutates nothing and never throws.
 */
export function computeSemanticTokens(doc: Document, scope: ScopeModel, profile: Profile): readonly SvcToken[] {
  const out: SvcToken[] = [];
  for (const token of doc.lex.tokens) {
    const kind = kindOf(token, scope, profile);
    if (!kind) continue;
    out.push({ span: { start: token.span.start, end: token.span.end }, kind });
  }
  // `//` line comments are recorded separately by the lexer (they are not tokens), so colour them here.
  // Order is unconstrained — the marshaler sorts spans by position when encoding.
  for (const span of doc.lex.comments) {
    out.push({ span: { start: span.start, end: span.end }, kind: 'comment' });
  }
  return out;
}

/** The semantic kind for a token, or `undefined` when it carries no colour (e.g. `eof`). */
function kindOf(token: Token, scope: ScopeModel, profile: Profile): SvcTokenKind | undefined {
  const type = token.type;
  const cat = lexicalCategory(type);
  if (cat === 'eof') return undefined;
  if (cat === 'keyword') return 'keyword';
  // `SvcTokenKind` splits `literal` into the specific `'string'`/`'number'` kinds.
  if (cat === 'literal') return type === 'string' ? 'string' : 'number';
  if (cat === 'ident') return refineIdent(token, scope, profile);
  if (cat === 'operator') return 'operator';
  return 'punctuation';
}

/** Refine an identifier token by name: an innermost visible binding's role > profile builtin > profile
 *  head > `'variable'`. A local binding wins over a same-named head/builtin, matching the evaluator. */
function refineIdent(token: Token, scope: ScopeModel, profile: Profile): SvcTokenKind {
  const name = token.value;
  const binding = scope.innermostVisibleAt(token.span.start, name);
  if (binding) return bindingKind(binding.kind);

  if (profile.builtins.has(name)) return 'builtin';
  if (profile.heads.has(name)) return 'head';

  return 'variable';
}

/** Map a {@link Binding.kind} to the token kind shown for a reference to it. */
function bindingKind(kind: Binding['kind']): SvcTokenKind {
  switch (kind) {
    case 'function':
    case 'component':
      return 'function';
    case 'param':
      return 'parameter';
    case 'const':
    case 'let':
    case 'for':
    default:
      return 'variable';
  }
}
