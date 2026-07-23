import type { Token } from '@metael/lang';
import type { BuiltinSpec, HeadParam, HeadSpec, Profile } from '@metael/lang/profile';
import type { Document } from '../document.ts';
import type { SvcParam, SvcSignature } from '../results.ts';

/** Upper bound on synthesized parameter names for an unbounded-arity builtin, so an `arity` of
 *  `[min, Infinity]` yields a finite, readable signature rather than an unbounded label. */
const MAX_SYNTH_PARAMS = 8;

/**
 * Compute signature help for the call enclosing a source `offset`, or `null` when the offset is not
 * inside a call whose callee resolves to a known head or builtin.
 *
 * @remarks
 * Signature help fires while an argument list is being typed — usually still unclosed (`foo(a, |`) —
 * so containment is derived from the TOKEN stream, not the AST: an incomplete call's node span ends at
 * the last typed argument and never reaches the cursor, and the parser stores no close-paren position.
 * Instead this scans {@link Document.lex}'s tokens for the innermost `(` still unmatched at `offset`
 * (the last-opened `lparen` before `offset` whose matching `rparen` is at or after `offset`, or absent),
 * and takes the identifier token immediately before it as the callee. When the token before that `(` is
 * not an identifier — a grouping paren `(a + b)` or a member/computed call target — there is no
 * resolvable callee and the result is `null`. The callee name is resolved in a fixed precedence:
 *
 * 1. {@link Profile.heads} → the signature parameters are the head's declared {@link HeadSpec.params}
 *    (name + doc).
 * 2. {@link Profile.builtins} → a {@link BuiltinSpec} carries no named params, so parameters are
 *    synthesized from its arity as `arg0, arg1, …` (one per slot up to the max, or the min when the
 *    max is unbounded, capped for readability).
 *
 * An unknown callee (neither head nor builtin) yields `null` rather than a misleading empty signature.
 * `activeParam` is the number of top-level commas between the open paren and `offset` (commas nested in
 * a deeper `(`/`[`/`{` do not count), so it is `0` right after the open paren and advances by one past
 * each argument separator of THIS call. Pure and total: it reads the document and profile but mutates
 * nothing and never throws.
 */
export function computeSignature(doc: Document, profile: Profile, offset: number): SvcSignature | null {
  const tokens = doc.lex.tokens;
  const openIdx = innermostOpenParen(tokens, offset);
  if (openIdx < 0) return null;

  // The callee is the identifier token immediately before the open paren; anything else is not a call.
  const calleeTok = tokens[openIdx - 1];
  if (!calleeTok || calleeTok.type !== 'ident' || !calleeTok.value) return null;
  const name = calleeTok.value;

  const params = resolveParams(name, profile);
  if (!params) return null;

  const label = `${name}(${params.map((p) => p.label).join(', ')})`;
  const activeParam = topLevelCommas(tokens, openIdx, offset);
  return { label, params, activeParam };
}

/** The parameter list for a call head: a head's declared params, a builtin's declared params when it
 *  carries them, else synthesized `argN` names from a builtin's arity. `null` when the name is neither a
 *  head nor a builtin. */
function resolveParams(name: string, profile: Profile): readonly SvcParam[] | null {
  const head: HeadSpec | undefined = profile.heads.get(name);
  if (head) return head.params.map(toParam);

  const builtin: BuiltinSpec | undefined = profile.builtins.get(name);
  if (builtin) return builtin.params?.length ? builtin.params.map(toParam) : synthParams(builtin.arity);

  return null;
}

/** Project a declared head/builtin parameter to a signature-help param, carrying its doc when present. */
function toParam(p: HeadParam): SvcParam {
  return p.doc !== undefined ? { label: p.name, doc: p.doc } : { label: p.name };
}

/** Synthesize `arg0, arg1, …` parameter labels from a builtin's `[min, max]` arity: one per slot up to
 *  the finite max, or the min when the max is unbounded, capped for readability. */
function synthParams(arity: readonly [number, number]): readonly SvcParam[] {
  const [min, max] = arity;
  const desired = Number.isFinite(max) ? max : min;
  const count = Math.max(0, Math.min(desired, MAX_SYNTH_PARAMS));
  const out: SvcParam[] = [];
  for (let i = 0; i < count; i++) out.push({ label: `arg${i}` });
  return out;
}

/**
 * Find the index (in `tokens`) of the innermost `(` that opens before `offset` and is still unmatched at
 * `offset`, or `-1` when the offset sits inside no open parenthesis.
 *
 * @remarks
 * Walks the tokens with a `(` depth counter, pushing the index of each `lparen` that opens at or before
 * `offset` and popping it on the matching `rparen`. A `rparen` at or after `offset` does NOT close its
 * `(` for this purpose, so the still-open paren the cursor sits within is retained. The last such
 * surviving open paren (the deepest, since parens nest) is the active call's open paren. Only `(`/`)`
 * participate — braces and brackets are ignored, since they do not delimit a call's argument list.
 */
function innermostOpenParen(tokens: readonly Token[], offset: number): number {
  const open: number[] = [];
  for (let i = 0; i < tokens.length; i++) {
    const t = tokens[i]!;
    // Tokens are ordered, so once one starts at/after the cursor no later token can affect the scan.
    if (t.type === 'eof' || t.span.start >= offset) break;
    if (t.type === 'lparen') {
      // Only parens that fully precede the cursor open an enclosing arg list (offset >= the paren's end).
      if (t.span.end <= offset) open.push(i);
    } else if (t.type === 'rparen') {
      // A close paren before the cursor matches (closes) the nearest open paren; one at/after the
      // cursor leaves the enclosing paren open so the cursor is still inside it.
      if (open.length > 0) open.pop();
    }
  }
  return open.length > 0 ? open[open.length - 1]! : -1;
}

/**
 * Count the top-level argument-separating commas of the call opened at `openIdx`, from just after the
 * open paren up to `offset`.
 *
 * @remarks
 * Walks tokens forward from `openIdx + 1`, tracking nesting depth across all three bracket kinds
 * (`(`/`[`/`{` raise it, `)`/`]`/`}` lower it) and counting a `comma` ONLY at depth `0` — the top level
 * of THIS call's argument list — so a comma inside a nested call, array, or object does not advance the
 * active parameter. Stops at the first token starting at or after `offset`.
 */
function topLevelCommas(tokens: readonly Token[], openIdx: number, offset: number): number {
  let depth = 0;
  let commas = 0;
  for (let i = openIdx + 1; i < tokens.length; i++) {
    const t = tokens[i]!;
    if (t.type === 'eof' || t.span.start >= offset) break;
    switch (t.type) {
      case 'lparen':
      case 'lbracket':
      case 'lbrace':
        depth++;
        break;
      case 'rparen':
      case 'rbracket':
      case 'rbrace':
        if (depth > 0) depth--;
        break;
      case 'comma':
        if (depth === 0) commas++;
        break;
      default:
        break;
    }
  }
  return commas;
}
