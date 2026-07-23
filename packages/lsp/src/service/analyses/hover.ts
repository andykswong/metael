import type { Token } from '@metael/lang';
import type { BuiltinSpec, HeadParam, HeadSpec, Profile, TypeDescriptorMeta } from '@metael/lang/profile';
import type { Document } from '../document.ts';
import type { ScopeModel, Binding } from '../scope-model.ts';
import type { SvcHover } from '../results.ts';

/**
 * Compute the hover card for the identifier at a source `offset`, or `null` when the offset does not
 * land on a resolvable identifier.
 *
 * @remarks
 * Finds the token whose span contains `offset` (see {@link tokenAt}). Only `'ident'` tokens hover — a
 * keyword, number, string, punctuation, or a position in whitespace yields `null`. The identifier's
 * name is resolved against the scope and profile in a fixed precedence that mirrors the evaluator (which
 * resolves a local binding before an injected head/builtin, so a user binding shadows a same-named head):
 *
 * 1. the innermost {@link Binding} visible at `offset` matching the name → a binding card (its declaration
 *    kind + name, plus the parameter list for a `function`/`component` signature).
 * 2. else {@link Profile.builtins} → a builtin card (signature + a portability-prefixed description + a
 *    per-arg list + what it returns).
 * 3. else {@link Profile.heads} → a head card (signature + description + a per-arg list + what it returns).
 * 4. else {@link Profile.types} → a type card (name + a member summary + doc).
 *
 * The first match wins; if nothing matches the result is `null`. Pure and total: it reads the document,
 * scope, and profile but mutates nothing and never throws.
 */
export function computeHover(doc: Document, scope: ScopeModel, profile: Profile, offset: number): SvcHover | null {
  const token = tokenAt(doc.lex.tokens, offset);
  if (!token || token.type !== 'ident') return null;

  const name = token.value;
  const span = { start: token.span.start, end: token.span.end };

  // A visible local binding wins over a profile head/builtin/type, matching the evaluator's precedence
  // (an inner shadow — the greatest `scopeStart` — beats an outer binding of the same name).
  const binding = scope.innermostVisibleAt(offset, name);
  if (binding) return { span, markdown: bindingCard(binding) };

  const builtin = profile.builtins.get(name);
  if (builtin) return { span, markdown: builtinCard(builtin) };

  const head = profile.heads.get(name);
  if (head) return { span, markdown: headCard(head) };

  const type = profile.types.get(name);
  if (type) return { span, markdown: typeCard(type) };

  return null;
}

/**
 * Find the token whose span contains `offset`, or `undefined` if none does.
 *
 * @remarks
 * Spans are half-open `[start, end)`, so the primary match is `span.start <= offset < span.end` — this
 * catches a cursor anywhere inside a token, including mid-token. As a fallback (e.g. the cursor sitting
 * exactly at the end of the last real token), a token ending at `offset` is accepted, preferring an
 * `'ident'` so hovering the trailing edge of an identifier still resolves it. The terminal `eof` token
 * is never returned.
 */
export function tokenAt(tokens: readonly Token[], offset: number): Token | undefined {
  let endMatch: Token | undefined;
  for (const t of tokens) {
    if (t.type === 'eof') continue;
    if (offset >= t.span.start && offset < t.span.end) return t;
    if (offset === t.span.end && (!endMatch || t.type === 'ident')) endMatch = t;
  }
  return endMatch;
}

/** The `  name — doc` lines for the params that carry a doc, in call order; `[]` when none do (so the
 *  caller renders no list rather than a run of bare `  name —` lines). Shared by both card renderers. */
function paramDocLines(params: readonly HeadParam[]): string[] {
  return params.filter((p) => p.doc).map((p) => `  ${p.name} — ${p.doc!}`);
}

/** Render a builtin's card: a param-named signature (when the spec declares its params), a description
 *  (prefixed with a compact portability marker unless the builtin is `'exact'`), a per-arg list of the
 *  documented params, what it returns (when the spec says), and — as a fallback when no param doc already
 *  covers it — a note that it takes a closure. */
function builtinCard(spec: BuiltinSpec): string {
  // A param-named signature (`blend(a, b, weight)`) when the spec declares its params — including the empty
  // list for a nullary builtin (`now()`); the `(…)` placeholder is only for a spec that declares none at all.
  const sig = spec.params ? `${spec.name}(${spec.params.map((p) => p.name).join(', ')})` : `${spec.name}(…)`;
  const lines = ['```metael', sig, '```'];
  // A portability prefix only when the result does NOT reproduce bit-for-bit; `'exact'` carries none.
  if (spec.doc) {
    const prefix = spec.portability === 'exact' ? '' : `(${spec.portability}) `;
    lines.push(`${prefix}${spec.doc}`);
  }
  const argLines = spec.params ? paramDocLines(spec.params) : [];
  lines.push(...argLines);
  if (spec.returnDoc) lines.push(`Returns ${spec.returnDoc}.`);
  // Keep the generic closure note only when no param doc already documents the closure argument.
  if (spec.takesClosure && argLines.length === 0) lines.push('Takes a closure argument.');
  return lines.join('\n');
}

/** Render a head's card: its param-named signature, a description, a per-arg list of the documented
 *  params, and what it returns (when the spec says). */
function headCard(spec: HeadSpec): string {
  const params = spec.params.map((p) => p.name).join(', ');
  const lines = ['```metael', `${spec.name}(${params})`, '```'];
  if (spec.doc) lines.push(spec.doc);
  lines.push(...paramDocLines(spec.params));
  if (spec.returnDoc) lines.push(`Returns ${spec.returnDoc}.`);
  return lines.join('\n');
}

/** Render a custom-value type's card: its name, a member summary, and any doc. */
function typeCard(spec: TypeDescriptorMeta): string {
  const members = spec.members.map((m) => m.name).join(', ');
  const lines = ['```metael', `type ${spec.name}`, '```'];
  if (members) lines.push(`Members: ${members}`);
  if (spec.doc) lines.push(spec.doc);
  return lines.join('\n');
}

/** Render a local binding's card: the declaration kind + name. A `function`/`component` binding also
 *  renders its parameter list as a signature (`component KPI(label, value)`, `function fib(n)`, or
 *  `component App()` when it takes none); a `const`/`let`/`param`/`for` binding has no param list and
 *  renders bare (`const xs`, `param a`). */
function bindingCard(binding: Binding): string {
  const sig = binding.params ? `${binding.kind} ${binding.name}(${binding.params.join(', ')})` : `${binding.kind} ${binding.name}`;
  return ['```metael', sig, '```'].join('\n');
}
