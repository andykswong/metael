import { didYouMean, KEYWORDS_SET } from '@metael/lang';
import type { Stmt, Expr } from '@metael/lang';
import { coreIntrinsicsProfile } from '@metael/lang/profile';
import type { Profile, TypeDescriptorMeta } from '@metael/lang/profile';
import type { Document } from '../document.ts';
import type { ScopeModel, Binding } from '../scope-model.ts';
import type { SvcCompletion, SvcCompletionKind } from '../results.ts';

/** Recursion ceiling for the declaration-init search, matching the parser/scope walk bound so a
 *  pathological or cyclic (partial) tree fails closed rather than overflowing the JS stack. */
const MAX_WALK_DEPTH = 512;

/**
 * Compute the completion candidates offered at a source `offset`.
 *
 * @remarks
 * Two contexts, selected by the text immediately before the cursor:
 *
 * - **Member context** — when the cursor follows `<ident> . <partial>`, the receiver identifier's
 *   custom-value TYPE is resolved (its declaration's constructor call name → the {@link Profile.types}
 *   entry whose `constructors` include that name) and that type's members are offered as `'member'`
 *   completions. Every resolution step is guarded; any miss yields no member completions (never a throw).
 * - **General context** — otherwise the visible {@link ScopeModel} bindings, the profile's builtins and
 *   heads, the language keywords, and the intrinsic `range` are merged and de-duplicated by label. In a
 *   child/wrap position (inside a block body) heads are ranked ahead of builtins; otherwise keywords are.
 *   If the cursor sits at the end of a partial word that is not itself a candidate, the closest known
 *   label (bounded Levenshtein) is nudged to the front so a near-miss typo surfaces first — a stable
 *   re-order, never a filter (the client keeps every candidate to fuzzy-match against).
 *
 * Pure and total: it reads the document, scope, and profile but mutates nothing and never throws.
 */
export function computeCompletion(doc: Document, scope: ScopeModel, profile: Profile, offset: number): readonly SvcCompletion[] {
  const text = doc.text;

  // 1. Member context: `<ident> . <partial>` immediately before the cursor.
  const dot = memberDotBefore(text, offset);
  if (dot) {
    const type = resolveReceiverType(dot.receiver, scope, profile, doc, dot.at);
    if (type) return type.members.map((m) => ({ label: m.name, kind: 'member' as const, doc: m.doc }));
    return [];
  }

  // 2. General context: visible bindings ∪ profile builtins ∪ heads (+ range), de-duplicated by label.
  const bindings: SvcCompletion[] = [];
  for (const b of scope.visibleAt(offset)) bindings.push({ label: b.name, kind: bindingKind(b.kind) });

  const builtins: SvcCompletion[] = [];
  for (const [name, spec] of profile.builtins) {
    // Surface the one-line doc beside the label (`detail`) and as the expanded panel (`doc`) — the same
    // shape heads use (`detail: head.doc`). Fall back to the profile tag only when a spec carries no doc.
    builtins.push({ label: name, kind: 'builtin', detail: spec.doc ?? `${spec.profile} builtin`, doc: spec.doc });
  }

  const heads: SvcCompletion[] = [];
  for (const [name, head] of profile.heads) heads.push({ label: name, kind: 'head', detail: head.doc });

  // The language's reserved words (the lexer's own keyword set), offered as `'keyword'` candidates in
  // the general (non-member) completion context — most relevant at statement start.
  const keywords: SvcCompletion[] = [...KEYWORDS_SET].map((kw) => ({ label: kw, kind: 'keyword' as const }));

  // In a child/wrap position (inside a block body) heads are the likely intent, so rank them first;
  // otherwise (statement-ish position) rank keywords ahead of builtins.
  const headsFirst = inChildPosition(text, offset);
  const items: SvcCompletion[] = headsFirst
    ? [...bindings, ...heads, ...builtins, ...keywords]
    : [...bindings, ...keywords, ...builtins, ...heads];
  // The `range` intrinsic is dispatched by the language kernel (not contributed by a profile module), so
  // offer it explicitly — using its real spec doc so its detail matches every other builtin.
  const rangeDoc = coreIntrinsicsProfile.builtins.get('range')?.doc;
  items.push({ label: 'range', kind: 'builtin', detail: rangeDoc, doc: rangeDoc });
  const deduped = dedupeByLabel(items);

  // Near-miss ranking: if the user is mid-typing a partial word that is not itself an exact candidate,
  // surface the closest known label (Levenshtein) at the front. A stable re-order — never a filter, so
  // the client keeps every candidate to fuzzy-match against; we only nudge the likely-intended typo up.
  return promoteNearMiss(deduped, partialWordBefore(text, offset));
}

/**
 * Move the single closest near-miss of `partial` to the front of `items`, or return `items` unchanged.
 *
 * @remarks
 * No-op when `partial` is empty or already an exact label in the set. Otherwise the nearest known label
 * (by {@link didYouMean}'s bounded Levenshtein) is moved to index 0, preserving the relative order of the
 * rest. Non-destructive: no candidate is dropped — this is a ranking nudge, not a filter.
 */
function promoteNearMiss(items: readonly SvcCompletion[], partial: string): readonly SvcCompletion[] {
  if (!partial) return items;
  const labels = new Set(items.map((c) => c.label));
  if (labels.has(partial)) return items; // an exact match needs no near-miss nudge
  const best = didYouMean(partial, labels);
  if (best === undefined) return items;
  const idx = items.findIndex((c) => c.label === best);
  if (idx <= 0) return items;
  const out = [...items];
  const [hit] = out.splice(idx, 1);
  out.unshift(hit!);
  return out;
}

/**
 * Read the partial identifier word ending at `offset` (the run of identifier chars immediately before the
 * cursor), or `''` when none. Requires a non-digit first char so a bare number is not treated as a word.
 */
function partialWordBefore(text: string, offset: number): string {
  let i = Math.max(0, Math.min(offset, text.length));
  while (i > 0 && isWordChar(text[i - 1]!)) i--;
  const word = text.slice(i, Math.max(0, Math.min(offset, text.length)));
  return word && isIdentStart(word[0]!) ? word : '';
}

/** A detected member-access context: the receiver identifier and the offset of its `.` separator. */
interface MemberDot {
  /** The identifier to the left of the `.` whose type supplies the member candidates. */
  readonly receiver: string;
  /** The offset of the `.` separator (used to resolve the receiver's binding at that position). */
  readonly at: number;
}

/**
 * Detect a `<ident> . <partial>` member-access context ending at `offset`.
 *
 * @remarks
 * Scans left from `offset` over an optional partial member word, then requires a `.`, then reads the
 * receiver identifier immediately before it. Returns `undefined` when the shape does not match (so the
 * caller falls back to general completion).
 */
function memberDotBefore(text: string, offset: number): MemberDot | undefined {
  let i = Math.max(0, Math.min(offset, text.length));
  // Skip the partial member word already typed after the dot (may be empty).
  while (i > 0 && isWordChar(text[i - 1]!)) i--;
  // The char immediately before the partial must be the `.` separator.
  if (i === 0 || text[i - 1] !== '.') return undefined;
  const dotPos = i - 1;
  // Read the receiver identifier immediately before the dot.
  let j = dotPos;
  while (j > 0 && isWordChar(text[j - 1]!)) j--;
  const receiver = text.slice(j, dotPos);
  if (!receiver || !isIdentStart(receiver[0]!)) return undefined;
  return { receiver, at: dotPos };
}

/**
 * Resolve the custom-value type of `receiver` at `dotPos`, or `undefined` on any miss.
 *
 * @remarks
 * Confirms `receiver` names a `const`/`let` binding visible at `dotPos`, reads that declaration's
 * initializer from the AST, takes its constructor call head (a `call` whose callee is an `ident`, e.g.
 * `vec3(...)` → `vec3`), and returns the {@link TypeDescriptorMeta} in {@link Profile.types} whose
 * `constructors` include that head. Every step is guarded — it never throws.
 */
function resolveReceiverType(
  receiver: string,
  scope: ScopeModel,
  profile: Profile,
  doc: Document,
  dotPos: number,
): TypeDescriptorMeta | undefined {
  // The receiver must be a value binding (const/let) visible at the dot.
  const binding = scope
    .visibleAt(dotPos)
    .find((b: Binding) => b.name === receiver && (b.kind === 'const' || b.kind === 'let'));
  if (!binding) return undefined;

  // Re-walk the AST to recover the declaration's initializer (Binding does not carry it).
  const init = findDeclInit(doc.parse?.program?.stmts ?? [], receiver);
  if (!init || init.kind !== 'call') return undefined;
  const callee = init.callee;
  if (!callee || callee.kind !== 'ident') return undefined;
  const ctor = callee.name;
  if (!ctor) return undefined;

  for (const type of profile.types.values()) {
    if (type.constructors?.includes(ctor)) return type;
  }
  return undefined;
}

/**
 * Find the initializer expression of the `const`/`let` declaration named `name`, recursing into block
 * bodies, or `undefined` if none is found. Bounded against cyclic/partial trees by a fixed depth.
 */
function findDeclInit(stmts: readonly Stmt[], name: string): Expr | undefined {
  const scan = (list: readonly Stmt[] | undefined, depth: number): Expr | undefined => {
    if (depth > MAX_WALK_DEPTH || !Array.isArray(list)) return undefined;
    for (const s of list) {
      if (!s || typeof s.kind !== 'string') continue;
      if ((s.kind === 'const' || s.kind === 'let') && s.name === name) return s.init;
      const child = childBlocks(s);
      for (const block of child) {
        const hit = scan(block, depth + 1);
        if (hit) return hit;
      }
    }
    return undefined;
  };
  return scan(stmts, 0);
}

/** The child statement blocks a statement opens (component/function/if/for/while bodies), for recursion. */
function childBlocks(s: Stmt): readonly (Stmt[] | undefined)[] {
  switch (s.kind) {
    case 'function':
    case 'component':
    case 'for':
    case 'while':
      return [s.body];
    case 'if':
      return [s.then, s.else];
    default:
      return [];
  }
}

/** Map a {@link Binding.kind} to the completion category shown for it. */
function bindingKind(kind: Binding['kind']): SvcCompletionKind {
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

/** Keep the first candidate for each distinct label, preserving order. */
function dedupeByLabel(items: readonly SvcCompletion[]): readonly SvcCompletion[] {
  const seen = new Set<string>();
  const out: SvcCompletion[] = [];
  for (const c of items) {
    if (seen.has(c.label)) continue;
    seen.add(c.label);
    out.push(c);
  }
  return out;
}

/** True when `offset` sits inside a block body (more `{` than `}` precede it) — a rough child/wrap
 *  position where head candidates are the likely intent. */
function inChildPosition(text: string, offset: number): boolean {
  const end = Math.max(0, Math.min(offset, text.length));
  let depth = 0;
  for (let i = 0; i < end; i++) {
    const c = text[i];
    if (c === '{') depth++;
    else if (c === '}') depth--;
  }
  return depth > 0;
}

/** True for identifier characters (letters, digits, `_`, `$`). */
function isWordChar(c: string): boolean {
  return /[A-Za-z0-9_$]/.test(c);
}

/** True for characters that may start an identifier (letters, `_`, `$`). */
function isIdentStart(c: string): boolean {
  return /[A-Za-z_$]/.test(c);
}
