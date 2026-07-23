import type { Token } from '@metael/lang';
import type { Document } from '../document.ts';
import type { SvcFold } from '../results.ts';

/** Recursion ceiling for the AST walk, matching the parser's nesting bound so a pathological or partial
 *  (mid-edit) tree fails closed rather than overflowing the JS stack. */
const MAX_WALK_DEPTH = 512;

/** A matched `{`…`}` region as offsets: `open` is the `{`'s start, `close` is one past the `}`. */
interface BracePair {
  readonly open: number;
  readonly close: number;
}

/**
 * Compute the foldable regions of a document — one fold per brace-delimited block body.
 *
 * @remarks
 * Folds are AST-driven so only real *blocks* fold, never object literals: the walk visits
 * {@link Document.parse}'s tree and collects a fold for each `function`/`component` body, `if`
 * `then`/`else` block, `while`/`for` body, wrapping-call `block`, and block-bodied `arrow`. Because a
 * statement's own span is only a keyword/operator anchor (not its full extent), each block's brace
 * region is recovered from the token stream: the fold snaps outward to the `{`…`}` pair that tightly
 * encloses the block's content, so a fold literally spans from its opening brace to just past its
 * closing brace. Empty blocks and any block whose braces cannot be matched (e.g. an unbalanced mid-edit
 * source) are skipped. Identical folds are de-duplicated and the result is ordered by start offset. Pure
 * and total.
 */
export function computeFolding(doc: Document): readonly SvcFold[] {
  const pairs = bracePairs(doc.lex.tokens);
  const folds: SvcFold[] = [];
  const seen = new Set<string>();

  const addBlock = (block: unknown): void => {
    if (!Array.isArray(block) || block.length === 0) return;
    let contentStart = Infinity;
    let contentEnd = -Infinity;
    for (const stmt of block) {
      const [lo, hi] = extentOf(stmt);
      if (lo < contentStart) contentStart = lo;
      if (hi > contentEnd) contentEnd = hi;
    }
    if (contentStart === Infinity) return;
    const pair = enclosingPair(pairs, contentStart, contentEnd);
    if (!pair) return;
    const key = `${pair.open}:${pair.close}`;
    if (seen.has(key)) return;
    seen.add(key);
    folds.push({ start: pair.open, end: pair.close });
  };

  const walk = (node: unknown, depth: number): void => {
    if (depth > MAX_WALK_DEPTH || !node || typeof node !== 'object') return;
    const kind = (node as { kind?: unknown }).kind;
    if (kind === 'function' || kind === 'component' || kind === 'while' || kind === 'for') {
      addBlock((node as { body?: unknown }).body);
    } else if (kind === 'if') {
      addBlock((node as { then?: unknown }).then);
      addBlock((node as { else?: unknown }).else);
    } else if (kind === 'call') {
      addBlock((node as { block?: unknown }).block);
    } else if (kind === 'arrow') {
      addBlock((node as { body?: unknown }).body); // ignored unless body is a Stmt[] block
    }
    const d = depth + 1;
    for (const key of Object.keys(node)) {
      if (key === 'span') continue;
      const child = (node as Record<string, unknown>)[key];
      if (Array.isArray(child)) for (const c of child) walk(c, d);
      else if (child && typeof child === 'object') walk(child, d);
    }
  };

  for (const stmt of doc.parse.program.stmts ?? []) walk(stmt, 0);
  folds.sort((a, b) => a.start - b.start || a.end - b.end);
  return folds;
}

/** The `[minStart, maxEnd]` extent of a node's whole subtree — statement/expression spans anchor only a
 *  keyword/operator, so the true source extent is the min/max over every spanned descendant. */
function extentOf(node: unknown): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  const visit = (n: unknown, depth: number): void => {
    if (depth > MAX_WALK_DEPTH || !n || typeof n !== 'object') return;
    const span = (n as { span?: { start?: unknown; end?: unknown } }).span;
    if (span && typeof span.start === 'number' && typeof span.end === 'number') {
      if (span.start < lo) lo = span.start;
      if (span.end > hi) hi = span.end;
    }
    for (const key of Object.keys(n as Record<string, unknown>)) {
      if (key === 'span') continue;
      const child = (n as Record<string, unknown>)[key];
      if (Array.isArray(child)) for (const c of child) visit(c, depth + 1);
      else if (child && typeof child === 'object') visit(child, depth + 1);
    }
  };
  visit(node, 0);
  return [lo, hi];
}

/** All matched `{`…`}` regions in token order, recovered by a stack so unbalanced braces are dropped. */
function bracePairs(tokens: readonly Token[]): readonly BracePair[] {
  const pairs: BracePair[] = [];
  const stack: number[] = [];
  for (const t of tokens) {
    if (t.type === 'lbrace') stack.push(t.span.start);
    else if (t.type === 'rbrace') {
      const open = stack.pop();
      if (open !== undefined) pairs.push({ open, close: t.span.end });
    }
  }
  return pairs;
}

/** The tightest brace pair that encloses the content range `[contentStart, contentEnd]` — the block's
 *  own braces — or `undefined` when none matches. */
function enclosingPair(pairs: readonly BracePair[], contentStart: number, contentEnd: number): BracePair | undefined {
  let best: BracePair | undefined;
  for (const p of pairs) {
    if (p.open < contentStart && p.close >= contentEnd && (!best || p.open > best.open)) best = p;
  }
  return best;
}
