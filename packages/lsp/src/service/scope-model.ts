import type { Program, Stmt, Expr, Pattern, Token } from '@metael/lang';
import type { Document } from './document.ts';
import type { SvcSpan } from './results.ts';

/**
 * A name introduced into scope by a declaration, parameter, or loop binding.
 *
 * @remarks
 * The resolver produces one `Binding` per bound name (a destructuring parameter such as `{ a, b }`
 * yields one `Binding` for `a` and one for `b`). {@link Binding.declSpan} locates the declaration
 * keyword/name in the source, while `[scopeStart, scopeEnd]` is the inclusive offset range over which
 * the name is considered visible — the enclosing block's extent, approximated forward from the
 * declaration to the end of that block.
 */
export interface Binding {
  /** The bound name. */
  readonly name: string;
  /** What introduced the name: a `const`/`let`/`function`/`component` declaration, a `param`, or a `for` loop binding. */
  readonly kind: 'const' | 'let' | 'function' | 'component' | 'param' | 'for';
  /** The source range of the declaration this name comes from (keyword/name span from the parser). */
  readonly declSpan: SvcSpan;
  /** The inclusive offset at which the name becomes visible (the start of its enclosing scope). */
  readonly scopeStart: number;
  /** The inclusive offset at which the name stops being visible (the end of its enclosing scope). */
  readonly scopeEnd: number;
  /**
   * The parameter names of a `function`/`component` binding, in declaration order (a destructuring
   * parameter contributes one name per bound field), so a hover card can render the signature. Present
   * only for `function`/`component` bindings — a zero-parameter one carries an empty array; `const`/`let`/
   * `param`/`for` bindings leave it `undefined`.
   */
  readonly params?: readonly string[];
}

/** The scope range of the program root — every top-level binding is visible across the whole document. */
const ROOT_SCOPE_END = Number.MAX_SAFE_INTEGER;

/** Recursion ceiling for the AST walk, matching the parser's nesting bound so a pathological or cyclic
 *  (partial) tree fails closed rather than overflowing the JS stack. */
const MAX_WALK_DEPTH = 512;

/**
 * A static scope tree over a parsed program: it resolves, without evaluating, which named bindings are
 * visible at any source offset.
 *
 * @remarks
 * The language binds names during evaluation, so this model reconstructs lexical visibility purely from
 * the AST for editor features (e.g. completion's visible-binding set). It walks {@link Program.stmts}
 * once at construction, recording a {@link Binding} for every `const`/`let`/`function`/`component`
 * declaration, every parameter name (destructuring patterns expand to one binding per bound name), and
 * every `for` loop variable, each tagged with the `[scopeStart, scopeEnd]` offset range over which it is
 * visible. It is tolerant of partial ASTs — a mid-edit parse may lack a span or body — and never throws.
 */
export class ScopeModel {
  private readonly bindings: Binding[] = [];
  /** All matched `{`…`}` regions in the source, precomputed once so a body scope can snap its end to
   *  the closing brace rather than the last statement's end. */
  private readonly bracePairs: readonly BracePair[];

  /** Build the scope tree by walking the document's parsed program once. */
  constructor(doc: Document) {
    this.bracePairs = bracePairs(doc.lex?.tokens ?? []);
    const program: Program | undefined = doc.parse?.program;
    const stmts = program?.stmts ?? [];
    for (const s of stmts) this.walkStmt(s, 0, ROOT_SCOPE_END, 0);
  }

  /**
   * The offset of the closing brace (inclusive) of the block whose body statements span
   * `[contentStart, contentEnd]`, or `undefined` when no brace pair encloses that content.
   *
   * @remarks
   * A block-body scope must remain in effect up to and including its `}` so that completion on a blank
   * line at the bottom of the block still sees the block's bindings — the last statement's `span.end`
   * stops short of the brace. The closing brace is not on the AST, so it is recovered from the matched
   * brace regions: the tightest pair enclosing the content is the block's own braces. Returns
   * `undefined` for an empty body (`contentStart` past `contentEnd`) or unbalanced/partial source, so
   * callers fall back to the subtree extent and never produce a NaN scope.
   */
  private braceCloseFor(contentStart: number, contentEnd: number): number | undefined {
    if (!Number.isFinite(contentStart) || !Number.isFinite(contentEnd)) return undefined;
    return enclosingPair(this.bracePairs, contentStart, contentEnd)?.close;
  }

  /** Every binding whose visibility range contains `offset`, in declaration order. */
  visibleAt(offset: number): readonly Binding[] {
    return this.bindings.filter((b) => offset >= b.scopeStart && offset <= b.scopeEnd);
  }

  /**
   * The binding named `name` that is visible at `offset` and belongs to the tightest (innermost) scope,
   * or `undefined` when none is visible.
   *
   * @remarks
   * Mirrors the evaluator's name resolution, which checks local bindings before injected heads/builtins,
   * so an editor analysis can let a local binding win over a same-named profile head or builtin. When a
   * name is shadowed across nested scopes (e.g. a param shadowing an outer `const`), the innermost wins —
   * chosen as the visible match with the greatest {@link Binding.scopeStart} (the most tightly nested, or
   * most recently opened, scope).
   */
  innermostVisibleAt(offset: number, name: string): Binding | undefined {
    let best: Binding | undefined;
    for (const b of this.bindings) {
      if (b.name !== name || offset < b.scopeStart || offset > b.scopeEnd) continue;
      if (!best || b.scopeStart > best.scopeStart) best = b;
    }
    return best;
  }

  /** Every binding the model recorded, regardless of scope. */
  allBindings(): readonly Binding[] {
    return this.bindings;
  }

  /** Record a binding, skipping anything without a usable name or declaration span. `params` (function/
   *  component parameter names) is recorded only when supplied; other kinds leave it off the object. */
  private add(
    name: string | undefined,
    kind: Binding['kind'],
    declSpan: SvcSpan | undefined,
    scopeStart: number,
    scopeEnd: number,
    params?: readonly string[],
  ): void {
    if (!name || !declSpan || typeof declSpan.start !== 'number' || typeof declSpan.end !== 'number') return;
    const binding: Binding = { name, kind, declSpan: { start: declSpan.start, end: declSpan.end }, scopeStart, scopeEnd };
    this.bindings.push(params ? { ...binding, params } : binding);
  }

  /** Walk one statement, adding its bindings and recursing into any child blocks it opens. */
  private walkStmt(s: Stmt | undefined, scopeStart: number, scopeEnd: number, depth: number): void {
    if (depth > MAX_WALK_DEPTH || !s || typeof s !== 'object' || typeof s.kind !== 'string') return;
    const d = depth + 1;
    switch (s.kind) {
      case 'const':
      case 'let': {
        this.add(s.name, s.kind, s.span, scopeStart, scopeEnd);
        this.walkExpr(s.init, scopeStart, scopeEnd, d);
        return;
      }
      case 'function':
      case 'component': {
        // Flatten each parameter pattern to its bound name(s) so the binding carries the signature.
        const paramNames = s.params?.flatMap((p) => patternNames(p)) ?? [];
        this.add(s.name, s.kind, s.span, scopeStart, scopeEnd, paramNames);
        // The declaration span is keyword-only; extend the body scope to its closing brace so bindings
        // stay visible on a blank line above the `}` (falling back to the subtree extent if unmatched).
        const bodyStart = s.span?.start ?? scopeStart;
        const [contentStart, contentEnd] = blockExtent(s.body);
        const bodyEnd = this.braceCloseFor(contentStart, contentEnd) ?? spanEndMax(s, bodyStart);
        for (const p of s.params ?? []) for (const n of patternNames(p)) this.add(n, 'param', s.span, bodyStart, bodyEnd);
        for (const b of s.body ?? []) this.walkStmt(b, bodyStart, bodyEnd, d);
        return;
      }
      case 'if': {
        this.walkExpr(s.test, scopeStart, scopeEnd, d);
        for (const b of s.then ?? []) this.walkStmt(b, scopeStart, scopeEnd, d);
        for (const b of s.else ?? []) this.walkStmt(b, scopeStart, scopeEnd, d);
        return;
      }
      case 'while': {
        this.walkExpr(s.test, scopeStart, scopeEnd, d);
        for (const b of s.body ?? []) this.walkStmt(b, scopeStart, scopeEnd, d);
        return;
      }
      case 'for': {
        const bodyStart = s.span?.start ?? scopeStart;
        const [contentStart, contentEnd] = blockExtent(s.body);
        const bodyEnd = this.braceCloseFor(contentStart, contentEnd) ?? spanEndMax(s, bodyStart);
        this.add(s.binding, 'for', s.span, bodyStart, bodyEnd);
        this.walkExpr(s.iterable, scopeStart, scopeEnd, d);
        for (const b of s.body ?? []) this.walkStmt(b, bodyStart, bodyEnd, d);
        return;
      }
      case 'return': {
        this.walkExpr(s.value, scopeStart, scopeEnd, d);
        return;
      }
      case 'assign': {
        this.walkExpr(s.target, scopeStart, scopeEnd, d);
        this.walkExpr(s.value, scopeStart, scopeEnd, d);
        return;
      }
      case 'expr': {
        this.walkExpr(s.expr, scopeStart, scopeEnd, d);
        return;
      }
      // Unrecognized (e.g. partial) statement kinds contribute no bindings.
    }
  }

  /**
   * Walk an expression, recursing through every kind's structural children so a nested `arrow` at any
   * depth (e.g. inside a call like `items.map(x => x)`) opens its parameter scope. Non-arrow kinds bind
   * nothing themselves; they exist here only as paths to reach a buried arrow.
   */
  private walkExpr(e: Expr | undefined, scopeStart: number, scopeEnd: number, depth: number): void {
    if (depth > MAX_WALK_DEPTH || !e || typeof e !== 'object' || typeof e.kind !== 'string') return;
    const d = depth + 1;
    switch (e.kind) {
      case 'arrow': {
        const bodyStart = e.span?.start ?? scopeStart;
        const body = e.body;
        // A block body (`=> { … }`) extends to its closing brace so bindings survive a blank line above
        // the `}`; an expression body has no brace, so its subtree extent is the correct end.
        let bodyEnd = spanEndMax(e, e.span?.end ?? scopeEnd);
        if (Array.isArray(body)) {
          const [contentStart, contentEnd] = blockExtent(body);
          bodyEnd = this.braceCloseFor(contentStart, contentEnd) ?? bodyEnd;
        }
        for (const p of e.params ?? []) for (const n of patternNames(p)) this.add(n, 'param', e.span, bodyStart, bodyEnd);
        if (Array.isArray(body)) for (const b of body) this.walkStmt(b, bodyStart, bodyEnd, d);
        else this.walkExpr(body, bodyStart, bodyEnd, d);
        return;
      }
      case 'call': {
        this.walkExpr(e.callee, scopeStart, scopeEnd, d);
        for (const a of e.args ?? []) this.walkExpr(a, scopeStart, scopeEnd, d);
        for (const b of e.block ?? []) this.walkStmt(b, scopeStart, scopeEnd, d);
        return;
      }
      case 'binary': {
        this.walkExpr(e.left, scopeStart, scopeEnd, d);
        this.walkExpr(e.right, scopeStart, scopeEnd, d);
        return;
      }
      case 'cond': {
        this.walkExpr(e.test, scopeStart, scopeEnd, d);
        this.walkExpr(e.then, scopeStart, scopeEnd, d);
        this.walkExpr(e.else, scopeStart, scopeEnd, d);
        return;
      }
      case 'unary': {
        this.walkExpr(e.operand, scopeStart, scopeEnd, d);
        return;
      }
      case 'member': {
        this.walkExpr(e.object, scopeStart, scopeEnd, d);
        return;
      }
      case 'index': {
        this.walkExpr(e.object, scopeStart, scopeEnd, d);
        this.walkExpr(e.index, scopeStart, scopeEnd, d);
        return;
      }
      case 'object': {
        for (const entry of e.entries ?? []) this.walkExpr(entry?.value, scopeStart, scopeEnd, d);
        return;
      }
      case 'array': {
        for (const element of e.elements ?? []) this.walkExpr(element?.value, scopeStart, scopeEnd, d);
        return;
      }
      // Literals (number/string/bool/null) and `ident` have no scope-opening children.
    }
  }
}

/** Extract the bound name(s) from a parameter pattern (a destructuring pattern binds several). */
function patternNames(p: Pattern | undefined): readonly string[] {
  if (!p || typeof p !== 'object') return [];
  switch (p.kind) {
    case 'name':
      return p.name ? [p.name] : [];
    case 'objectPattern':
      return p.fields ?? [];
    case 'arrayPattern':
      return p.elements ?? [];
    default:
      return [];
  }
}

/**
 * The `[minStart, maxEnd]` source extent of `node`'s whole subtree, or `[Infinity, -Infinity]` when no
 * descendant carries a span.
 *
 * @remarks
 * Statement/declaration spans from the parser cover only a keyword/operator anchor, not the full
 * construct, so the true source extent is the min start / max end over every spanned descendant.
 * Bounded against cyclic/partial structures by a fixed recursion depth.
 */
function subtreeExtent(node: unknown): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  const visit = (n: unknown, depth: number): void => {
    if (depth > 64 || !n || typeof n !== 'object') return;
    const span = (n as { span?: { start?: unknown; end?: unknown } }).span;
    if (span && typeof span.start === 'number' && span.start < lo) lo = span.start;
    if (span && typeof span.end === 'number' && span.end > hi) hi = span.end;
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

/** The greatest span end found anywhere in `node`'s subtree, floored at `fallback` — the fallback body
 *  extent used when the closing brace of a block cannot be recovered from the token stream. */
function spanEndMax(node: unknown, fallback: number): number {
  const hi = subtreeExtent(node)[1];
  return hi > fallback ? hi : fallback;
}

/** The `[minStart, maxEnd]` content extent over a block's statements, or `[Infinity, -Infinity]` for a
 *  missing/empty block — used to locate the enclosing brace pair. */
function blockExtent(body: unknown): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  if (Array.isArray(body)) {
    for (const stmt of body) {
      const [s, e] = subtreeExtent(stmt);
      if (s < lo) lo = s;
      if (e > hi) hi = e;
    }
  }
  return [lo, hi];
}

/** A matched `{`…`}` region as offsets: `open` is the `{`'s start, `close` is one past the `}`. */
interface BracePair {
  readonly open: number;
  readonly close: number;
}

/** All matched `{`…`}` regions in token order, recovered with a stack so unbalanced braces in a
 *  partial (mid-edit) source are simply dropped rather than mispaired. */
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

/** The tightest brace pair enclosing the content range `[contentStart, contentEnd]` — the block's own
 *  braces — or `undefined` when none matches. */
function enclosingPair(pairs: readonly BracePair[], contentStart: number, contentEnd: number): BracePair | undefined {
  let best: BracePair | undefined;
  for (const p of pairs) {
    if (p.open < contentStart && p.close >= contentEnd && (!best || p.open > best.open)) best = p;
  }
  return best;
}
