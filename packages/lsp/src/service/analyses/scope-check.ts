import { FORBIDDEN_KEYS } from '@metael/lang';
import type { Stmt, Expr } from '@metael/lang';
import type { Profile } from '@metael/lang/profile';
import type { Document } from '../document.ts';
import type { ScopeModel } from '../scope-model.ts';
import type { SvcDiagnostic, SvcSpan } from '../results.ts';

/** Recursion ceiling for the AST walk, matching the parser/scope-model bound so a pathological or
 *  cyclic (partial) tree fails closed rather than overflowing the JS stack. */
const MAX_WALK_DEPTH = 512;

/**
 * Compute the scope-driven diagnostics the parser cannot — undeclared value-reads and block-scope
 * redeclarations — for a parsed document under an active vocabulary {@link Profile}.
 *
 * @remarks
 * This mirrors the evaluator's own author-error diagnostics WITHOUT evaluating, and is deliberately
 * conservative: it only reports what the interpreter would also report, preferring a false-negative (a
 * missed error) to any false-positive (flagging valid code).
 *
 * - **Undeclared value-reads → `ML-LANG-UNKNOWN-VAR`.** An `ident` in value-read position is flagged
 *   only when its name resolves to none of: a {@link FORBIDDEN_KEYS} entry (the evaluator emits a
 *   different, security-specific code for those); the implicit `data` root binding; a binding visible at
 *   the ident's offset per {@link ScopeModel.visibleAt}; a profile builtin/head/type name; or the
 *   `range` intrinsic. Call heads (`foo(...)`), member property strings, object-entry keys, `for`
 *   bindings, and assignment target idents are never treated as value-reads — matching how the
 *   evaluator resolves each of those through a separate path (or not at all). Because
 *   {@link ScopeModel.visibleAt} uses forward-visibility (a binding is visible from its scope start),
 *   this pass does not flag use-before-declaration; that is an accepted, safe false-negative.
 * - **Block-scope redeclaration → `ML-LANG-REDECL`.** The parser already reports top-level redeclarations;
 *   this adds only redeclarations the evaluator would raise inside a NON-root block frame. It mirrors the
 *   interpreter's per-frame `env.hasOwn` check by detecting two `const`/`function`/`component`
 *   declarations of the same name among the DIRECT siblings of one block body (component/function/arrow
 *   bodies, `if`/`else`/`while`/`for` bodies) — never across nested blocks, which get their own frames.
 *   `let` is excluded: its redecl diagnostic is gated on being inside a component, a condition this
 *   static pass cannot reproduce without risking a false-positive.
 *
 * Pure and total: it reads the document, scope, and profile but mutates nothing and never throws.
 *
 * @param doc - the parsed document whose AST is walked.
 * @param scope - the static scope model used to test binding visibility at an offset.
 * @param profile - the active vocabulary, whose builtin/head/type names are treated as declared.
 * @returns the added scope diagnostics (undeclared value-reads then block-scope redeclarations).
 */
export function computeScopeChecks(doc: Document, scope: ScopeModel, profile: Profile): readonly SvcDiagnostic[] {
  const out: SvcDiagnostic[] = [];
  const stmts = doc.parse?.program?.stmts ?? [];
  const checker = new ScopeChecker(scope, profile, out);
  // Walk each top-level statement for value-reads + nested block redeclarations. The top-level (root)
  // scope's own redeclarations are already reported by the parser, so this pass never checks them.
  for (const s of stmts) checker.walkStmt(s, 0);
  return out;
}

/** The stateful walk backing {@link computeScopeChecks}: it carries the scope/profile lookups and pushes
 *  diagnostics as it recurses, guarded by a fixed depth against partial/cyclic trees. */
class ScopeChecker {
  constructor(
    private readonly scope: ScopeModel,
    private readonly profile: Profile,
    private readonly out: SvcDiagnostic[],
  ) {}

  /** Whether `name` is resolvable — visible at `offset`, the implicit `data`, a profile name, or
   *  `range` — so an ident reference to it must NOT be flagged as undeclared. */
  private isDeclared(name: string, offset: number): boolean {
    if (name === 'data' || name === 'range') return true;
    if (this.profile.builtins.has(name) || this.profile.heads.has(name) || this.profile.types.has(name)) return true;
    return this.scope.visibleAt(offset).some((b) => b.name === name);
  }

  /** Flag an ident in value-read position when its name is neither forbidden, declared, nor a profile
   *  name. Uses the ident's own span so the diagnostic points exactly at the read. */
  private checkValueRead(e: Extract<Expr, { kind: 'ident' }>): void {
    const name = e.name;
    if (!name || FORBIDDEN_KEYS.has(name)) return;
    const span = e.span;
    if (!span || typeof span.start !== 'number' || typeof span.end !== 'number') return;
    if (this.isDeclared(name, span.start)) return;
    this.out.push({
      span: { start: span.start, end: span.end },
      severity: 'error',
      code: 'ML-LANG-UNKNOWN-VAR',
      message: `unknown variable '${name}'`,
    });
  }

  /** Walk a statement, checking any value-read expressions it contains and recursing into child blocks. */
  walkStmt(s: Stmt | undefined, depth: number): void {
    if (depth > MAX_WALK_DEPTH || !s || typeof s !== 'object' || typeof s.kind !== 'string') return;
    const d = depth + 1;
    switch (s.kind) {
      case 'const':
      case 'let':
        this.walkExpr(s.init, d);
        return;
      case 'assign':
        // The target ident is a WRITE (the evaluator resolves it via execAssign, not evalIdent) — skip it
        // as a value-read. A member/index target's OBJECT part IS a value-read, so recurse it.
        this.walkAssignTarget(s.target, d);
        this.walkExpr(s.value, d);
        return;
      case 'function':
      case 'component':
        // Params are declarations (ScopeModel scopes them); the body idents are reads within that scope.
        for (const b of s.body ?? []) this.walkStmt(b, d);
        this.redeclInBlock(s.body ?? []);
        return;
      case 'if':
        this.walkExpr(s.test, d);
        for (const b of s.then ?? []) this.walkStmt(b, d);
        for (const b of s.else ?? []) this.walkStmt(b, d);
        this.redeclInBlock(s.then ?? []);
        this.redeclInBlock(s.else ?? []);
        return;
      case 'while':
        this.walkExpr(s.test, d);
        for (const b of s.body ?? []) this.walkStmt(b, d);
        this.redeclInBlock(s.body ?? []);
        return;
      case 'for':
        // The `for` binding is a declaration (visible in the body), not a read — skip it.
        this.walkExpr(s.iterable, d);
        for (const b of s.body ?? []) this.walkStmt(b, d);
        this.redeclInBlock(s.body ?? []);
        return;
      case 'return':
        this.walkExpr(s.value, d);
        return;
      case 'expr':
        this.walkExpr(s.expr, d);
        return;
      // Unrecognized (partial) statement kinds contribute nothing.
    }
  }

  /** Walk the OBJECT part of a member/index assignment target as a value-read (`o.a = 1` reads `o`); a
   *  bare-ident target is a write, handled by the evaluator separately, so it is intentionally not read. */
  private walkAssignTarget(target: Expr | undefined, depth: number): void {
    if (depth > MAX_WALK_DEPTH || !target || typeof target !== 'object') return;
    if (target.kind === 'member') this.walkExpr(target.object, depth + 1);
    else if (target.kind === 'index') {
      this.walkExpr(target.object, depth + 1);
      this.walkExpr(target.index, depth + 1);
    }
    // A bare `ident` target is a write — not a value-read; skip it.
  }

  /** Walk an expression, checking every value-read ident and recursing through each kind's children. */
  walkExpr(e: Expr | undefined, depth: number): void {
    if (depth > MAX_WALK_DEPTH || !e || typeof e !== 'object' || typeof e.kind !== 'string') return;
    const d = depth + 1;
    switch (e.kind) {
      case 'ident':
        this.checkValueRead(e);
        return;
      case 'member':
        // `object.property`: the object is a value-read; `property` is a string, never an ident.
        this.walkExpr(e.object, d);
        return;
      case 'index':
        this.walkExpr(e.object, d);
        this.walkExpr(e.index, d);
        return;
      case 'binary':
        this.walkExpr(e.left, d);
        this.walkExpr(e.right, d);
        return;
      case 'unary':
        this.walkExpr(e.operand, d);
        return;
      case 'cond':
        this.walkExpr(e.test, d);
        this.walkExpr(e.then, d);
        this.walkExpr(e.else, d);
        return;
      case 'object':
        // Entry keys are strings, not idents; only the entry values are value-reads.
        for (const entry of e.entries ?? []) this.walkExpr(entry?.value, d);
        return;
      case 'array':
        for (const element of e.elements ?? []) this.walkExpr(element?.value, d);
        return;
      case 'arrow':
        // Params are declarations (ScopeModel scopes them); the body idents are reads within the arrow's
        // scope, so visibleAt at each ident's offset already sees the params. A block body opens its own
        // frame, so its direct siblings are redecl-checked.
        if (Array.isArray(e.body)) {
          for (const b of e.body) this.walkStmt(b, d);
          this.redeclInBlock(e.body);
        } else {
          this.walkExpr(e.body, d);
        }
        return;
      case 'call':
        // The callee head is resolved via evalCall (binding → builtin → range → host), NOT evalIdent —
        // so a call head is never an UNKNOWN-VAR. Leave it entirely alone; recurse args + any block.
        for (const a of e.args ?? []) this.walkExpr(a, d);
        for (const b of e.block ?? []) this.walkStmt(b, d);
        return;
      // Literals (number/string/bool/null) have no value-read children.
    }
  }

  /**
   * Report a block-scope redeclaration for the DIRECT `const`/`function`/`component` siblings of one
   * (non-root) block body, mirroring the evaluator's per-frame `env.hasOwn` check.
   *
   * @remarks
   * Only same-name declarations that share the SAME frame are a redeclaration — nested `if`/`while`/`for`
   * bodies and arrow/function bodies each open their own frame and are handled by their own call. The
   * root (top-level) scope is never passed here: the parser already emits `ML-LANG-REDECL` for it, and
   * this pass must not double-report. `let` is excluded because its redecl diagnostic is gated on
   * `insideComponent`, which a static pass cannot reliably reproduce. The diagnostic anchors on the 2nd+
   * declaration's span, matching the evaluator's message text.
   */
  private redeclInBlock(body: readonly Stmt[]): void {
    if (!Array.isArray(body)) return;
    const seen = new Set<string>();
    for (const s of body) {
      if (!s || typeof s.kind !== 'string') continue;
      // Mirror the evaluator's unconditional redecl guards: const/function/component. `let` is gated on
      // insideComponent, so excluding it avoids a false-positive on a top-level-ish `let` we can't classify.
      if (s.kind !== 'const' && s.kind !== 'function' && s.kind !== 'component') continue;
      const name = (s as { name?: string }).name;
      if (!name) continue; // recovery placeholder — never a real binding.
      if (seen.has(name)) {
        const span: SvcSpan | undefined = s.span && typeof s.span.start === 'number' && typeof s.span.end === 'number'
          ? { start: s.span.start, end: s.span.end }
          : undefined;
        if (span) {
          this.out.push({ span, severity: 'error', code: 'ML-LANG-REDECL', message: `'${name}' already declared` });
        }
      }
      seen.add(name);
    }
  }
}
