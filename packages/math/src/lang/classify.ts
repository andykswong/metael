// A pure, static analysis over a function's AST: is it "core-compliant" — i.e. does it use ONLY
// closure-free, heap-free constructs so it could lower to a restricted (compile-to-shader / fixed-
// data-model) target, not just the interpreter? Reports why-not for each disqualifier. No evaluation,
// no side effects: purely reads the AST + the builtins catalog.
import type { Expr, Stmt, Diagnostic } from '@metael/lang';
import { makeDiagnostic } from '@metael/lang';
import { BUILTINS } from './registry-data.ts';

/** The outcome of {@link classifyProfile}: whether a function is core-compliant, plus a why-not
 *  diagnostic for every disqualifying construct found. */
export interface ProfileResult {
  /** True iff the function uses only core-compliant constructs (no heap types, no closures, no host
   *  builtins, no unresolvable calls). */
  readonly core: boolean;
  /** One informational diagnostic per disqualifier (empty iff core). Code: 'ML-LANG-PROFILE'. */
  readonly reasons: Diagnostic[];
}

/** Classify a `function`/`component`-shaped node (anything with a `body: Stmt[]`). Only `function`
 *  bodies are meaningfully core-classifiable, but the walk is shape-driven so it accepts either. */
export function classifyProfile(fn: { readonly body: readonly Stmt[] }): ProfileResult {
  const reasons: Diagnostic[] = [];
  const flag = (message: string, span?: Expr['span']): void => {
    reasons.push(makeDiagnostic('ML-LANG-PROFILE', message, span));
  };

  const walkExpr = (e: Expr): void => {
    switch (e.kind) {
      case 'number': case 'bool': case 'ident': return;               // scalar-safe
      case 'null': flag('null is a heap/reference value, not core-compliant', e.span); return;
      case 'string': flag('string literals are not core-compliant (no string type on a restricted target)', e.span); return;
      case 'object': flag('object literals are a heap type, not core-compliant', e.span); e.entries.forEach((en) => walkExpr(en.value)); return;
      case 'array': flag('array literals are a dynamic heap type, not core-compliant', e.span); e.elements.forEach((el) => walkExpr(el.value)); return;
      case 'arrow': flag('closures (arrows) are not core-compliant', e.span); return;
      case 'member': flag('member access implies a heap value, not core-compliant', e.span); walkExpr(e.object); return;
      case 'index': flag('indexing implies a heap value, not core-compliant', e.span); walkExpr(e.object); walkExpr(e.index); return;
      case 'unary': walkExpr(e.operand); return;
      case 'binary': walkExpr(e.left); walkExpr(e.right); return;
      case 'cond': walkExpr(e.test); walkExpr(e.then); walkExpr(e.else); return;
      case 'call': {
        if (e.callee.kind === 'ident') {
          const spec = BUILTINS[e.callee.name];
          if (spec) {
            if (spec.profile === 'host') flag(`calls host builtin '${e.callee.name}' (heap/closure), not core-compliant`, e.span);
            // a 'core' builtin is fine; still walk args
          } else {
            // Not a builtin: could be a user function. Conservatively non-core unless proven —
            // a single-function classifier cannot see callee bodies.
            flag(`calls '${e.callee.name}' which cannot be proven core-compliant`, e.span);
          }
        } else {
          flag('an indirect call target cannot be proven core-compliant', e.span);
          walkExpr(e.callee);
        }
        e.args.forEach(walkExpr);
        if (e.block) e.block.forEach(walkStmt);
        return;
      }
    }
  };

  const walkStmt = (s: Stmt): void => {
    switch (s.kind) {
      case 'const': walkExpr(s.init); return;
      case 'let': flag('reactive let is not core-compliant', s.span); walkExpr(s.init); return;
      case 'assign': walkExpr(s.value); walkExpr(s.target); return;
      case 'expr': walkExpr(s.expr); return;
      case 'return': if (s.value) walkExpr(s.value); return;
      case 'if': walkExpr(s.test); s.then.forEach(walkStmt); s.else?.forEach(walkStmt); return;
      case 'for': flag('for-of iterates a heap collection, not core-compliant', s.span); walkExpr(s.iterable); s.body.forEach(walkStmt); return;
      case 'while': walkExpr(s.test); s.body.forEach(walkStmt); return;
      case 'function': case 'component': flag('nested function/component is not core-compliant', s.span); return;
    }
  };

  fn.body.forEach(walkStmt);
  return { core: reasons.length === 0, reasons };
}
