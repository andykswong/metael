// Free-identifier resolution for a kernel body: each bare name binds to a thread coordinate (a param),
// an input buffer, a vec/mat or scalar uniform, or a callee helper. Buffers/uniforms are classified
// through descriptorOf(v).lower (NOT a bespoke typed-array check), so a vec/mat/future-numeric-type
// input flows through the same table with zero gpu changes.
import type { UserFn, Lowering, Expr, Stmt, ReactiveHost } from '@metael/lang';
import { descriptorOf, readClosureValue, isUserFn } from '@metael/lang';

export type Binding =
  | { readonly role: 'coord'; readonly name: string; readonly axis: number }
  | { readonly role: 'buffer'; readonly name: string; readonly value: unknown; readonly lower: Lowering }
  | { readonly role: 'uniform'; readonly name: string; readonly value: unknown; readonly lower: Lowering }
  | { readonly role: 'scalar'; readonly name: string; readonly value: number }
  | { readonly role: 'callee'; readonly name: string; readonly fn: UserFn };

export interface BindingTable {
  readonly params: readonly string[];
  readonly byName: ReadonlyMap<string, Binding>;
}

/** Resolve the kernel's params + free identifiers against its closure. `freeNames` is the set of bare
 *  identifiers the body references (collected by collectFreeNames). A name that resolves to a buffer/vec/
 *  scalar/callee is bound; an unresolved or unsupported value is left absent (the gate rejects it). */
export function buildBindingTable(kernel: UserFn, freeNames: ReadonlySet<string>, host: ReactiveHost): BindingTable {
  const params = kernel.params.map((p) => (p.kind === 'name' ? p.name : ''));   // v1: name params only
  const paramSet = new Set(params);
  const byName = new Map<string, Binding>();
  params.forEach((name, axis) => { if (name) byName.set(name, { role: 'coord', name, axis }); });
  for (const name of freeNames) {
    if (paramSet.has(name)) continue;
    const value = readClosureValue(kernel, name, host);
    if (value === undefined) continue;
    const d = descriptorOf(value);
    if (d?.lower?.access === 'linear-buffer') { byName.set(name, { role: 'buffer', name, value, lower: d.lower }); continue; }
    if (d?.lower?.access === 'value' && (d.lower.shape === 'vecN' || d.lower.shape === 'matMxN')) { byName.set(name, { role: 'uniform', name, value, lower: d.lower }); continue; }
    if (typeof value === 'number') { byName.set(name, { role: 'scalar', name, value }); continue; }
    if (isUserFn(value)) { byName.set(name, { role: 'callee', name, fn: value }); continue; }
    // any other value (string, normal array/object, closure) is left absent → the gate rejects it.
  }
  return { params, byName };
}

/** Collect free identifiers referenced by the kernel body (names not locally bound). A lexical walk:
 *  track let/const/param/loop-binding introductions per scope; anything read not locally introduced is free. */
export function collectFreeNames(kernel: UserFn): Set<string> {
  const free = new Set<string>();
  const walkExpr = (e: Expr, bound: Set<string>): void => {
    switch (e.kind) {
      case 'ident': if (!bound.has(e.name)) free.add(e.name); return;
      case 'number': case 'string': case 'bool': case 'null': return;
      case 'member': walkExpr(e.object, bound); return;
      case 'index': walkExpr(e.object, bound); walkExpr(e.index, bound); return;
      case 'unary': walkExpr(e.operand, bound); return;
      case 'binary': walkExpr(e.left, bound); walkExpr(e.right, bound); return;
      case 'cond': walkExpr(e.test, bound); walkExpr(e.then, bound); walkExpr(e.else, bound); return;
      case 'call': {
        if (e.callee.kind === 'ident') { if (!bound.has(e.callee.name)) free.add(e.callee.name); } else walkExpr(e.callee, bound);
        e.args.forEach((a) => walkExpr(a, bound));
        if (e.block) { const b = new Set(bound); e.block.forEach((s) => walkStmt(s, b)); }
        return;
      }
      case 'object': e.entries.forEach((en) => walkExpr(en.value, bound)); return;
      case 'array': e.elements.forEach((el) => walkExpr(el.value, bound)); return;
      case 'arrow': { const inner = new Set(bound); e.params.forEach((p) => { if (p.kind === 'name') inner.add(p.name); }); if (Array.isArray(e.body)) e.body.forEach((s) => walkStmt(s, inner)); else walkExpr(e.body, inner); return; }
    }
  };
  const walkStmt = (s: Stmt, bound: Set<string>): void => {
    switch (s.kind) {
      case 'const': case 'let': walkExpr(s.init, bound); bound.add(s.name); return;
      case 'assign': walkExpr(s.value, bound); walkExpr(s.target, bound); return;
      case 'expr': walkExpr(s.expr, bound); return;
      case 'return': if (s.value) walkExpr(s.value, bound); return;
      case 'if': walkExpr(s.test, bound); { const b = new Set(bound); s.then.forEach((st) => walkStmt(st, b)); } if (s.else) { const b = new Set(bound); s.else.forEach((st) => walkStmt(st, b)); } return;
      case 'for': walkExpr(s.iterable, bound); { const b = new Set(bound); b.add(s.binding); s.body.forEach((st) => walkStmt(st, b)); } return;
      case 'while': walkExpr(s.test, bound); { const b = new Set(bound); s.body.forEach((st) => walkStmt(st, b)); } return;
      case 'function': case 'component': return;   // nested decls not supported in a kernel (the gate rejects)
    }
  };
  const bound = new Set<string>(kernel.params.map((p) => (p.kind === 'name' ? p.name : '')).filter(Boolean));
  kernel.body.forEach((s) => walkStmt(s, bound));
  return free;
}

/** True iff the kernel body references any of `names` — as a free identifier OR a call callee. Used to
 *  detect a vec/mat-bearing kernel that the CPU emitter delegates to the interpreter (authoritative for
 *  vec math) rather than hand-walking. */
export function bodyReferencesAny(kernel: UserFn, names: ReadonlySet<string>): boolean {
  // collectFreeNames already surfaces free idents AND call callees (its `call` case adds the callee name).
  for (const n of collectFreeNames(kernel)) if (names.has(n)) return true;
  return false;
}
