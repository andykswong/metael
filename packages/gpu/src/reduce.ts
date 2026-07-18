// The REDUCTION kernel kind: a 2-arg ASSOCIATIVE + COMMUTATIVE reducer (`component r(acc, x) { return acc <op>
// x }`) + an input buffer + an identity, folded to a scalar. This is a DISTINCT kernel kind from the map kernel
// (whose params are thread coordinates), so it gets its OWN head (`gpuReduce`) and its OWN gate (`gateReducer`).
//
// This file ships the reducer CONTRACT — the gate (arity + a scalar-lowerable, pure body) and the CPU fold
// (`cpuReduce`, the exact linear left-fold that is the correctness oracle floor). The GPU reduction legs (a
// WebGL2 ping-pong, a WGSL workgroup-shared tree) are follow-on work; the multi-pass orchestration lands with
// them. The CPU linear fold + the WebGL2 tile fold both preserve element ORDER, so associativity alone would
// suffice for them; but the WGSL workgroup tree fold combines `_reduce(scratch[k], scratch[k+stride])` — it
// REORDERS operands across lanes, e.g. `op(op(s0,s2), op(s1,s3))`. So a reducer must be COMMUTATIVE as well as
// associative or it diverges across backends (a non-commutative associative reducer gives a backend-dependent
// scalar — silent with verify off; `verify: true` catches it). Neither property is statically decidable from
// the AST, so both are documented caller contracts, not gate rules. A tree reduction also reorders the fold →
// a float-associativity tolerance is a concern for those legs; this linear left-fold seeded by `identity` is
// exact for integer sums. The canonical examples (sum, product, min, max) are all associative AND commutative.
import type { UserFn, Diagnostic, ReactiveHost, HostEnvironment, Expr, Stmt } from '@metael/lang';
import { makeDiagnostic, makeCallable } from '@metael/lang';
import { gateKernel, type GateVerdict } from './gate.ts';

/** Gate a REDUCER for lowerability. A reducer is DISTINCT from a map kernel: exactly 2 scalar params
 *  (`acc`, `x`), a scalar-lowerable body (`return acc <op> x` / `return acc > x ? acc : x` — arithmetic,
 *  comparisons, ternary, the math builtins), and PURE over those two params — NO buffer read, NO vec/mat
 *  uniform, NO helper call. Reuses the map-kernel gate machinery (`gateKernel` with comps=1) for the body's
 *  lowerability + a scalar-output-shape check — the reducer's 2 params bind as scalar coords, so `acc`/`x`
 *  read fine and any non-lowerable expression is flagged — then ADDS the two reducer-specific rules the
 *  map-kernel gate does not enforce: exact arity, and purity (a closed-over buffer/vec/helper is rejected).
 *
 *  NOT gated here: the NEUTRAL-IDENTITY contract AND the ASSOCIATIVE + COMMUTATIVE requirement (see `cpuReduce`
 *  / `ReduceConfig.identity`). The `identity` must be the reducer's neutral element, and the reducer must be
 *  associative AND commutative — the WGSL workgroup tree fold reorders operands across lanes, so a
 *  non-commutative associative reducer diverges across backends. Neither property is statically decidable from
 *  the AST (they are properties of the reducer relative to its seed / its operands), so no gate reason can flag
 *  them. They are documented caller contracts that `verify: true` catches at run time. */
export function gateReducer(reducer: UserFn, host: ReactiveHost): GateVerdict {
  const reasons: Diagnostic[] = [];
  const span = reducer.body[0]?.span;
  // A reducer takes EXACTLY 2 name params (acc, x) — both scalars. Not 2 name params (or a destructuring
  // param) → not a binary associative reducer.
  const nameParams = reducer.params.filter((p) => p.kind === 'name');
  if (reducer.params.length !== 2 || nameParams.length !== 2) {
    reasons.push(makeDiagnostic('MLGPU-REDUCER-ARITY', `a reducer must take exactly 2 parameters (acc, x) — got ${reducer.params.length}`, span));
  }
  // Reuse the map-kernel gate for the body's lowerability + a SCALAR (comps=1) output-shape check: the
  // reducer body is scalar arithmetic/comparison/ternary + the math builtins — exactly the lowerable-scalar
  // surface gateKernel validates. Its 2 params bind as coords (scalars), so acc/x read fine and any buffer/
  // helper/vec/string/object use is flagged with the existing MLGPU-* reasons.
  const v = gateKernel(reducer, host, 1);
  reasons.push(...v.reasons);
  // A reducer must be PURE over its two scalar params. gateKernel ACCEPTS a valid buffer index (`acc + buf[x]`)
  // and a scalar/vec uniform — legal in a map kernel — but a reducer's only inputs are acc + x, so a
  // closed-over buffer/vec/mat/helper has no place in a binary fold. Reject any such binding so the reducer
  // gate is strictly narrower than the map-kernel gate. (A closed-over scalar CONSTANT — `role:'scalar'` — is
  // allowed: it is a uniform of the fold, resolved by the interpreter in cpuReduce and lowerable later.)
  for (const b of v.bindings.byName.values()) {
    if (b.role === 'buffer' || b.role === 'uniform' || b.role === 'callee') {
      const what = b.role === 'callee' ? 'helper function' : b.role === 'uniform' ? 'vec/mat' : 'buffer';
      reasons.push(makeDiagnostic('MLGPU-NOT-LOWERABLE', `a reducer must be pure over its two parameters (acc, x) — it may not reference a ${what} ('${b.name}')`, span));
    }
  }
  // A reducer's two params are SCALARS — indexing one (`acc[x]` / `x[i]`) is meaningless. The shared map-kernel
  // gate accepts it (it only refuses to descend into a BUFFER ident, not a scalar/coord param index): harmless
  // on the CPU (the interpreter returns null → 0) but the reduce GLSL emit would emit `_fetch` on a non-buffer,
  // a compile error. Since 4.2 makes the GPU emit real, reject a reducer whose body indexes either param.
  const paramNames = new Set(nameParams.map((p) => (p as Extract<typeof p, { kind: 'name' }>).name));
  if (indexesAnyParam(reducer.body, paramNames)) {
    reasons.push(makeDiagnostic('MLGPU-NOT-LOWERABLE', 'a reducer parameter is a scalar and cannot be indexed', span));
  }
  return { core: reasons.length === 0, reasons, bindings: v.bindings };
}

/** True iff any `index` expression in the reducer body has one of the reducer's 2 scalar params as its object
 *  (`acc[...]` / `x[...]`) — a meaningless scalar index the shared map-kernel gate does not catch. A shallow
 *  structural walk over the reducer body (which is scalar arithmetic/comparison/ternary/builtins by the gate). */
function indexesAnyParam(body: readonly Stmt[], params: ReadonlySet<string>): boolean {
  const inExpr = (e: Expr): boolean => {
    switch (e.kind) {
      case 'index': return (e.object.kind === 'ident' && params.has(e.object.name)) || inExpr(e.object) || inExpr(e.index);
      case 'member': return inExpr(e.object);
      case 'unary': return inExpr(e.operand);
      case 'binary': return inExpr(e.left) || inExpr(e.right);
      case 'cond': return inExpr(e.test) || inExpr(e.then) || inExpr(e.else);
      case 'call': return (e.callee.kind !== 'ident' && inExpr(e.callee)) || e.args.some(inExpr) || (e.block?.some(inStmt) ?? false);
      case 'object': return e.entries.some((en) => inExpr(en.value));
      case 'array': return e.elements.some((el) => inExpr(el.value));
      default: return false;
    }
  };
  const inStmt = (s: Stmt): boolean => {
    switch (s.kind) {
      case 'const': case 'let': return inExpr(s.init);
      case 'assign': return inExpr(s.value) || inExpr(s.target);
      case 'expr': return inExpr(s.expr);
      case 'return': return s.value ? inExpr(s.value) : false;
      case 'if': return inExpr(s.test) || s.then.some(inStmt) || (s.else?.some(inStmt) ?? false);
      case 'for': return inExpr(s.iterable) || s.body.some(inStmt);
      case 'while': return inExpr(s.test) || s.body.some(inStmt);
      default: return false;
    }
  };
  return body.some(inStmt);
}

/** The CPU fold — the correctness ORACLE floor. An EXACT linear left-fold of `inputValues` seeded by
 *  `identity`, evaluating the reducer through the shipped interpreter (via makeCallable) so the CPU reduce is
 *  identical-by-construction to the interpreter (the descriptor handlers + arithmetic coercion the emitters
 *  also delegate to). A FRESH callable per fold step gives each reducer invocation its own budget (mirrors
 *  the oracle's per-cell callable — makeCallable's Runner budget is aggregate across calls of one callable).
 *  `Number(...)` on each result mirrors the interpreter's downstream buffer-write coercion (a null from a /0
 *  → 0). The GPU tree reduction (a follow-on) reorders this fold → a float-associativity tolerance; this
 *  linear fold is exact for integer sums and is the reference every GPU leg is verified against.
 *
 *  ASSOCIATIVE + COMMUTATIVE CONTRACT: this CPU linear fold and the WebGL2 tile fold both preserve element
 *  ORDER (associativity suffices for them), but the WGSL workgroup tree fold combines
 *  `_reduce(scratch[k], scratch[k+stride])` — it REORDERS operands across lanes. So the reducer must be
 *  COMMUTATIVE as well as associative, or the WGSL leg diverges from this oracle (a backend-dependent scalar,
 *  silent with verify off; `verify: true` flags it). sum/product/min/max are all associative AND commutative.
 *
 *  NEUTRAL-IDENTITY CONTRACT: `identity` MUST be the reducer's NEUTRAL element (the `e` for which
 *  `reduce(e, x) === x` for all `x`: 0 for a sum, 1 for a product, a very-negative sentinel for max, a
 *  very-large one for min). This linear fold applies `identity` EXACTLY ONCE (the seed); the GPU tree fold
 *  re-seeds it into every tile on every pass (a backend-dependent count). For a neutral identity both agree;
 *  a non-neutral identity diverges between backends. Not enforceable — neutrality is not statically decidable —
 *  so it is a caller contract that `verify: true` flags (a non-neutral identity → `match.ok === false`). */
export function cpuReduce(reducer: UserFn, inputValues: readonly number[], identity: number, host: ReactiveHost): number {
  const declineEnv: HostEnvironment = { resolveCall: () => ({ handled: false }) };
  let acc = identity;
  for (const x of inputValues) {
    const call = makeCallable(reducer, { host, env: declineEnv });   // fresh budget per fold step
    acc = Number(call(acc, x));
  }
  return acc;
}
