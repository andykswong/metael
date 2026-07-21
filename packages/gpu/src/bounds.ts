// A CONSERVATIVE static out-of-bounds bounds-prover for a kernel's buffer-index expressions. It runs a
// SOUND interval over-approximation of each `bufferIdent[indexExpr]` against the buffer's length and rejects
// (MLGPU-INDEX-STATIC) an index that is PROVABLY out of range for EVERY coordinate value — and ONLY that.
//
// THE SOUNDNESS INVARIANT (non-negotiable): a false rejection is worse than a missed OOB. An index whose
// interval overlaps [0, length) AT ALL — or that we cannot bound at all (a `null`/⊤ interval) — is LEFT to
// the sampled oracle (`verify`) and the runtime backends; it is NEVER rejected here. We only reject when the
// entire interval provably lies outside [0, length). The interval bounds are exact affine over-
// approximations of the (coord/loop/const)-parameterized index, so a rejection means every concrete coord
// combination produces an out-of-range access.
import type { UserFn, Expr, Stmt, Diagnostic } from '@metael/lang';
import { descriptorOf, makeDiagnostic } from '@metael/lang';
import type { BindingTable } from './binding.ts';

/** A proven range for an integer/real quantity, or `null` = ⊤ (unprovable / data-dependent — pass). */
export interface Interval {
  /** The inclusive lower bound of the quantity's proven range. */
  readonly lo: number;
  /** The inclusive upper bound of the quantity's proven range. */
  readonly hi: number;
}
type Iv = Interval | null;

// Interval arithmetic. `+`/`-` are the obvious bounds; `*` is the min/max of the four corner products
// (so a negative operand — e.g. `i - 1` reaching below 0 — is handled correctly). A result that overflows
// to a non-finite value (e.g. Inf − Inf → NaN from a pathological scalar) collapses to ⊤ (null), never a
// bogus finite bound.
const guard = (iv: Interval): Iv => (Number.isFinite(iv.lo) && Number.isFinite(iv.hi) ? iv : null);
const add = (a: Interval, b: Interval): Iv => guard({ lo: a.lo + b.lo, hi: a.hi + b.hi });
const sub = (a: Interval, b: Interval): Iv => guard({ lo: a.lo - b.hi, hi: a.hi - b.lo });
const mul = (a: Interval, b: Interval): Iv => {
  const c = [a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi];
  return guard({ lo: Math.min(...c), hi: Math.max(...c) });
};

/** The proven interval of an expression under `env` (coord/loop/local intervals) + the binding table
 *  (a closed-over scalar is a compile-time constant `[v, v]`). Only number literals, in-scope coord/loop/
 *  local names, closed-over scalars, and `+`/`-`/`*`/unary-`-` over those are provable; everything else
 *  (`/`, `%`, a buffer read, a function call, a ternary, a comparison, a member/swizzle) is ⊤ (null). */
export function intervalOf(e: Expr, env: ReadonlyMap<string, Iv>, bindings: BindingTable): Iv {
  switch (e.kind) {
    case 'number': return Number.isFinite(e.value) ? { lo: e.value, hi: e.value } : null;
    case 'ident': {
      if (env.has(e.name)) return env.get(e.name)!;
      const b = bindings.byName.get(e.name);
      // A closed-over scalar is a concrete number at gpu() time → an exact point interval. Coord/loop names
      // live in `env`; a buffer/uniform/callee ident is not a numeric quantity → ⊤.
      if (b?.role === 'scalar' && Number.isFinite(b.value)) return { lo: b.value, hi: b.value };
      return null;
    }
    case 'unary': {
      if (e.op !== '-') return null;   // `!x` is a bool
      const x = intervalOf(e.operand, env, bindings);
      return x ? guard({ lo: -x.hi, hi: -x.lo }) : null;
    }
    case 'binary': {
      if (e.op !== '+' && e.op !== '-' && e.op !== '*') return null;   // `/`,`%`,compares,logical → ⊤
      const l = intervalOf(e.left, env, bindings); const r = intervalOf(e.right, env, bindings);
      if (!l || !r) return null;
      return e.op === '+' ? add(l, r) : e.op === '-' ? sub(l, r) : mul(l, r);
    }
    default: return null;   // member/index/call/cond/object/array/arrow/string/bool/null → ⊤
  }
}

/** Collect every local name that is EVER the target of an `assign` statement. Such a name is REASSIGNED,
 *  so a single tracked interval (from its declaration) cannot be trusted — it is forced to ⊤ (null) wherever
 *  it is read. This is the conservative rule the soundness invariant demands (a `let s = 0` accumulated in a
 *  loop, `s = s + …`, becomes data-dependent → ⊤). A `const`/`let` DECLARATION is not a reassignment; a
 *  const declared inside a loop is re-bound to the loop var each iteration and is analyzed in-context. */
function collectAssigned(body: readonly Stmt[]): Set<string> {
  const out = new Set<string>();
  const walk = (s: Stmt): void => {
    switch (s.kind) {
      case 'assign': if (s.target.kind === 'ident') out.add(s.target.name); return;
      case 'if': s.then.forEach(walk); s.else?.forEach(walk); return;
      case 'for': case 'while': s.body.forEach(walk); return;
      default: return;   // const/let/return/expr introduce no reassignment target
    }
  };
  body.forEach(walk);
  return out;
}

/** Prove-or-pass every `bufferIdent[indexExpr]` in the kernel body against the buffer's length, pushing an
 *  MLGPU-INDEX-STATIC diagnostic into `reasons` for each index whose interval lies ENTIRELY outside
 *  [0, length). `dims` (the requested output shape) bounds the coordinate params (axis a ∈ [0, dims[a]-1]).
 *  A separate pass from gateKernel (which stays dims-agnostic): the engine calls it right after the gate,
 *  pushing into the SAME reasons array, so a provable OOB makes the resource non-core. */
export function checkStaticBounds(kernel: UserFn, bindings: BindingTable, dims: readonly number[], reasons: Diagnostic[]): void {
  // Coordinate params: axis a is bounded by the output dimension dims[a] (a positive integer from cfg.output).
  // A missing/invalid dim (a param with no matching output axis, or a non-positive-integer dim caught later by
  // the cost gate) leaves the coord ⊤ (unprovable) — never a bogus bound.
  const base = new Map<string, Iv>();
  for (const b of bindings.byName.values()) {
    if (b.role === 'coord') {
      const d = dims[b.axis];
      base.set(b.name, d !== undefined && Number.isInteger(d) && d >= 1 ? { lo: 0, hi: d - 1 } : null);
    }
  }
  const assigned = collectAssigned(kernel.body);
  // Soundness: a name that is REASSIGNED anywhere in the body cannot be trusted at a read site — even a coord
  // param (`i = 5`) or a closed-over scalar (`base = 2`), whose base/binding interval would otherwise leak the
  // pre-assignment value. Force every assigned name to ⊤ in the base env (env is consulted before the scalar
  // binding in intervalOf, so this shadows it). const/let/for re-set their own names based on the same set.
  for (const name of assigned) base.set(name, null);

  // The buffer's element count, or null if it can't be read (a missing/zero length → skip: unprovable, and
  // an empty buffer's every access is the runtime's/oracle's to catch, not a static all-OOB proof worth the
  // false-rejection risk). Works for both a typed-array input and a resident GpuBufferHandle (both expose
  // `.length` via getMember).
  const bufLen = (name: string): number | null => {
    const b = bindings.byName.get(name);
    if (!b || b.role !== 'buffer') return null;
    // A PLAIN metael array buffer input (no descriptor) exposes its element count directly — so a provable
    // OOB on a plain array is caught statically exactly like a typed-array buffer (a missing plain-array
    // length would leave it unprovable → an OOB slips to the oracle; equal treatment is the consistent call).
    if (Array.isArray(b.value)) return b.value.length > 0 ? b.value.length : null;
    const len = Number(descriptorOf(b.value)?.getMember?.(b.value, 'length'));
    return Number.isFinite(len) && len > 0 ? len : null;
  };

  // An interval-of that ALSO resolves `bufferIdent.length` to the buffer's element count (a compile-time
  // point interval), so a length-guard like `i + N >= x.length` becomes provable. Everything else defers to
  // the plain `intervalOf` (which is ⊤ on a member). Used ONLY to decide guard-always-true for reachability
  // suppression — never to reject — so ⊤ (null) here just means "can't prove the guard" (no suppression).
  const ivExt = (e: Expr, env: ReadonlyMap<string, Iv>): Iv => {
    if (e.kind === 'member' && e.property === 'length' && e.object.kind === 'ident') {
      const len = bufLen(e.object.name);
      return len !== null ? { lo: len, hi: len } : null;
    }
    if (e.kind === 'unary' && e.op === '-') { const x = ivExt(e.operand, env); return x ? guard({ lo: -x.hi, hi: -x.lo }) : null; }
    if (e.kind === 'binary' && (e.op === '+' || e.op === '-' || e.op === '*')) {
      const l = ivExt(e.left, env); const r = ivExt(e.right, env);
      if (!l || !r) return null;
      return e.op === '+' ? add(l, r) : e.op === '-' ? sub(l, r) : mul(l, r);
    }
    return intervalOf(e, env, bindings);
  };

  // Is a boolean guard PROVABLY true for EVERY coord combination? Sound over-approximation for a numeric
  // comparison: `L < R` is always-true iff max(L) < min(R) i.e. `l.hi < r.lo`; `L >= R` iff `l.lo >= r.hi`;
  // etc. A ⊤ operand, a non-comparison, or `==`/`!=` (not decidable from a range) → false. Used ONLY to
  // suppress a rejection (mark a fall-through path dead) — the SAFE direction: a too-permissive `true` merely
  // misses an OOB (the oracle covers it); it can never cause a FALSE rejection. Being too strict (`false`)
  // just keeps the trailing statement reachable — which is exactly what the mirror case (a non-provable
  // guard) needs so a genuinely-reachable trailing OOB still rejects.
  const guardProvablyTrue = (test: Expr, env: ReadonlyMap<string, Iv>): boolean => {
    if (test.kind !== 'binary') return false;
    const l = ivExt(test.left, env); const r = ivExt(test.right, env);
    if (!l || !r) return false;
    switch (test.op) {
      case '<': return l.hi < r.lo;
      case '<=': return l.hi <= r.lo;
      case '>': return l.lo > r.hi;
      case '>=': return l.lo >= r.hi;
      default: return false;   // ==, !=, &&, ||, + … → not a decidable always-true range guard
    }
  };

  // Does a statement DEFINITELY return (exit the function) on every path? Then every SUBSEQUENT statement in
  // its block is UNREACHABLE dead code and must NOT be rejected (the runtime never runs it). A `return` always
  // returns; an `if` returns iff — with an `else` — BOTH arms definitely return, or — without an `else` — its
  // guard is provably-always-true AND its then-block definitely returns (the fall-through is then infeasible).
  // A bare `if` with a non-provable guard, a `for`/`while`, or anything else does NOT (control can fall
  // through), so a reachable trailing statement is still analyzed + rejected. A block definitely returns if ANY
  // of its top-level statements does (everything after that one is itself dead).
  const blockReturns = (stmts: readonly Stmt[], env: ReadonlyMap<string, Iv>): boolean => stmts.some((st) => definitelyReturns(st, env));
  const definitelyReturns = (s: Stmt, env: ReadonlyMap<string, Iv>): boolean => {
    if (s.kind === 'return') return true;
    if (s.kind === 'if') {
      return s.else ? blockReturns(s.then, env) && blockReturns(s.else, env) : guardProvablyTrue(s.test, env) && blockReturns(s.then, env);
    }
    return false;
  };

  // `guaranteed` = this expression is CERTAIN to be evaluated for every coord (a top-level statement, or the
  // body of a `for … of range(C)` with a provable constant C ≥ 1 — which runs ≥ 1 time). We only ever REJECT
  // when guaranteed: an all-OOB index that MIGHT not execute (inside an `if`/`while`, or a `for range(m)` that
  // could iterate zero times) is NOT a proven-for-every-coord OOB — its guard could exclude exactly the OOB
  // coords, so rejecting would be a false positive. Intervals are still tracked everywhere (so a nested index
  // reads the right loop-var bound); only the rejection is suppressed off the guaranteed path. Missing such an
  // OOB is fine — the sampled oracle covers a data-/control-dependent OOB. (Soundness ≫ completeness.)
  const walkExpr = (e: Expr, env: Map<string, Iv>, guaranteed: boolean): void => {
    switch (e.kind) {
      case 'index': {
        if (e.object.kind === 'ident' && bindings.byName.get(e.object.name)?.role === 'buffer') {
          const len = bufLen(e.object.name);
          if (len !== null && guaranteed) {
            const iv = intervalOf(e.index, env, bindings);
            // Round OUTWARD to match the runtime's `round(idx)`: an index rounds to [0,len) iff its raw value
            // is in [-0.5, len-0.5). We reject ONLY when the whole interval provably rounds outside that band —
            // floor(lo) >= len (the smallest value rounds ≥ len) OR ceil(hi) < 0 (the largest rounds < 0). Using
            // floor(lo)/ceil(hi) is the conservative direction: a fractional index that COULD round into range
            // (e.g. lo = len-0.4, or hi = -0.4) is never rejected. An overlapping or ⊤ interval passes.
            if (iv !== null && (Math.floor(iv.lo) >= len || Math.ceil(iv.hi) < 0)) {
              reasons.push(makeDiagnostic('MLGPU-INDEX-STATIC', `index is provably out of range: [${iv.lo}, ${iv.hi}] vs buffer length ${len}`, e.span));
            }
          }
          walkExpr(e.index, env, guaranteed);   // still descend for a nested `a[b[i]]` inner index
        } else { walkExpr(e.object, env, guaranteed); walkExpr(e.index, env, guaranteed); }
        return;
      }
      case 'member': walkExpr(e.object, env, guaranteed); return;
      case 'unary': walkExpr(e.operand, env, guaranteed); return;
      case 'binary': walkExpr(e.left, env, guaranteed); walkExpr(e.right, env, guaranteed); return;
      // A ternary's test always runs; its branches are conditional → not guaranteed.
      case 'cond': walkExpr(e.test, env, guaranteed); walkExpr(e.then, env, false); walkExpr(e.else, env, false); return;
      case 'call': {
        if (e.callee.kind !== 'ident') walkExpr(e.callee, env, guaranteed);
        e.args.forEach((a) => walkExpr(a, env, guaranteed));
        if (e.block) walkStmts(e.block, new Map(env), false);
        return;
      }
      case 'object': e.entries.forEach((en) => walkExpr(en.value, env, guaranteed)); return;
      case 'array': e.elements.forEach((el) => walkExpr(el.value, env, guaranteed)); return;
      default: return;   // number/string/bool/null/ident/arrow → nothing to check
    }
  };

  const walkStmts = (stmts: readonly Stmt[], env: Map<string, Iv>, guaranteed: boolean): void => {
    // Track reachability WITHIN the block: once a statement DEFINITELY returns, every subsequent statement is
    // dead — still walked (so local intervals stay consistent for any later analysis) but with guaranteed=false
    // so its indices are NEVER rejected (the runtime never executes them → rejecting would be a false positive,
    // violating the never-falsely-reject invariant). `definitelyReturns` is false for a bare non-provable `if`
    // and any loop, so a genuinely-reachable trailing statement stays on the guaranteed path and still rejects.
    let reachable = true;
    for (const s of stmts) {
      walkStmt(s, env, guaranteed && reachable);
      if (reachable && definitelyReturns(s, env)) reachable = false;
    }
  };
  const walkStmt = (s: Stmt, env: Map<string, Iv>, guaranteed: boolean): void => {
    switch (s.kind) {
      case 'const': case 'let':
        walkExpr(s.init, env, guaranteed);
        // A reassigned name is ⊤ everywhere (collectAssigned); else track the declaration's proven interval.
        env.set(s.name, assigned.has(s.name) ? null : intervalOf(s.init, env, bindings));
        return;
      case 'assign': walkExpr(s.value, env, guaranteed); walkExpr(s.target, env, guaranteed); return;   // target already forced ⊤
      case 'expr': walkExpr(s.expr, env, guaranteed); return;
      case 'return': if (s.value) walkExpr(s.value, env, guaranteed); return;
      // The test always runs (guaranteed inherits); the branches are conditional → their indices might not
      // execute for a given coord, so never REJECT inside them (analyze with guaranteed=false).
      case 'if': walkExpr(s.test, env, guaranteed); walkStmts(s.then, new Map(env), false); if (s.else) walkStmts(s.else, new Map(env), false); return;
      case 'for': {
        const it = s.iterable;
        let loopIv: Iv = null;
        // A `for … of range(C)` body is guaranteed to run (≥1 time) ONLY when C is a provable integer constant
        // ≥ 1. An unprovable/⊤ bound (e.g. range(m), m data-dependent) could iterate ZERO times → the body is
        // NOT guaranteed → suppress rejections inside it. A constant C ≤ 0 makes the body DEAD → skip it.
        let bodyGuaranteed = false;
        let dead = false;
        if (it.kind === 'call' && it.callee.kind === 'ident' && it.callee.name === 'range' && it.args.length >= 1) {
          walkExpr(it.args[0]!, env, guaranteed);
          const bnd = intervalOf(it.args[0]!, env, bindings);
          if (bnd !== null && bnd.lo === bnd.hi && Number.isInteger(bnd.lo)) {
            if (bnd.lo >= 1) { loopIv = { lo: 0, hi: bnd.lo - 1 }; bodyGuaranteed = guaranteed; }
            else dead = true;   // range(0)/range(negative): the body never executes → don't analyze it
          }
        } else { walkExpr(it, env, guaranteed); }
        if (!dead) {
          const child = new Map(env);
          child.set(s.binding, assigned.has(s.binding) ? null : loopIv);
          walkStmts(s.body, child, bodyGuaranteed);
        }
        return;
      }
      case 'while': walkExpr(s.test, env, guaranteed); walkStmts(s.body, new Map(env), false); return;
      default: return;   // function/component: not lowerable (the gate rejects) — nothing to prove
    }
  };

  walkStmts(kernel.body, new Map(base), true);
}
