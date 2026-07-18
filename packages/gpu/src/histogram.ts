// The HISTOGRAM kernel kind: a 1-arg BIN-MAPPER (`component binOf(x) { return <bin index expr> }`) + an
// input buffer + a bin count; the result is a per-bin COUNT array. This is a DISTINCT kernel kind from the
// map kernel (whose params are thread coordinates) AND from the reducer (2 params, folds to a scalar) — it
// is a DATA-DEPENDENT ATOMIC SCATTER: each input element maps to a bin index, and that bin's count is
// incremented. So it gets its OWN head (`gpuHistogram`) and its OWN gate (`gateBinMapper`), mirroring the
// reduce head/gate/engine shape.
//
// This file ships the bin-mapper CONTRACT — the gate (arity + a scalar-lowerable, pure body) and the exact
// CPU count (`cpuHistogram`, the correctness ORACLE floor). Per-backend at run time (engine routing in
// resource.ts): CPU → cpuHistogram (the exact oracle); WebGPU → an `atomicAdd(&bins[binIndex], 1)` scatter
// (a real compute stage with `var<storage, read_write> bins: array<atomic<u32>>`); WebGL2 → cpuHistogram
// (WebGL2's fragment stage has NO atomics, so a histogram FALLS TO CPU on WebGL2 — a documented per-backend
// difference; the settled backend reflects that honestly as 'cpu' + a note).
//
// OUT-OF-RANGE BINS ARE DROPPED (not clamped): a bin index < 0 or >= bins is not counted (the standard
// histogram bounds behavior). The CPU oracle drops via `b >= 0 && b < bins`; the WGSL scatter drops via the
// same guard `if (_b >= 0 && _b < i32(bins))` before `atomicAdd` — so both backends AGREE.
import type { UserFn, Diagnostic, ReactiveHost, HostEnvironment, Expr, Stmt } from '@metael/lang';
import { makeDiagnostic, makeCallable } from '@metael/lang';
import { gateKernel, type GateVerdict } from './gate.ts';

/** Gate a BIN-MAPPER for lowerability. A bin-mapper is DISTINCT from a map kernel and a reducer: exactly 1
 *  scalar param (`x`), a scalar-lowerable body (`return x % bins` / `return floor(x / width)` — arithmetic,
 *  comparisons, ternary, the math builtins), and PURE over that one param — NO buffer read, NO vec/mat
 *  uniform, NO helper call. Reuses the map-kernel gate machinery (`gateKernel` with comps=1) for the body's
 *  lowerability + a scalar-output-shape check — the mapper's param binds as a scalar coord, so `x` reads fine
 *  and any non-lowerable expression is flagged — then ADDS the two mapper-specific rules the map-kernel gate
 *  does not enforce: exact arity (1 param), and purity (a closed-over buffer/vec/helper is rejected). Carries
 *  the Phase-4 fix forward: a bin-mapper's one param is a SCALAR, so indexing it (`x[i]`) is meaningless and
 *  rejected (the shared map-kernel gate accepts a scalar index — harmless on the CPU but a GPU compile error).
 *
 *  NOT gated here: the RANGE of the returned bin index. An out-of-range index (< 0 or >= bins) is DROPPED at
 *  scatter time (both CPU + WGSL agree via a bounds guard), not a gate error — the mapper's output range is
 *  data-dependent and not statically decidable. */
export function gateBinMapper(binMapper: UserFn, host: ReactiveHost): GateVerdict {
  const reasons: Diagnostic[] = [];
  const span = binMapper.body[0]?.span;
  // A bin-mapper takes EXACTLY 1 name param (x) — a scalar. Not 1 name param (or a destructuring param) →
  // not a 1-arg bin index function.
  const nameParams = binMapper.params.filter((p) => p.kind === 'name');
  if (binMapper.params.length !== 1 || nameParams.length !== 1) {
    reasons.push(makeDiagnostic('MLGPU-BINMAPPER-ARITY', `a bin-mapper must take exactly 1 parameter (x) — got ${binMapper.params.length}`, span));
  }
  // Reuse the map-kernel gate for the body's lowerability + a SCALAR (comps=1) output-shape check: the
  // bin-mapper body is scalar arithmetic/comparison/ternary + the math builtins — exactly the lowerable-scalar
  // surface gateKernel validates. Its 1 param binds as a coord (a scalar), so `x` reads fine and any buffer/
  // helper/vec/string/object use is flagged with the existing MLGPU-* reasons.
  const v = gateKernel(binMapper, host, 1);
  reasons.push(...v.reasons);
  // A bin-mapper must be PURE over its one scalar param. gateKernel ACCEPTS a valid buffer index (`x + buf[0]`)
  // and a scalar/vec uniform — legal in a map kernel — but a bin-mapper's only input is `x`, so a closed-over
  // buffer/vec/mat/helper has no place in the mapping. Reject any such binding so the mapper gate is strictly
  // narrower than the map-kernel gate. (A closed-over scalar CONSTANT — `role:'scalar'` — is allowed: it is a
  // uniform of the mapping, resolved by the interpreter in cpuHistogram and lowerable later.)
  for (const b of v.bindings.byName.values()) {
    if (b.role === 'buffer' || b.role === 'uniform' || b.role === 'callee') {
      const what = b.role === 'callee' ? 'helper function' : b.role === 'uniform' ? 'vec/mat' : 'buffer';
      reasons.push(makeDiagnostic('MLGPU-NOT-LOWERABLE', `a bin-mapper must be pure over its one parameter (x) — it may not reference a ${what} ('${b.name}')`, span));
    }
  }
  // A bin-mapper's param is a SCALAR — indexing it (`x[i]`) is meaningless. The shared map-kernel gate accepts
  // it (it only refuses to descend into a BUFFER ident, not a scalar/coord param index): harmless on the CPU
  // (the interpreter returns null → 0) but the GPU emit would emit an index on a non-buffer, a compile error.
  // Reject a mapper whose body indexes its param (carrying the Phase-4 reducer fix forward).
  const paramNames = new Set(nameParams.map((p) => (p as Extract<typeof p, { kind: 'name' }>).name));
  if (indexesAnyParam(binMapper.body, paramNames)) {
    reasons.push(makeDiagnostic('MLGPU-NOT-LOWERABLE', 'a bin-mapper parameter is a scalar and cannot be indexed', span));
  }
  return { core: reasons.length === 0, reasons, bindings: v.bindings };
}

/** True iff any `index` expression in the mapper body has its scalar param as its object (`x[...]`) — a
 *  meaningless scalar index the shared map-kernel gate does not catch. A shallow structural walk over the
 *  mapper body (which is scalar arithmetic/comparison/ternary/builtins by the gate). */
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

/** The CPU histogram — the correctness ORACLE floor. For each input element, compute its bin index via the
 *  bin-mapper (evaluated through the shipped interpreter via makeCallable, so the CPU count is
 *  identical-by-construction to the interpreter — the same descriptor handlers + arithmetic coercion the
 *  emitters delegate to) and increment `counts[binIndex]`. A FRESH callable per element gives each mapper
 *  invocation its own budget (mirrors cpuReduce's per-element callable — makeCallable's Runner budget is
 *  aggregate across calls of one callable). `Math.trunc(Number(...))` mirrors the interpreter's numeric
 *  coercion + the WGSL `i32(_binOf(x))` truncation, so the CPU count agrees with the WGSL scatter.
 *
 *  OUT-OF-RANGE DROP: a bin index < 0, >= bins, or non-finite (a /0 → null → 0 is finite; a NaN would come
 *  from an out-of-domain builtin) is NOT counted — the standard histogram bounds behavior. The WGSL scatter
 *  drops the same way (`if (_b >= 0 && _b < bins) atomicAdd(...)`), so both backends agree. This is the
 *  ORACLE the (adapter-gated) WGSL scatter is verified against. */
export function cpuHistogram(binMapper: UserFn, inputValues: readonly number[], bins: number, host: ReactiveHost): number[] {
  const counts = new Array(bins).fill(0);
  const declineEnv: HostEnvironment = { resolveCall: () => ({ handled: false }) };
  for (const x of inputValues) {
    const call = makeCallable(binMapper, { host, env: declineEnv });   // fresh budget per element
    const b = Math.trunc(Number(call(x)));
    if (Number.isFinite(b) && b >= 0 && b < bins) counts[b] += 1;   // drop out-of-range / non-finite bins
  }
  return counts;
}
