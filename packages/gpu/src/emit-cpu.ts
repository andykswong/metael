// The CPU emitter: the kernel AST → a JS closure computing output[coord]. Eval-free (no new Function).
// Buffer/vec element access + operators DELEGATE to each value's descriptor handlers (the SAME getIndex/
// binary the interpreter oracle runs) → the CPU path and the oracle are identical by construction. Scalar
// arithmetic mirrors the interpreter (toNum coercion incl. boolean→0/1, /0 → null, %0 → null, loose-equals
// for ==/!=), so a dividing/comparing kernel stays bit-identical to the oracle.
import type { UserFn, Expr, Stmt, ReactiveHost, TypeDescriptor, HostEnvironment } from '@metael/lang';
import { descriptorOf, NOT_HANDLED, MAX_RANGE, makeCallable } from '@metael/lang';
import { MATH_BUILTINS } from '@metael/math/lang';
import * as core from '@metael/math';
import type { Binding, BindingTable } from './binding.ts';
import { bodyReferencesAny } from './binding.ts';

// The vec/mat + numeric builtin names. A kernel whose body references ANY of these is delegated WHOLE
// to the interpreter — authoritative for all vec math. The vec/mat constructors + dot/cross/normalize/
// length build a tagged vec/mat value the hand-walk cannot construct (makeVec/makeMat are lang-internal).
// The numeric builtins (min/abs/clamp/sqrt/… — GLSL-componentwise) apply over a vec arg's components,
// which the hand-walk's scalar applyBuiltin cannot; the interpreter maps them componentwise, so a kernel
// using any of them delegates. (A scalar-only use also delegates — harmless: the delegate IS the oracle.)
const VEC_NAMES = new Set(['vec2', 'vec3', 'vec4', 'mat2', 'mat3', 'mat4', 'dot', 'cross', 'normalize', 'length',
  'transpose', 'determinant', 'inverse', 'mat2x3', 'mat2x4', 'mat3x2', 'mat3x4', 'mat4x2', 'mat4x3',
  'distance', 'reflect', 'refract', 'faceforward',
  'qmul', 'qconj', 'qinvert', 'qaxisangle', 'qrotate', 'qslerp', 'qmat',
  'min', 'max', 'abs', 'sign', 'floor', 'ceil', 'round', 'clamp', 'sqrt', 'pow', 'sin', 'cos', 'exp', 'log', 'fract', 'step', 'mix', 'smoothstep',
  'tan', 'sinh', 'cosh', 'tanh', 'asin', 'acos', 'atan', 'atan2', 'exp2', 'log2', 'inverseSqrt',
  'degrees', 'radians', 'trunc',
  // The componentwise SCALAR-map builtins that also promote over a vec arg (like sin/cos): a vec-arg use must
  // delegate WHOLE to the interpreter (the hand-walk's scalar applyBuiltin can't map over a vec's components).
  'mod', 'asinh', 'acosh', 'atanh', 'countOneBits', 'reverseBits']);

type Scope = Map<string, unknown>;

// The interpreter's bad-domain / non-coercible sentinel is deepFreeze([]): as a FINAL output it coerces
// via Number([]) = 0, but MID-ARITHMETIC it coerces via toNum([]) = NaN (a plain array has no descriptor).
// Returning this exact sentinel — not a bare 0 — keeps a bad builtin (sqrt(neg), log(<=0), a NaN/null arg)
// bit-identical to the interpreter both when it flows onward (…+1 → NaN) and when it is the cell value (→0).
const BAD: readonly unknown[] = Object.freeze([]);

/** Compile the kernel AST into an eval-free JS closure that computes one output cell from its thread
 *  coordinates. The returned function takes a cell's `coords` and returns its `comps` output components
 *  (a single-element array for a scalar output). Buffer/vec access + operators delegate to the value
 *  descriptors the interpreter oracle uses, so the CPU backend and the oracle are identical by construction;
 *  a vec/builtin-bearing kernel delegates WHOLE to the interpreter (authoritative for vec math). */
export function emitCpu(kernel: UserFn, bindings: BindingTable, host: ReactiveHost, comps = 1): (coords: readonly number[]) => number[] {
  // Extract a cell's `comps` components from an interpreter return value: a scalar wrapped as `[Number(v)]`
  // for comps=1; a vecN read component-wise (x,y,z,w) via the descriptor's getMember for comps>1 — the
  // interleaved layout every backend produces. Mirrors applyVecBuiltin's `comps(v)` helper.
  const getMemberComp = (v: unknown, prop: string): unknown => { const d = descriptorOf(v); if (!d?.getMember) return null; try { const res = d.getMember(v, prop); return res === NOT_HANDLED ? null : res; } catch { return null; } };
  const extractComps = (r: unknown): number[] => {
    if (comps === 1) return [Number(r)];
    const out: number[] = [];
    for (let k = 0; k < comps; k++) out.push(Number(getMemberComp(r, 'xyzw'[k] as string) ?? 0));
    return out;
  };
  // A vec/mat-bearing kernel: the hand-walk can't construct a tagged vec (makeVec is lang-internal), so
  // delegate the WHOLE cell to the interpreter (authoritative for vec math — cross/normalize included).
  // The scalar fast path (no vec/mat) keeps the hand-walk below. A FRESH makeCallable per cell so each cell
  // gets its own budget (makeCallable's Runner budget is aggregate across calls of one callable). For a
  // vecN output (comps>1) the interpreter returns a vec value → read its N components; for a scalar (comps=1)
  // the old `Number(r ?? 0)` collapse, now wrapped as `[value]` (byte-identical to the pre-vecN cell).
  if (bodyReferencesAny(kernel, VEC_NAMES)) {
    const declineEnv: HostEnvironment = { resolveCall: () => ({ handled: false }) };
    return (coords: readonly number[]): number[] => {
      const call = makeCallable(kernel, { host, env: declineEnv, maxSteps: 1_000_000, builtins: [MATH_BUILTINS] });
      const r = call(...coords);
      if (comps === 1) return [typeof r === 'number' ? r : Number(r ?? 0)];
      return extractComps(r);
    };
  }
  const callee = (name: string): UserFn | undefined => { const b = bindings.byName.get(name); return b?.role === 'callee' ? b.fn : undefined; };
  // Scalar coercion — the interpreter's toNum EXACTLY: number→v; boolean→(v?1:0) (a comparison result used
  // arithmetically, e.g. `(x[i] > 0) * mask`); string→trimmed Number|NaN; anything else (null, [], object)→NaN.
  const toNum = (v: unknown): number => {
    if (typeof v === 'number') return v;
    if (typeof v === 'boolean') return v ? 1 : 0;
    if (typeof v === 'string') { const t = v.trim(); return t === '' ? NaN : Number(t); }
    return NaN;
  };
  const looseEq = (l: unknown, r: unknown): boolean => {
    if (l === r) return true;
    if (l === null || l === undefined) return r === null || r === undefined;
    if (r === null || r === undefined) return false;
    if (typeof l === 'number' && typeof r === 'string') return l === toNum(r) && r.trim() !== '';
    if (typeof l === 'string' && typeof r === 'number') return toNum(l) === r && l.trim() !== '';
    return false;
  };
  const applyBinary = (op: string, l: unknown, r: unknown): unknown => {
    const dl = descriptorOf(l); const dr = descriptorOf(r);
    if (dl?.binary) { const res = dl.binary(op as never, l, r); if (res !== NOT_HANDLED) return res; }
    if (dr?.binary) { const res = dr.binary(op as never, l, r); if (res !== NOT_HANDLED) return res; }
    const a = toNum(l); const b = toNum(r);
    switch (op) {
      case '+': return a + b; case '-': return a - b; case '*': return a * b;
      case '/': return b === 0 ? null : a / b;
      case '%': return b === 0 ? null : a % b;
      case '<': return a < b; case '<=': return a <= b; case '>': return a > b; case '>=': return a >= b;
      case '==': return looseEq(l, r); case '!=': return !looseEq(l, r); default: return NaN;
    }
  };
  const truthyE = (v: unknown): boolean => { const d = descriptorOf(v); if (d?.truthy) return d.truthy(v); return !(v === false || v === null || v === undefined || v === 0 || v === '' || (typeof v === 'number' && Number.isNaN(v))); };
  const getIndexSafe = (obj: unknown, key: unknown): unknown => {
    const d = descriptorOf(obj);
    if (d?.getIndex) { try { const res = d.getIndex(obj, key as number); return res === NOT_HANDLED ? null : res; } catch { return null; } }
    // A PLAIN metael array buffer input (no descriptor): mirror the interpreter's readMember — a valid,
    // in-range integer index returns the element (?? null); anything else (OOB / non-integer) reads as null
    // (the interpreter errors → null), so this hand-walk stays IDENTICAL to the interpreter oracle.
    if (Array.isArray(obj) && typeof key === 'number' && Number.isInteger(key) && key >= 0 && key < obj.length) return obj[key] ?? null;
    return null;
  };
  const getMemberSafe = (obj: unknown, prop: string): unknown => {
    const d = descriptorOf(obj);
    if (d?.getMember) { try { const res = d.getMember(obj, prop); return res === NOT_HANDLED ? null : res; } catch { return null; } }
    // A PLAIN metael array buffer input: `.length` is the only lowerable member read (the gate allows just
    // `buffer.length` + vec swizzles); mirror the interpreter (the element count).
    if (Array.isArray(obj) && prop === 'length') return obj.length;
    return null;
  };

  const evalE = (e: Expr, scope: Scope): unknown => {
    switch (e.kind) {
      case 'number': return e.value; case 'bool': return e.value; case 'null': return null; case 'string': return e.value;
      case 'ident': { if (scope.has(e.name)) return scope.get(e.name); const b = bindings.byName.get(e.name); return b ? bindingValue(b) : null; }
      case 'index': return getIndexSafe(evalE(e.object, scope), evalE(e.index, scope));
      case 'member': return getMemberSafe(evalE(e.object, scope), e.property);
      case 'unary': { const x = evalE(e.operand, scope); if (e.op === '-') { const d = descriptorOf(x); if (d?.neg) return d.neg(x); return -toNum(x); } return !truthyE(x); }
      case 'binary': {
        if (e.op === '&&') { const l = evalE(e.left, scope); return truthyE(l) ? evalE(e.right, scope) : l; }
        if (e.op === '||') { const l = evalE(e.left, scope); return truthyE(l) ? l : evalE(e.right, scope); }
        return applyBinary(e.op, evalE(e.left, scope), evalE(e.right, scope));
      }
      case 'cond': return truthyE(evalE(e.test, scope)) ? evalE(e.then, scope) : evalE(e.else, scope);
      case 'call': return evalCall(e, scope);
      default: return null;
    }
  };
  const evalCall = (e: Extract<Expr, { kind: 'call' }>, scope: Scope): unknown => {
    const name = e.callee.kind === 'ident' ? e.callee.name : '';
    const args = e.args.map((a) => evalE(a, scope));
    const fn = callee(name);
    if (fn) return runBody(fn, args);
    return applyBuiltin(name, args);
  };
  const runBody = (fn: UserFn, args: readonly unknown[]): number => {
    const scope: Scope = new Map();
    fn.params.forEach((p, i) => { if (p.kind === 'name') scope.set(p.name, args[i] ?? null); });
    let ret: unknown = null;
    runInner(fn.body, scope, (v) => { ret = v; });
    return toNum(ret);
  };
  const execS = (s: Stmt, scope: Scope, ret: (v: unknown) => void): boolean => {
    switch (s.kind) {
      case 'const': case 'let': scope.set(s.name, evalE(s.init, scope)); return false;
      case 'assign': if (s.target.kind === 'ident') scope.set(s.target.name, evalE(s.value, scope)); return false;
      case 'expr': evalE(s.expr, scope); return false;
      case 'return': ret(s.value ? evalE(s.value, scope) : null); return true;
      case 'if': if (truthyE(evalE(s.test, scope))) return runInner(s.then, scope, ret); else if (s.else) return runInner(s.else, scope, ret); return false;
      case 'for': {
        // Match range(n): Math.floor(n) iterations, and 0 iterations when n < 0, n > MAX_RANGE, or n is
        // non-finite (range() short-circuits those to []). The gate guarantees `for … of range(EXPR)` with
        // at least one arg, so args[0] is present — the `?? NaN` only guards a malformed AST defensively
        // (a missing bound → NaN → 0 iterations, matching range()'s empty result).
        const boundExpr = (s.iterable as Extract<Expr, { kind: 'call' }>).args[0];
        const raw = boundExpr ? toNum(evalE(boundExpr, scope)) : NaN;
        const n = (!Number.isFinite(raw) || raw < 0 || raw > MAX_RANGE) ? 0 : Math.floor(raw);
        for (let i = 0; i < n; i++) { scope.set(s.binding, i); if (runInner(s.body, scope, ret)) return true; }
        return false;
      }
      default: return false;
    }
  };
  const runInner = (body: readonly Stmt[], scope: Scope, ret: (v: unknown) => void): boolean => { for (const s of body) if (execS(s, scope, ret)) return true; return false; };

  function bindingValue(b: Binding): unknown { return b.role === 'coord' ? 0 : b.role === 'callee' ? b.fn : (b as { value: unknown }).value; }
  // A numeric/transcendental builtin mirrors the interpreter's guards EXACTLY: any required arg that
  // coerces to a non-finite value (a NaN, or a null from a /0 or an OOB read) → the bad-domain sentinel
  // BAD (== the interpreter's deepFreeze([]): NaN mid-arithmetic, 0 as an output); `sqrt` also fails x<0,
  // `log` also fails x<=0. Never a raw Math.sqrt(-1)=NaN / Math.log(0)=-Infinity that would diverge.
  function applyBuiltin(name: string, a: readonly unknown[]): unknown {
    const n = (x: unknown): number => toNum(x);
    const guard = (...xs: number[]): boolean => xs.some((x) => Number.isNaN(x));   // a NaN/null-derived arg → interpreter bad()→[]
    switch (name) {
      case 'min': { const x = n(a[0]), y = n(a[1]); return guard(x, y) ? BAD : Math.min(x, y); }
      case 'max': { const x = n(a[0]), y = n(a[1]); return guard(x, y) ? BAD : Math.max(x, y); }
      case 'abs': { const x = n(a[0]); return guard(x) ? BAD : Math.abs(x); }
      case 'sign': { const x = n(a[0]); return guard(x) ? BAD : Math.sign(x); }
      case 'floor': { const x = n(a[0]); return guard(x) ? BAD : Math.floor(x); }
      case 'ceil': { const x = n(a[0]); return guard(x) ? BAD : Math.ceil(x); }
      case 'round': { const x = n(a[0]); if (guard(x)) return BAD; const r = Math.round(x); return (Math.abs(x % 1) === 0.5 && r % 2 !== 0) ? r - 1 : r; }
      case 'clamp': { const x = n(a[0]), lo = n(a[1]), hi = n(a[2]); return guard(x, lo, hi) ? BAD : Math.min(Math.max(x, lo), hi); }
      case 'sqrt': { const x = n(a[0]); return (guard(x) || x < 0) ? BAD : Math.sqrt(x); }
      case 'pow': { const x = n(a[0]), y = n(a[1]); return guard(x, y) ? BAD : Math.pow(x, y); }
      case 'sin': { const x = n(a[0]); return guard(x) ? BAD : Math.sin(x); }
      case 'cos': { const x = n(a[0]); return guard(x) ? BAD : Math.cos(x); }
      case 'tan': { const x = n(a[0]); return guard(x) ? BAD : Math.tan(x); }
      case 'sinh': { const x = n(a[0]); return guard(x) ? BAD : Math.sinh(x); }
      case 'cosh': { const x = n(a[0]); return guard(x) ? BAD : Math.cosh(x); }
      case 'tanh': { const x = n(a[0]); return guard(x) ? BAD : Math.tanh(x); }
      case 'asin': { const x = n(a[0]); return (guard(x) || x < -1 || x > 1) ? BAD : Math.asin(x); }
      case 'acos': { const x = n(a[0]); return (guard(x) || x < -1 || x > 1) ? BAD : Math.acos(x); }
      case 'atan': { const x = n(a[0]); return guard(x) ? BAD : Math.atan(x); }
      case 'atan2': { const y = n(a[0]), x = n(a[1]); return guard(y, x) ? BAD : Math.atan2(y, x); }
      // Inverse hyperbolics: raw Math.* result (NaN out-of-domain for acosh x<1 / atanh |x|>=1), matching the
      // interpreter's un-guarded scalar path + the native shader — a NaN-derived ARG still fails closed to BAD.
      case 'asinh': { const x = n(a[0]); return guard(x) ? BAD : Math.asinh(x); }
      case 'acosh': { const x = n(a[0]); return guard(x) ? BAD : Math.acosh(x); }
      case 'atanh': { const x = n(a[0]); return guard(x) ? BAD : Math.atanh(x); }
      // Floored modulo (sign follows the divisor — core.mod, NOT JS %). A NaN-derived arg → BAD.
      case 'mod': { const x = n(a[0]), y = n(a[1]); return guard(x, y) ? BAD : core.mod(x, y); }
      // Bit ops (32-bit unsigned): the CPU-emit leg of the tri-target lowering — CPU-emit ≡ interpreter by
      // delegating to the core impls, which treat the arg as x>>>0 = TRUNCATE toward zero then wrap mod 2^32
      // (matching the interpreter + the WGSL `bitcast<u32>(i32(x))` / GLSL `uint(int(x))` truncating coercion —
      // NOT a round). A NaN/null-derived arg fails closed to BAD (a 0 output), like the other scalar ops.
      case 'countOneBits': { const x = n(a[0]); return guard(x) ? BAD : core.countOneBits(x); }
      case 'reverseBits': { const x = n(a[0]); return guard(x) ? BAD : core.reverseBits(x); }
      case 'exp': { const x = n(a[0]); return guard(x) ? BAD : Math.exp(x); }
      case 'exp2': { const x = n(a[0]); return guard(x) ? BAD : Math.pow(2, x); }
      case 'log': { const x = n(a[0]); return (guard(x) || x <= 0) ? BAD : Math.log(x); }
      case 'log2': { const x = n(a[0]); return (guard(x) || x <= 0) ? BAD : Math.log2(x); }
      case 'inverseSqrt': { const x = n(a[0]); return (guard(x) || x <= 0) ? BAD : 1 / Math.sqrt(x); }
      case 'degrees': { const x = n(a[0]); return guard(x) ? BAD : x * 180 / Math.PI; }
      case 'radians': { const x = n(a[0]); return guard(x) ? BAD : x * Math.PI / 180; }
      case 'trunc': { const x = n(a[0]); return guard(x) ? BAD : Math.trunc(x); }
      case 'fract': { const x = n(a[0]); return guard(x) ? BAD : x - Math.floor(x); }
      case 'step': { const e = n(a[0]), x = n(a[1]); return guard(e, x) ? BAD : (x < e ? 0 : 1); }
      case 'mix': { const p = n(a[0]), q = n(a[1]), t = n(a[2]); return guard(p, q, t) ? BAD : p + (q - p) * t; }
      case 'smoothstep': { const e0 = n(a[0]), e1 = n(a[1]), x = n(a[2]); if (guard(e0, e1, x)) return BAD; const t = e1 === e0 ? 0 : Math.min(Math.max((x - e0) / (e1 - e0), 0), 1); return t * t * (3 - 2 * t); }
      case 'dot': case 'length': case 'cross': case 'normalize': return applyVecBuiltin(name, a);
      default: return NaN;
    }
  }
  function applyVecBuiltin(name: string, a: readonly unknown[]): unknown {
    const comps = (v: unknown): number[] => { const d: TypeDescriptor | undefined = descriptorOf(v); if (!d || d.lower?.shape !== 'vecN') return []; const nn = d.lower.rows ?? 0; const out: number[] = []; for (let i = 0; i < nn; i++) out.push(toNum(getMemberSafe(v, 'xyzw'[i] as string))); return out; };
    if (name === 'dot') { const x = comps(a[0]); const y = comps(a[1]); return x.reduce((s, xi, i) => s + xi * (y[i] ?? 0), 0); }
    if (name === 'length') { const x = comps(a[0]); return Math.sqrt(x.reduce((s, xi) => s + xi * xi, 0)); }
    // DEAD for a vec-bearing kernel: emitCpu delegates any body referencing a vec/mat name (vec2/3/4,
    // mat2/3/4, dot, cross, normalize, length) WHOLE to the interpreter — authoritative for all vec math,
    // including cross/normalize (which produce a vec intermediate the hand-walk can't build). This branch
    // is retained only defensively; the delegate returns before applyBuiltin/applyVecBuiltin is reached.
    return NaN;
  }

  return (coords) => {
    const scope: Scope = new Map();
    kernel.params.forEach((p, i) => { if (p.kind === 'name') scope.set(p.name, coords[i] ?? 0); });
    let ret: unknown = null;
    runInner(kernel.body, scope, (v) => { ret = v; });
    // The OUTPUT-buffer coercion — the value(s) written to output for this cell. It mirrors the DOWNSTREAM
    // coercion the interpreter oracle applies to a cell (Number(call(...))): a null (a /0 or %0 result) or
    // the BAD sentinel ([]) becomes 0, never NaN. (Intermediate arithmetic uses toNum, which is null/[]→NaN
    // like the interpreter's toNum — only this final buffer write coerces to a number, so CPU-emit ≡ oracle.)
    // This scalar hand-walk is reached only for a non-vec kernel; comps>1 kernels reference a vec name and
    // take the interpreter-delegate path above, so extractComps here just wraps the scalar as `[value]`.
    return extractComps(ret);
  };
}
