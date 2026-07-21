// The numeric builtin module — one Builtin per numeric name, injected by a consumer at evaluateProgram.
// Each `invoke` translates a call into the capability-context API: it evaluates args ONCE (binding each to
// a local, since ctx.evalArg re-evaluates), delegates the arithmetic to @metael/math core, boxes the result
// through the vec/mat/buffer descriptors, and applies the fail-loud domain-guard diagnostics the language
// surface expects (core returns raw NaN; the binding raises the loud diagnostic in scalar position only).
import type { Builtin, BuiltinModule, BuiltinCtx, LowerElement } from '@metael/lang';
import { descriptorOf, isUserFn } from '@metael/lang';
import { makeVec, makeMat, identityMat, vecStoreOf } from './descriptors.ts';
import { makeTypedArray, BUFFER_KINDS } from './buffers.ts';
import { BUILTINS } from './registry-data.ts';
import * as core from '@metael/math';

// A typed-array construction cap: 2^24 elements. Over this, construction fails closed with ML-LANG-BUDGET.
const MAX_BUFFER_LENGTH = 16_777_216;

// ─────────────────────────────────────────── shared helpers ───────────────────────────────────────────

/** Numeric coercion mirroring the interpreter's toNum: number→v; boolean→0/1; trimmed string→Number|NaN;
 *  anything else→NaN. */
function toNum(v: unknown): number {
  if (typeof v === 'number') return v;
  if (typeof v === 'boolean') return v ? 1 : 0;
  if (typeof v === 'string') { const t = v.trim(); return t === '' ? NaN : Number(t); }
  return NaN;
}
/** Coerce an arg to a finite/coercible number, or null if not (a NaN or a non-coercible value). */
function num(v: unknown): number | null {
  if (typeof v === 'number') return Number.isNaN(v) ? null : v;
  const n = toNum(v);
  return Number.isNaN(n) ? null : n;
}
/** A strict numeric read: a real (non-NaN) number passes through, anything else → null (no coercion of
 *  strings/booleans). Used for the geometric/quaternion scalar slots that only ever accept a literal number. */
function strictNum(v: unknown): number | null {
  return (typeof v === 'number' && !Number.isNaN(v)) ? v : null;
}
/** The components of a value IFF it is a vecN (single-column), else null. */
function vecComps(v: unknown): number[] | null {
  const d = descriptorOf(v);
  if (!d || d.lower?.shape !== 'vecN') return null;
  return Array.from(vecStoreOf(v).c);
}
/** True iff the value is a vecN (single-column) custom value. */
function isVecValue(v: unknown): boolean {
  return descriptorOf(v)?.lower?.shape === 'vecN';
}
/** The result precision for a computed value: f64 if ANY vec/mat operand is f64, else f32. */
function elemOfArgs(args: readonly unknown[]): LowerElement {
  for (const a of args) if (descriptorOf(a)?.lower?.element === 'f64') return 'f64';
  return 'f32';
}
/** Read all supplied call args ONCE (ctx.evalArg re-evaluates), matching the old single-pass arg map. */
function evalAll(ctx: BuiltinCtx): unknown[] {
  const n = ctx.argCount();
  const out = new Array<unknown>(n);
  for (let i = 0; i < n; i++) out[i] = ctx.evalArg(i);
  return out;
}

// ─────────────────────────────────────────── typed-array constructors ───────────────────────────────────────────

function makeBufferBuiltin(kind: 'f32' | 'f64' | 'i32' | 'u32'): Builtin {
  return {
    spec: BUILTINS[kind]!,
    invoke: (ctx) => {
      ctx.tick();
      const spec = BUFFER_KINDS[kind];
      const a0 = ctx.evalArg(0);   // null when no arg (evalArg substitutes a null literal)
      let store: { [i: number]: number; length: number };
      if (typeof a0 === 'number') {
        const n = Math.floor(a0);
        if (!Number.isFinite(n) || n < 0) { ctx.error('ML-LANG-BUILTIN-ARG', `${kind}(n) — n must be a non-negative number`); return ctx.freeze([]); }
        if (n > MAX_BUFFER_LENGTH) { ctx.error('ML-LANG-BUDGET', `${kind}(${n}) exceeds the ${MAX_BUFFER_LENGTH}-element cap`); return ctx.freeze([]); }
        store = new spec.ctor(n);
        const genFn = ctx.argCount() > 1 ? ctx.evalArg(1) : undefined;
        if (genFn !== undefined) {
          if (typeof genFn !== 'function' && !isUserFn(genFn)) { ctx.error('ML-LANG-BUILTIN-ARG', `${kind}(n, fn) — the second argument must be a function`); return ctx.freeze([]); }
          for (let i = 0; i < n; i++) { ctx.tick(); const val = ctx.callClosure(genFn, [i]); store[i] = spec.coerce(typeof val === 'number' ? val : NaN); }
        }
      } else if (Array.isArray(a0)) {
        if (a0.length > MAX_BUFFER_LENGTH) { ctx.error('ML-LANG-BUDGET', `${kind}([…]) exceeds the ${MAX_BUFFER_LENGTH}-element cap`); return ctx.freeze([]); }
        store = new spec.ctor(a0.length);
        for (let i = 0; i < a0.length; i++) { ctx.tick(); const val = a0[i]; store[i] = spec.coerce(typeof val === 'number' ? val : NaN); }
      } else {
        ctx.error('ML-LANG-BUILTIN-ARG', `${kind}(n | […] | (n, fn)) — bad argument`);
        return ctx.freeze([]);
      }
      const gen = ctx.allocateGeneration();
      return makeTypedArray(kind, store, gen);
    },
  };
}

// ─────────────────────────────────────────── vec/mat constructors (with §GLSL/WGSL composition) ───────────────────────────────────────────

function makeVecBuiltin(head: 'vec2' | 'vec3' | 'vec4', N: number): Builtin {
  return {
    spec: BUILTINS[head]!,
    invoke: (ctx) => {
      ctx.tick();
      const args = evalAll(ctx);
      // A single number splats to N copies (f32).
      if (args.length === 1 && typeof args[0] === 'number') return makeVec(new Array<number>(N).fill(args[0]));
      // Composition: each arg is a number or a vecM; flatten left-to-right; the total must be exactly N.
      const flat: number[] = [];
      let f64 = false;
      for (const a of args) {
        if (typeof a === 'number') { flat.push(a); continue; }
        if (isVecValue(a)) { const s = vecStoreOf(a); for (let i = 0; i < s.c.length; i++) flat.push(s.c[i] as number); if (s.element === 'f64') f64 = true; continue; }
        ctx.error('ML-LANG-BUILTIN-ARG', `${head}(...) — arguments must be numbers or vectors`); return ctx.freeze([]);
      }
      if (flat.length !== N) { ctx.error('ML-LANG-BUILTIN-ARG', `${head}(...) — components must total ${N}`); return ctx.freeze([]); }
      return makeVec(flat, f64 ? 'f64' : 'f32');
    },
  };
}

/** Build a matrix builtin for a rows×cols shape. Square shapes (rows===cols) additionally accept the
 *  zero-arg identity form. All shapes accept EITHER `rows*cols` numbers (column-major) OR exactly `cols`
 *  column vectors (each a vec of width `rows`). */
function makeMatBuiltin(head: string, rows: number, cols: number, allowIdentity: boolean): Builtin {
  const total = rows * cols;
  return {
    spec: BUILTINS[head]!,
    invoke: (ctx) => {
      ctx.tick();
      const args = evalAll(ctx);
      if (args.length === 0 && allowIdentity) return makeMat(identityMat(rows), rows, cols);
      // Column-vector form: exactly `cols` args, each a vec of width `rows` (column-major layout).
      if (args.length === cols && args.every((a) => isVecValue(a) && vecStoreOf(a).rows === rows)) {
        const flat: number[] = [];
        let f64 = false;
        for (const a of args) { const s = vecStoreOf(a); for (let i = 0; i < s.c.length; i++) flat.push(s.c[i] as number); if (s.element === 'f64') f64 = true; }
        return makeMat(flat, rows, cols, f64 ? 'f64' : 'f32');
      }
      // All-numbers form (column-major).
      if (args.every((x) => typeof x === 'number') && args.length === total) return makeMat(args as number[], rows, cols);
      ctx.error('ML-LANG-BUILTIN-ARG', allowIdentity
        ? `${head}() (identity), ${head}(${total} numbers, column-major), or ${head}(${cols} column vectors)`
        : `${head}(${total} numbers, column-major) or ${head}(${cols} column vectors)`);
      return ctx.freeze([]);
    },
  };
}

// ─────────────────────────────────────────── matrix ops ───────────────────────────────────────────

const transposeBuiltin: Builtin = {
  spec: BUILTINS.transpose!,
  invoke: (ctx) => {
    ctx.tick();
    const m = ctx.evalArg(0);
    const d = descriptorOf(m);
    if (!d || d.lower?.shape !== 'matMxN') { ctx.error('ML-LANG-BUILTIN-ARG', 'transpose(mat)'); return ctx.freeze([]); }
    const s = vecStoreOf(m);
    return makeMat(core.transpose(s.c as number[], s.rows, s.cols), s.cols, s.rows, s.element);
  },
};

const determinantBuiltin: Builtin = {
  spec: BUILTINS.determinant!,
  invoke: (ctx) => {
    ctx.tick();
    const m = ctx.evalArg(0);
    const d = descriptorOf(m);
    const s = d?.lower?.shape === 'matMxN' ? vecStoreOf(m) : null;
    if (!s || s.rows !== s.cols) { ctx.error('ML-LANG-BUILTIN-ARG', 'determinant(square mat)'); return ctx.freeze([]); }
    return core.determinant(s.c as number[], s.rows);
  },
};

const inverseBuiltin: Builtin = {
  spec: BUILTINS.inverse!,
  invoke: (ctx) => {
    ctx.tick();
    const m = ctx.evalArg(0);
    const d = descriptorOf(m);
    const s = d?.lower?.shape === 'matMxN' ? vecStoreOf(m) : null;
    if (!s || s.rows !== s.cols) { ctx.error('ML-LANG-BUILTIN-ARG', 'inverse(square mat)'); return ctx.freeze([]); }
    return makeMat(core.inverse(s.c as number[], s.rows), s.rows, s.rows, s.element);
  },
};

const matrixCompMultBuiltin: Builtin = {
  spec: BUILTINS.matrixCompMult!,
  invoke: (ctx) => {
    ctx.tick();
    const a = ctx.evalArg(0); const b = ctx.evalArg(1);
    const da = descriptorOf(a); const db = descriptorOf(b);
    const sa = da?.lower?.shape === 'matMxN' ? vecStoreOf(a) : null;
    const sb = db?.lower?.shape === 'matMxN' ? vecStoreOf(b) : null;
    if (!sa || !sb || sa.rows !== sb.rows || sa.cols !== sb.cols) { ctx.error('ML-LANG-BUILTIN-ARG', 'matrixCompMult(mat, mat) — matrices must be the same shape'); return ctx.freeze([]); }
    return makeMat(core.matrixCompMult(sa.c as number[], sb.c as number[]), sa.rows, sa.cols, sa.element === 'f64' || sb.element === 'f64' ? 'f64' : 'f32');
  },
};

const qmatBuiltin: Builtin = {
  spec: BUILTINS.qmat!,
  invoke: (ctx) => {
    ctx.tick();
    const q = ctx.evalArg(0);
    const d = descriptorOf(q);
    if (!d || d.lower?.shape !== 'vecN' || (d.lower.rows ?? 0) !== 4) { ctx.error('ML-LANG-BUILTIN-ARG', 'qmat(vec4 quaternion)'); return ctx.freeze([]); }
    const s = vecStoreOf(q);
    return makeMat(core.qmat(s.c as number[]), 3, 3, s.element);
  },
};

// ─────────────────────────────────────────── vec/quat geometric builtins ───────────────────────────────────────────

/** One builtin per geometric vec/quat op. Each reads its args once, validates vector shapes, delegates to
 *  core, and boxes a vec result (f64 iff any vec arg is f64). A bad-shape arg → ML-LANG-BUILTIN-ARG + []. */
const geometricBuiltins: Builtin[] = (() => {
  const names = ['dot', 'cross', 'normalize', 'length', 'distance', 'reflect', 'refract', 'faceforward',
    'qmul', 'qconj', 'qinvert', 'qaxisangle', 'qrotate', 'qslerp'] as const;
  return names.map<Builtin>((head) => ({
    spec: BUILTINS[head]!,
    invoke: (ctx) => {
      ctx.tick();
      const args = evalAll(ctx);
      const elem = elemOfArgs(args);
      const badVec = (): unknown => { ctx.error('ML-LANG-BUILTIN-ARG', `${head}(vec…) — argument must be a vector`); return ctx.freeze([]); };
      switch (head) {
        case 'dot': { const x = vecComps(args[0]); const y = vecComps(args[1]); if (!x || !y || x.length !== y.length) return badVec(); return core.dot(x, y); }
        case 'cross': { const x = vecComps(args[0]); const y = vecComps(args[1]); if (!x || !y || x.length !== 3 || y.length !== 3) return badVec(); return makeVec(core.cross(x, y), elem); }
        case 'length': { const x = vecComps(args[0]); if (!x) return badVec(); return core.length(x); }
        case 'distance': { const x = vecComps(args[0]); const y = vecComps(args[1]); if (!x || !y || x.length !== y.length) return badVec(); return core.distance(x, y); }
        case 'reflect': { const I = vecComps(args[0]); const N = vecComps(args[1]); if (!I || !N || I.length !== N.length) return badVec(); return makeVec(core.reflect(I, N), elem); }
        case 'refract': { const I = vecComps(args[0]); const N = vecComps(args[1]); const eta = strictNum(args[2]); if (!I || !N || eta === null || I.length !== N.length) return badVec(); return makeVec(core.refract(I, N, eta), elem); }
        case 'faceforward': { const N = vecComps(args[0]); const I = vecComps(args[1]); const Nref = vecComps(args[2]); if (!N || !I || !Nref || N.length !== I.length || N.length !== Nref.length) return badVec(); return makeVec(core.faceforward(N, I, Nref), elem); }
        case 'qconj': { const q = vecComps(args[0]); if (!q || q.length !== 4) return badVec(); return makeVec(core.qconj(q), elem); }
        case 'qinvert': { const q = vecComps(args[0]); if (!q || q.length !== 4) return badVec(); return makeVec(core.qinvert(q), elem); }
        case 'qmul': { const A = vecComps(args[0]); const B = vecComps(args[1]); if (!A || A.length !== 4 || !B || B.length !== 4) return badVec(); return makeVec(core.qmul(A, B), elem); }
        case 'qaxisangle': { const ax = vecComps(args[0]); const ang = strictNum(args[1]); if (!ax || ax.length !== 3 || ang === null) return badVec(); return makeVec(core.qaxisangle(ax, ang), elem); }
        case 'qrotate': { const q = vecComps(args[0]); const v = vecComps(args[1]); if (!q || q.length !== 4 || !v || v.length !== 3) return badVec(); return makeVec(core.qrotate(q, v), elem); }
        case 'qslerp': { const A = vecComps(args[0]); const B = vecComps(args[1]); const t = strictNum(args[2]); if (!A || A.length !== 4 || !B || B.length !== 4 || t === null) return badVec(); return makeVec(core.qslerp(A, B, t), elem); }
        case 'normalize': default: { const x = vecComps(args[0]); if (!x) return badVec(); return makeVec(core.normalize(x), elem); }
      }
    },
  }));
})();

// ─────────────────────────────────────────── affine transform / camera builtins ───────────────────────────────────────────

const transformBuiltins: Builtin[] = [
  {
    spec: BUILTINS.transformation!,
    invoke: (ctx) => {
      ctx.tick();
      const args = evalAll(ctx);
      const t = vecComps(args[0]); const r = vecComps(args[1]); const s = vecComps(args[2]);
      if (!t || t.length !== 3 || !r || r.length !== 4 || !s || s.length !== 3) { ctx.error('ML-LANG-BUILTIN-ARG', 'transformation(t:vec3, r:vec4, s:vec3)'); return ctx.freeze([]); }
      return makeMat(core.transformation(t, r, s), 4, 4, elemOfArgs(args));
    },
  },
  {
    spec: BUILTINS.decompose!,
    invoke: (ctx) => {
      ctx.tick();
      const m = ctx.evalArg(0);
      const d = descriptorOf(m);
      const s = d?.lower?.shape === 'matMxN' ? vecStoreOf(m) : null;
      if (!s || s.rows !== 4 || s.cols !== 4) { ctx.error('ML-LANG-BUILTIN-ARG', 'decompose(mat4)'); return ctx.freeze([]); }
      const { t, r, sc } = ((): { t: number[]; r: number[]; sc: number[] } => { const dec = core.decompose(s.c as number[]); return { t: dec.t, r: dec.r, sc: dec.s }; })();
      return ctx.freeze({ t: makeVec(t, s.element), r: makeVec(r, s.element), s: makeVec(sc, s.element) });
    },
  },
  {
    spec: BUILTINS.perspective!,
    invoke: (ctx) => {
      ctx.tick();
      const args = evalAll(ctx);
      const fovy = num(args[0]); const aspect = num(args[1]); const near = num(args[2]); const far = num(args[3]);
      if (fovy === null || aspect === null || near === null || far === null) { ctx.error('ML-LANG-BUILTIN-ARG', 'perspective(fovy, aspect, near, far) — numeric arguments'); return ctx.freeze([]); }
      return makeMat(core.perspective(fovy, aspect, near, far), 4, 4);
    },
  },
  {
    spec: BUILTINS.ortho!,
    invoke: (ctx) => {
      ctx.tick();
      const args = evalAll(ctx);
      const xs = [num(args[0]), num(args[1]), num(args[2]), num(args[3]), num(args[4]), num(args[5])];
      if (xs.some((x) => x === null)) { ctx.error('ML-LANG-BUILTIN-ARG', 'ortho(left, right, bottom, top, near, far) — numeric arguments'); return ctx.freeze([]); }
      const [l, r, b, t, n, f] = xs as number[];
      return makeMat(core.ortho(l!, r!, b!, t!, n!, f!), 4, 4);
    },
  },
  {
    spec: BUILTINS.lookAt!,
    invoke: (ctx) => {
      ctx.tick();
      const args = evalAll(ctx);
      const eye = vecComps(args[0]); const center = vecComps(args[1]); const up = vecComps(args[2]);
      if (!eye || eye.length !== 3 || !center || center.length !== 3 || !up || up.length !== 3) { ctx.error('ML-LANG-BUILTIN-ARG', 'lookAt(eye:vec3, center:vec3, up:vec3)'); return ctx.freeze([]); }
      return makeMat(core.lookAt(eye, center, up), 4, 4, elemOfArgs(args));
    },
  },
];

// ─────────────────────────────────────────── scalar math (with GLSL componentwise-vec promotion) ───────────────────────────────────────────

/** The pure scalar math for the numeric builtins, over already-coerced numbers. Domain-guarded funcs
 *  (sqrt/asin/acos/log/log2/inverseSqrt) return raw NaN out-of-domain — the scalar-position fail-loud
 *  diagnostic is layered on in the invoke; the vec-componentwise path keeps the NaN component. */
const SCALAR: Readonly<Record<string, (xs: readonly number[]) => number>> = {
  min: (xs) => core.min(xs[0]!, xs[1]!),
  max: (xs) => core.max(xs[0]!, xs[1]!),
  abs: (xs) => core.abs(xs[0]!),
  sign: (xs) => core.sign(xs[0]!),
  floor: (xs) => core.floor(xs[0]!),
  ceil: (xs) => core.ceil(xs[0]!),
  round: (xs) => core.round(xs[0]!),
  clamp: (xs) => core.clamp(xs[0]!, xs[1]!, xs[2]!),
  sqrt: (xs) => core.sqrt(xs[0]!),
  pow: (xs) => core.pow(xs[0]!, xs[1]!),
  mod: (xs) => core.mod(xs[0]!, xs[1]!),
  sin: (xs) => core.sin(xs[0]!),
  cos: (xs) => core.cos(xs[0]!),
  tan: (xs) => core.tan(xs[0]!),
  sinh: (xs) => core.sinh(xs[0]!),
  cosh: (xs) => core.cosh(xs[0]!),
  tanh: (xs) => core.tanh(xs[0]!),
  asin: (xs) => core.asin(xs[0]!),
  acos: (xs) => core.acos(xs[0]!),
  atan: (xs) => core.atan(xs[0]!),
  atan2: (xs) => core.atan2(xs[0]!, xs[1]!),
  asinh: (xs) => core.asinh(xs[0]!),
  // acosh (domain x>=1) and atanh (domain |x|<1) return raw NaN out-of-domain — NO fail-loud guard (unlike
  // asin/acos/log below). This is deliberate: these lower to the native shader asinh/acosh/atanh, which also
  // yield NaN out-of-domain, so returning NaN here (not a loud diagnostic + 0) is what keeps the interpreter
  // oracle identical to the GPU result. asinh has no restricted domain at all.
  acosh: (xs) => core.acosh(xs[0]!),
  atanh: (xs) => core.atanh(xs[0]!),
  exp: (xs) => core.exp(xs[0]!),
  exp2: (xs) => core.exp2(xs[0]!),
  log: (xs) => core.log(xs[0]!),
  log2: (xs) => core.log2(xs[0]!),
  inverseSqrt: (xs) => core.inverseSqrt(xs[0]!),
  degrees: (xs) => core.degrees(xs[0]!),
  radians: (xs) => core.radians(xs[0]!),
  trunc: (xs) => core.trunc(xs[0]!),
  fract: (xs) => core.fract(xs[0]!),
  step: (xs) => core.step(xs[0]!, xs[1]!),
  mix: (xs) => core.mix(xs[0]!, xs[1]!, xs[2]!),
  smoothstep: (xs) => core.smoothstep(xs[0]!, xs[1]!, xs[2]!),
  countOneBits: (xs) => core.countOneBits(xs[0]!),
  reverseBits: (xs) => core.reverseBits(xs[0]!),
};

const SCALAR_NAMES = Object.keys(SCALAR);

function makeScalarBuiltin(head: string): Builtin {
  const spec = BUILTINS[head]!;
  const arity = spec.arity[0];
  const fn = SCALAR[head]!;
  return {
    spec,
    invoke: (ctx) => {
      ctx.tick();
      const args = evalAll(ctx);
      const bad = (): unknown => { ctx.error('ML-LANG-BUILTIN-ARG', `${head}(number, …) — non-numeric argument`); return ctx.freeze([]); };
      const relevant = args.slice(0, arity);
      // Componentwise vec application (GLSL semantics): if any arity-relevant arg is a vecN, map the scalar
      // op over its components, broadcasting a plain scalar to every component. All vec args in that prefix
      // must share ONE width (a mismatch is fail-loud — the interpreter oracle never truncates). A NaN
      // component (out-of-domain, non-numeric broadcast) is KEPT — native shaders never abort the vector.
      const vecArgs = relevant.map(vecComps).filter((c): c is number[] => c !== null);
      if (vecArgs.length > 0) {
        const width = vecArgs[0]!.length;
        if (vecArgs.some((c) => c.length !== width)) { ctx.error('ML-LANG-BUILTIN-ARG', `${head}(vec…) — vector arguments must be the same width`); return ctx.freeze([]); }
        const scalarAt = (arg: unknown, i: number): number => { const c = vecComps(arg); return c ? (c[i] ?? NaN) : (num(arg) ?? NaN); };
        const out: number[] = [];
        for (let i = 0; i < width; i++) {
          const xs: number[] = [];
          for (let k = 0; k < arity; k++) xs.push(scalarAt(relevant[k], i));
          out.push(fn(xs));
        }
        return makeVec(out, elemOfArgs(relevant));
      }
      // Scalar path: coerce each needed arg (null → fail loud), intercept the domain-restricted funcs for
      // their specific diagnostics, then compute.
      const xs: number[] = [];
      for (let k = 0; k < arity; k++) { const v = num(args[k]); if (v === null) return bad(); xs.push(v); }
      if (head === 'sqrt' && xs[0]! < 0) { ctx.error('ML-LANG-BUILTIN-ARG', `sqrt(x) — x must be >= 0`); return ctx.freeze([]); }
      if (head === 'log' && xs[0]! <= 0) { ctx.error('ML-LANG-BUILTIN-ARG', `log(x) — x must be > 0`); return ctx.freeze([]); }
      if ((head === 'asin' || head === 'acos') && (xs[0]! < -1 || xs[0]! > 1)) { ctx.error('ML-LANG-BUILTIN-ARG', `${head}(x) — x must be in [-1, 1]`); return ctx.freeze([]); }
      if ((head === 'log2' || head === 'inverseSqrt') && xs[0]! <= 0) { ctx.error('ML-LANG-BUILTIN-ARG', `${head}(x) — x must be > 0`); return ctx.freeze([]); }
      return fn(xs);
    },
  };
}

// ─────────────────────────────────────────── the module ───────────────────────────────────────────

const builtins: Builtin[] = [
  makeBufferBuiltin('f32'), makeBufferBuiltin('f64'), makeBufferBuiltin('i32'), makeBufferBuiltin('u32'),
  makeVecBuiltin('vec2', 2), makeVecBuiltin('vec3', 3), makeVecBuiltin('vec4', 4),
  makeMatBuiltin('mat2', 2, 2, true), makeMatBuiltin('mat3', 3, 3, true), makeMatBuiltin('mat4', 4, 4, true),
  // Non-square matCxR: name matCxR → C columns × R rows, so [rows, cols] = [R, C].
  makeMatBuiltin('mat2x3', 3, 2, false), makeMatBuiltin('mat2x4', 4, 2, false), makeMatBuiltin('mat3x2', 2, 3, false),
  makeMatBuiltin('mat3x4', 4, 3, false), makeMatBuiltin('mat4x2', 2, 4, false), makeMatBuiltin('mat4x3', 3, 4, false),
  transposeBuiltin, determinantBuiltin, inverseBuiltin, matrixCompMultBuiltin, qmatBuiltin,
  ...geometricBuiltins,
  ...transformBuiltins,
  ...SCALAR_NAMES.map(makeScalarBuiltin),
];

/** The numeric standard-library module: every numeric builtin (constructors, vec/mat/quat ops, transforms,
 *  scalar math, bit ops). A consumer injects it via `evaluateProgram(src, { …, builtins: [MATH_BUILTINS] })`. */
export const MATH_BUILTINS: BuiltinModule = { builtins };
