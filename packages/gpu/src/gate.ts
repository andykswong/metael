// The compute-lowerability gate: walk the kernel AST + the binding table and decide GPU-lowerability,
// reusing BUILTINS data + descriptorOf(v).lower. NOT classifyProfile (which rejects array index / for-of /
// member / every user call — it would reject every kernel this engine runs). Reasons are MLGPU-* +
// span-anchored. A buffer has no whole-value form in a shader — it is valid ONLY as `a[i]` or `a.length`;
// anywhere else it is flagged. Helper (callee) bodies are gated recursively against their own closures.
import type { UserFn, Expr, Stmt, Diagnostic, ReactiveHost } from '@metael/lang';
import { BUILTINS, makeDiagnostic } from '@metael/lang';
import { buildBindingTable, collectFreeNames } from './binding.ts';
import type { BindingTable } from './binding.ts';

export interface GateVerdict { readonly core: boolean; readonly reasons: Diagnostic[]; readonly bindings: BindingTable }

export function gateKernel(kernel: UserFn, host: ReactiveHost, comps = 1): GateVerdict {
  const reasons: Diagnostic[] = [];
  const visited = new Set<UserFn>();
  const bindings = gateFn(kernel, host, reasons, visited);
  // The output-shape check: a vecN output (comps>1) requires every `return` to yield a vecN of that width;
  // a scalar output (comps=1) requires every `return` to yield a scalar (no vecN). Structural + conservative
  // (an expr it can't prove is a vec reads as scalar width 0 → a vecN output over it is rejected, so the
  // user must return an obvious vecN). Applied ONLY to the kernel body — a helper's return isn't the output,
  // and a kernel that calls a helper is already rejected as non-lowerable above.
  checkOutputShape(kernel, comps, bindings, reasons);
  return { core: reasons.length === 0, reasons, bindings };
}

// A compare op yields a bool (scalar), never a vec — so `a[i] > 0` reads as width 0, not a vec operand.
const COMPARE = new Set(['==', '!=', '<', '<=', '>', '>=']);
// The ONLY swizzle chars the interpreter's vec descriptor evaluates (getMember indexes 'xyzw'). A color
// (.rgb) / texture (.stpq) swizzle returns NOT_HANDLED → 0 on the CPU oracle + CPU emitter while the
// WGSL/GLSL shaders compute the real components → a silent GPU-vs-CPU divergence. Only xyzw is lowerable.
const SWIZZLE_CHARS = new Set(['x', 'y', 'z', 'w']);
const isXyzwSwizzle = (prop: string): boolean => [...prop].every((ch) => SWIZZLE_CHARS.has(ch));
// The xyzw index order the interpreter's vec descriptor indexes (getMember maps 'xyzw'.indexOf(ch) → component).
// A char whose index is >= the source vec's width is OUT OF RANGE: the interpreter NOT_HANDLEs it → the CPU
// oracle + emitter coerce to 0 (a false-green verify), while WGSL/GLSL emit a spec-invalid swizzle (a real-
// adapter compile error). So an over-range swizzle must be rejected even though its chars are all xyzw.
const SWIZZLE_ORDER = 'xyzw';

// The matrix constructors, name → [rows, cols]. A matCxR builds C columns of R rows, so its shape is R×C.
// The square ctors (matN) are N×N; the six non-square ctors spell out both dimensions.
const MAT_CTORS: Readonly<Record<string, [number, number]>> = {
  mat2: [2, 2], mat3: [3, 3], mat4: [4, 4],
  mat2x3: [3, 2], mat2x4: [4, 2], mat3x2: [2, 3], mat3x4: [4, 3], mat4x2: [2, 4], mat4x3: [3, 4],
};
/** The matrix shape an expression evaluates to, or null if it is not a matrix. Recognizes the matrix
 *  constructors plus the matrix-returning ops (`transpose` swaps rows/cols; `inverse` keeps the shape;
 *  `qmat` — a quaternion → rotation matrix — is 3×3). Structural + conservative: an unknown call, or an op
 *  whose argument is not itself a matrix, is not a matrix here → null. */
export function matShapeOf(e: Expr): { rows: number; cols: number } | null {
  if (e.kind !== 'call' || e.callee.kind !== 'ident') return null;
  const n = e.callee.name;
  if (Object.hasOwn(MAT_CTORS, n)) { const [rows, cols] = MAT_CTORS[n]!; return { rows, cols }; }   // own-property check: a prototype-inherited callee name (constructor/toString/…) is NOT a ctor
  if (n === 'transpose') { const s = e.args[0] ? matShapeOf(e.args[0]) : null; return s ? { rows: s.cols, cols: s.rows } : null; }
  if (n === 'inverse') return e.args[0] ? matShapeOf(e.args[0]) : null;
  if (n === 'qmat') return { rows: 3, cols: 3 };
  return null;
}

/** A read-only map of local name → square matrix size (N for an N×N matrix). The WGSL emitter — unlike the GLSL
 *  emitter's `tenv` — carries no local type env, so this is the ONE piece of local matrix-shape knowledge the
 *  gate + emitter SHARE: it lets `inverse(M)` (M a local) resolve its `_invN` size. Only SQUARE locals are
 *  recorded (a non-square matrix is never invertible). Exported so `emit-wgsl.ts` uses the SAME resolution the
 *  gate does — a gate-accepted `inverse` is therefore always emitter-resolvable (no silent bare-arg WGSL). */
export type LocalMats = ReadonlyMap<string, number>;

/** The SQUARE size (N for N×N) of a matrix expression, consulting local matrix bindings — the single lowerability
 *  rule for `inverse(E)`. Resolves: a matrix constructor / `qmat` leaf (square only); a local matrix binding
 *  (`const M = mat3(...)`); and the shape-preserving/swapping ops `transpose`/`inverse` RECURSIVELY (so
 *  `inverse(transpose(M))` — the normal-matrix idiom — resolves when M is a square local). A computed matrix
 *  expression (a `mat*mat` product, `mat±mat`, `mat*scalar`, a ternary, an un-recorded local) is NOT statically
 *  sized → null. `matShapeOf` alone can't do this: it is not locals-aware and its transpose/inverse recursion
 *  bottoms out at a non-locals-aware `matShapeOf`, so a local-rooted chain returns null there. */
export function matSizeOf(e: Expr, locals: LocalMats): number | null {
  if (e.kind === 'ident') return locals.get(e.name) ?? null;
  if (e.kind === 'call' && e.callee.kind === 'ident') {
    const n = e.callee.name;
    if (n === 'transpose' || n === 'inverse') return e.args[0] ? matSizeOf(e.args[0], locals) : null;
    const m = matShapeOf(e);   // a ctor / qmat leaf — square only (transpose of a non-square ctor is caught above)
    return m && m.rows === m.cols ? m.rows : null;
  }
  return null;
}

/** Build the local name → square-matrix-size map for a kernel body, in DECLARATION order (a later binding's init
 *  can reference an earlier matrix local, e.g. `const B = transpose(A)`). A local rebound to a non-matrix (or a
 *  computed matrix `matSizeOf` can't size) is REMOVED — so a later `inverse(that local)` is conservatively
 *  unresolvable → gate-rejected. The gate and the emitter both call this over the same body, so they agree. */
export function buildLocalMats(kernel: UserFn): Map<string, number> {
  const locals = new Map<string, number>();
  const record = (name: string, init: Expr): void => { const n = matSizeOf(init, locals); if (n !== null) locals.set(name, n); else locals.delete(name); };
  const walkStmt = (s: Stmt): void => {
    switch (s.kind) {
      case 'const': case 'let': record(s.name, s.init); return;
      case 'assign': if (s.target.kind === 'ident') record(s.target.name, s.value); return;
      case 'if': s.then.forEach(walkStmt); s.else?.forEach(walkStmt); return;
      case 'for': s.body.forEach(walkStmt); return;
      case 'while': s.body.forEach(walkStmt); return;
      default: return;   // expr / return contribute no new local matrix binding
    }
  };
  kernel.body.forEach(walkStmt);
  return locals;
}

/** A read-only map of local name → its FULL matrix/vector shape ({rows, cols}). The superset companion to
 *  `LocalMats`: it keeps vecs (cols:1) and NON-square matrices too, so a shape-driven emitter/gate choice
 *  (the WGSL divide-guard, mat-negate, and the determinant square check) can resolve a vec/mat LOCAL — not
 *  just the square-invertible ones `LocalMats` records for `inverse`. */
export type LocalShapes = ReadonlyMap<string, { rows: number; cols: number }>;

/** The full {rows, cols} shape of a matrix/vector expression, consulting local bindings. Like `matSizeOf` but
 *  returns the full shape (NON-square incl.) and also resolves a VEC (cols:1). A vec ctor → {rows:n, cols:1};
 *  a mat ctor / `qmat` leaf → its {rows, cols}; `transpose` swaps; `inverse` keeps; a local → its recorded
 *  shape. Unresolvable (a computed expr, an un-recorded local) → null. `matShapeOf` alone isn't locals-aware
 *  and doesn't know vecs, so this is the ONE resolver the shape-driven emitter + the determinant gate SHARE. */
export function shapeOfExpr(e: Expr, locals: LocalShapes): { rows: number; cols: number } | null {
  if (e.kind === 'ident') return locals.get(e.name) ?? null;
  if (e.kind === 'call' && e.callee.kind === 'ident') {
    const n = e.callee.name;
    if (n === 'transpose') { const s = e.args[0] ? shapeOfExpr(e.args[0], locals) : null; return s ? { rows: s.cols, cols: s.rows } : null; }
    if (n === 'inverse') return e.args[0] ? shapeOfExpr(e.args[0], locals) : null;
    const m = matShapeOf(e); if (m) return m;                          // mat ctor / qmat leaf (square + non-square)
    if (n === 'vec2') return { rows: 2, cols: 1 }; if (n === 'vec3') return { rows: 3, cols: 1 }; if (n === 'vec4') return { rows: 4, cols: 1 };
  }
  return null;
}

/** Build the local name → full-shape map for a kernel body, in DECLARATION order (mirrors `buildLocalMats`
 *  but keeps vecs + non-square mats). A rebinding to an unresolvable value REMOVES the local (conservative —
 *  a later shape query on it returns null, the same not-proven behavior `buildLocalMats` gives). */
export function buildLocalShapes(kernel: UserFn): Map<string, { rows: number; cols: number }> {
  const locals = new Map<string, { rows: number; cols: number }>();
  const record = (name: string, init: Expr): void => { const s = shapeOfExpr(init, locals); if (s) locals.set(name, s); else locals.delete(name); };
  const walkStmt = (s: Stmt): void => {
    switch (s.kind) {
      case 'const': case 'let': record(s.name, s.init); return;
      case 'assign': if (s.target.kind === 'ident') record(s.target.name, s.value); return;
      case 'if': s.then.forEach(walkStmt); s.else?.forEach(walkStmt); return;
      case 'for': case 'while': s.body.forEach(walkStmt); return;
      default: return;   // expr / return contribute no new local shape binding
    }
  };
  kernel.body.forEach(walkStmt);
  return locals;
}

/** The vec width an expression evaluates to: 0 = scalar/unknown, 2/3/4 = a vecN. Structural + conservative:
 *  a `vec2/3/4(...)` call is its width; `normalize`/`mix` preserve their first arg's width; `cross` is 3;
 *  a vec-bound local/input is its n; a swizzle `v.xy` is the swizzle length (a single `.x` is scalar); a
 *  binary/cond/neg follows its vec operand. Anything unproven is 0 (scalar). `dot`/`length` are scalar. */
export function returnVecWidth(e: Expr, bindings: BindingTable, localWidth: ReadonlyMap<string, number>): number {
  switch (e.kind) {
    case 'call': {
      if (e.callee.kind === 'ident') {
        const n = e.callee.name;
        if (n === 'vec2') return 2; if (n === 'vec3') return 3; if (n === 'vec4') return 4;
        if (n === 'normalize' || n === 'mix') return e.args[0] ? returnVecWidth(e.args[0], bindings, localWidth) : 0;
        if (n === 'cross') return 3;
        // reflect/refract/faceforward return a vector of their first (incident/normal) arg's width — so a
        // vecN output over one is accepted, not read as scalar and rejected.
        if (n === 'reflect' || n === 'refract' || n === 'faceforward') return e.args[0] ? returnVecWidth(e.args[0], bindings, localWidth) : 0;
        // Quaternion ops that produce a quat (a vec4): qmul/qconj/qinvert/qaxisangle/qslerp always return a vec4.
        if (n === 'qmul' || n === 'qconj' || n === 'qinvert' || n === 'qaxisangle' || n === 'qslerp') return 4;
        // qrotate produces the rotated 3-vector (a vec3).
        if (n === 'qrotate') return 3;
        if (n === 'distance' || n === 'determinant') return 0;   // fold a vec/mat down to a scalar
      }
      return 0;   // dot/length/scalar builtins/helper calls → scalar
    }
    case 'ident': {
      if (localWidth.has(e.name)) return localWidth.get(e.name)!;
      const b = bindings.byName.get(e.name);
      if (b && (b.role === 'buffer' || b.role === 'uniform') && b.lower.shape === 'vecN') return b.lower.rows ?? 0;
      return 0;
    }
    case 'member': {
      const w = returnVecWidth(e.object, bindings, localWidth);
      // A swizzle of an INCONSISTENT-shape object (w === -1 — the object is a `vec±mat`, a mismatched-width
      // vec binary, a mismatched matmul, or a mismatched-branch ternary) stays inconsistent: `(vec2+mat2).x`
      // has no lowerable value, so the -1 must propagate (not collapse to 0 via the `w >= 2` guard below and
      // read as a legal scalar) so checkOutputShape still rejects it. Only an already-invalid expr is ever -1
      // here — a valid vec object is always width 0/2/3/4 — so this never falsely rejects a real kernel.
      if (w === -1) return -1;
      // A swizzle on a vec: `.xy`→2, single `.x`→scalar. A NON-xyzw swizzle (.rgb/.stpq) OR an OVER-RANGE
      // swizzle (a char indexing past the source vec's width, e.g. `.z` on a vec2) has no width here (returns 0)
      // — the gate's walkExpr rejects both outright, and reporting a phantom width (the raw swizzle length)
      // would let the shape check pass a kernel the interpreter can't evaluate (it NOT_HANDLEs an out-of-range
      // component → a silent 0). Belt-and-suspenders with that walkExpr rejection.
      if (w >= 2) {
        const len = e.property.length;
        const inRange = isXyzwSwizzle(e.property) && [...e.property].every((ch) => SWIZZLE_ORDER.indexOf(ch) < w);
        return len >= 2 && len <= 4 && inRange ? len : 0;
      }
      return 0;
    }
    case 'unary': return e.op === '!' ? 0 : returnVecWidth(e.operand, bindings, localWidth);
    case 'binary': {
      if (COMPARE.has(e.op)) return 0;   // a comparison is a bool → scalar
      // The interpreter's vec/mat descriptor `binary` handler (the correctness oracle) evaluates ONLY a fixed
      // set of (op, left-shape, right-shape) triples; every other combo returns NOT_HANDLED → ML-LANG-OP-
      // UNSUPPORTED → the cell is 0. A shader emits the NATIVE op and computes a real value, so accepting a
      // NOT_HANDLED combo is a SILENT GPU-vs-interpreter divergence (verify is off by default). We model that
      // table EXACTLY here: an accepted combo returns its real result width; a NOT_HANDLED combo returns -1
      // so checkOutputShape rejects it (never Math.max, which would fabricate a width for a combo the oracle
      // can't evaluate). A matrix RESULT reads as width 0 — a matrix isn't a vecN, and a computed matrix isn't
      // matShapeOf-recognized (it recognizes only ctor/transpose/inverse/qmat call nodes), so it flows as a
      // scalar downstream, matching how the interpreter chains e.g. `((mat*scalar) * vec)`.
      const lm = matShapeOf(e.left); const rm = matShapeOf(e.right);
      const l = returnVecWidth(e.left, bindings, localWidth); const r = returnVecWidth(e.right, bindings, localWidth);
      // An operand that is itself inconsistent (a nested -1) makes the whole binary inconsistent → propagate.
      if (l === -1 || r === -1) return -1;
      // Operand shape classes (mutually exclusive: a mat has returnVecWidth 0 + a matShapeOf; a vec has
      // width ≥ 2 + no matShapeOf; a scalar has width 0 + no matShapeOf).
      const lMat = lm !== null; const rMat = rm !== null;
      const lVec = l >= 2 && !lMat; const rVec = r >= 2 && !rMat;
      const lScalar = l === 0 && !lMat; const rScalar = r === 0 && !rMat;
      // The carve-out: plain scalar∘scalar arithmetic never touches the vec/mat descriptor (both operands are
      // numbers), so it is always lowerable → width 0. (Math.max over two 0s = 0; kept for clarity.)
      if (lScalar && rScalar) return Math.max(l, r);
      // At least one operand is a vec/mat — apply the interpreter's table, keyed by op.
      const op = e.op;
      if (op === '*') {
        // matmul: the LEFT operand is a matrix; the right is a mat|vec; inner dims must match (lm.cols === the
        // right's row count). `vec * mat` (a vec on the LEFT of a matrix) is NOT matmul in the interpreter
        // (`lMat && rs && o==='*'`), so it falls through to the trailing -1.
        if (lMat && (rVec || rMat)) { const inner = rMat ? rm!.rows : r; return lm!.cols === inner ? (rMat ? 0 : lm!.rows) : -1; }
        if (lVec && rScalar) return l;               // vec * scalar → scale
        if (lScalar && rVec) return r;               // scalar * vec → scale
        if (lVec && rVec) return l === r ? l : -1;   // vec * vec (equal width) → componentwise; differing → -1
        if (lMat && rScalar) return 0;               // mat * scalar → a matrix (reads as scalar/mat downstream)
        if (lScalar && rMat) return 0;               // scalar * mat → a matrix
        return -1;                                   // vec * mat (vec left) → NOT_HANDLED
      }
      if (op === '/') {
        if (lVec && rScalar) return l;               // vec / scalar → scale
        if (lVec && rVec) return l === r ? l : -1;   // vec / vec (equal width) → componentwise
        if (lMat && rScalar) return 0;               // mat / scalar → a matrix
        return -1;                                   // scalar/vec, scalar/mat, mat/vec, mat/mat, vec/mat → NOT_HANDLED
      }
      if (op === '+' || op === '-') {
        if (lVec && rVec) return l === r ? l : -1;   // vec ± vec (equal width) → componentwise; differing → -1
        return -1;                                   // vec±scalar, scalar±vec, mat±mat, mat±scalar, scalar±mat, mat±vec, vec±mat → NOT_HANDLED
      }
      // Any other op with a vec/mat operand (`%`, `&&`, `||`) is undefined on the vec/mat descriptor → -1.
      return -1;
    }
    case 'cond': {
      const t = returnVecWidth(e.then, bindings, localWidth); const el = returnVecWidth(e.else, bindings, localWidth);
      // A ternary whose two branches disagree on vec width (`t ? vec3(..) : vec2(..)`) can't be given one
      // consistent output width. Math.max would collapse (3,2)→3 and let it reach the emitter (a raw shader
      // compile error / silent pad-or-truncate). Signal -1 ("inconsistent") so checkOutputShape rejects it.
      if (t !== el) return -1;
      return t;
    }
    default: return 0;
  }
}

/** Validate every `return` in the kernel body against the requested output width. Tracks local const/let vec
 *  widths so a `const v = vec3(...); return normalize(v)` is proven. Pushes MLGPU-OUTPUT-SHAPE on a mismatch. */
function checkOutputShape(kernel: UserFn, comps: number, bindings: BindingTable, reasons: Diagnostic[]): void {
  const localWidth = new Map<string, number>();
  const expected = comps === 1 ? 'a scalar (f32)' : `a vec${comps}`;
  const walk = (stmts: readonly Stmt[]): void => {
    for (const s of stmts) {
      switch (s.kind) {
        case 'const': case 'let': localWidth.set(s.name, returnVecWidth(s.init, bindings, localWidth)); break;
        case 'assign': if (s.target.kind === 'ident') localWidth.set(s.target.name, returnVecWidth(s.value, bindings, localWidth)); break;
        case 'return': {
          // A matrix has no output-cell form: a kernel writes one scalar or one vecN per element, never a
          // whole matrix. Catch it before the scalar/vec width check (a mat ctor otherwise reads as width 0 →
          // a false "scalar" pass) and reject with a shape reason of its own.
          if (s.value && matShapeOf(s.value)) {
            reasons.push(makeDiagnostic('MLGPU-OUTPUT-SHAPE', `a matrix return has no output-cell form — a kernel returns a scalar or a vecN`, s.span));
            break;
          }
          const w = s.value ? returnVecWidth(s.value, bindings, localWidth) : 0;   // width 0 = scalar (vec is never 1)
          // w === -1 = "inconsistent" (a return whose width can't be determined — a mismatched-width ternary
          // or vec binary). Always a shape error regardless of the requested comps: the two branches disagree
          // on the output width, so no single output element fits.
          const bad = w === -1 || (comps === 1 ? w >= 2 : w !== comps);
          const actual = w === -1 ? 'an inconsistent-width return (branches disagree)' : w >= 2 ? `a vec${w}` : 'a scalar';
          if (bad) reasons.push(makeDiagnostic('MLGPU-OUTPUT-SHAPE', `kernel return shape does not match the requested output element: expected ${expected} but the return is ${actual}`, s.span));
          break;
        }
        case 'if': walk(s.then); if (s.else) walk(s.else); break;
        case 'for': case 'while': walk(s.body); break;
      }
    }
  };
  walk(kernel.body);
}

// ─── Multi-output (a named-object return writes several output buffers) ───
// A kernel `return { sum: EXPR_s, diff: EXPR_d }` is equivalent to N single-output kernels, one per named
// output, each returning that entry's EXPR. `synthOutputKernel(kernel, key)` produces that per-key kernel
// (same params + closure, every object-literal return rewritten to `return <entry-for-key>`); the engine
// then runs each through the PROVEN single-output path (gate → emit → dispatch → oracle) with zero new
// emitter/backend code. `checkMultiOutputShape` validates the object structure the single-output gate never
// sees (it rejects object literals): every `return` must be an object literal whose keys exactly match the
// declared outputs, no spread, and each entry's expr width must match its element's comps.

/** Produce the single-output kernel for one named output: the original kernel with every object-literal
 *  `return { … }` rewritten to `return <the entry whose key === name>`. Same params/closure/body-shape, so
 *  the existing single-output machinery (gate/emit/dispatch/oracle) handles it unchanged. Assumes the shape
 *  check passed (every return is a matching object literal); a defensive `0` covers a malformed AST. */
export function synthOutputKernel(kernel: UserFn, name: string): UserFn {
  return { ...kernel, body: rewriteReturns(kernel.body, name) };
}
function rewriteReturns(body: readonly Stmt[], name: string): Stmt[] {
  return body.map((s) => rewriteReturnStmt(s, name));
}
function rewriteReturnStmt(s: Stmt, name: string): Stmt {
  switch (s.kind) {
    case 'return': {
      if (s.value && s.value.kind === 'object') {
        const entry = s.value.entries.find((en) => !en.spread && en.key === name);
        return { ...s, value: entry ? entry.value : { kind: 'number', value: 0, span: s.span } };
      }
      return s;
    }
    case 'if': {
      const out: Extract<Stmt, { kind: 'if' }> = { ...s, then: rewriteReturns(s.then, name) };
      if (s.else) out.else = rewriteReturns(s.else, name);
      return out;
    }
    case 'for': case 'while': return { ...s, body: rewriteReturns(s.body, name) };
    default: return s;
  }
}

/** Validate a multi-output (`outputs`) kernel's return STRUCTURE — the check the single-output gate cannot do
 *  (it rejects object literals outright). Every `return` must be an OBJECT LITERAL whose keys EXACTLY match
 *  the declared output names, with NO spread (the keys must be statically listable). Returns MLGPU-OUTPUT-
 *  SHAPE diagnostics; empty ⇒ the structure is valid. Per-entry WIDTH (a vecN entry vs its declared element)
 *  is NOT checked here — the engine synthesizes a single-output kernel per key (`synthOutputKernel`) and
 *  gates it with that output's comps, so the existing `checkOutputShape` validates each entry's width for
 *  free (one width-check source, no duplicate diagnostics). */
export function checkMultiOutputShape(kernel: UserFn, names: readonly string[]): Diagnostic[] {
  const reasons: Diagnostic[] = [];
  const keySet = new Set(names);
  // A well-formed return's keys are a SET exactly equal to the declared outputs: no duplicates (a typo like
  // `{ sum, sum }` — accepted by a length+membership check alone — would leave a declared key ('diff')
  // unsatisfied, and synthesis for that missing key emits `return 0` → a silently all-zeros output that
  // verify falsely reports ok), no missing keys, no extras.
  const sameKeys = (entryKeys: readonly string[]): boolean =>
    new Set(entryKeys).size === entryKeys.length && entryKeys.length === keySet.size && entryKeys.every((k) => keySet.has(k));
  const list = `{${names.join(', ')}}`;
  const walk = (stmts: readonly Stmt[]): void => {
    for (const s of stmts) {
      switch (s.kind) {
        case 'return': {
          const v = s.value;
          if (!v || v.kind !== 'object') {
            reasons.push(makeDiagnostic('MLGPU-OUTPUT-SHAPE', `an 'outputs' kernel must return an object literal with keys ${list} — this return is not an object literal`, s.span));
            break;
          }
          if (v.entries.some((en) => en.spread)) {
            reasons.push(makeDiagnostic('MLGPU-OUTPUT-SHAPE', `an 'outputs' return may not use spread — list every key ${list} explicitly`, s.span));
            break;
          }
          const entryKeys = v.entries.map((en) => en.key);
          if (!sameKeys(entryKeys)) {
            reasons.push(makeDiagnostic('MLGPU-OUTPUT-SHAPE', `an 'outputs' return's keys {${entryKeys.join(', ')}} must exactly match the declared outputs ${list}`, s.span));
          }
          break;
        }
        case 'if': walk(s.then); if (s.else) walk(s.else); break;
        case 'for': case 'while': walk(s.body); break;
      }
    }
  };
  walk(kernel.body);
  return reasons;
}

/** Gate one function body against its own closure; recurse into every helper it calls. Returns the fn's
 *  binding table (the caller keeps the kernel's). A visited set guards mutual/self recursion. */
function gateFn(fn: UserFn, host: ReactiveHost, reasons: Diagnostic[], visited: Set<UserFn>): BindingTable {
  visited.add(fn);
  const free = collectFreeNames(fn);
  const bindings = buildBindingTable(fn, free, host);
  const flag = (code: string, msg: string, span?: Expr['span']) => reasons.push(makeDiagnostic(code, msg, span));

  // Every free name must resolve to a supported binding; else reject with the reason for WHY.
  for (const name of free) {
    if (BUILTINS[name]) continue;                       // a builtin call (validated in walkExpr)
    const b = bindings.byName.get(name);
    if (!b) {
      // A common shape is indexing a member of a resource wrapper (`rA.value[i]`) — the kernel closes over
      // `rA` (a GpuResource), not a buffer. Point at the fix: bind the buffer to a local OUTSIDE the kernel
      // (`const bufA = rA.value`) and index that. Detect `name.<member>` used anywhere in the body.
      const member = memberAccessedOn(fn, name);
      const hint = member
        ? ` — it looks like a member of another value; bind the buffer to a local OUTSIDE the kernel first (e.g. \`const buf = ${name}.${member}\`) and index \`buf\` instead`
        : ` — declare it as a typed array (f32/i32/u32), a number, or a helper function`;
      flag('MLGPU-BAD-INPUT', `'${name}' is not a kernel input${hint}`, fn.body[0]?.span); continue;
    }
    // v1 defers vec/mat INPUTS: a closed-over vec/mat gets role:'uniform', but the WGSL emitter neither
    // declares nor packs it (the vec/mat-in-kernel path is a later increment). Reject so gate ↔ emitter
    // agree. A vec/mat built INSIDE the kernel (`const v = vec3(a[i], …)`) is a `vec2/3/4` call + a local
    // const — NOT a uniform binding — so it is unaffected here.
    if (b.role === 'uniform') flag('MLGPU-NOT-LOWERABLE', `a vec/mat input ('${name}') is not yet supported as a kernel uniform`, fn.body[0]?.span);
  }

  const isBufferIdent = (x: Expr): boolean => x.kind === 'ident' && bindings.byName.get(x.name)?.role === 'buffer';

  // Track local const/let/assign vec widths (mirrors checkOutputShape's map) so walkExpr can validate an
  // over-range swizzle on a vec-typed local (`const v = vec2(...); return v.z`) — not just a direct `vecN(...).z`
  // ctor. Populated as statements are walked in source order; a swizzle in a later statement sees the local's
  // width. A `for`-binding is not vec-typed (it's a range index → scalar), so it stays width 0 (unset).
  const localWidth = new Map<string, number>();

  // Local matrix sizes for the `inverse` lowerability check — precomputed over the WHOLE body (declaration order)
  // so `inverse(M)` resolves M regardless of walk order, matching how the emitter precomputes then emits. The
  // SAME `buildLocalMats`/`matSizeOf` the emitter's inverse resolution derives from → gate ↔ emitter cannot
  // disagree on which inverses lower.
  const localMats = buildLocalMats(fn);
  // Local FULL shapes for the `determinant` square check — the superset companion to `localMats` that keeps
  // vecs + NON-square mats. Precomputed over the whole body (same as the emitter's `buildLocalShapes`) so a
  // non-square matrix bound to a LOCAL (`const m = mat2x3(...); determinant(m)`) resolves its {rows, cols}
  // here and the determinant check below can reject it — `matShapeOf` alone (not locals-aware) could not.
  const localShapes = buildLocalShapes(fn);

  // A kernel-local (const/let/for-binding/param) name may not start with `_`: the emitters reserve the
  // `_`-prefix for their own temporaries (`_r`/`_flat`/`_out`/`_p`/`_frag`/`_fetch`/`_fx`/…, `_u_`-scalars).
  // A user local named `_r` would collide with the vecN-return temp (a WGSL/GLSL redefinition) or shadow a
  // reserved name — a silent compile error / wrong lowering the no-adapter emit path can't catch. One rule
  // (reject any `_`-prefixed local) closes the whole class without enumerating the reserved set.
  const checkReservedName = (name: string, span?: Expr['span']): void => {
    if (name.startsWith('_')) flag('MLGPU-NOT-LOWERABLE', `a kernel-local name ('${name}') may not start with '_' (reserved for the compiler)`, span);
  };
  for (const p of fn.params) if (p.kind === 'name') checkReservedName(p.name, fn.body[0]?.span);

  const walkExpr = (e: Expr): void => {
    switch (e.kind) {
      case 'number': case 'bool': return;
      case 'ident': {
        // A buffer has no whole-value form; it is valid only as the object of `a[i]` or `a.length`
        // (handled in the `index`/`member` cases, which do NOT descend into that ident). Reaching a
        // buffer ident HERE means it appears in a whole-value position → flag.
        if (bindings.byName.get(e.name)?.role === 'buffer') flag('MLGPU-NOT-LOWERABLE', `a buffer has no whole-value form — index it (${e.name}[i]) or read ${e.name}.length`, e.span);
        return;
      }
      case 'string': flag('MLGPU-NOT-LOWERABLE', 'strings are not lowerable to a compute kernel', e.span); return;
      case 'null': flag('MLGPU-NOT-LOWERABLE', 'null is not lowerable', e.span); return;
      case 'object': flag('MLGPU-NOT-LOWERABLE', 'object literals are not lowerable', e.span); return;
      case 'array': flag('MLGPU-NOT-LOWERABLE', 'array literals are not lowerable (use a typed-array input)', e.span); return;
      case 'arrow': flag('MLGPU-NOT-LOWERABLE', 'closures are not lowerable', e.span); return;
      case 'member': {
        // `bufferIdent.length` is the one OK whole-buffer read; anything else descends into the object.
        if (isBufferIdent(e.object) && e.property === 'length') return;
        // The only OTHER member read a compute kernel makes is a vec swizzle (`v.xy`). A swizzle the
        // interpreter can evaluate is xyzw ONLY (its descriptor indexes 'xyzw'); a color (.rgb) / texture
        // (.stpq) swizzle returns NOT_HANDLED → 0 on the CPU oracle + CPU emitter while the WGSL/GLSL shaders
        // pass it through verbatim and compute the real components — a SILENT GPU-vs-CPU divergence (verify is
        // off by default). Reject a 1-4-char all-alpha member (a swizzle shape) on a non-buffer object unless
        // every char ∈ xyzw. (A plain object member read isn't lowerable in a kernel anyway — the only member
        // reads are `buffer.length` and vec swizzles — so requiring xyzw here is safe.)
        const prop = e.property;
        if (!isBufferIdent(e.object) && prop.length >= 1 && prop.length <= 4 && /^[a-z]+$/i.test(prop) && !isXyzwSwizzle(prop)) {
          flag('MLGPU-NOT-LOWERABLE', `only xyzw swizzles are lowerable (a color/texture swizzle like '.${prop}' is not — the interpreter cannot evaluate it)`, e.span);
        }
        // An in-alphabet-but-OVER-RANGE xyzw swizzle (`.z` on a vec2, `.xyz` on a vec2) is spec-invalid: the
        // interpreter NOT_HANDLEs an out-of-range component → the CPU oracle + emitter coerce to 0 (a
        // false-green verify) while WGSL/GLSL emit a swizzle past the source vec's width (a real-adapter
        // compile error). When the object's provable vec width `w` is known (a direct `vecN(...)` ctor or a
        // vec-typed local — via localWidth built in walkStmt), reject any swizzle char indexing >= w. If the
        // object isn't a provable vec (w == 0 — e.g. a complexly-derived local), the return-site
        // `returnVecWidth`/`checkOutputShape` check catches it; a non-vec `.foo` member is handled above.
        else if (!isBufferIdent(e.object) && isXyzwSwizzle(prop) && prop.length >= 1 && prop.length <= 4) {
          const w = returnVecWidth(e.object, bindings, localWidth);
          if (w >= 2 && [...prop].some((ch) => SWIZZLE_ORDER.indexOf(ch) >= w)) {
            flag('MLGPU-NOT-LOWERABLE', `swizzle '.${prop}' reads past the width of a vec${w} (the interpreter cannot evaluate an out-of-range component) — index only x${w >= 2 ? 'y' : ''}${w >= 3 ? 'z' : ''}${w >= 4 ? 'w' : ''}`, e.span);
          }
        }
        walkExpr(e.object);
        return;
      }
      case 'index': {
        walkExpr(e.index);
        // `bufferIdent[i]` is the one OK buffer position — do NOT descend into that ident (it isn't a
        // whole-value use). Any other object expr is walked normally.
        if (!isBufferIdent(e.object)) walkExpr(e.object);
        return;
      }
      case 'unary': walkExpr(e.operand); return;
      case 'binary': walkExpr(e.left); walkExpr(e.right); return;
      case 'cond': walkExpr(e.test); walkExpr(e.then); walkExpr(e.else); return;
      case 'call': {
        if (e.callee.kind === 'ident') {
          const spec = BUILTINS[e.callee.name];
          const b = bindings.byName.get(e.callee.name);
          // A user function BINDING wins over a builtin of the same name — the interpreter resolves the
          // closure binding first, so `function abs(x){…}` SHADOWS the `abs` intrinsic. Check the callee
          // binding BEFORE the builtin spec: a shadowing (or any) helper call is rejected in v1 (the emitter
          // inlines only the top-level body, emits no `fn name(){}`), so gate ↔ emitter agree instead of the
          // emitter silently lowering the intrinsic while the oracle runs the user's body. Helper bodies are
          // still gated recursively below.
          if (b?.role === 'callee') { flag('MLGPU-NOT-LOWERABLE', `calling a helper function ('${e.callee.name}') is not yet lowerable — inline it into the kernel`, e.span); }
          else if (spec) {
            if (e.callee.name === 'rand') flag('MLGPU-NOT-LOWERABLE', `rand() cannot be lowered to a compute kernel — it cannot match the deterministic interpreter oracle`, e.span);
            else if (spec.profile === 'host' || (spec.portability === 'cpu-only' && !spec.lowerName)) flag('MLGPU-NOT-LOWERABLE', `builtin '${e.callee.name}' has no shader lowering`, e.span);
            // WGSL has NO builtin inverse() — the emitter hand-emits a per-SIZE `_invN` helper, so the arg's square
            // size must be STATICALLY resolvable (a mat ctor, a matrix-typed local, or a transpose/inverse chain
            // over one — via `matSizeOf` over `localMats`, the SAME rule the emitter uses). A computed-matrix arg
            // (a `mat*mat` product, `mat±mat`, `mat*scalar`, a ternary, an un-recorded local) is unresolvable →
            // REJECT here so the kernel falls back to CPU/interpreter (both correct) rather than the emitter
            // silently dropping the inverse (a WGSL-only, adapter-only wrong result). This keeps gate ↔ emitter in
            // lockstep: no gate-accepted `inverse` can reach the emitter's un-inverted-arg fallback.
            else if (e.callee.name === 'inverse' && (!e.args[0] || matSizeOf(e.args[0], localMats) === null)) {
              flag('MLGPU-NOT-LOWERABLE', `inverse(...) requires a statically-sized square matrix argument (a mat ctor or a matrix-typed local) — a computed matrix expression is not yet lowerable`, e.span);
            }
            // determinant is defined only for a SQUARE matrix (both shaders' native determinant + the interpreter
            // oracle reject a non-square arg). Reject a non-square arg here so gate ↔ oracle agree. `shapeOfExpr`
            // is locals-aware, so this catches BOTH a DIRECT non-square ctor (`determinant(mat2x3(...))`) AND a
            // non-square matrix bound to a LOCAL (`const M = mat2x3(...); determinant(M)`) — the local's shape is
            // recorded in `localShapes`. (A computed / un-recorded matrix arg resolves to null and is not flagged
            // here; a non-square such arg still errors in the oracle + fails to compile on both shaders, so no
            // silent wrong value results.)
            else if (e.callee.name === 'determinant' && e.args[0]) {
              const sh = shapeOfExpr(e.args[0], localShapes);
              if (sh && sh.rows !== sh.cols) flag('MLGPU-NOT-LOWERABLE', `determinant(...) requires a square matrix`, e.span);
            }
          }
          else flag('MLGPU-NOT-LOWERABLE', `call '${e.callee.name}' is not a lowerable builtin or helper`, e.span);
        } else { flag('MLGPU-NOT-LOWERABLE', 'an indirect call is not lowerable', e.span); walkExpr(e.callee); }
        if (e.block) flag('MLGPU-NOT-LOWERABLE', 'a wrapping-block call is not lowerable in a compute kernel', e.span);
        e.args.forEach(walkExpr);
        return;
      }
    }
  };
  const walkStmt = (s: Stmt): void => {
    switch (s.kind) {
      case 'const': case 'let': checkReservedName(s.name, s.span); walkExpr(s.init); localWidth.set(s.name, returnVecWidth(s.init, bindings, localWidth)); return;
      case 'assign': {
        // An input buffer is bound read-only in every emitter (WGSL `var<storage, read>`; the CPU/GLSL
        // paths read it through the descriptor and never store into it), so an index-write `buf[i] = …`
        // to one has no lowering. Flag it — otherwise the interpreter oracle would honor the write
        // (mutating the caller's live buffer + bumping its generation) while the dispatched shader drops
        // it, so the two references diverge. gate ↔ emitter must agree.
        const t = s.target;
        if (t.kind === 'index' && isBufferIdent(t.object)) {
          const name = (t.object as Extract<Expr, { kind: 'ident' }>).name;
          flag('MLGPU-INPUT-WRITE', `cannot assign to an input buffer ('${name}') inside a kernel — inputs are read-only in a compute dispatch`, s.span);
        }
        walkExpr(s.value); walkExpr(s.target);
        if (t.kind === 'ident') localWidth.set(t.name, returnVecWidth(s.value, bindings, localWidth));
        return;
      }
      case 'expr': walkExpr(s.expr); return;
      case 'return': if (s.value) walkExpr(s.value); return;
      case 'if': walkExpr(s.test); s.then.forEach(walkStmt); s.else?.forEach(walkStmt); return;
      case 'for': {
        checkReservedName(s.binding, s.span);
        const it = s.iterable;
        if (!(it.kind === 'call' && it.callee.kind === 'ident' && it.callee.name === 'range')) {
          flag('MLGPU-NOT-LOWERABLE', 'only `for (… of range(n))` (a bounded loop) is lowerable', s.span);
        } else if (it.args.length === 0) {
          flag('MLGPU-NOT-LOWERABLE', '`range()` needs a bound argument', s.span);
        } else {
          walkExpr(it.args[0]!);
        }
        s.body.forEach(walkStmt);
        return;
      }
      case 'while': flag('MLGPU-NOT-LOWERABLE', 'a data-dependent `while` loop is not lowerable (use a bounded `for … of range(n)`)', s.span); s.body.forEach(walkStmt); return;
      case 'function': case 'component': flag('MLGPU-NOT-LOWERABLE', 'a nested function/component is not lowerable', s.span); return;
    }
  };
  fn.body.forEach(walkStmt);

  // Recurse into helper callees (gate their bodies against their own closures).
  for (const b of bindings.byName.values()) {
    if (b.role === 'callee' && !visited.has(b.fn)) gateFn(b.fn, host, reasons, visited);
  }
  return bindings;
}

/** If `name` is read as `name.<member>` (or `name.<member>[…]`, `name.<member>.…`) ANYWHERE in `fn`'s body,
 *  return that first member property; else null. Used only to enrich the not-a-kernel-input diagnostic with
 *  a targeted "bind a local first" hint (a resource wrapper indexed as `rA.value[i]`). A read-only AST scan. */
function memberAccessedOn(fn: UserFn, name: string): string | null {
  let found: string | null = null;
  const visitExpr = (e: Expr): void => {
    if (found) return;
    switch (e.kind) {
      case 'member': if (e.object.kind === 'ident' && e.object.name === name) { found = e.property; return; } visitExpr(e.object); return;
      case 'index': visitExpr(e.object); visitExpr(e.index); return;
      case 'unary': visitExpr(e.operand); return;
      case 'binary': visitExpr(e.left); visitExpr(e.right); return;
      case 'cond': visitExpr(e.test); visitExpr(e.then); visitExpr(e.else); return;
      case 'call': if (e.callee.kind !== 'ident') visitExpr(e.callee); e.args.forEach(visitExpr); if (e.block) e.block.forEach(visitStmt); return;
      case 'object': e.entries.forEach((en) => visitExpr(en.value)); return;
      case 'array': e.elements.forEach((el) => visitExpr(el.value)); return;
      case 'arrow': if (Array.isArray(e.body)) e.body.forEach(visitStmt); else visitExpr(e.body); return;
      default: return;   // ident / number / string / bool / null — no member access
    }
  };
  const visitStmt = (s: Stmt): void => {
    if (found) return;
    switch (s.kind) {
      case 'const': case 'let': visitExpr(s.init); return;
      case 'assign': visitExpr(s.value); visitExpr(s.target); return;
      case 'expr': visitExpr(s.expr); return;
      case 'return': if (s.value) visitExpr(s.value); return;
      case 'if': visitExpr(s.test); s.then.forEach(visitStmt); if (s.else) s.else.forEach(visitStmt); return;
      case 'for': visitExpr(s.iterable); s.body.forEach(visitStmt); return;
      case 'while': visitExpr(s.test); s.body.forEach(visitStmt); return;
      default: return;   // nested function/component — the gate rejects these separately
    }
  };
  fn.body.forEach(visitStmt);
  return found;
}
