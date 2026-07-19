// The WGSL compute-shader emitter: kernel AST → a @compute entry, type-disciplined. Storage bindings +
// the _Params uniform + the workgroup/bounds prologue are DERIVED from the binding-table descriptor
// lowering (one var<storage,read> per buffer input, one read_write _out, one uniform _p). The WHOLE body
// is emitted in the scalar type f32 (metael's numeric model is float) — coords, locals, loop vars,
// literals and scalar uniforms are ALL f32; WGSL has NO implicit i32/f32 conversion, so any mix is a hard
// compile error. Integer domains appear ONLY at two boundaries: the OUTPUT flat index `_flat` (u32,
// computed straight from gid, never from an f32 coord copy) and a BUFFER INDEX (cast u32(round(<f32>)) at
// the access site — f32-exact for an in-bounds index since MAX_BUFFER_LENGTH = 2^24 ≤ f32's exact range).
import type { UserFn, Expr, Stmt } from '@metael/lang';
import { BUILTINS } from '@metael/lang';
import type { Binding, BindingTable } from './binding.ts';
import { bodyReferencesAny } from './binding.ts';
import { REDUCE_TILE } from './emit-glsl.ts';
import { matShapeOf, buildLocalShapes, shapeOfExpr, returnVecWidth } from './gate.ts';
import type { LocalShapes } from './gate.ts';

// A conservative operand-shape probe for the emitter's width-driven choices (e.g. the divide guard). WGSL
// carries no local type env like the GLSL emitter's `tenv`, so this recovers an operand's shape structurally
// from the local-shape map + the binding table + the gate's width recognizers. `shapeOfExpr` (locals-aware)
// resolves a vec/mat LOCAL (`const v = vec2(...)`, `const m = mat2(...)`) FIRST — otherwise a bare local would
// fall through to `returnVecWidth` with no local widths and read as scalar, mis-lowering the divide-guard +
// mat-negate. `matShapeOf` catches a direct ctor; else `returnVecWidth` yields 2/3/4 for a vecN and 0 for a
// scalar/unknown (a vec is never width 1). Any proven vec/mat operand takes the native-componentwise path.
type Shape = { kind: 'scalar' } | { kind: 'vec'; n: number } | { kind: 'mat'; rows: number; cols: number };
function shapeOf(e: Expr, bindings: BindingTable, localShapes: LocalShapes): Shape {
  const ls = shapeOfExpr(e, localShapes);
  // cols===1 is a vec of width rows; a defensive rows===1 (never emitted — min vec is 2) reads as scalar.
  if (ls) return ls.cols === 1 ? (ls.rows >= 2 ? { kind: 'vec', n: ls.rows } : { kind: 'scalar' }) : { kind: 'mat', rows: ls.rows, cols: ls.cols };
  const m = matShapeOf(e); if (m) return { kind: 'mat', rows: m.rows, cols: m.cols };
  const w = returnVecWidth(e, bindings, new Map());
  return w >= 2 ? { kind: 'vec', n: w } : { kind: 'scalar' };
}

/** The SQUARE size (N for N×N) of a matrix expression from the local-shape map — the `_invN` helper's size.
 *  A locals-aware square filter over `shapeOfExpr` (matSizeOf-equivalent): a non-square or non-matrix shape,
 *  or an unresolvable arg, → null. The gate REJECTS an `inverse(E)` whose square size is null via the SAME
 *  rule, so a gate-accepted kernel that reaches the emitter always resolves a size here. */
function squareSizeOf(e: Expr, localShapes: LocalShapes): number | null {
  const s = shapeOfExpr(e, localShapes);
  return s && s.rows === s.cols ? s.rows : null;
}

const WORKGROUP_1D = 64;
const WORKGROUP_2D = 8;
// The histogram scatter's workgroup size — one thread per input element, ceil(inLen/G) workgroups. Exported
// so the WebGPU driver's `dispatchWorkgroups(ceil(n/G))` uses the SAME G the shader's @workgroup_size bakes
// (they cannot drift), mirroring how the reduce path shares REDUCE_TILE.
export const HISTOGRAM_WORKGROUP = WORKGROUP_1D;
// f16: the emitter emits self-consistent f16 WGSL (enable f16; array<f16>) — the STORAGE buffers + the body
// arithmetic + the scalar-uniform members are all the f16 scalar type S. The WebGPU backend now requests the
// `shader-f16` device feature + packs/reads f16 storage buffers, so an f16 dispatch runs end-to-end on a
// capable device. The engine (resource.ts) DOWNGRADES an f16 request to f32 (with a resource note) on any
// backend lacking shader-f16 (cpu/webgl2, or a WebGPU device without the feature) AND on any kernel with a
// scalar uniform (the f16 uniform-block PACKING — half members at 16-byte-aligned offsets — is a deferred
// v1 scope; the backend packs the uniform block as f32), so the f16 storage path shipped here only ever runs
// for a shader-f16 device + a uniform-free kernel — keeping f16 safe (never wrong values, only a clean f32
// fallback). Default precision is still 'f32'.
const wgslScalar = (p: 'f16' | 'f32') => (p === 'f16' ? 'f16' : 'f32');

// Core-exact builtins the gate accepts but that carry no registry `lowerName` — they map to a native WGSL
// function of the same name (round → round, which is ties-to-even in WGSL, matching the interpreter/CPU).
const WGSL_CORE_FN: Readonly<Record<string, string>> = { min: 'min', max: 'max', abs: 'abs', sign: 'sign', floor: 'floor', ceil: 'ceil', clamp: 'clamp', round: 'round' };

// ─── `inverse` — WGSL has NO builtin inverse() (unlike GLSL, where it is native for mat2/3/4) ───
// So a matrix inverse is HAND-EMITTED here as a per-size helper function injected into the shader PRELUDE:
// `_inv2`/`_inv3`/`_inv4`, keyed by which sizes the kernel actually uses (a call site emits just `_invN(<arg>)`,
// so the arg is evaluated ONCE — the cleaner of the two approaches; the inline alternative re-evaluates the arg
// once per matrix element). The helper body is GENERATED from the closed-form adjugate/determinant, NOT
// hand-transcribed: `_invN(m) = (1/detN(m)) * matNxN(adjugate entries)`. Each entry inv[col][row] = C(col,row)/det
// where the cofactor C(a,b) = (−1)^(a+b) · minor(a,b) and minor(a,b) is the determinant of the (N−1)×(N−1)
// submatrix with COLUMN a and ROW b deleted. WGSL indexes a matNxR as `m[col][row]` and constructs it
// column-major (matNxN<S>(col0row0, col0row1, …, col1row0, …)), so an entry written as inv[col][row] pushed in
// (col outer, row inner) order lands in exactly the constructor's column-major slot. Since the constructor's
// k-th argument IS inv[col][row] for (col,row) = (⌊k/N⌋, k mod N), the emitted order = the interpreter's
// column-major `out[col*N+row]` order — the two are identical by construction. WGSL's native `determinant()`
// COULD supply the scalar det, but generating detN from the SAME cofactor recursion (over m[col][row]) keeps the
// helper self-contained + provably consistent with each entry's minor. det≈0 (singular) yields ±Inf/NaN — the
// same undefined result the interpreter + GLSL give (no guard), which is acceptable.
//
// The generated WGSL is verified correct-by-construction: transpiling each generated m[col][row] expression to a
// column-major flat array read reproduces the interpreter's matInverse for the mat2/mat3/mat4 fixtures, and this
// environment has no WebGPU adapter (the on-adapter VALUE parity is a browser-gated leg). Only the SIZES the
// kernel uses are injected, so a kernel that never calls inverse pays nothing.

/** The WGSL determinant of the submatrix over the given remaining columns × remaining rows (a size-k minor),
 *  by cofactor expansion along the FIRST remaining column — a scalar-valued expression over `m[col][row]`. */
function wgslDetExpr(cols: readonly number[], rows: readonly number[]): string {
  const k = cols.length;
  const el = (c: number, r: number): string => `m[${c}][${r}]`;
  if (k === 1) return el(cols[0]!, rows[0]!);
  if (k === 2) return `(${el(cols[0]!, rows[0]!)} * ${el(cols[1]!, rows[1]!)} - ${el(cols[1]!, rows[0]!)} * ${el(cols[0]!, rows[1]!)})`;
  const c0 = cols[0]!; const restCols = cols.slice(1);
  // Cofactor expansion along the first column: term i carries sign (−1)^i. WGSL has NO unary `+`, so the
  // FIRST term must be emitted WITHOUT a leading sign (a bare `m..*..`) — only the subsequent terms get an
  // explicit `+ `/`- ` prefix. Emitting `+ ` on term 0 produces `(+ m.. …)`, which fails WGSL validation.
  const terms = rows.map((r, i) => {
    const restRows = rows.filter((_, idx) => idx !== i);
    const body = `${el(c0, r)} * ${wgslDetExpr(restCols, restRows)}`;
    return i === 0 ? body : `${i % 2 === 0 ? '+' : '-'} ${body}`;
  });
  return `(${terms.join(' ')})`;
}
/** The full body of the `_invN` helper: `let d = 1 / <detN>; return matNxN<S>(<adjugate entries>) * d;`. The
 *  entries are emitted in the constructor's column-major slot order (col outer, row inner). */
function wgslInverseHelper(n: number, S: string): string {
  const all = Array.from({ length: n }, (_, i) => i);
  const det = wgslDetExpr(all, all);
  const entries: string[] = [];
  for (let col = 0; col < n; col++) {
    for (let row = 0; row < n; row++) {
      const sign = (col + row) % 2 === 0 ? '' : '-';
      // Entry inv[col][row] = C(row,col)/det, where the adjugate transposes the cofactor matrix. The cofactor
      // C(row,col) is the minor of M with M's ROW=row and COLUMN=col deleted. So the minor keeps the columns
      // c ≠ row and the rows r ≠ col — matching the interpreter's `subMat(c, n, dr=col, dc=row)` exactly (both
      // numerically verified to give inverse(M)·M = I for mat2/mat3/mat4).
      const cols = all.filter((c) => c !== row);   // remaining column indices (M's row=row deleted)
      const rows = all.filter((r) => r !== col);   // remaining row indices (M's column=col deleted)
      entries.push(`${sign}(${wgslDetExpr(cols, rows)})`);
    }
  }
  return [
    `fn _inv${n}(m: mat${n}x${n}<${S}>) -> mat${n}x${n}<${S}> {`,
    `  let _d = ${S}(1) / (${det});`,
    `  return mat${n}x${n}<${S}>(${entries.join(', ')}) * _d;`,
    `}`,
  ].join('\n');
}
/** Emit the `_inv{2,3,4}` prelude helper definitions for exactly the sizes used, in ascending order. */
function wgslInversePrelude(used: ReadonlySet<number>, S: string): string[] {
  return [2, 3, 4].filter((n) => used.has(n)).map((n) => wgslInverseHelper(n, S));
}

// ─── Quaternion prelude helpers (`_qslerp`, `_qmat`) — injected ONCE per shader, keyed on body references ───
// The simple quat ops (qmul/qconj/qinvert/qaxisangle/qrotate) emit inline; these two get a prelude helper because
// each has structure an inline expression can't express readably: `_qslerp` has a branch, `_qmat` builds a 3×3
// matrix from nine products of the components. Both are the SAME computation the interpreter oracle runs.
//   _qslerp: antipodal fix (dot<0 → negate b + flip dot so the shorter arc is taken), a small-angle NORMALIZED-lerp
//     fallback (dot>0.9995 → sin θ ≈ 0 would divide-by-near-zero), and the sin-weighted great-circle blend. Both
//     branches are computed and `select`ed (a kernel is pure, so the unpicked branch's cost is harmless).
//   _qmat: the column-major rotation matrix; `mat3x3<S>(...)` takes columns, so the nine entries are pushed in
//     (col outer, row inner) order — exactly the interpreter's column-major `out[c*3+r]`, so the two agree.
function wgslQuatPrelude(kernel: UserFn, S: string): string[] {
  const out: string[] = [];
  if (bodyReferencesAny(kernel, QSLERP_NAME)) out.push([
    `fn _qslerp(a: vec4<${S}>, b_in: vec4<${S}>, t: ${S}) -> vec4<${S}> {`,
    `  var d = dot(a, b_in);`,
    `  var b = b_in;`,
    `  if (d < ${S}(0)) { b = -b_in; d = -d; }`,
    `  let _lerp = normalize(a + (b - a) * t);`,
    `  let th = acos(clamp(d, ${S}(-1), ${S}(1)));`,
    `  let s = sin(th);`,
    `  let _slerp = (a * (sin((${S}(1) - t) * th) / s)) + (b * (sin(t * th) / s));`,
    `  return select(_slerp, _lerp, d > ${S}(0.9995));`,
    `}`,
  ].join('\n'));
  if (bodyReferencesAny(kernel, QMAT_NAME)) out.push([
    `fn _qmat(q: vec4<${S}>) -> mat3x3<${S}> {`,
    `  let x = q.x; let y = q.y; let z = q.z; let w = q.w;`,
    `  return mat3x3<${S}>(`,
    `    ${S}(1) - ${S}(2)*(y*y + z*z), ${S}(2)*(x*y + w*z),           ${S}(2)*(x*z - w*y),`,
    `    ${S}(2)*(x*y - w*z),           ${S}(1) - ${S}(2)*(x*x + z*z), ${S}(2)*(y*z + w*x),`,
    `    ${S}(2)*(x*z + w*y),           ${S}(2)*(y*z - w*x),           ${S}(1) - ${S}(2)*(x*x + y*y));`,
    `}`,
  ].join('\n'));
  return out;
}
const QSLERP_NAME: ReadonlySet<string> = new Set(['qslerp']);
const QMAT_NAME: ReadonlySet<string> = new Set(['qmat']);

// `LocalShapes`, `shapeOfExpr`, and `buildLocalShapes` live in gate.ts — the SHARED locals-aware shape resolver
// the emitter's `shapeOf`/`squareSizeOf` and the gate use. The gate rejects an `inverse(E)` whose square size is
// null via the SAME resolution, so a gate-accepted kernel that reaches this emitter always has a resolvable
// inverse size (the emitter never falls back to a bare, un-inverted argument).

/** Scan a kernel body for `inverse(<square mat>)` calls and collect the SIZES (2/3/4) whose `_invN` helper the
 *  prelude must define. A read-only structural AST walk (mirrors the gate's walkers); the size comes from
 *  `squareSizeOf` (locals-aware), so `inverse(mat4(...))` AND `const M = mat3(...); inverse(M)` both register. */
function collectInverseSizes(kernel: UserFn, locals: LocalShapes): Set<number> {
  const sizes = new Set<number>();
  const visitExpr = (e: Expr): void => {
    switch (e.kind) {
      case 'member': visitExpr(e.object); return;
      case 'index': visitExpr(e.object); visitExpr(e.index); return;
      case 'unary': visitExpr(e.operand); return;
      case 'binary': visitExpr(e.left); visitExpr(e.right); return;
      case 'cond': visitExpr(e.test); visitExpr(e.then); visitExpr(e.else); return;
      case 'call': {
        if (e.callee.kind === 'ident' && e.callee.name === 'inverse' && e.args[0]) { const n = squareSizeOf(e.args[0], locals); if (n !== null) sizes.add(n); }
        if (e.callee.kind !== 'ident') visitExpr(e.callee);
        e.args.forEach(visitExpr);
        if (e.block) e.block.forEach(visitStmt);
        return;
      }
      case 'object': e.entries.forEach((en) => visitExpr(en.value)); return;
      case 'array': e.elements.forEach((el) => visitExpr(el.value)); return;
      case 'arrow': if (Array.isArray(e.body)) e.body.forEach(visitStmt); else visitExpr(e.body); return;
      default: return;   // ident / number / string / bool / null
    }
  };
  const visitStmt = (s: Stmt): void => {
    switch (s.kind) {
      case 'const': case 'let': visitExpr(s.init); return;
      case 'assign': visitExpr(s.value); visitExpr(s.target); return;
      case 'expr': visitExpr(s.expr); return;
      case 'return': if (s.value) visitExpr(s.value); return;
      case 'if': visitExpr(s.test); s.then.forEach(visitStmt); s.else?.forEach(visitStmt); return;
      case 'for': visitExpr(s.iterable); s.body.forEach(visitStmt); return;
      case 'while': visitExpr(s.test); s.body.forEach(visitStmt); return;
      default: return;
    }
  };
  kernel.body.forEach(visitStmt);
  return sizes;
}

// WGSL requires `bool` at if/select-test/logical-operand positions and has NO implicit f32↔bool conversion.
// metael/JS truthiness is "non-zero is true", so we split emission into two typed modes: `emitExpr` ALWAYS
// yields the scalar type (f32) — coercing an inherently-bool subexpr to f32 via select(0,1,<bool>) — and
// `boolExpr` ALWAYS yields bool (coercing a value via `!= f32(0)`). Every value position (return, init,
// assignment RHS, arithmetic operand) flows through emitExpr, so a bool used as a number is coerced for
// free; every bool position flows through boolExpr.
const COMPARE_OPS = new Set(['==', '!=', '<', '<=', '>', '>=']);
function isBoolExpr(e: Expr): boolean {
  if (e.kind === 'binary') return COMPARE_OPS.has(e.op) || e.op === '&&' || e.op === '||';
  if (e.kind === 'unary') return e.op === '!';
  return false;
}

export function emitWgsl(kernel: UserFn, bindings: BindingTable, precision: 'f16' | 'f32', comps = 1): string {
  const S = wgslScalar(precision);
  const params = kernel.params.map((p) => (p.kind === 'name' ? p.name : ''));
  const rank = params.length;
  const buffers: Binding[] = []; const uniforms: Binding[] = [];
  for (const b of bindings.byName.values()) { if (b.role === 'buffer') buffers.push(b); else if (b.role === 'scalar' || b.role === 'uniform') uniforms.push(b); }
  const scalars = uniforms.filter((u) => u.role === 'scalar');
  const lines: string[] = [];
  if (precision === 'f16') lines.push('enable f16;');
  let binding = 0;
  for (const b of buffers) lines.push(`@group(0) @binding(${binding++}) var<storage, read> ${b.name}: array<${S}>;`);
  lines.push(`@group(0) @binding(${binding++}) var<storage, read_write> _out: array<${S}>;`);
  // Scalar-uniform members are namespaced `_u_<name>` so a scalar named `rows`/`cols`/`deps` cannot collide
  // with (or shadow) the reserved dispatch-dim members. The dispatch dims are three u32 (rows, cols, deps) —
  // present for EVERY rank so the uniform layout is rank-independent (the backend always packs 3 u32 then the
  // scalars). A rank<3 kernel simply never reads `deps`/`cols`. The backend packs scalars by POSITION
  // (Float32Array at byte offset 12, after the three u32), not by name, so this rename is emitter-internal.
  lines.push(`struct _Params { rows: u32, cols: u32, deps: u32,${scalars.map((u) => ` _u_${u.name}: ${S},`).join('')} };`);
  lines.push(`@group(0) @binding(${binding}) var<uniform> _p: _Params;`);
  // Resolve local matrix sizes ONCE (used by `inverse` to pick its `_invN` helper + emit the arg once) and inject
  // an `_invN` prelude helper for each square size the kernel inverts (none if unused). `inverse` has no WGSL builtin.
  const locals = buildLocalShapes(kernel);
  lines.push(...wgslInversePrelude(collectInverseSizes(kernel, locals), S));
  lines.push(...wgslQuatPrelude(kernel, S));
  const wg = rank === 3 ? `@workgroup_size(4, 4, 4)` : rank === 2 ? `@workgroup_size(${WORKGROUP_2D}, ${WORKGROUP_2D})` : `@workgroup_size(${WORKGROUP_1D})`;
  lines.push(`@compute ${wg}`);
  lines.push(`fn main(@builtin(global_invocation_id) gid: vec3<u32>) {`);
  if (rank === 3) {
    // Row-major flatten for output[W,H,D]: dims[0]→rows (W), dims[1]→cols (H), dims[2]→deps (D), so the flat
    // index is (x*H + y)*D + z = (gid.x*_p.cols + gid.y)*_p.deps + gid.z.
    lines.push(`  if (gid.x >= _p.rows || gid.y >= _p.cols || gid.z >= _p.deps) { return; }`);
    lines.push(`  let ${params[0]} = ${S}(gid.x);`);
    lines.push(`  let ${params[1]} = ${S}(gid.y);`);
    lines.push(`  let ${params[2]} = ${S}(gid.z);`);
    lines.push(`  let _flat = (gid.x * _p.cols + gid.y) * _p.deps + gid.z;`);
  } else if (rank === 2) {
    lines.push(`  if (gid.x >= _p.rows || gid.y >= _p.cols) { return; }`);
    lines.push(`  let ${params[0]} = ${S}(gid.x);`);
    lines.push(`  let ${params[1]} = ${S}(gid.y);`);
    lines.push(`  let _flat = gid.x * _p.cols + gid.y;`);
  } else {
    lines.push(`  if (gid.x >= _p.rows) { return; }`);
    lines.push(`  let ${params[0]} = ${S}(gid.x);`);
    lines.push(`  let _flat = gid.x;`);
  }
  lines.push(emitBody(kernel.body, S, bindings, 1, comps, { n: 0 }, locals));
  lines.push(`}`);
  return lines.join('\n');
}

// A monotonic counter for the vecN-return temp name, threaded through the body walk so EACH `return <vec>`
// gets a distinct temp (`_r0`, `_r1`, …). Two returns in one lexical scope must not both emit `let _r` — a
// duplicate `let _r` is a WGSL redefinition compile error the no-adapter emit path can't catch.
interface TempCtr { n: number }

function emitBody(body: readonly Stmt[], S: string, bindings: BindingTable, indent: number, comps: number, ctr: TempCtr, locals: LocalShapes): string {
  const out: string[] = [];
  for (const s of body) out.push(emitStmt(s, S, bindings, indent, comps, ctr, locals));
  return out.join('\n');
}
// The `return <expr>` write. A scalar output (comps=1) writes one flat slot: `_out[_flat] = <expr>;`. A
// vecN output (comps>1) computes the vec into a temp then writes N flat slots — `_out[_flat*Nu + ku] = _r.<c>`
// — keeping `_out` a flat `array<f32>` (NOT array<vec3>, which carries a 16-byte stride). `_flat` is u32, so
// the index arithmetic stays u32 (`_flat * ${comps}u + ${k}u`).
function emitReturn(s: Extract<Stmt, { kind: 'return' }>, S: string, bindings: BindingTable, pad: string, comps: number, ctr: TempCtr, locals: LocalShapes): string {
  const expr = emitExpr(s.value ?? { kind: 'number', value: 0, span: s.span }, S, bindings, locals);
  if (comps === 1) return `${pad}_out[_flat] = ${expr};\n${pad}return;`;
  const t = `_r${ctr.n++}`;   // a distinct temp per vecN return → no duplicate `let _r` in one scope
  const writes: string[] = [`${pad}let ${t} = ${expr};`];
  for (let k = 0; k < comps; k++) writes.push(`${pad}_out[_flat * ${comps}u + ${k}u] = ${t}.${'xyzw'[k]};`);
  writes.push(`${pad}return;`);
  return writes.join('\n');
}
function emitStmt(s: Stmt, S: string, bindings: BindingTable, indent: number, comps: number, ctr: TempCtr, locals: LocalShapes): string {
  const pad = '  '.repeat(indent);
  switch (s.kind) {
    case 'const': return `${pad}let ${s.name} = ${emitExpr(s.init, S, bindings, locals)};`;
    case 'let': return `${pad}var ${s.name} = ${emitExpr(s.init, S, bindings, locals)};`;
    case 'assign': return s.target.kind === 'ident' ? `${pad}${s.target.name} = ${emitExpr(s.value, S, bindings, locals)};` : `${pad}// unsupported assign`;
    case 'return': return emitReturn(s, S, bindings, pad, comps, ctr, locals);
    case 'if': return `${pad}if (${boolExpr(s.test, S, bindings, locals)}) {\n${emitBody(s.then, S, bindings, indent + 1, comps, ctr, locals)}\n${pad}}` + (s.else ? ` else {\n${emitBody(s.else, S, bindings, indent + 1, comps, ctr, locals)}\n${pad}}` : '');
    case 'for': {
      const bound = emitExpr((s.iterable as Extract<Expr, { kind: 'call' }>).args[0]!, S, bindings, locals);
      return `${pad}for (var ${s.binding} = 0.0; ${s.binding} < ${bound}; ${s.binding} = ${s.binding} + 1.0) {\n${emitBody(s.body, S, bindings, indent + 1, comps, ctr, locals)}\n${pad}}`;
    }
    case 'expr': return `${pad}${emitExpr(s.expr, S, bindings, locals)};`;
    default: return `${pad}// unsupported stmt`;
  }
}
/** Emit a bool-typed expression (for if/select-test/logical-operand positions). An inherently-bool expr
 *  (comparison / && / || / !) is emitted directly; any other (value) expr coerces via `!= f32(0)`. */
function boolExpr(e: Expr, S: string, bindings: BindingTable, locals: LocalShapes): string {
  return isBoolExpr(e) ? emitBoolCore(e, S, bindings, locals) : `(${emitExpr(e, S, bindings, locals)} != ${S}(0))`;
}
/** The bool-producing core for the inherently-bool operators. Logical operands (&&/||) and `!` operands
 *  are themselves bool positions → recurse through boolExpr; comparison operands are values → emitExpr. */
function emitBoolCore(e: Expr, S: string, bindings: BindingTable, locals: LocalShapes): string {
  if (e.kind === 'unary' && e.op === '!') return `(!${boolExpr(e.operand, S, bindings, locals)})`;
  if (e.kind === 'binary' && (e.op === '&&' || e.op === '||')) return `(${boolExpr(e.left, S, bindings, locals)} ${e.op} ${boolExpr(e.right, S, bindings, locals)})`;
  if (e.kind === 'binary') return `(${emitExpr(e.left, S, bindings, locals)} ${wgslOp(e.op)} ${emitExpr(e.right, S, bindings, locals)})`;   // a comparison: operands are f32 values
  return `(${emitExpr(e, S, bindings, locals)} != ${S}(0))`;   // unreachable given isBoolExpr, but keeps this total
}

function emitExpr(e: Expr, S: string, bindings: BindingTable, locals: LocalShapes): string {
  // `&&`/`||` in a VALUE position return an OPERAND value (JS/interpreter short-circuit: `a && b` → a if
  // falsy else b; `a || b` → a if truthy else b), NOT a 0/1 bool. Emit a value-returning select on the
  // left's truthiness (kernels are pure, so eager eval of both operands is safe — the unpicked branch's
  // value is discarded). A comparison / `!` DOES yield a bool → coerced to 0/1 below (matches the oracle's
  // Number(true)=1 / toNum(false)=0 downstream coercion).
  if (e.kind === 'binary' && (e.op === '&&' || e.op === '||')) {
    const l = emitExpr(e.left, S, bindings, locals); const r = emitExpr(e.right, S, bindings, locals); const lb = boolExpr(e.left, S, bindings, locals);
    return e.op === '&&' ? `select(${l}, ${r}, ${lb})` : `select(${r}, ${l}, ${lb})`;
  }
  // A bool-typed subexpression (comparison / !) reaching a VALUE position (return, arithmetic operand, init)
  // must coerce back to the scalar type — WGSL can't write a bool to array<f32> or add it.
  if (isBoolExpr(e)) return `select(${S}(0), ${S}(1), ${emitBoolCore(e, S, bindings, locals)})`;
  switch (e.kind) {
    case 'number': return Number.isInteger(e.value) ? `${S}(${e.value})` : String(e.value);
    case 'bool': return e.value ? `${S}(1)` : `${S}(0)`;
    case 'ident': return bindings.byName.get(e.name)?.role === 'scalar' ? `_p._u_${e.name}` : e.name;
    case 'index': return `${emitExpr(e.object, S, bindings, locals)}[u32(round(${emitExpr(e.index, S, bindings, locals)}))]`;
    case 'member': {
      const obj = e.object;
      // `bufferIdent.length` is the one whole-buffer read the gate allows: a storage array's length is
      // arrayLength(&buf) (a u32) — cast to the scalar domain since it flows into f32 arithmetic.
      if (obj.kind === 'ident' && bindings.byName.get(obj.name)?.role === 'buffer' && e.property === 'length') {
        return `${S}(arrayLength(&${obj.name}))`;
      }
      return `${emitExpr(e.object, S, bindings, locals)}.${e.property}`;   // a vec swizzle (.x/.xy) — valid WGSL
    }
    case 'unary': {
      // WGSL has NO unary `-` for matrices (only scalars/vectors). Negate a matrix componentwise via a
      // matrix–scalar multiply `(m * S(-1))` — WGSL-legal + identical to the interpreter/GLSL componentwise
      // negate. A scalar/vec operand keeps the native unary `-`. (`!` is inherently-bool → handled above.)
      if (shapeOf(e.operand, bindings, locals).kind === 'mat') return `(${emitExpr(e.operand, S, bindings, locals)} * ${S}(-1))`;
      return `(-${emitExpr(e.operand, S, bindings, locals)})`;
    }
    case 'binary': {
      const l = emitExpr(e.left, S, bindings, locals); const r = emitExpr(e.right, S, bindings, locals);
      // A zero divisor: the interpreter maps `/0` and `%0` to null → 0 as a cell (NOT the native Inf/NaN a
      // raw shader division yields). Guard both so a gate-accepted divide-by-a-possibly-zero-denominator
      // matches the oracle instead of silently writing Inf/NaN. (`%` is JS remainder — sign of the DIVIDEND
      // — lowered to the truncated remainder `a - b*trunc(a/b)` since WGSL `%` is integer-only.)
      // The `/0` guard is width-aware: a scalar `select(l/r, S(0), r==S(0))` is only type-correct when BOTH
      // operands are scalar. When either is a vec/mat, that scalar false-branch + the `vec == S(0)` test are a
      // type mismatch (a real-adapter compile error). A vec/mat divide is emitted as the NATIVE componentwise
      // divide (no guard) — `Inf`/`NaN` on a zero component, which is exactly what the interpreter's unguarded
      // componentwise divide yields, so the oracle still matches.
      if (e.op === '/') {
        const ls = shapeOf(e.left, bindings, locals); const rs = shapeOf(e.right, bindings, locals);
        if (ls.kind === 'scalar' && rs.kind === 'scalar') return `select(${l} / ${r}, ${S}(0), ${r} == ${S}(0))`;
        return `(${l} / ${r})`;
      }
      if (e.op === '%') return `select(${l} - ${r} * trunc(${l} / ${r}), ${S}(0), ${r} == ${S}(0))`;
      return `(${l} ${wgslOp(e.op)} ${r})`;
    }
    case 'cond': return `select(${emitExpr(e.else, S, bindings, locals)}, ${emitExpr(e.then, S, bindings, locals)}, ${boolExpr(e.test, S, bindings, locals)})`;
    case 'call': {
      const name = e.callee.kind === 'ident' ? e.callee.name : '';
      // A user function BINDING shadows a builtin of the same name (the interpreter resolves the closure
      // first). The gate rejects such a kernel (helper calls aren't lowerable in v1); emit a benign
      // placeholder rather than the native builtin so a non-core kernel emitted anyway never masquerades as
      // the intrinsic. Checked BEFORE the builtin branches so `function abs(){…}` never lowers to abs().
      if (bindings.byName.get(name)?.role === 'callee') return `/* shadowed builtin ${name} (helper — gate-rejected) */ ${S}(0)`;
      const spec = BUILTINS[name];
      const args = e.args.map((a) => emitExpr(a, S, bindings, locals)).join(', ');
      if (name === 'vec2' || name === 'vec3' || name === 'vec4') return `${name}<${S}>(${args})`;
      // Every matrix constructor (square matN + the six non-square matCxR) emits WGSL `matCxR<S>(...)`,
      // where C = cols, R = rows. matShapeOf reports rows/cols from the ctor name (matN → N×N), so a square
      // mat2/mat3/mat4 emits mat2x2/mat3x3/mat4x4 exactly as before.
      if (/^mat[2-4](x[2-4])?$/.test(name)) { const m = matShapeOf(e)!; return `mat${m.cols}x${m.rows}<${S}>(${args})`; }
      // Domain-restricted transcendentals: the interpreter maps an out-of-domain input to 0 (a cell). Guard
      // so a gate-accepted kernel matches the oracle instead of writing the native NaN. sqrt(x<0)→0; log(x<=0)→0.
      if (name === 'sqrt') return `select(sqrt(${args}), ${S}(0), (${args}) < ${S}(0))`;
      if (name === 'log') return `select(log(${args}), ${S}(0), (${args}) <= ${S}(0))`;
      // asin/acos: |x|>1 is out of domain; the interpreter maps it to 0 (a cell). Guard so a gate-accepted
      // kernel matches the oracle instead of the native NaN.
      if (name === 'asin' || name === 'acos') return `select(${name}(${args}), ${S}(0), abs(${args}) > ${S}(1))`;
      // atan2(y, x) has a per-target native name (no registry lowerName): WGSL spells it atan2.
      if (name === 'atan2') return `atan2(${args})`;
      // faceforward has a per-target native name (no registry lowerName): WGSL spells it faceForward.
      if (name === 'faceforward') return `faceForward(${args})`;
      // ─── quaternions (vec4 layout (x,y,z,w) = imaginary xyz + real w; hand-emitted — WGSL has no quat type) ───
      // qconj negates the imaginary part; qinvert = conj / dot(q,q) (the multiplicative inverse). Both emit the
      // arg once as a whole vec4. qmul is the Hamilton product spelled out over the .xyzw components; each operand
      // is emitted once per component reference (kernels are pure, so re-evaluating a pure sub-expression is safe).
      if (name === 'qconj') { const q = emitExpr(e.args[0]!, S, bindings, locals); return `(${q} * vec4<${S}>(-1, -1, -1, 1))`; }
      if (name === 'qinvert') { const q = emitExpr(e.args[0]!, S, bindings, locals); return `((${q} * vec4<${S}>(-1, -1, -1, 1)) / dot(${q}, ${q}))`; }
      if (name === 'qmul') { const a = emitExpr(e.args[0]!, S, bindings, locals); const b = emitExpr(e.args[1]!, S, bindings, locals); return `vec4<${S}>(${a}.w*${b}.x + ${a}.x*${b}.w + ${a}.y*${b}.z - ${a}.z*${b}.y, ${a}.w*${b}.y - ${a}.x*${b}.z + ${a}.y*${b}.w + ${a}.z*${b}.x, ${a}.w*${b}.z + ${a}.x*${b}.y - ${a}.y*${b}.x + ${a}.z*${b}.w, ${a}.w*${b}.w - ${a}.x*${b}.x - ${a}.y*${b}.y - ${a}.z*${b}.z)`; }
      // qaxisangle(axis:vec3, angle) → (axis·sin(θ/2), cos(θ/2)); qrotate(q:vec4, v:vec3) → the rotated vec3
      // via `v + 2·cross(q.xyz, cross(q.xyz, v) + q.w·v)` — the same optimized formula the interpreter uses.
      if (name === 'qaxisangle') { const ax = emitExpr(e.args[0]!, S, bindings, locals); const an = emitExpr(e.args[1]!, S, bindings, locals); return `vec4<${S}>((${ax}) * sin((${an}) * 0.5), cos((${an}) * 0.5))`; }
      if (name === 'qrotate') { const q = emitExpr(e.args[0]!, S, bindings, locals); const v = emitExpr(e.args[1]!, S, bindings, locals); return `((${v}) + 2.0 * cross((${q}).xyz, cross((${q}).xyz, (${v})) + (${q}).w * (${v})))`; }
      // qslerp — the branch (small-angle fallback + antipodal fix) lives in the `_qslerp` prelude helper (injected once).
      if (name === 'qslerp') { const a = emitExpr(e.args[0]!, S, bindings, locals); const b = emitExpr(e.args[1]!, S, bindings, locals); const t = emitExpr(e.args[2]!, S, bindings, locals); return `_qslerp(${a}, ${b}, ${t})`; }
      // qmat — the 3×3 column-major rotation matrix lives in the `_qmat` prelude helper (injected once).
      if (name === 'qmat') { const q = emitExpr(e.args[0]!, S, bindings, locals); return `_qmat(${q})`; }
      // inverse: WGSL has NO builtin inverse() — call the per-size prelude helper `_invN` (defined once in the
      // shader header for exactly the sizes used). `squareSizeOf` resolves the square size from the arg — a `matN(...)`
      // ctor, a local matrix binding (`const M = mat3(...); inverse(M)`), or a transpose/inverse chain over one. The
      // arg is emitted ONCE (as the single helper argument), so the matrix is not re-evaluated. The gate REJECTS an
      // `inverse(E)` whose square size is null (MLGPU-NOT-LOWERABLE), so a gate-accepted kernel ALWAYS resolves a
      // size here. The null branch is therefore unreachable: emit a LOUD self-flagging comment (a shader-compile
      // failure a future regression would surface) — NEVER the bare, un-inverted argument (a silent wrong result).
      if (name === 'inverse' && e.args[0]) {
        const n = squareSizeOf(e.args[0], locals);
        return n !== null
          ? `_inv${n}(${emitExpr(e.args[0], S, bindings, locals)})`
          : `/* MLGPU internal error: inverse(...) reached the emitter with an unresolvable matrix size — the gate must reject this */ _INVERSE_SIZE_UNRESOLVED_`;
      }
      // log2 / inverseSqrt: x<=0 is out of domain; the interpreter maps it to 0 (a cell). Guard so a
      // gate-accepted kernel matches the oracle instead of the native NaN. inverseSqrt has no registry
      // lowerName (its native name is the same in WGSL, but it needs the domain guard).
      if (name === 'log2') return `select(log2(${args}), ${S}(0), (${args}) <= ${S}(0))`;
      if (name === 'inverseSqrt') return `select(inverseSqrt(${args}), ${S}(0), (${args}) <= ${S}(0))`;
      if (spec?.lowerName) return `${spec.lowerName}(${args})`;
      if (WGSL_CORE_FN[name]) return `${WGSL_CORE_FN[name]}(${args})`;   // core-exact builtins (abs/clamp/round/…)
      // Unreachable for a gate-accepted kernel (every emittable head is handled above); a defensive placeholder.
      return `/* unsupported call ${name} */ ${S}(0)`;
    }
    default: return `${S}(0)`;
  }
}
function wgslOp(op: string): string {
  // The direct-mapping binary operators pass through unchanged. `%` never reaches here — it is lowered to a
  // truncated-remainder expression at the `binary` case (WGSL `%` is integer-only + would diverge on floats).
  return op;
}

// ─── The REDUCTION emitter: a WGSL workgroup-shared tree reduction (a WebGPU multi-pass fold) ───
// A reduction folds N inputs → 1 scalar. Unlike the WebGL2 ping-pong (fragment passes over ping-pong textures),
// WebGPU has a real compute stage with workgroup-shared memory, so each workgroup of G threads loads G elements
// into `var<workgroup> _scratch: array<f32, G>`, tree-folds them in shared memory with `workgroupBarrier()`
// between halving steps, and thread 0 writes the workgroup's partial to `_out[workgroup_id]`. The driver
// (webgpu's `dispatchReduce`) runs this MULTI-PASS: N → ceil(N/G) partials → … → 1. This ONE shader runs every
// pass (only the `_in`/`_out` buffers + `_p.inLen` change). The reducer's binary op is emitted ONCE as
// `fn _reduce(acc, x) -> f32` reusing `emitExpr` — the SAME scalar lowering the map emitter uses (arithmetic,
// comparison, ternary→select, the math builtins), so `_reduce`'s body is the tested emit path, not hand-rolled.
//
// The reduce path is f32-only (metael's numeric model; parity with the highp reduce GLSL) — no precision knob.
// A reducer is PURE over (acc, x) (gateReducer), so there are NO buffer/coord/vec bindings — only its 2 scalar
// params + optional closed-over scalar CONSTANTS (role:'scalar'), which ride `_RParams` as `_u_<name>` (the
// emitExpr scalar lowering) and are set by the driver. G is baked as a const: `@workgroup_size(G)`,
// `array<f32, Gu>`, and an initial fold stride of G/2. G is the SHARED `REDUCE_TILE` — the same constant the
// WebGL2 reduce tile, webgpu's per-pass grid (`ceil(len/G)` workgroups), and the engine's dispatch-limit bound
// use, so a workgroup a lane loads == a tile a partial folds across every backend (they cannot drift).
const REDUCE_WORKGROUP = REDUCE_TILE;

/** A statement in the reducer's `_reduce(acc, x)` body — like `emitStmt` but a `return` emits a NORMAL
 *  `return <expr>;` (not the map path's `_out[_flat] = …` cell write). Reuses `emitExpr`/`boolExpr`. */
function emitReduceStmt(s: Stmt, S: string, bindings: BindingTable, indent: number, locals: LocalShapes): string {
  const pad = '  '.repeat(indent);
  const sub = (body: readonly Stmt[], ind: number): string => body.map((x) => emitReduceStmt(x, S, bindings, ind, locals)).join('\n');
  switch (s.kind) {
    case 'const': return `${pad}let ${s.name} = ${emitExpr(s.init, S, bindings, locals)};`;
    case 'let': return `${pad}var ${s.name} = ${emitExpr(s.init, S, bindings, locals)};`;
    case 'assign': return s.target.kind === 'ident' ? `${pad}${s.target.name} = ${emitExpr(s.value, S, bindings, locals)};` : `${pad}// unsupported assign`;
    case 'return': return `${pad}return ${emitExpr(s.value ?? { kind: 'number', value: 0, span: s.span }, S, bindings, locals)};`;
    case 'if': return `${pad}if (${boolExpr(s.test, S, bindings, locals)}) {\n${sub(s.then, indent + 1)}\n${pad}}` + (s.else ? ` else {\n${sub(s.else, indent + 1)}\n${pad}}` : '');
    case 'for': {
      const bound = emitExpr((s.iterable as Extract<Expr, { kind: 'call' }>).args[0]!, S, bindings, locals);
      return `${pad}for (var ${s.binding} = 0.0; ${s.binding} < ${bound}; ${s.binding} = ${s.binding} + 1.0) {\n${sub(s.body, indent + 1)}\n${pad}}`;
    }
    case 'expr': return `${pad}${emitExpr(s.expr, S, bindings, locals)};`;
    default: return `${pad}// unsupported stmt`;
  }
}

/** Emit the WGSL workgroup-shared tree reduction for `reducer` (its 2 scalar params `acc`, `x` bind as f32).
 *  The driver (webgpu's `dispatchReduce`) runs this ONCE PER PASS, binding the current element buffer as `_in`,
 *  a fresh ceil(len/G)-partial buffer as `_out`, and a `_RParams` uniform (inLen + identity + any closed-over
 *  scalar constant). `identity` is baked into the doc comment only for provenance; at run time it flows through
 *  the `_p.identity` uniform (the driver sets it per pass) so the out-of-range lanes fold the fold-neutral. */
export function emitReduceWgsl(reducer: UserFn, bindings: BindingTable, _identity: number): string {
  const S = 'f32';   // the reduce path is f32-only (parity with the highp reduce GLSL)
  const G = REDUCE_WORKGROUP;
  const params = reducer.params.map((p) => (p.kind === 'name' ? p.name : ''));
  const acc = params[0] || 'acc';
  const x = params[1] || 'x';
  const scalars: Binding[] = [];
  for (const b of bindings.byName.values()) if (b.role === 'scalar') scalars.push(b);
  const locals = buildLocalShapes(reducer);
  const body = reducer.body.map((s) => emitReduceStmt(s, S, bindings, 1, locals)).join('\n');

  const L: string[] = [];
  L.push(`@group(0) @binding(0) var<storage, read> _in: array<${S}>;`);
  L.push(`@group(0) @binding(1) var<storage, read_write> _out: array<${S}>;`);
  // The params block: the current element count (`inLen`, a u32 the load guard compares against), the fold
  // `identity` (f32), then any closed-over scalar constant as `_u_<name>` (the emitExpr scalar lowering). The
  // driver packs inLen as a u32 then identity + the scalars as f32 (offsets 0,4,8,…) each pass. Declared BEFORE
  // the `var<uniform>` that uses it, mirroring the map emitter's `struct _Params` (safe against toolchains
  // stricter than the WGSL spec's out-of-order module rule).
  L.push(`struct _RParams { inLen: u32, identity: ${S},${scalars.map((s) => ` _u_${s.name}: ${S},`).join('')} };`);
  L.push(`@group(0) @binding(2) var<uniform> _p: _RParams;`);
  // Workgroup-shared scratch: G elements, one per thread. G baked as a const array size.
  L.push(`var<workgroup> _scratch: array<${S}, ${G}u>;`);
  // `inverse` has no WGSL builtin — inject an `_invN` helper for each square size the reducer inverts (none if unused).
  L.push(...wgslInversePrelude(collectInverseSizes(reducer, locals), S));
  L.push(...wgslQuatPrelude(reducer, S));
  L.push(`fn _reduce(${acc}: ${S}, ${x}: ${S}) -> ${S} {`);
  L.push(body);
  L.push(`}`);
  L.push(`@compute @workgroup_size(${G})`);
  L.push(`fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {`);
  L.push(`  let _i = gid.x;`);
  // Load G elements into shared memory: element `_i` for an in-range lane, else the fold-neutral IDENTITY —
  // NEVER a garbage/OOB fold. (WGSL bounds-clamps an OOB storage read, and `select` discards it anyway; the
  // identity is the CORRECT neutral for a partial workgroup's unused lanes — the analogue of the WebGL2
  // `if (idx < _inLen)` guard.) `select(f, t, cond)` → t when cond true.
  L.push(`  _scratch[lid.x] = select(_p.identity, _in[_i], _i < _p.inLen);`);
  L.push(`  workgroupBarrier();`);   // all lanes' loads are visible before the fold begins
  // Tree fold in shared memory: halve the active range each step, a barrier after EACH step so every lane sees
  // the writes. The stride evolves identically across all invocations (uniform control flow → the barrier is
  // reached the same number of times by every lane, as WGSL requires). `_reduce` combines two scratch lanes
  // only (never re-applies identity), so the neutral element is folded once-per-unused-lane at load — matching
  // the neutral-identity contract the linear oracle + the WebGL2 leg share.
  L.push(`  var _stride = ${G / 2}u;`);
  L.push(`  loop {`);
  L.push(`    if (_stride == 0u) { break; }`);
  L.push(`    if (lid.x < _stride) { _scratch[lid.x] = _reduce(_scratch[lid.x], _scratch[lid.x + _stride]); }`);
  L.push(`    workgroupBarrier();`);
  L.push(`    _stride = _stride / 2u;`);
  L.push(`  }`);
  // Thread 0 writes this workgroup's partial. The driver sizes `_out` as ceil(inLen/G), so `wid.x` is in range.
  L.push(`  if (lid.x == 0u) { _out[wid.x] = _scratch[0]; }`);
  L.push(`}`);
  return L.join('\n');
}

// ─── The HISTOGRAM emitter: a WGSL data-dependent ATOMIC SCATTER (a WebGPU single-pass count) ───
// A histogram maps each input element to a BIN INDEX and increments that bin's COUNT. WebGPU has real storage
// atomics, so one thread per input element computes its bin index via the bin-mapper (`_binOf`, emitted ONCE
// via `emitExpr` — the SAME scalar lowering the map/reduce emitters use) and does `atomicAdd(&_bins[b], 1u)`.
// `_bins` is a `var<storage, read_write> array<atomic<u32>>` the driver ZEROES before the dispatch (a fresh
// WebGPU storage buffer created with mappedAtCreation:false is zero-initialized per spec). An OUT-OF-RANGE bin
// index (`_b < 0 || _b >= bins`) is DROPPED (not counted) via the bounds guard — the same drop the CPU oracle
// (`cpuHistogram`) does, so the two agree. This path is f32-only (metael's numeric model) — no precision knob.
//
// NOTE: there is NO WebGPU adapter in this environment, so this leg's VALUE path is NOT runtime-tested here —
// the WGSL is structurally snapshotted + compiled on a real device via the gated browser test (which skips
// absent an adapter). The scatter is written correct-by-inspection against the CPU oracle.

/** A statement in the bin-mapper's `_binOf(x)` body — like `emitReduceStmt` but for the 1-param mapper: a
 *  `return` emits a NORMAL `return <expr>;`. Reuses `emitExpr`/`boolExpr` (the tested scalar lowering). */
function emitHistogramStmt(s: Stmt, S: string, bindings: BindingTable, indent: number, locals: LocalShapes): string {
  const pad = '  '.repeat(indent);
  const sub = (body: readonly Stmt[], ind: number): string => body.map((x) => emitHistogramStmt(x, S, bindings, ind, locals)).join('\n');
  switch (s.kind) {
    case 'const': return `${pad}let ${s.name} = ${emitExpr(s.init, S, bindings, locals)};`;
    case 'let': return `${pad}var ${s.name} = ${emitExpr(s.init, S, bindings, locals)};`;
    case 'assign': return s.target.kind === 'ident' ? `${pad}${s.target.name} = ${emitExpr(s.value, S, bindings, locals)};` : `${pad}// unsupported assign`;
    case 'return': return `${pad}return ${emitExpr(s.value ?? { kind: 'number', value: 0, span: s.span }, S, bindings, locals)};`;
    case 'if': return `${pad}if (${boolExpr(s.test, S, bindings, locals)}) {\n${sub(s.then, indent + 1)}\n${pad}}` + (s.else ? ` else {\n${sub(s.else, indent + 1)}\n${pad}}` : '');
    case 'for': {
      const bound = emitExpr((s.iterable as Extract<Expr, { kind: 'call' }>).args[0]!, S, bindings, locals);
      return `${pad}for (var ${s.binding} = 0.0; ${s.binding} < ${bound}; ${s.binding} = ${s.binding} + 1.0) {\n${sub(s.body, indent + 1)}\n${pad}}`;
    }
    case 'expr': return `${pad}${emitExpr(s.expr, S, bindings, locals)};`;
    default: return `${pad}// unsupported stmt`;
  }
}

/** Emit the WGSL atomic-scatter histogram for `binMapper` (its 1 scalar param `x` binds as f32). One thread per
 *  input element maps `x` → a bin index, dropping out-of-range indices, and `atomicAdd`s that bin. `bins` is
 *  baked into the doc comment only for provenance; at run time it flows through the `_p.bins` uniform (the
 *  driver sets it) so the emitter is bins-agnostic. The scatter path is f32-only. */
export function emitHistogramWgsl(binMapper: UserFn, bindings: BindingTable, _bins: number): string {
  const S = 'f32';   // the histogram path is f32-only (metael's numeric model)
  const params = binMapper.params.map((p) => (p.kind === 'name' ? p.name : ''));
  const x = params[0] || 'x';
  const scalars: Binding[] = [];
  for (const b of bindings.byName.values()) if (b.role === 'scalar') scalars.push(b);
  const locals = buildLocalShapes(binMapper);
  const body = binMapper.body.map((s) => emitHistogramStmt(s, S, bindings, 1, locals)).join('\n');

  const L: string[] = [];
  L.push(`@group(0) @binding(0) var<storage, read> _in: array<${S}>;`);
  // The bin counts: `array<atomic<u32>>` so concurrent lanes scatter into the same bin safely via atomicAdd.
  // The driver ZEROES this buffer before the dispatch (a fresh WebGPU storage buffer is zero-initialized).
  L.push(`@group(0) @binding(1) var<storage, read_write> _bins: array<atomic<u32>>;`);
  // The params block: the input element count (`inLen`, a u32 the load guard compares against), the bin count
  // (`bins`, a u32 the bounds guard compares against), then any closed-over scalar constant as `_u_<name>`
  // (the emitExpr scalar lowering). Declared BEFORE the `var<uniform>` that uses it, mirroring the map/reduce
  // emitters (safe against toolchains stricter than the WGSL spec's out-of-order module rule).
  L.push(`struct _HParams { inLen: u32, bins: u32,${scalars.map((s) => ` _u_${s.name}: ${S},`).join('')} };`);
  L.push(`@group(0) @binding(2) var<uniform> _p: _HParams;`);
  // `inverse` has no WGSL builtin — inject an `_invN` helper for each square size the bin-mapper inverts (none if unused).
  L.push(...wgslInversePrelude(collectInverseSizes(binMapper, locals), S));
  L.push(...wgslQuatPrelude(binMapper, S));
  L.push(`fn _binOf(${x}: ${S}) -> ${S} {`);
  L.push(body);
  L.push(`}`);
  L.push(`@compute @workgroup_size(${HISTOGRAM_WORKGROUP})`);
  L.push(`fn main(@builtin(global_invocation_id) gid: vec3<u32>) {`);
  L.push(`  let _i = gid.x;`);
  L.push(`  if (_i >= _p.inLen) { return; }`);
  // Compute the bin index as f32 FIRST, then drop a NON-FINITE index BEFORE the i32 cast: the CPU oracle drops
  // a non-finite bin (`Number.isFinite(b)`), but `i32(NaN)` in WGSL is INDETERMINATE (often 0) → a NaN bin
  // would be miscounted as bin 0, diverging from the oracle. `_bf == _bf` is false ONLY for NaN (NaN != NaN),
  // so it drops NaN. (An Inf bin index: `i32(±Inf)` is impl-defined, but the `_b < bins`/`_b >= 0` bounds guard
  // below drops it on the common saturating impl; the NaN self-compare handles the divergence that mattered.)
  // Then truncate toward zero (i32(f32)) — matches the CPU oracle's Math.trunc(Number(...)) — and drop an
  // out-of-range bin (< 0 or >= bins) via the bounds guard: the SAME drops cpuHistogram does, so both agree.
  L.push(`  let _bf = _binOf(_in[_i]);`);
  L.push(`  if (_bf == _bf) {`);   // NaN != NaN → false for NaN, dropping it (matches Number.isFinite drop)
  L.push(`    let _b = i32(_bf);`);
  L.push(`    if (_b >= 0 && _b < i32(_p.bins)) { atomicAdd(&_bins[u32(_b)], 1u); }`);
  L.push(`  }`);
  L.push(`}`);
  return L.join('\n');
}
