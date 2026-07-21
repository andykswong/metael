// The GLSL-ES-3.0 compute-via-fragment emitter: kernel AST → a fragment shader that computes ONE output
// cell per fragment. WebGL2 has no compute stage, so a compute kernel is run as a fullscreen-quad fragment
// shader whose gl_FragCoord picks the cell; inputs are float textures (sampler2D, R32F, one element per
// texel), the output is written to an RGBA32F render target's R channel and read back with readPixels.
//
// The body mirrors the WGSL emitter's walker: the whole scalar body is `float` (metael's numeric model),
// bool positions are coerced (if/ternary tests via `!= 0.0`, a bool value via `float(...)`), a bounded
// `for … of range(n)` becomes a float-counter C loop, `%` lowers to `mod()` (GLSL `%` is integer-only),
// vec/mat construct + operate natively. Unlike WGSL, GLSL has no `let` type inference — a `const`/`let`
// must be declared with a concrete type, so each initializer's GLSL type is inferred (float / vecN / matN).
import type { UserFn, Expr, Stmt } from '@metael/lang';
import { BUILTINS } from '@metael/math/lang';
import type { Binding, BindingTable } from './binding.ts';
import { bodyReferencesAny } from './binding.ts';

// Core-exact builtins the gate accepts but that carry no registry `lowerName` — they map to a native GLSL
// function of the same name (round → roundEven so ties-to-even matches the interpreter/CPU + WGSL).
// `mod` maps to GLSL's native `mod`, which is ALREADY floored (sign follows the divisor) — exactly metael's
// core.mod — so no bespoke case is needed (unlike WGSL, whose `%` is truncated). Its result is componentwise
// for a vec arg (like the other core fns), which `glslType`'s GLSL_CORE_FN branch declares correctly.
const GLSL_CORE_FN: Readonly<Record<string, string>> = { min: 'min', max: 'max', abs: 'abs', sign: 'sign', floor: 'floor', ceil: 'ceil', clamp: 'clamp', round: 'roundEven', mod: 'mod' };
const VEC_CTORS = new Set(['vec2', 'vec3', 'vec4', 'mat2', 'mat3', 'mat4', 'mat2x3', 'mat2x4', 'mat3x2', 'mat3x4', 'mat4x2', 'mat4x3']);
const COMPARE_OPS = new Set(['==', '!=', '<', '<=', '>', '>=']);

function isBoolExpr(e: Expr): boolean {
  if (e.kind === 'binary') return COMPARE_OPS.has(e.op) || e.op === '&&' || e.op === '||';
  if (e.kind === 'unary') return e.op === '!';
  return false;
}

const QSLERP_NAME: ReadonlySet<string> = new Set(['qslerp']);
const QMAT_NAME: ReadonlySet<string> = new Set(['qmat']);
// ─── Quaternion GLSL prelude helpers (`_qslerp`, `_qmat`) — mirror the WGSL versions; injected ONCE before `main` ───
// _qslerp: antipodal fix (dot<0 → negate b + flip dot) + small-angle NORMALIZED-lerp fallback (dot>0.9995) + the
//   sin-weighted great-circle blend — the SAME three cases the interpreter oracle computes. GLSL-ES-3.0 allows an
//   `if`/early-`return` inside a function, so it reads straightforwardly.
// _qmat: the column-major rotation matrix. GLSL `mat3(...)` takes columns, so the nine entries are given in
//   (col outer, row inner) order — exactly the interpreter's column-major `out[c*3+r]`, so the two agree.
function glslQuatPrelude(kernel: UserFn): string[] {
  const out: string[] = [];
  if (bodyReferencesAny(kernel, QSLERP_NAME)) out.push([
    'vec4 _qslerp(vec4 a, vec4 b_in, float t) {',
    '  float d = dot(a, b_in);',
    '  vec4 b = b_in;',
    '  if (d < 0.0) { b = -b_in; d = -d; }',
    '  if (d > 0.9995) { return normalize(a + (b - a) * t); }',
    '  float th = acos(clamp(d, -1.0, 1.0));',
    '  float s = sin(th);',
    '  return a * (sin((1.0 - t) * th) / s) + b * (sin(t * th) / s);',
    '}',
  ].join('\n'));
  if (bodyReferencesAny(kernel, QMAT_NAME)) out.push([
    'mat3 _qmat(vec4 q) {',
    '  float x = q.x; float y = q.y; float z = q.z; float w = q.w;',
    '  return mat3(',
    '    1.0 - 2.0*(y*y + z*z), 2.0*(x*y + w*z),       2.0*(x*z - w*y),',
    '    2.0*(x*y - w*z),       1.0 - 2.0*(x*x + z*z), 2.0*(y*z + w*x),',
    '    2.0*(x*z + w*y),       2.0*(y*z - w*x),       1.0 - 2.0*(x*x + y*y));',
    '}',
  ].join('\n'));
  return out;
}

const COUNTONEBITS_NAME: ReadonlySet<string> = new Set(['countOneBits']);
const REVERSEBITS_NAME: ReadonlySet<string> = new Set(['reverseBits']);
// ─── the 32-bit bit ops as GLSL ES 3.00 prelude helpers (injected ONCE before `main`, keyed on body references) ───
// GLSL ES 3.00 (the WebGL2 target) has NO bitCount/bitfieldReverse (an ES 3.10 feature), but it DOES have `uint`,
// the bitwise operators, and int↔uint reinterpret casts — enough to hand-roll both ops (mirroring the quat prelude
// pattern). Each helper takes/returns `float` (the scalar body type): the arg is reinterpreted as a 32-bit unsigned
// integer via `uint(int(x))` — TRUNCATE toward zero (matching the interpreter's ToUint32 / `x >>> 0` coercion), NOT
// round. `int(float)` truncates toward zero per the GLSL ES 3.00 spec; the double cast goes through `int` first
// because `uint(negativeFloat)` is undefined, whereas int→uint is a defined bit-pattern reinterpret — so a negative
// input wraps like `>>>0` (uint(int(-1.0)) = 0xFFFFFFFF). Truncation (not roundEven) is what `>>>0` does, so a
// fractional input like 3.9 counts bits of 3, matching the interpreter, rather than rounding up to 4.
//   _countOneBits: the SWAR population count — pairwise bit sums (0x55), 2-bit sums (0x33), nibble sums (0x0F),
//     then a `* 0x01010101 >> 24` horizontal byte sum. The result is 0..32 → always f32-exact.
//   _reverseBits: the standard 32-bit reverse — swap adjacent bits, then 2-bit fields, nibbles, bytes, 16-bit
//     halves. `float(v)` is f32-exact for an f32-exact integer input (reversal preserves the ≤24-bit span).
function glslBitPrelude(kernel: UserFn): string[] {
  const out: string[] = [];
  if (bodyReferencesAny(kernel, COUNTONEBITS_NAME)) out.push([
    'float _countOneBits(float x) {',
    '  uint v = uint(int(x));',
    '  v = v - ((v >> 1u) & 0x55555555u);',
    '  v = (v & 0x33333333u) + ((v >> 2u) & 0x33333333u);',
    '  v = (v + (v >> 4u)) & 0x0F0F0F0Fu;',
    '  return float((v * 0x01010101u) >> 24u);',
    '}',
  ].join('\n'));
  if (bodyReferencesAny(kernel, REVERSEBITS_NAME)) out.push([
    'float _reverseBits(float x) {',
    '  uint v = uint(int(x));',
    '  v = ((v & 0xAAAAAAAAu) >> 1u) | ((v & 0x55555555u) << 1u);',
    '  v = ((v & 0xCCCCCCCCu) >> 2u) | ((v & 0x33333333u) << 2u);',
    '  v = ((v & 0xF0F0F0F0u) >> 4u) | ((v & 0x0F0F0F0Fu) << 4u);',
    '  v = ((v & 0xFF00FF00u) >> 8u) | ((v & 0x00FF00FFu) << 8u);',
    '  v = (v >> 16u) | (v << 16u);',
    '  return float(v);',
    '}',
  ].join('\n'));
  return out;
}

// A local-name → GLSL-type environment threaded through emission so a `const`/`let` can be declared with the
// right type (float by default; vecN/matN for vec-bearing intermediates).
type TypeEnv = Map<string, string>;

/** Emit the GLSL-ES 3.0 compute-via-fragment shader for a map kernel: one sampler/texture per buffer input,
 *  the packed uniforms, and the per-fragment coordinate decode are derived from `bindings`. `precision` maps
 *  to the float qualifier (`highp` for f32, `mediump` for f16); `comps` sets the per-cell output width. Throws
 *  (a loud gate↔emitter drift) if a gate-accepted construct has no GLSL lowering. */
export function emitGlsl(kernel: UserFn, bindings: BindingTable, precision: 'f16' | 'f32', comps = 1): string {
  const fprec = precision === 'f16' ? 'mediump' : 'highp';
  const params = kernel.params.map((p) => (p.kind === 'name' ? p.name : ''));
  const rank = params.length;
  const buffers: Binding[] = []; const scalars: Binding[] = [];
  for (const b of bindings.byName.values()) { if (b.role === 'buffer') buffers.push(b); else if (b.role === 'scalar') scalars.push(b); }
  const tenv: TypeEnv = new Map();
  for (const p of params) if (p) tenv.set(p, 'float');
  for (const s of scalars) tenv.set(s.name, 'float');

  const L: string[] = [];
  L.push('#version 300 es');
  L.push(`precision ${fprec} float;`);
  L.push('precision highp int;');
  L.push('precision highp sampler2D;');
  // Output dims + the output texture width (the flat-index → fragment map). Set by the backend as uniforms.
  L.push('uniform int _rows; uniform int _cols; uniform int _texW; uniform int _deps;');
  // Each input buffer: a float texture + its texture width (for the texel map) + its element count (.length).
  for (const b of buffers) L.push(`uniform sampler2D ${b.name}; uniform int ${b.name}_texW; uniform int ${b.name}_len;`);
  // Scalar uniforms are namespaced `_u_<name>` (mirroring the WGSL emitter's `_p._u_<name>`): a user scalar
  // named `_rows`/`_cols`/`_texW` would otherwise redeclare a reserved dispatch uniform (a GLSL compile
  // error). The backend sets these by the same `_u_`-prefixed name.
  for (const s of scalars) L.push(`uniform float _u_${s.name};`);
  L.push('out vec4 _frag;');
  L.push('float _fetch(sampler2D t, int idx, int w) { return texelFetch(t, ivec2(idx % w, idx / w), 0).r; }');
  L.push(...glslQuatPrelude(kernel));
  L.push(...glslBitPrelude(kernel));
  L.push('void main() {');
  L.push('  int _fx = int(gl_FragCoord.x); int _fy = int(gl_FragCoord.y);');
  if (rank === 3) {
    // No 3-D render target in WebGL2: the output texture is FLAT (one texel per cell, like the 1-D case), so
    // `_flat` is the texel index `_fy * _texW + _fx`. Decompose it back into (x,y,z) via the shared row-major
    // flatten `_flat = (x*H + y)*D + z` for output[W,H,D] (W=_rows, H=_cols, D=_deps): z = _flat % D,
    // y = (_flat / D) % H, x = _flat / (H*D). Inlined into the float casts (parity with the WGSL/CPU coords).
    L.push('  int _flat = _fy * _texW + _fx;');
    L.push(`  float ${params[0]} = float(_flat / (_cols * _deps)); float ${params[1]} = float((_flat / _deps) % _cols); float ${params[2]} = float(_flat % _deps);`);
  } else if (rank === 2) {
    L.push('  int _flat = _fy * _cols + _fx;');
    L.push(`  float ${params[0]} = float(_fy); float ${params[1]} = float(_fx);`);
  } else {
    L.push('  int _flat = _fy * _texW + _fx;');
    L.push(`  float ${params[0]} = float(_flat);`);
  }
  // The full cell count is _rows*_cols for rank 1/2 (cols=1 for 1-D) and _rows*_cols*_deps for rank 3.
  L.push(`  if (_flat >= ${rank === 3 ? '_rows * _cols * _deps' : '_rows * _cols'}) { discard; }`);
  L.push(emitBody(kernel.body, tenv, bindings, 1, comps, { n: 0 }));
  L.push('}');
  return L.join('\n');
}

// A monotonic counter for the vecN-return temp name, threaded through the body walk so EACH `return <vec>`
// gets a distinct temp (`_r0`, `_r1`, …). Two returns in one lexical scope must not both emit `<t> _r` — a
// duplicate declaration is a GLSL redefinition compile error the no-adapter emit path can't catch.
interface TempCtr { n: number }

function emitBody(body: readonly Stmt[], tenv: TypeEnv, bindings: BindingTable, indent: number, comps: number, ctr: TempCtr): string {
  return body.map((s) => emitStmt(s, tenv, bindings, indent, comps, ctr)).join('\n');
}
// The `return <expr>` write into the cell's RGBA32F texel. A scalar output (comps=1) writes the R channel:
// `_frag = vec4(<expr>, 0.0, 0.0, 1.0);`. A vecN output packs the cell's N components into the texel's
// leading channels (R,G,B,A), padding the unused ones with 0 — the readback gathers channel k as component
// k. The output texture stays CELL-indexed (one texel per cell); only the texel's channels carry N comps.
function emitReturn(s: Extract<Stmt, { kind: 'return' }>, tenv: TypeEnv, bindings: BindingTable, pad: string, comps: number, ctr: TempCtr): string {
  const expr = emitExpr(s.value ?? { kind: 'number', value: 0, span: s.span }, tenv, bindings);
  if (comps === 1) return `${pad}_frag = vec4(${expr}, 0.0, 0.0, 1.0);\n${pad}return;`;
  const ty = ['', '', 'vec2', 'vec3', 'vec4'][comps]!;
  const t = `_r${ctr.n++}`;   // a distinct temp per vecN return → no duplicate declaration in one scope
  // Pack the vec's N components into the texel's leading channels; the readback gathers exactly the first
  // `comps` channels, so the unused ones are inert — pad any gap with 0.0 and the alpha with 1.0 (mirroring
  // the scalar path's `vec4(e, 0.0, 0.0, 1.0)`). vec4 → all four are components.
  const chans = [`${t}.x`, `${t}.y`, `${t}.z`, `${t}.w`].slice(0, comps);
  while (chans.length < 3) chans.push('0.0');
  if (chans.length < 4) chans.push('1.0');
  return `${pad}${ty} ${t} = ${expr};\n${pad}_frag = vec4(${chans.join(', ')});\n${pad}return;`;
}
function emitStmt(s: Stmt, tenv: TypeEnv, bindings: BindingTable, indent: number, comps: number, ctr: TempCtr): string {
  const pad = '  '.repeat(indent);
  switch (s.kind) {
    case 'const': case 'let': {
      const t = glslType(s.init, tenv, bindings);
      tenv.set(s.name, t);
      return `${pad}${t} ${s.name} = ${emitExpr(s.init, tenv, bindings)};`;
    }
    case 'assign': return s.target.kind === 'ident' ? `${pad}${s.target.name} = ${emitExpr(s.value, tenv, bindings)};` : `${pad}// unsupported assign`;
    case 'return': return emitReturn(s, tenv, bindings, pad, comps, ctr);
    case 'if': return `${pad}if (${boolExpr(s.test, tenv, bindings)}) {\n${emitBody(s.then, tenv, bindings, indent + 1, comps, ctr)}\n${pad}}` + (s.else ? ` else {\n${emitBody(s.else, tenv, bindings, indent + 1, comps, ctr)}\n${pad}}` : '');
    case 'for': {
      const bound = emitExpr((s.iterable as Extract<Expr, { kind: 'call' }>).args[0]!, tenv, bindings);
      tenv.set(s.binding, 'float');   // the loop var feeds float index arithmetic (parity with WGSL/CPU)
      return `${pad}for (float ${s.binding} = 0.0; ${s.binding} < ${bound}; ${s.binding} = ${s.binding} + 1.0) {\n${emitBody(s.body, tenv, bindings, indent + 1, comps, ctr)}\n${pad}}`;
    }
    case 'expr': return `${pad}${emitExpr(s.expr, tenv, bindings)};`;
    default: return `${pad}// unsupported stmt`;
  }
}
/** A bool-typed expression (if/ternary-test position). An inherently-bool expr is emitted directly; any
 *  value expr is coerced via `!= 0.0`. */
function boolExpr(e: Expr, tenv: TypeEnv, bindings: BindingTable): string {
  return isBoolExpr(e) ? emitBoolCore(e, tenv, bindings) : `(${emitExpr(e, tenv, bindings)} != 0.0)`;
}
function emitBoolCore(e: Expr, tenv: TypeEnv, bindings: BindingTable): string {
  if (e.kind === 'unary' && e.op === '!') return `(!${boolExpr(e.operand, tenv, bindings)})`;
  if (e.kind === 'binary' && (e.op === '&&' || e.op === '||')) return `(${boolExpr(e.left, tenv, bindings)} ${e.op} ${boolExpr(e.right, tenv, bindings)})`;
  if (e.kind === 'binary') return `(${emitExpr(e.left, tenv, bindings)} ${e.op} ${emitExpr(e.right, tenv, bindings)})`;   // a comparison: operands are float values
  return `(${emitExpr(e, tenv, bindings)} != 0.0)`;   // unreachable given isBoolExpr, but keeps this total
}

function emitExpr(e: Expr, tenv: TypeEnv, bindings: BindingTable): string {
  // `&&`/`||` in a VALUE position return an OPERAND value (JS/interpreter short-circuit: `a && b` → a if
  // falsy else b; `a || b` → a if truthy else b), NOT a 0/1 bool. Emit a value-returning ternary on the
  // left's truthiness (kernels are pure → eager eval of both operands is safe). A comparison / `!` DOES
  // yield a bool → coerced to float(...) below (matches the oracle's downstream Number(true)/toNum(false)).
  if (e.kind === 'binary' && (e.op === '&&' || e.op === '||')) {
    const l = emitExpr(e.left, tenv, bindings); const r = emitExpr(e.right, tenv, bindings); const lb = boolExpr(e.left, tenv, bindings);
    return e.op === '&&' ? `(${lb} ? ${r} : ${l})` : `(${lb} ? ${l} : ${r})`;
  }
  // A bool-typed subexpr (comparison / !) reaching a VALUE position must coerce back to float — GLSL can't add a bool.
  if (isBoolExpr(e)) return `float(${emitBoolCore(e, tenv, bindings)})`;
  switch (e.kind) {
    case 'number': return Number.isInteger(e.value) ? `${e.value}.0` : String(e.value);
    case 'bool': return e.value ? '1.0' : '0.0';
    case 'ident': return bindings.byName.get(e.name)?.role === 'scalar' ? `_u_${e.name}` : e.name;   // scalar uniforms are namespaced; locals are bare
    case 'index': {
      const obj = e.object;
      // Round (not truncate) the float index to the nearest int, matching the WGSL emitter's u32(round(...)):
      // an in-bounds integer index is f32-exact, but float ACCUMULATION (row*N+k) can land a hair below the
      // integer (8.9999995 for 9) — round recovers it, truncation would read the wrong (lower) element.
      if (obj.kind === 'ident' && bindings.byName.get(obj.name)?.role === 'buffer') {
        return `_fetch(${obj.name}, int(roundEven(${emitExpr(e.index, tenv, bindings)})), ${obj.name}_texW)`;
      }
      return `${emitExpr(obj, tenv, bindings)}[int(roundEven(${emitExpr(e.index, tenv, bindings)}))]`;   // dynamic vec index
    }
    case 'member': {
      const obj = e.object;
      // `bufferIdent.length` is the one whole-buffer read the gate allows — the backend passes the element
      // count as a uniform int (a sampler2D has no length in the shader).
      if (obj.kind === 'ident' && bindings.byName.get(obj.name)?.role === 'buffer' && e.property === 'length') {
        return `float(${obj.name}_len)`;
      }
      return `${emitExpr(obj, tenv, bindings)}.${e.property}`;   // a vec swizzle (.x/.xy)
    }
    case 'unary': return `(-${emitExpr(e.operand, tenv, bindings)})`;   // `!` is inherently-bool → handled above
    case 'binary': {
      const l = emitExpr(e.left, tenv, bindings); const r = emitExpr(e.right, tenv, bindings);
      // A zero divisor: the interpreter maps `/0` and `%0` to null → 0 as a cell (NOT the native inf/NaN a
      // raw shader division yields). Guard both so a gate-accepted divide matches the oracle. (`%` is JS
      // remainder — sign of the DIVIDEND — lowered to the truncated remainder `a - b*trunc(a/b)`; GLSL `%`
      // is integer-only and its mod() takes the sign of the divisor.)
      // The `/0` guard is width-aware: the scalar `(r == 0.0 ? 0.0 : l/r)` is only type-correct when BOTH
      // operands are `float`. When either is a vec/mat, the scalar 0.0 false-branch + the `vec == 0.0` test
      // are a type mismatch (a real-adapter compile error). A vec/mat divide is emitted as the NATIVE
      // componentwise divide (no guard) — `inf`/`NaN` on a zero component, matching the interpreter's
      // unguarded componentwise divide, so the oracle still agrees.
      if (e.op === '/') {
        const lt = glslType(e.left, tenv, bindings); const rt = glslType(e.right, tenv, bindings);
        if (lt === 'float' && rt === 'float') return `(${r} == 0.0 ? 0.0 : ${l} / ${r})`;
        return `(${l} / ${r})`;
      }
      if (e.op === '%') return `(${r} == 0.0 ? 0.0 : ${l} - ${r} * trunc(${l} / ${r}))`;
      return `(${l} ${e.op} ${r})`;
    }
    case 'cond': return `(${boolExpr(e.test, tenv, bindings)} ? ${emitExpr(e.then, tenv, bindings)} : ${emitExpr(e.else, tenv, bindings)})`;
    case 'call': {
      const name = e.callee.kind === 'ident' ? e.callee.name : '';
      // A user function BINDING shadows a builtin of the same name (the interpreter resolves the closure
      // first). The gate rejects such a kernel (helper calls aren't lowerable in v1); emit a benign
      // placeholder rather than the native builtin. Checked BEFORE the builtin branches so `function abs(){…}`
      // never lowers to abs().
      if (bindings.byName.get(name)?.role === 'callee') return `/* shadowed builtin ${name} (helper — gate-rejected) */ 0.0`;
      const spec = BUILTINS[name];
      const args = e.args.map((a) => emitExpr(a, tenv, bindings)).join(', ');
      if (VEC_CTORS.has(name)) return `${name}(${args})`;   // GLSL vecN/matN are not generic
      // Domain-restricted transcendentals: the interpreter maps an out-of-domain input to 0 (a cell). Guard
      // so a gate-accepted kernel matches the oracle instead of the native NaN. sqrt(x<0)→0; log(x<=0)→0.
      if (name === 'sqrt') return `((${args}) < 0.0 ? 0.0 : sqrt(${args}))`;
      if (name === 'log') return `((${args}) <= 0.0 ? 0.0 : log(${args}))`;
      // asin/acos: |x|>1 is out of domain; the interpreter maps it to 0 (a cell). Guard so a gate-accepted
      // kernel matches the oracle instead of the native NaN.
      if (name === 'asin' || name === 'acos') return `(abs(${args}) > 1.0 ? 0.0 : ${name}(${args}))`;
      // atan2(y, x) has a per-target native name (no registry lowerName): GLSL spells it as 2-arg atan.
      if (name === 'atan2') return `atan(${args})`;
      // faceforward has no registry lowerName (its WGSL name differs); GLSL spells it faceforward.
      if (name === 'faceforward') return `faceforward(${args})`;
      // The 32-bit bit ops call the injected `_countOneBits`/`_reverseBits` prelude helpers (GLSL ES 3.00 has no
      // bitCount/bitfieldReverse). The gate rejects a VEC arg, so `${args}` is a single scalar here.
      if (name === 'countOneBits') return `_countOneBits(${args})`;
      if (name === 'reverseBits') return `_reverseBits(${args})`;
      // ─── quaternions (vec4 layout (x,y,z,w) = imaginary xyz + real w; hand-emitted — GLSL has no quat type) ───
      // qconj negates the imaginary part; qinvert = conj / dot(q,q); qmul is the Hamilton product (see WGSL emitter).
      if (name === 'qconj') { const q = emitExpr(e.args[0]!, tenv, bindings); return `(${q} * vec4(-1.0, -1.0, -1.0, 1.0))`; }
      if (name === 'qinvert') { const q = emitExpr(e.args[0]!, tenv, bindings); return `((${q} * vec4(-1.0, -1.0, -1.0, 1.0)) / dot(${q}, ${q}))`; }
      if (name === 'qmul') { const a = emitExpr(e.args[0]!, tenv, bindings); const b = emitExpr(e.args[1]!, tenv, bindings); return `vec4(${a}.w*${b}.x + ${a}.x*${b}.w + ${a}.y*${b}.z - ${a}.z*${b}.y, ${a}.w*${b}.y - ${a}.x*${b}.z + ${a}.y*${b}.w + ${a}.z*${b}.x, ${a}.w*${b}.z + ${a}.x*${b}.y - ${a}.y*${b}.x + ${a}.z*${b}.w, ${a}.w*${b}.w - ${a}.x*${b}.x - ${a}.y*${b}.y - ${a}.z*${b}.z)`; }
      // qaxisangle(axis:vec3, angle) → (axis·sin(θ/2), cos(θ/2)); qrotate(q:vec4, v:vec3) → the rotated vec3 (see WGSL emitter).
      if (name === 'qaxisangle') { const ax = emitExpr(e.args[0]!, tenv, bindings); const an = emitExpr(e.args[1]!, tenv, bindings); return `vec4((${ax}) * sin((${an}) * 0.5), cos((${an}) * 0.5))`; }
      if (name === 'qrotate') { const q = emitExpr(e.args[0]!, tenv, bindings); const v = emitExpr(e.args[1]!, tenv, bindings); return `((${v}) + 2.0 * cross((${q}).xyz, cross((${q}).xyz, (${v})) + (${q}).w * (${v})))`; }
      // qslerp — the branch lives in the `_qslerp` prelude helper (injected once before main).
      if (name === 'qslerp') { const a = emitExpr(e.args[0]!, tenv, bindings); const b = emitExpr(e.args[1]!, tenv, bindings); const t = emitExpr(e.args[2]!, tenv, bindings); return `_qslerp(${a}, ${b}, ${t})`; }
      // qmat — the 3×3 column-major rotation matrix lives in the `_qmat` prelude helper (injected once before main).
      if (name === 'qmat') { const q = emitExpr(e.args[0]!, tenv, bindings); return `_qmat(${q})`; }
      // inverse has no registry lowerName (WGSL has NO inverse() — hand-emitted there per size); GLSL ES 3.0
      // has a native inverse() for mat2/mat3/mat4, so it lowers to the same-named native call here.
      if (name === 'inverse') return `inverse(${args})`;
      // log2 / inverseSqrt: x<=0 is out of domain; the interpreter maps it to 0 (a cell). Guard so a
      // gate-accepted kernel matches the oracle instead of the native NaN. GLSL spells it inversesqrt.
      if (name === 'log2') return `((${args}) <= 0.0 ? 0.0 : log2(${args}))`;
      if (name === 'inverseSqrt') return `((${args}) <= 0.0 ? 0.0 : inversesqrt(${args}))`;
      if (spec?.lowerName) return `${spec.lowerName}(${args})`;
      if (GLSL_CORE_FN[name]) return `${GLSL_CORE_FN[name]}(${args})`;
      // Unreachable for a gate-accepted kernel (the gate rejects every head with no shader lowering, so gate ↔
      // emitter stay in lockstep). Fail LOUD if one ever reaches here anyway (a future gate↔emitter drift) —
      // a thrown error the engine catches into a local diagnostic — rather than the old silent `0.0`
      // placeholder, which compiled to a wrong-but-quiet 0 that diverged from the interpreter oracle unnoticed.
      throw new Error(`metael-gpu: no GLSL lowering for builtin '${name}' — the gate should have rejected this kernel`);
    }
    default: return '0.0';
  }
}

// ─── The REDUCTION fragment emitter: a reducer-fold-over-a-tile shader (a WebGL2 multi-pass tree reduction) ───
// A reduction folds N inputs → 1 scalar. WebGL2 has no compute/shared-memory, so the driver runs a MULTI-PASS
// tree reduction over ping-pong textures: each pass reads M elements and each output texel folds a TILE of
// `REDUCE_TILE` consecutive elements (seeded by the identity) → ceil(M/TILE) partials, until 1 remains. This
// ONE fragment shader is reused for EVERY pass (only the input texture + element count + output width change);
// the reducer's binary op is emitted ONCE as a GLSL `float _reduce(acc, x)` function. The tile size is a
// COMPILE-TIME CONSTANT (a constant-bound `for` — required by GLSL-ES-3.00, which does not accept a uniform
// loop bound) and the driver sizes each pass's output as ceil(currentLen/TILE). The per-lane guard
// `if (idx < _inLen)` makes the LAST (partial) tile read the IDENTITY for lanes past the element count — never
// a garbage texel (the partial-tile correctness). The identity is a `uniform float _identity` (no rebuild per
// identity). NOTE: a reducer is PURE over (acc, x) (gateReducer), so there are NO buffer/coord bindings — only
// acc/x + optional closed-over scalar CONSTANTS (role:'scalar', emitted `_u_<name>`); emitExpr handles these.
export const REDUCE_TILE = 256;

/** A statement in the reducer's `_reduce(acc, x)` body — like emitStmt but a `return` emits a NORMAL
 *  `return <expr>;` (not the map path's `_frag = …` cell write). Reuses emitExpr/glslType/boolExpr. */
function emitReduceStmt(s: Stmt, tenv: TypeEnv, bindings: BindingTable, indent: number): string {
  const pad = '  '.repeat(indent);
  const sub = (body: readonly Stmt[], ind: number): string => body.map((x) => emitReduceStmt(x, tenv, bindings, ind)).join('\n');
  switch (s.kind) {
    case 'const': case 'let': {
      const t = glslType(s.init, tenv, bindings);
      tenv.set(s.name, t);
      return `${pad}${t} ${s.name} = ${emitExpr(s.init, tenv, bindings)};`;
    }
    case 'assign': return s.target.kind === 'ident' ? `${pad}${s.target.name} = ${emitExpr(s.value, tenv, bindings)};` : `${pad}// unsupported assign`;
    case 'return': return `${pad}return ${emitExpr(s.value ?? { kind: 'number', value: 0, span: s.span }, tenv, bindings)};`;
    case 'if': return `${pad}if (${boolExpr(s.test, tenv, bindings)}) {\n${sub(s.then, indent + 1)}\n${pad}}` + (s.else ? ` else {\n${sub(s.else, indent + 1)}\n${pad}}` : '');
    case 'for': {
      const bound = emitExpr((s.iterable as Extract<Expr, { kind: 'call' }>).args[0]!, tenv, bindings);
      tenv.set(s.binding, 'float');
      return `${pad}for (float ${s.binding} = 0.0; ${s.binding} < ${bound}; ${s.binding} = ${s.binding} + 1.0) {\n${sub(s.body, indent + 1)}\n${pad}}`;
    }
    case 'expr': return `${pad}${emitExpr(s.expr, tenv, bindings)};`;
    default: return `${pad}// unsupported stmt`;
  }
}

/** Emit the reduction fragment shader for `reducer` (its 2 scalar params `acc`, `x` bind as `float`). The
 *  driver (webgl2's `dispatchReduce`) runs it once per tree-reduction pass, binding the current element
 *  texture as `_in` and setting `_inLen`/`_inTexW`/`_outTexW`/`_identity` per pass; the tile size is the
 *  baked `TILE` constant (= REDUCE_TILE). Any closed-over scalar constant is a `uniform float _u_<name>` the
 *  driver also sets. */
export function emitReduceGlsl(reducer: UserFn, bindings: BindingTable): string {
  const params = reducer.params.map((p) => (p.kind === 'name' ? p.name : ''));
  const acc = params[0] || 'acc';
  const x = params[1] || 'x';
  const scalars: Binding[] = [];
  for (const b of bindings.byName.values()) if (b.role === 'scalar') scalars.push(b);
  const tenv: TypeEnv = new Map();
  tenv.set(acc, 'float'); tenv.set(x, 'float');
  for (const s of scalars) tenv.set(s.name, 'float');
  const body = reducer.body.map((s) => emitReduceStmt(s, tenv, bindings, 1)).join('\n');

  const L: string[] = [];
  L.push('#version 300 es');
  L.push('precision highp float;');
  L.push('precision highp int;');
  L.push('precision highp sampler2D;');
  // Per-pass uniforms: the current input texture + its element count/width, this pass's output width (the
  // flat-index → fragment map), and the fold identity. Set by the driver each pass (except _identity, once).
  L.push('uniform sampler2D _in; uniform int _inLen; uniform int _inTexW; uniform int _outTexW; uniform float _identity;');
  for (const s of scalars) L.push(`uniform float _u_${s.name};`);
  L.push('out vec4 _frag;');
  L.push(`const int TILE = ${REDUCE_TILE};`);   // a COMPILE-TIME constant loop bound (GLSL-ES-3.00 requires it)
  L.push('float _fetch(sampler2D t, int idx, int w) { return texelFetch(t, ivec2(idx % w, idx / w), 0).r; }');
  L.push(...glslQuatPrelude(reducer));
  L.push(...glslBitPrelude(reducer));
  L.push(`float _reduce(float ${acc}, float ${x}) {`);
  L.push(body);
  L.push('}');
  L.push('void main() {');
  L.push('  int _fx = int(gl_FragCoord.x); int _fy = int(gl_FragCoord.y);');
  L.push('  int _flat = _fy * _outTexW + _fx;');   // this fragment's partial index
  L.push('  float _acc = _identity;');
  L.push('  int _base = _flat * TILE;');
  L.push('  for (int _j = 0; _j < TILE; _j++) {');
  L.push('    int _idx = _base + _j;');
  // The partial-tile guard: lanes past _inLen read the IDENTITY (are not folded), never a garbage texel.
  L.push('    if (_idx < _inLen) { _acc = _reduce(_acc, _fetch(_in, _idx, _inTexW)); }');
  L.push('  }');
  L.push('  _frag = vec4(_acc, 0.0, 0.0, 1.0);');
  L.push('}');
  return L.join('\n');
}

/** Infer the GLSL type of an expression so a `const`/`let` declares correctly (float unless a vec/mat flows
 *  through). Mirrors the value the interpreter/emitter produce; a bool value position is `float`. */
function glslType(e: Expr, tenv: TypeEnv, bindings: BindingTable): string {
  // `&&`/`||` in a value position yield an operand value (see emitExpr), so the declared type follows the
  // operands (a comparison / `!` yields a bool → float). Both operands share a type for a well-typed kernel.
  if (e.kind === 'binary' && (e.op === '&&' || e.op === '||')) return glslType(e.right, tenv, bindings);
  if (isBoolExpr(e)) return 'float';
  switch (e.kind) {
    case 'number': case 'bool': return 'float';
    case 'ident': return tenv.get(e.name) ?? 'float';
    case 'index': {
      // A matrix column read `m[i]` is a vec of the matrix's ROW count (native GLSL — indexing a mat yields a
      // column vec), so a `const c = m[i];` must declare `vecR c`, not `float c` (a GLSL compile error). The
      // object's GLSL type comes from `tenv` (a mat local) or a direct mat ctor: matN (N×N) → vecN; matCxR
      // (C cols × R rows — GLSL's ColumnsxRows naming, matching metael's) → vecR. A BUFFER index stays float.
      const ot = glslType(e.object, tenv, bindings);
      const m = /^mat(\d)(?:x(\d))?$/.exec(ot);
      if (m) return `vec${m[2] ?? m[1]}`;
      return 'float';   // buffer[i] → float; a vec[i] → float
    }
    case 'member': {
      const obj = e.object;
      if (obj.kind === 'ident' && bindings.byName.get(obj.name)?.role === 'buffer' && e.property === 'length') return 'float';
      const ot = glslType(obj, tenv, bindings);
      if (ot.startsWith('vec')) { const len = e.property.length; return len === 1 ? 'float' : `vec${len}`; }   // swizzle
      return 'float';
    }
    case 'unary': return e.op === '!' ? 'float' : glslType(e.operand, tenv, bindings);
    case 'binary': {
      const lt = glslType(e.left, tenv, bindings); const rt = glslType(e.right, tenv, bindings);
      if (lt.startsWith('mat') && rt.startsWith('vec')) return rt;   // mat * vec → vec
      if (lt.startsWith('mat')) return lt;                           // mat * mat / mat * scalar → mat
      if (rt.startsWith('mat')) return rt;
      if (lt.startsWith('vec')) return lt;                           // vec op {scalar,vec} → vec
      if (rt.startsWith('vec')) return rt;
      return 'float';
    }
    case 'cond': return glslType(e.then, tenv, bindings);
    case 'call': {
      const name = e.callee.kind === 'ident' ? e.callee.name : '';
      if (VEC_CTORS.has(name)) return name;
      if (name === 'cross') return 'vec3';
      if (name === 'normalize' || name === 'mix') return e.args[0] ? glslType(e.args[0], tenv, bindings) : 'float';
      if (name === 'atan2' || name === 'inverseSqrt') return e.args[0] ? glslType(e.args[0], tenv, bindings) : 'float';   // no lowerName; componentwise
      // faceforward(N, I, Nref) has no registry lowerName; it returns a vector of N's (first arg's) type.
      if (name === 'faceforward') return e.args[0] ? glslType(e.args[0], tenv, bindings) : 'float';
      // Quaternion ops that produce a quat (a vec4): qmul/qconj/qinvert/qaxisangle/qslerp; qrotate → vec3; qmat → mat3.
      if (name === 'qmul' || name === 'qconj' || name === 'qinvert' || name === 'qaxisangle' || name === 'qslerp') return 'vec4';
      if (name === 'qrotate') return 'vec3';
      if (name === 'qmat') return 'mat3';
      // inverse returns the same square-matrix type as its argument (mat2→mat2, mat3→mat3, mat4→mat4).
      if (name === 'inverse') return e.args[0] ? glslType(e.args[0], tenv, bindings) : 'float';
      // dot/length/distance fold a vector down to a scalar; determinant folds a matrix down to a scalar.
      // determinant carries a registry lowerName, so it must be caught BEFORE the componentwise lowerName
      // fallback below (which would wrongly declare `mat2 d = determinant(...)`).
      if (name === 'dot' || name === 'length' || name === 'distance' || name === 'determinant') return 'float';
      if (GLSL_CORE_FN[name]) return e.args[0] ? glslType(e.args[0], tenv, bindings) : 'float';
      const spec = BUILTINS[name];
      if (spec?.lowerName && e.args[0]) return glslType(e.args[0], tenv, bindings);   // transcendentals + reflect/refract: componentwise vec
      return 'float';
    }
    default: return 'float';
  }
}
