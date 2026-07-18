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
import { BUILTINS } from '@metael/lang';
import type { Binding, BindingTable } from './binding.ts';

// Core-exact builtins the gate accepts but that carry no registry `lowerName` — they map to a native GLSL
// function of the same name (round → roundEven so ties-to-even matches the interpreter/CPU + WGSL).
const GLSL_CORE_FN: Readonly<Record<string, string>> = { min: 'min', max: 'max', abs: 'abs', sign: 'sign', floor: 'floor', ceil: 'ceil', clamp: 'clamp', round: 'roundEven' };
const VEC_CTORS = new Set(['vec2', 'vec3', 'vec4', 'mat2', 'mat3', 'mat4']);
const COMPARE_OPS = new Set(['==', '!=', '<', '<=', '>', '>=']);

function isBoolExpr(e: Expr): boolean {
  if (e.kind === 'binary') return COMPARE_OPS.has(e.op) || e.op === '&&' || e.op === '||';
  if (e.kind === 'unary') return e.op === '!';
  return false;
}

// A local-name → GLSL-type environment threaded through emission so a `const`/`let` can be declared with the
// right type (float by default; vecN/matN for vec-bearing intermediates).
type TypeEnv = Map<string, string>;

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
  L.push('uniform int _rows; uniform int _cols; uniform int _texW;');
  // Each input buffer: a float texture + its texture width (for the texel map) + its element count (.length).
  for (const b of buffers) L.push(`uniform sampler2D ${b.name}; uniform int ${b.name}_texW; uniform int ${b.name}_len;`);
  // Scalar uniforms are namespaced `_u_<name>` (mirroring the WGSL emitter's `_p._u_<name>`): a user scalar
  // named `_rows`/`_cols`/`_texW` would otherwise redeclare a reserved dispatch uniform (a GLSL compile
  // error). The backend sets these by the same `_u_`-prefixed name.
  for (const s of scalars) L.push(`uniform float _u_${s.name};`);
  L.push('out vec4 _frag;');
  L.push('float _fetch(sampler2D t, int idx, int w) { return texelFetch(t, ivec2(idx % w, idx / w), 0).r; }');
  L.push('void main() {');
  L.push('  int _fx = int(gl_FragCoord.x); int _fy = int(gl_FragCoord.y);');
  if (rank === 2) {
    L.push('  int _flat = _fy * _cols + _fx;');
    L.push(`  float ${params[0]} = float(_fy); float ${params[1]} = float(_fx);`);
  } else {
    L.push('  int _flat = _fy * _texW + _fx;');
    L.push(`  float ${params[0]} = float(_flat);`);
  }
  L.push('  if (_flat >= _rows * _cols) { discard; }');
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
      if (e.op === '/') return `(${r} == 0.0 ? 0.0 : ${l} / ${r})`;
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
      if (spec?.lowerName) return `${spec.lowerName}(${args})`;
      if (GLSL_CORE_FN[name]) return `${GLSL_CORE_FN[name]}(${args})`;
      // Unreachable for a gate-accepted kernel (every emittable head is handled above); a defensive placeholder.
      return `/* unsupported call ${name} */ 0.0`;
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
    case 'index': return 'float';   // buffer[i] → float; a vec[i] → float
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
      if (name === 'dot' || name === 'length') return 'float';
      if (GLSL_CORE_FN[name]) return e.args[0] ? glslType(e.args[0], tenv, bindings) : 'float';
      const spec = BUILTINS[name];
      if (spec?.lowerName && e.args[0]) return glslType(e.args[0], tenv, bindings);   // transcendentals: componentwise
      return 'float';
    }
    default: return 'float';
  }
}
