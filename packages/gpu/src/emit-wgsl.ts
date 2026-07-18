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
import { REDUCE_TILE } from './emit-glsl.ts';

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
  // Scalar-uniform members are namespaced `_u_<name>` so a scalar named `rows`/`cols` cannot collide with
  // (or shadow) the reserved dispatch-dim members. The backend packs scalars by POSITION (Float32Array at
  // offset 8), not by name, so this rename is purely emitter-internal — no backend change is needed.
  lines.push(`struct _Params { rows: u32, cols: u32,${scalars.map((u) => ` _u_${u.name}: ${S},`).join('')} };`);
  lines.push(`@group(0) @binding(${binding}) var<uniform> _p: _Params;`);
  const wg = rank === 2 ? `@workgroup_size(${WORKGROUP_2D}, ${WORKGROUP_2D})` : `@workgroup_size(${WORKGROUP_1D})`;
  lines.push(`@compute ${wg}`);
  lines.push(`fn main(@builtin(global_invocation_id) gid: vec3<u32>) {`);
  if (rank === 2) {
    lines.push(`  if (gid.x >= _p.rows || gid.y >= _p.cols) { return; }`);
    lines.push(`  let ${params[0]} = ${S}(gid.x);`);
    lines.push(`  let ${params[1]} = ${S}(gid.y);`);
    lines.push(`  let _flat = gid.x * _p.cols + gid.y;`);
  } else {
    lines.push(`  if (gid.x >= _p.rows) { return; }`);
    lines.push(`  let ${params[0]} = ${S}(gid.x);`);
    lines.push(`  let _flat = gid.x;`);
  }
  lines.push(emitBody(kernel.body, S, bindings, 1, comps, { n: 0 }));
  lines.push(`}`);
  return lines.join('\n');
}

// A monotonic counter for the vecN-return temp name, threaded through the body walk so EACH `return <vec>`
// gets a distinct temp (`_r0`, `_r1`, …). Two returns in one lexical scope must not both emit `let _r` — a
// duplicate `let _r` is a WGSL redefinition compile error the no-adapter emit path can't catch.
interface TempCtr { n: number }

function emitBody(body: readonly Stmt[], S: string, bindings: BindingTable, indent: number, comps: number, ctr: TempCtr): string {
  const out: string[] = [];
  for (const s of body) out.push(emitStmt(s, S, bindings, indent, comps, ctr));
  return out.join('\n');
}
// The `return <expr>` write. A scalar output (comps=1) writes one flat slot: `_out[_flat] = <expr>;`. A
// vecN output (comps>1) computes the vec into a temp then writes N flat slots — `_out[_flat*Nu + ku] = _r.<c>`
// — keeping `_out` a flat `array<f32>` (NOT array<vec3>, which carries a 16-byte stride). `_flat` is u32, so
// the index arithmetic stays u32 (`_flat * ${comps}u + ${k}u`).
function emitReturn(s: Extract<Stmt, { kind: 'return' }>, S: string, bindings: BindingTable, pad: string, comps: number, ctr: TempCtr): string {
  const expr = emitExpr(s.value ?? { kind: 'number', value: 0, span: s.span }, S, bindings);
  if (comps === 1) return `${pad}_out[_flat] = ${expr};\n${pad}return;`;
  const t = `_r${ctr.n++}`;   // a distinct temp per vecN return → no duplicate `let _r` in one scope
  const writes: string[] = [`${pad}let ${t} = ${expr};`];
  for (let k = 0; k < comps; k++) writes.push(`${pad}_out[_flat * ${comps}u + ${k}u] = ${t}.${'xyzw'[k]};`);
  writes.push(`${pad}return;`);
  return writes.join('\n');
}
function emitStmt(s: Stmt, S: string, bindings: BindingTable, indent: number, comps: number, ctr: TempCtr): string {
  const pad = '  '.repeat(indent);
  switch (s.kind) {
    case 'const': return `${pad}let ${s.name} = ${emitExpr(s.init, S, bindings)};`;
    case 'let': return `${pad}var ${s.name} = ${emitExpr(s.init, S, bindings)};`;
    case 'assign': return s.target.kind === 'ident' ? `${pad}${s.target.name} = ${emitExpr(s.value, S, bindings)};` : `${pad}// unsupported assign`;
    case 'return': return emitReturn(s, S, bindings, pad, comps, ctr);
    case 'if': return `${pad}if (${boolExpr(s.test, S, bindings)}) {\n${emitBody(s.then, S, bindings, indent + 1, comps, ctr)}\n${pad}}` + (s.else ? ` else {\n${emitBody(s.else, S, bindings, indent + 1, comps, ctr)}\n${pad}}` : '');
    case 'for': {
      const bound = emitExpr((s.iterable as Extract<Expr, { kind: 'call' }>).args[0]!, S, bindings);
      return `${pad}for (var ${s.binding} = 0.0; ${s.binding} < ${bound}; ${s.binding} = ${s.binding} + 1.0) {\n${emitBody(s.body, S, bindings, indent + 1, comps, ctr)}\n${pad}}`;
    }
    case 'expr': return `${pad}${emitExpr(s.expr, S, bindings)};`;
    default: return `${pad}// unsupported stmt`;
  }
}
/** Emit a bool-typed expression (for if/select-test/logical-operand positions). An inherently-bool expr
 *  (comparison / && / || / !) is emitted directly; any other (value) expr coerces via `!= f32(0)`. */
function boolExpr(e: Expr, S: string, bindings: BindingTable): string {
  return isBoolExpr(e) ? emitBoolCore(e, S, bindings) : `(${emitExpr(e, S, bindings)} != ${S}(0))`;
}
/** The bool-producing core for the inherently-bool operators. Logical operands (&&/||) and `!` operands
 *  are themselves bool positions → recurse through boolExpr; comparison operands are values → emitExpr. */
function emitBoolCore(e: Expr, S: string, bindings: BindingTable): string {
  if (e.kind === 'unary' && e.op === '!') return `(!${boolExpr(e.operand, S, bindings)})`;
  if (e.kind === 'binary' && (e.op === '&&' || e.op === '||')) return `(${boolExpr(e.left, S, bindings)} ${e.op} ${boolExpr(e.right, S, bindings)})`;
  if (e.kind === 'binary') return `(${emitExpr(e.left, S, bindings)} ${wgslOp(e.op)} ${emitExpr(e.right, S, bindings)})`;   // a comparison: operands are f32 values
  return `(${emitExpr(e, S, bindings)} != ${S}(0))`;   // unreachable given isBoolExpr, but keeps this total
}

function emitExpr(e: Expr, S: string, bindings: BindingTable): string {
  // `&&`/`||` in a VALUE position return an OPERAND value (JS/interpreter short-circuit: `a && b` → a if
  // falsy else b; `a || b` → a if truthy else b), NOT a 0/1 bool. Emit a value-returning select on the
  // left's truthiness (kernels are pure, so eager eval of both operands is safe — the unpicked branch's
  // value is discarded). A comparison / `!` DOES yield a bool → coerced to 0/1 below (matches the oracle's
  // Number(true)=1 / toNum(false)=0 downstream coercion).
  if (e.kind === 'binary' && (e.op === '&&' || e.op === '||')) {
    const l = emitExpr(e.left, S, bindings); const r = emitExpr(e.right, S, bindings); const lb = boolExpr(e.left, S, bindings);
    return e.op === '&&' ? `select(${l}, ${r}, ${lb})` : `select(${r}, ${l}, ${lb})`;
  }
  // A bool-typed subexpression (comparison / !) reaching a VALUE position (return, arithmetic operand, init)
  // must coerce back to the scalar type — WGSL can't write a bool to array<f32> or add it.
  if (isBoolExpr(e)) return `select(${S}(0), ${S}(1), ${emitBoolCore(e, S, bindings)})`;
  switch (e.kind) {
    case 'number': return Number.isInteger(e.value) ? `${S}(${e.value})` : String(e.value);
    case 'bool': return e.value ? `${S}(1)` : `${S}(0)`;
    case 'ident': return bindings.byName.get(e.name)?.role === 'scalar' ? `_p._u_${e.name}` : e.name;
    case 'index': return `${emitExpr(e.object, S, bindings)}[u32(round(${emitExpr(e.index, S, bindings)}))]`;
    case 'member': {
      const obj = e.object;
      // `bufferIdent.length` is the one whole-buffer read the gate allows: a storage array's length is
      // arrayLength(&buf) (a u32) — cast to the scalar domain since it flows into f32 arithmetic.
      if (obj.kind === 'ident' && bindings.byName.get(obj.name)?.role === 'buffer' && e.property === 'length') {
        return `${S}(arrayLength(&${obj.name}))`;
      }
      return `${emitExpr(e.object, S, bindings)}.${e.property}`;   // a vec swizzle (.x/.xy) — valid WGSL
    }
    case 'unary': return `(-${emitExpr(e.operand, S, bindings)})`;   // `!` is inherently-bool → handled above
    case 'binary': {
      const l = emitExpr(e.left, S, bindings); const r = emitExpr(e.right, S, bindings);
      // A zero divisor: the interpreter maps `/0` and `%0` to null → 0 as a cell (NOT the native Inf/NaN a
      // raw shader division yields). Guard both so a gate-accepted divide-by-a-possibly-zero-denominator
      // matches the oracle instead of silently writing Inf/NaN. (`%` is JS remainder — sign of the DIVIDEND
      // — lowered to the truncated remainder `a - b*trunc(a/b)` since WGSL `%` is integer-only.)
      if (e.op === '/') return `select(${l} / ${r}, ${S}(0), ${r} == ${S}(0))`;
      if (e.op === '%') return `select(${l} - ${r} * trunc(${l} / ${r}), ${S}(0), ${r} == ${S}(0))`;
      return `(${l} ${wgslOp(e.op)} ${r})`;
    }
    case 'cond': return `select(${emitExpr(e.else, S, bindings)}, ${emitExpr(e.then, S, bindings)}, ${boolExpr(e.test, S, bindings)})`;
    case 'call': {
      const name = e.callee.kind === 'ident' ? e.callee.name : '';
      // A user function BINDING shadows a builtin of the same name (the interpreter resolves the closure
      // first). The gate rejects such a kernel (helper calls aren't lowerable in v1); emit a benign
      // placeholder rather than the native builtin so a non-core kernel emitted anyway never masquerades as
      // the intrinsic. Checked BEFORE the builtin branches so `function abs(){…}` never lowers to abs().
      if (bindings.byName.get(name)?.role === 'callee') return `/* shadowed builtin ${name} (helper — gate-rejected) */ ${S}(0)`;
      const spec = BUILTINS[name];
      const args = e.args.map((a) => emitExpr(a, S, bindings)).join(', ');
      if (name === 'vec2' || name === 'vec3' || name === 'vec4') return `${name}<${S}>(${args})`;
      if (name === 'mat2' || name === 'mat3' || name === 'mat4') { const n = name.slice(3); return `mat${n}x${n}<${S}>(${args})`; }
      // Domain-restricted transcendentals: the interpreter maps an out-of-domain input to 0 (a cell). Guard
      // so a gate-accepted kernel matches the oracle instead of writing the native NaN. sqrt(x<0)→0; log(x<=0)→0.
      if (name === 'sqrt') return `select(sqrt(${args}), ${S}(0), (${args}) < ${S}(0))`;
      if (name === 'log') return `select(log(${args}), ${S}(0), (${args}) <= ${S}(0))`;
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
function emitReduceStmt(s: Stmt, S: string, bindings: BindingTable, indent: number): string {
  const pad = '  '.repeat(indent);
  const sub = (body: readonly Stmt[], ind: number): string => body.map((x) => emitReduceStmt(x, S, bindings, ind)).join('\n');
  switch (s.kind) {
    case 'const': return `${pad}let ${s.name} = ${emitExpr(s.init, S, bindings)};`;
    case 'let': return `${pad}var ${s.name} = ${emitExpr(s.init, S, bindings)};`;
    case 'assign': return s.target.kind === 'ident' ? `${pad}${s.target.name} = ${emitExpr(s.value, S, bindings)};` : `${pad}// unsupported assign`;
    case 'return': return `${pad}return ${emitExpr(s.value ?? { kind: 'number', value: 0, span: s.span }, S, bindings)};`;
    case 'if': return `${pad}if (${boolExpr(s.test, S, bindings)}) {\n${sub(s.then, indent + 1)}\n${pad}}` + (s.else ? ` else {\n${sub(s.else, indent + 1)}\n${pad}}` : '');
    case 'for': {
      const bound = emitExpr((s.iterable as Extract<Expr, { kind: 'call' }>).args[0]!, S, bindings);
      return `${pad}for (var ${s.binding} = 0.0; ${s.binding} < ${bound}; ${s.binding} = ${s.binding} + 1.0) {\n${sub(s.body, indent + 1)}\n${pad}}`;
    }
    case 'expr': return `${pad}${emitExpr(s.expr, S, bindings)};`;
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
  const body = reducer.body.map((s) => emitReduceStmt(s, S, bindings, 1)).join('\n');

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
function emitHistogramStmt(s: Stmt, S: string, bindings: BindingTable, indent: number): string {
  const pad = '  '.repeat(indent);
  const sub = (body: readonly Stmt[], ind: number): string => body.map((x) => emitHistogramStmt(x, S, bindings, ind)).join('\n');
  switch (s.kind) {
    case 'const': return `${pad}let ${s.name} = ${emitExpr(s.init, S, bindings)};`;
    case 'let': return `${pad}var ${s.name} = ${emitExpr(s.init, S, bindings)};`;
    case 'assign': return s.target.kind === 'ident' ? `${pad}${s.target.name} = ${emitExpr(s.value, S, bindings)};` : `${pad}// unsupported assign`;
    case 'return': return `${pad}return ${emitExpr(s.value ?? { kind: 'number', value: 0, span: s.span }, S, bindings)};`;
    case 'if': return `${pad}if (${boolExpr(s.test, S, bindings)}) {\n${sub(s.then, indent + 1)}\n${pad}}` + (s.else ? ` else {\n${sub(s.else, indent + 1)}\n${pad}}` : '');
    case 'for': {
      const bound = emitExpr((s.iterable as Extract<Expr, { kind: 'call' }>).args[0]!, S, bindings);
      return `${pad}for (var ${s.binding} = 0.0; ${s.binding} < ${bound}; ${s.binding} = ${s.binding} + 1.0) {\n${sub(s.body, indent + 1)}\n${pad}}`;
    }
    case 'expr': return `${pad}${emitExpr(s.expr, S, bindings)};`;
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
  const body = binMapper.body.map((s) => emitHistogramStmt(s, S, bindings, 1)).join('\n');

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
