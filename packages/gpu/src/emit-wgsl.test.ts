import { describe, it, expect } from 'vitest';
import { evaluateProgram, isUserFn } from '@metael/lang';
import type { UserFn } from '@metael/lang';
import { PlainStorageHost, RecordingHostEnv } from '@metael/lang';
import { gateKernel } from './gate.ts';
import { emitWgsl } from './emit-wgsl.ts';

function kernelOf(src: string) { const host = new PlainStorageHost(); const res = evaluateProgram(src, { host, env: new RecordingHostEnv() }); if (!isUserFn(res.value)) throw new Error('kernel'); return { fn: res.value as UserFn, host }; }

describe('WGSL emitter', () => {
  it('emits a compute entry with workgroup size, a bounds guard, storage buffers, and a flat-index write', () => {
    const { fn, host } = kernelOf(`
      const N = 4
      const a = f32(N, (i) => i)
      component k(i) { return a[i] * 2 }
      k`);
    const { bindings } = gateKernel(fn, host);
    const wgsl = emitWgsl(fn, bindings, 'f32');
    expect(wgsl).toContain('@compute');
    expect(wgsl).toContain('@workgroup_size');
    expect(wgsl).toContain('global_invocation_id');
    expect(wgsl).toMatch(/var<storage,\s*read>/);
    expect(wgsl).toMatch(/var<storage,\s*read_write>/);
    expect(wgsl).toContain('return;');
    expect(wgsl).toContain('a[');
    expect(wgsl).toContain('struct _Params');
    expect(wgsl).not.toMatch(/let\s+i\s*=\s*i32\(gid/);
    expect(wgsl).toContain('u32(round(');
  });
  it('lowers a bounded for-of range to a wgsl for loop', () => {
    const { fn, host } = kernelOf(`
      const N = 3
      const a = f32(N * N, (i) => i)
      const b = f32(N * N, (i) => i)
      component product(row, col) { let sum = 0; for (const k of range(N)) { sum = sum + a[row * N + k] * b[k * N + col] } return sum }
      product`);
    const { bindings } = gateKernel(fn, host);
    const wgsl = emitWgsl(fn, bindings, 'f32');
    expect(wgsl).toMatch(/for\s*\(/);
    expect(wgsl).toContain('sum');
  });
  it('lowers a transcendental to the native wgsl builtin', () => {
    const { fn, host } = kernelOf(`const a = f32(4, (i) => i)\ncomponent k(i) { return sin(a[i]) }\nk`);
    const { bindings } = gateKernel(fn, host);
    expect(emitWgsl(fn, bindings, 'f32')).toContain('sin(');
  });
  it('coerces a non-bool if-condition to bool (WGSL requires bool)', () => {
    const { fn, host } = kernelOf(`const a = f32(4, (i) => i)\ncomponent k(i) { if (a[i]) { return 1 } return 0 }\nk`);
    const wgsl = emitWgsl(fn, gateKernel(fn, host).bindings, 'f32');
    expect(wgsl).toMatch(/!= f32\(0\)/);      // the bare f32 condition is coerced
    expect(wgsl).not.toMatch(/if \(a\[u32\(round\(i\)\)\]\) \{/);   // NOT a bare-f32 if
  });
  it('coerces a bool-valued return to f32 via select(0,1,bool) (a bool cannot be written to array<f32>)', () => {
    const { fn, host } = kernelOf(`const a = f32(4, (i) => i)\ncomponent k(i) { return a[i] > 0 }\nk`);
    const wgsl = emitWgsl(fn, gateKernel(fn, host).bindings, 'f32');
    expect(wgsl).toMatch(/_out\[_flat\] = select\(f32\(0\), f32\(1\), /);   // bool value → f32 at the write
  });
  it('a.length emits arrayLength, not .length', () => {
    const { fn, host } = kernelOf(`const a = f32(4, (i) => i)\ncomponent k(i) { let s = 0; for (const j of range(a.length)) { s = s + 1 } return s }\nk`);
    const wgsl = emitWgsl(fn, gateKernel(fn, host).bindings, 'f32');
    expect(wgsl).toContain('arrayLength(&a)');
    expect(wgsl).not.toMatch(/a\.length/);
  });
  it('lowers a float % to a truncated remainder (sign-of-dividend, matching the interpreter — not integer %)', () => {
    const { fn, host } = kernelOf(`const a = f32(4, (i) => i)\ncomponent k(i) { return a[i] % 2 }\nk`);
    const wgsl = emitWgsl(fn, gateKernel(fn, host).bindings, 'f32');
    expect(wgsl).toContain('trunc(');            // a - b*trunc(a/b) — WGSL % is integer-only + would diverge on negatives
    expect(wgsl).not.toMatch(/\]\s*%\s*/);       // no bare `%` operator left in the body
  });
  it('emits && / || as value-returning short-circuit (operand value, not a 0/1 bool — matches the interpreter)', () => {
    const { fn, host } = kernelOf(`const a = f32(4, (i) => i)\nconst b = f32(4, (i) => i + 1)\ncomponent k(i) { return a[i] && b[i] }\nk`);
    const wgsl = emitWgsl(fn, gateKernel(fn, host).bindings, 'f32');
    expect(wgsl).toMatch(/_out\[_flat\] = select\(a\[.*b\[/);   // select(a, b, bool(a)) — returns b when a truthy, else a
    expect(wgsl).not.toMatch(/select\(f32\(0\), f32\(1\), \(a\[.* != f32\(0\)\) &&/);   // NOT coerced to 0/1
  });
  it('guards a zero divisor for / and % (interpreter maps /0 and %0 to 0, not the native Inf/NaN)', () => {
    const div = kernelOf(`const a = f32(4, (i) => i)\ncomponent k(i) { return 1 / a[i] }\nk`);
    const wdiv = emitWgsl(div.fn, gateKernel(div.fn, div.host).bindings, 'f32');
    expect(wdiv).toMatch(/select\(.* \/ .*, f32\(0\), .* == f32\(0\)\)/);   // r==0 → 0, not +Inf
    const mod = kernelOf(`const a = f32(4, (i) => i)\ncomponent k(i) { return 1 % a[i] }\nk`);
    const wmod = emitWgsl(mod.fn, gateKernel(mod.fn, mod.host).bindings, 'f32');
    expect(wmod).toMatch(/select\(.*trunc.*, f32\(0\), .* == f32\(0\)\)/);   // r==0 → 0, not NaN
  });
  it('vec/scalar divide emits native componentwise divide, not a scalar-typed select', () => {
    // A scalar-typed `select(l/r, f32(0), r==f32(0))` over a VEC left operand is a type-mismatched WGSL
    // shader — the false-branch f32(0) + the r==f32(0) test can't apply to a vec. The width-aware guard
    // keeps only the scalar path guarded and emits the native componentwise divide when an operand is a vec.
    const { fn, host } = kernelOf('component k(i) { return (vec2(i, i) / 2).x } k');
    const wgsl = emitWgsl(fn, gateKernel(fn, host).bindings, 'f32');
    expect(wgsl).not.toContain('select(vec2'); // no scalar-guarded select over a vec
    // a plain SCALAR divide STILL emits the scalar /0 guard (path preserved, not blanket-removed)
    const sc = kernelOf('component k(i) { return i / 2 } k');
    const wsc = emitWgsl(sc.fn, gateKernel(sc.fn, sc.host).bindings, 'f32');
    expect(wsc).toMatch(/select\(.* \/ .*, f32\(0\), .* == f32\(0\)\)/);
  });
  it('guards bad-domain transcendentals (sqrt(neg)/log(<=0) → 0, matching the interpreter, not native NaN)', () => {
    const sq = kernelOf(`const a = f32(4, (i) => i)\ncomponent k(i) { return sqrt(a[i] - 2) }\nk`);
    const wsq = emitWgsl(sq.fn, gateKernel(sq.fn, sq.host).bindings, 'f32');
    expect(wsq).toMatch(/select\(sqrt\(.*\), f32\(0\), .* < f32\(0\)\)/);
    const lg = kernelOf(`const a = f32(4, (i) => i)\ncomponent k(i) { return log(a[i]) }\nk`);
    const wlg = emitWgsl(lg.fn, gateKernel(lg.fn, lg.host).bindings, 'f32');
    expect(wlg).toMatch(/select\(log\(.*\), f32\(0\), .* <= f32\(0\)\)/);
  });
  it('lowers the core-exact builtins (abs/clamp) to native WGSL (parity with the CPU emitter)', () => {
    const { fn, host } = kernelOf(`const a = f32(4, (i) => i)\ncomponent k(i) { return clamp(abs(a[i] - 2), 0, 1) }\nk`);
    const wgsl = emitWgsl(fn, gateKernel(fn, host).bindings, 'f32');
    expect(wgsl).toContain('abs(');
    expect(wgsl).toContain('clamp(');
    expect(wgsl).not.toContain('unsupported call');   // NOT dropped to an f32(0) placeholder
  });
  it('a scalar uniform named rows does not collide with the dims struct', () => {
    const { fn, host } = kernelOf(`const rows = 8\nconst a = f32(4, (i) => i)\ncomponent k(i) { return a[i] + rows }\nk`);
    const wgsl = emitWgsl(fn, gateKernel(fn, host).bindings, 'f32');
    expect(wgsl).toContain('_u_rows');       // namespaced scalar member
    expect(wgsl).toContain('rows: u32');     // the dims member still present
  });
  it('gives each vecN `return` a DISTINCT temp (two returns never emit `let _r` twice → no WGSL redefinition)', () => {
    // Two vecN returns in one body (here in separate branches) must not both emit `let _r` — a monotonic
    // per-emit counter makes each temp unique (`_r0`, `_r1`, …). A repeated `let _r` would be a WGSL
    // redefinition compile error the no-adapter emit path can't catch.
    const { fn, host } = kernelOf(`const x = f32(8, (i) => i)\ncomponent k(i) { if (i > 0) { return vec2(x[i*2], x[i*2+1]) } return vec2(0, 0) }\nk`);
    const wgsl = emitWgsl(fn, gateKernel(fn, host).bindings, 'f32', 2);
    const temps = [...wgsl.matchAll(/let (_r\d+) =/g)].map((m) => m[1]);
    expect(temps.length).toBe(2);                       // one temp per vecN return
    expect(new Set(temps).size).toBe(temps.length);     // all distinct — no duplicate `let _r`
  });
  it('neg(mat2) emits a componentwise WGSL scale, not unary -mat (WGSL has no unary - for matrices)', () => {
    const { fn, host } = kernelOf('component k(i) { const m = -mat2(1,2,3,4) return (m * vec2(1,1)).x } k');
    const gate = gateKernel(fn, host);
    expect(gate.core).toBe(true);   // the kernel must be gate-accepted for emit to run
    const wgsl = emitWgsl(fn, gate.bindings, 'f32');
    expect(wgsl).toContain('* f32(-1)');        // componentwise scale by -1 (matrix–scalar multiply)
    expect(wgsl).not.toMatch(/\(-mat2x2/);      // NOT the illegal unary -mat form
  });
  it('a vec local in a divide emits native componentwise, not a scalar select over the vec', () => {
    // `const v = vec2(...); v / 2` — the operand-shape probe must resolve the LOCAL `v` as a vec so the
    // divide emits the native componentwise `(v / f32(2))`, not a scalar-typed `select(v / .., f32(0), ..)`
    // (which is a WGSL type error — a scalar false-branch + `vec == f32(0)` test over a vec).
    const { fn, host } = kernelOf('component k(i) { const v = vec2(i, i) return (v / 2).x } k');
    const wgsl = emitWgsl(fn, gateKernel(fn, host).bindings, 'f32');
    expect(wgsl).not.toMatch(/select\(v \//); // no scalar guard over the vec local
  });
  it('a mat local negation emits a componentwise scale, not unary -mat', () => {
    // `const m = mat2(...); -m` — the operand-shape probe must resolve the LOCAL `m` as a mat so the negate
    // emits the componentwise scale `(m * f32(-1))`, not the illegal bare unary `(-m)` (WGSL has no unary -mat).
    const { fn, host } = kernelOf('component k(i) { const m = mat2(1,2,3,4) const w = -m return (w * vec2(i,i)).x } k');
    const wgsl = emitWgsl(fn, gateKernel(fn, host).bindings, 'f32');
    expect(wgsl).not.toMatch(/\(-m\)/);      // no bare unary -mat
    expect(wgsl).toContain('* f32(-1)');     // componentwise scale
  });
  it('inverse(mat2) hand-emits a closed-form _inv2 prelude helper (WGSL has no builtin inverse())', () => {
    // A kernel that inverts a mat2 then reads a component of inverse(M) * v. WGSL has NO inverse(), so the
    // emitter must inject a `_inv2` prelude helper (built from determinant + adjugate) and CALL it — never a
    // bare `inverse(` (which would be a WGSL compile error on a real adapter).
    const { fn, host } = kernelOf('component k(i) { const M = mat2(4,2,7,6) const w = inverse(M) * vec2(1,1) return w.x } k');
    const gate = gateKernel(fn, host);
    expect(gate.core).toBe(true);   // must be gate-accepted for emit to run
    const wgsl = emitWgsl(fn, gate.bindings, 'f32');
    expect(wgsl).toContain('fn _inv2(m: mat2x2<f32>) -> mat2x2<f32>');   // the injected helper definition
    expect(wgsl).toContain('mat2x2<f32>(');                              // the adjugate is built column-major
    expect(wgsl).toContain('_inv2(');                                    // the call site uses the helper
    expect(wgsl).not.toMatch(/[^_]inverse\(/);                           // NO bare native inverse( — WGSL has none
  });
  it('inverse(mat3) hand-emits a _inv3 helper; unused sizes (_inv2/_inv4) are NOT injected', () => {
    const { fn, host } = kernelOf('component k(i) { const M = mat3(2,0,1, 1,3,0, 0,2,1) const w = inverse(M) * vec3(1,1,1) return w.x } k');
    const gate = gateKernel(fn, host);
    expect(gate.core).toBe(true);
    const wgsl = emitWgsl(fn, gate.bindings, 'f32');
    expect(wgsl).toContain('fn _inv3(m: mat3x3<f32>) -> mat3x3<f32>');
    expect(wgsl).toContain('_inv3(');
    expect(wgsl).not.toContain('fn _inv2');   // only the used size's helper is emitted
    expect(wgsl).not.toContain('fn _inv4');
    expect(wgsl).not.toMatch(/[^_]inverse\(/);
    // WGSL has NO unary `+` operator — a cofactor expansion must not open a parenthesised term group with
    // `(+ …` (only the SUBSEQUENT terms carry an explicit sign). A leading `+` fails module validation on a
    // real adapter → the dispatch silently writes zeros (caught only by a real WebGPU device, not WebGL2's
    // native inverse()). Assert no `(+ ` appears anywhere in the emitted shader.
    expect(wgsl).not.toContain('(+ ');
  });
  it('inverse(transpose(M)) — the normal-matrix idiom over a local — hand-emits _inv3(transpose(M)), never a bare transpose(M)', () => {
    // This is the exact case the pre-fix emitter silently mis-lowered: matShapeOf(transpose(M)) was null (M an
    // ident), so it dropped the inverse and emitted the bare `transpose(M)`. The locals-aware `matSizeOf` now
    // resolves it (gate-accepted), so the emitter calls `_inv3(transpose(...))` — the inverse is NOT dropped.
    const { fn, host } = kernelOf('component k(i) { const M = mat3(2,0,1, 1,3,0, 0,2,1) const w = inverse(transpose(M)) * vec3(1,1,1) return w.x } k');
    const gate = gateKernel(fn, host);
    expect(gate.core).toBe(true);
    const wgsl = emitWgsl(fn, gate.bindings, 'f32');
    expect(wgsl).toContain('fn _inv3(m: mat3x3<f32>) -> mat3x3<f32>');
    expect(wgsl).toMatch(/_inv3\(transpose\(/);   // the inverse wraps the transpose — NOT dropped
    expect(wgsl).not.toMatch(/[^_]inverse\(/);     // NO bare native inverse( — WGSL has none
    expect(wgsl).not.toContain('_INVERSE_SIZE_UNRESOLVED_');   // the loud unreachable marker never fires for a gate-accepted kernel
  });
  it('a kernel that never calls inverse injects NO _invN prelude helper', () => {
    const { fn, host } = kernelOf('component k(i) { const M = mat2(1,2,3,4) return (M * vec2(1,1)).x } k');
    const wgsl = emitWgsl(fn, gateKernel(fn, host).bindings, 'f32');
    expect(wgsl).not.toContain('fn _inv');   // pay nothing when inverse is unused
  });
  it('a single vecN return emits a valid unique temp + N flat writes (normal case still works)', () => {
    const { fn, host } = kernelOf(`const x = f32(12, (i) => i)\ncomponent k(i) { return vec3(x[i*3], x[i*3+1], x[i*3+2]) }\nk`);
    const wgsl = emitWgsl(fn, gateKernel(fn, host).bindings, 'f32', 3);
    const temps = [...wgsl.matchAll(/let (_r\d+) =/g)].map((m) => m[1]);
    expect(temps.length).toBe(1);
    const t = temps[0]!;
    expect(wgsl).toContain(`_out[_flat * 3u + 0u] = ${t}.x;`);
    expect(wgsl).toContain(`_out[_flat * 3u + 2u] = ${t}.z;`);
  });
  it('a rank-3 kernel emits workgroup_size(4,4,4), gid.z, and the (x*H+y)*D+z flatten', () => {
    const { fn, host } = kernelOf('component k(x, y, z) { return x + y + z } k');
    const wgsl = emitWgsl(fn, gateKernel(fn, host).bindings, 'f32');
    expect(wgsl).toContain('@workgroup_size(4, 4, 4)');
    expect(wgsl).toContain('gid.z');
    expect(wgsl).toContain('(gid.x * _p.cols + gid.y) * _p.deps + gid.z');
  });
});
