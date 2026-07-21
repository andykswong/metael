import { describe, it, expect } from 'vitest';
import { MATH_BUILTINS } from '@metael/math/lang';
import { evaluateProgram, isUserFn } from '@metael/lang';
import type { UserFn } from '@metael/lang';
import { PlainStorageHost, RecordingHostEnv } from '@metael/lang';
import { gateKernel } from './gate.ts';
import { emitGlsl } from './emit-glsl.ts';

function kernelOf(src: string) {
  const host = new PlainStorageHost();
  const res = evaluateProgram(src, { host, env: new RecordingHostEnv(), builtins: [MATH_BUILTINS] });
  if (!isUserFn(res.value)) throw new Error('kernel');
  return { fn: res.value as UserFn, host };
}

describe('GLSL-ES-3.0 emitter (compute-via-fragment)', () => {
  it('emits a #version 300 es fragment shader with a float-texture sampler + an RGBA32F output', () => {
    const { fn, host } = kernelOf(`const a = f32(16, (i) => i)\ncomponent k(i) { return a[i] * 2 }\nk`);
    const { bindings } = gateKernel(fn, host);
    const glsl = emitGlsl(fn, bindings, 'f32');
    expect(glsl).toContain('#version 300 es');
    expect(glsl).toContain('precision highp float;');
    expect(glsl).toMatch(/uniform sampler2D a;/);      // input buffer as a texture
    expect(glsl).toContain('gl_FragCoord');
    expect(glsl).toMatch(/out vec4 _frag/);            // RGBA32F output
    expect(glsl).toContain('_fetch(a,');               // buffer index → texelFetch helper
    expect(glsl).toContain('_frag = vec4(');           // packs the result into the R channel
  });

  it('maps precision f16 → mediump', () => {
    const { fn, host } = kernelOf(`const a = f32(4, (i) => i)\ncomponent k(i) { return a[i] }\nk`);
    expect(emitGlsl(fn, gateKernel(fn, host).bindings, 'f16')).toContain('precision mediump float;');
  });

  it('lowers transcendentals native (sin)', () => {
    const { fn, host } = kernelOf(`const a = f32(4, (i) => i)\ncomponent k(i) { return sin(a[i]) }\nk`);
    expect(emitGlsl(fn, gateKernel(fn, host).bindings, 'f32')).toContain('sin(');
  });

  it('lowers a bounded for-of range to a float-counter for loop (parity with the CPU/WGSL float model)', () => {
    const { fn, host } = kernelOf(`
      const N = 3
      const a = f32(N * N, (i) => i)
      const b = f32(N * N, (i) => i)
      component product(row, col) { let sum = 0; for (const k of range(N)) { sum = sum + a[row * N + k] * b[k * N + col] } return sum }
      product`);
    const glsl = emitGlsl(fn, gateKernel(fn, host).bindings, 'f32');
    expect(glsl).toMatch(/for \(float k = 0\.0;/);   // float loop var (k feeds float index arithmetic)
    expect(glsl).toContain('sum');
  });

  it('coerces a non-bool if-condition to bool (!= 0.0) and a bool return to float()', () => {
    const cond = kernelOf(`const a = f32(4, (i) => i)\ncomponent k(i) { if (a[i]) { return 1 } return 0 }\nk`);
    const gcond = emitGlsl(cond.fn, gateKernel(cond.fn, cond.host).bindings, 'f32');
    expect(gcond).toMatch(/!= 0\.0/);                 // the bare-float condition is coerced to bool
    const ret = kernelOf(`const a = f32(4, (i) => i)\ncomponent k(i) { return a[i] > 0 }\nk`);
    const gret = emitGlsl(ret.fn, gateKernel(ret.fn, ret.host).bindings, 'f32');
    expect(gret).toMatch(/_frag = vec4\(float\(/);    // bool value → float at the RGBA pack
  });

  it('lowers a float % to a truncated remainder (sign-of-dividend, matching the interpreter — NOT mod())', () => {
    const { fn, host } = kernelOf(`const a = f32(4, (i) => i)\ncomponent k(i) { return a[i] % 2 }\nk`);
    const glsl = emitGlsl(fn, gateKernel(fn, host).bindings, 'f32');
    expect(glsl).toContain('trunc(');       // a - b*trunc(a/b), NOT the sign-of-divisor mod() builtin
    expect(glsl).not.toContain('mod(');     // GLSL mod() would diverge on negatives
  });

  it('emits && / || as value-returning short-circuit (operand value, not a 0/1 bool — matches the interpreter)', () => {
    const { fn, host } = kernelOf(`const a = f32(4, (i) => i)\nconst b = f32(4, (i) => i + 1)\ncomponent k(i) { return a[i] && b[i] }\nk`);
    const glsl = emitGlsl(fn, gateKernel(fn, host).bindings, 'f32');
    // `a && b` returns b when a is truthy, else a — a ternary on a's truthiness, NOT float(a && b) → 0/1.
    expect(glsl).toMatch(/\? _fetch\(b,.*: _fetch\(a,/);
  });

  it('emits vec constructors natively (no <f32> generic) and dot()', () => {
    const { fn, host } = kernelOf(`
      const a = f32(48, (i) => i)
      component k(i) { const u = vec3(a[i*3], a[i*3+1], a[i*3+2]); return dot(u, u) }
      k`);
    const glsl = emitGlsl(fn, gateKernel(fn, host).bindings, 'f32');
    expect(glsl).toContain('vec3(');
    expect(glsl).not.toContain('vec3<');   // GLSL vec3 is not generic
    expect(glsl).toContain('dot(');
  });

  it('reads a buffer .length as a uniform element count (not a sampler property)', () => {
    const { fn, host } = kernelOf(`const a = f32(4, (i) => i)\ncomponent k(i) { let s = 0; for (const j of range(a.length)) { s = s + 1 } return s }\nk`);
    const glsl = emitGlsl(fn, gateKernel(fn, host).bindings, 'f32');
    expect(glsl).toContain('a_len');       // the buffer's element count is a uniform int
    expect(glsl).not.toMatch(/a\.length/);
  });

  it('declares a determinant local as float (it folds a matrix down to a scalar, not a mat)', () => {
    // determinant carries a registry lowerName, so the componentwise lowerName fallback would wrongly declare
    // `mat2 d = determinant(...)`. glslType must special-case determinant → float BEFORE that fallback.
    const { fn, host } = kernelOf('component k(i) { const d = determinant(mat2(1,2,3,4)) return d } k');
    const glsl = emitGlsl(fn, gateKernel(fn, host).bindings, 'f32');
    expect(glsl).toMatch(/float d = determinant\(/);
    expect(glsl).not.toMatch(/mat2 d = determinant\(/);
  });

  it('guards a zero divisor for / and % (interpreter maps /0 and %0 to 0, not native inf/NaN)', () => {
    const div = kernelOf(`const a = f32(4, (i) => i)\ncomponent k(i) { return 1 / a[i] }\nk`);
    const gdiv = emitGlsl(div.fn, gateKernel(div.fn, div.host).bindings, 'f32');
    expect(gdiv).toMatch(/\(.* == 0\.0 \? 0\.0 : .* \/ .*\)/);
    const mod = kernelOf(`const a = f32(4, (i) => i)\ncomponent k(i) { return 1 % a[i] }\nk`);
    const gmod = emitGlsl(mod.fn, gateKernel(mod.fn, mod.host).bindings, 'f32');
    expect(gmod).toMatch(/\(.* == 0\.0 \? 0\.0 : .*trunc.*\)/);
  });
  it('vec/vec divide emits native componentwise divide (no scalar 0.0 guard)', () => {
    // A scalar `(r == 0.0 ? 0.0 : l/r)` guard over VEC operands is a type-mismatched GLSL shader (the 0.0
    // false-branch + the r==0.0 test are scalar). The width-aware guard branches on glslType: only the
    // float/float path stays guarded; a vec operand emits the native componentwise divide.
    const { fn, host } = kernelOf('component k(i) { return (vec2(i,1) / vec2(2,i)).x } k');
    const glsl = emitGlsl(fn, gateKernel(fn, host).bindings, 'f32');
    expect(glsl).not.toMatch(/== 0\.0 \? 0\.0/); // no scalar guard applied to the vec divide
    // a plain SCALAR divide STILL emits the scalar /0 guard (path preserved, not blanket-removed)
    const sc = kernelOf('component k(i) { return i / 2 } k');
    const gsc = emitGlsl(sc.fn, gateKernel(sc.fn, sc.host).bindings, 'f32');
    expect(gsc).toMatch(/\(.* == 0\.0 \? 0\.0 : .* \/ .*\)/);
  });
  it('guards bad-domain transcendentals (sqrt(neg)/log(<=0) → 0, matching the interpreter, not native NaN)', () => {
    const sq = kernelOf(`const a = f32(4, (i) => i)\ncomponent k(i) { return sqrt(a[i] - 2) }\nk`);
    expect(emitGlsl(sq.fn, gateKernel(sq.fn, sq.host).bindings, 'f32')).toMatch(/\(.* < 0\.0 \? 0\.0 : sqrt\(/);
    const lg = kernelOf(`const a = f32(4, (i) => i)\ncomponent k(i) { return log(a[i]) }\nk`);
    expect(emitGlsl(lg.fn, gateKernel(lg.fn, lg.host).bindings, 'f32')).toMatch(/\(.* <= 0\.0 \? 0\.0 : log\(/);
  });
  it('namespaces a scalar uniform (_u_ prefix) so a user scalar named _rows cannot redeclare a reserved uniform', () => {
    const { fn, host } = kernelOf(`const _rows = 8\nconst a = f32(4, (i) => i)\ncomponent k(i) { return a[i] + _rows }\nk`);
    const glsl = emitGlsl(fn, gateKernel(fn, host).bindings, 'f32');
    expect(glsl).toContain('uniform float _u__rows;');   // the user scalar, namespaced
    expect(glsl).toContain('uniform int _rows;');         // the reserved dispatch uniform, intact
    expect(glsl).toContain('_u__rows');                   // referenced by the namespaced name in the body
  });
  it('rounds a buffer index with roundEven (ties-to-even, matching WGSL round + survives float accumulation)', () => {
    const { fn, host } = kernelOf(`const a = f32(16, (i) => i)\ncomponent k(i) { return a[i] }\nk`);
    const glsl = emitGlsl(fn, gateKernel(fn, host).bindings, 'f32');
    expect(glsl).toMatch(/_fetch\(a, int\(roundEven\(/);   // round-to-nearest (ties-even), not truncation
  });
  it('lowers the core-exact builtins (abs/clamp) to native GLSL (parity with the CPU emitter + WGSL)', () => {
    const { fn, host } = kernelOf(`const a = f32(4, (i) => i)\ncomponent k(i) { return clamp(abs(a[i] - 2), 0, 1) }\nk`);
    const glsl = emitGlsl(fn, gateKernel(fn, host).bindings, 'f32');
    expect(glsl).toContain('abs(');
    expect(glsl).toContain('clamp(');
    expect(glsl).not.toContain('unsupported');   // these are NOT dropped to a 0.0 placeholder
  });
  it('a rank-3 GLSL kernel decomposes _flat into (x,y,z)', () => {
    const { fn, host } = kernelOf('component k(x, y, z) { return x + y + z } k');
    const glsl = emitGlsl(fn, gateKernel(fn, host).bindings, 'f32');
    expect(glsl).toContain('_deps'); // 3rd-dim uniform
    expect(glsl).toMatch(/float .* = float\(.*% _deps\)/); // z = _flat % D
  });

  // ─── The structured constructors + column access (native to GLSL) ───
  it('vecN composition passes vec args through natively (vec3(vec2(..), ..))', () => {
    const { fn, host } = kernelOf('component k(i) { return vec3(vec2(i, i * 2), i + 7) } k');
    const glsl = emitGlsl(fn, gateKernel(fn, host, 3).bindings, 'f32', 3);
    expect(glsl).toContain('vec3(vec2(');
  });
  it('matMxN from column vecs emits the native column constructor (mat2(vec2(..), vec2(..)))', () => {
    const { fn, host } = kernelOf('component k(i) { const m = mat2(vec2(1, 2), vec2(3, 4)) return (m * vec2(i, i)).x } k');
    const glsl = emitGlsl(fn, gateKernel(fn, host, 1).bindings, 'f32', 1);
    expect(glsl).toContain('mat2 m = mat2(vec2(1.0, 2.0), vec2(3.0, 4.0))');   // the mat local declared mat2 + column ctor
  });
  it('a mat column read m[i] DECLARES the local as vecR (a float declaration would be a compile error)', () => {
    // The glslType fix: `const c = m[0]` over a mat4 must declare `vec4 c`, not `float c`.
    const { fn, host } = kernelOf('component k(i) { const m4 = mat4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 10, 11, 12, 1) const c = m4[0] return c.x } k');
    const glsl = emitGlsl(fn, gateKernel(fn, host, 1).bindings, 'f32', 1);
    expect(glsl).toMatch(/vec4 c = m4\[int\(roundEven\(0\.0\)\)\];/);   // vec4, NOT float
    expect(glsl).not.toMatch(/float c = m4\[/);
  });
  it('a mat3 column read into a vec3 output DECLARES the vec3 temp (m[i].xyz → vec3)', () => {
    const { fn, host } = kernelOf('component k(i) { const m = mat3(1, 2, 3, 4, 5, 6, 7, 8, 9) return m[1].xyz } k');
    const glsl = emitGlsl(fn, gateKernel(fn, host, 3).bindings, 'f32', 3);
    expect(glsl).toMatch(/vec3 _r0 = m\[int\(roundEven\(1\.0\)\)\]\.xyz;/);
  });
  it('a mat column divide emits the NATIVE componentwise divide, not a scalar ternary-guard (a compile error)', () => {
    const { fn, host } = kernelOf('component k(i) { const m = mat3(1, 2, 3, 4, 5, 6, 7, 8, 9) return (m[1] / 2).x } k');
    const glsl = emitGlsl(fn, gateKernel(fn, host, 1).bindings, 'f32', 1);
    expect(glsl).toContain('(m[int(roundEven(1.0))] / 2.0).x');
    expect(glsl).not.toMatch(/2\.0 == 0\.0 \? 0\.0 : m\[int/);   // no scalar guard over the column vec
  });

  // ─── newly-lowered scalar builtins: mod (native, already floored in GLSL) + the inverse hyperbolics ───
  it('lowers mod to the native GLSL mod (already floored — sign follows the divisor), not a truncated form', () => {
    const { fn, host } = kernelOf('const a = f32(4, (i) => i)\ncomponent k(i) { return mod(a[i], 3) }\nk');
    const glsl = emitGlsl(fn, gateKernel(fn, host).bindings, 'f32');
    expect(glsl).toContain('mod(');                    // native GLSL mod IS floored — no hand-emit
    expect(glsl).not.toMatch(/trunc\(/);               // NOT the WGSL-style x - y*floor(x/y)/trunc form
  });
  it('lowers asinh/acosh/atanh to the native GLSL ES 3.00 builtins of the same name (via lowerName)', () => {
    for (const name of ['asinh', 'acosh', 'atanh']) {
      const { fn, host } = kernelOf(`const a = f32(4, (i) => i)\ncomponent k(i) { return ${name}(a[i]) }\nk`);
      expect(emitGlsl(fn, gateKernel(fn, host).bindings, 'f32')).toContain(`${name}(`);
    }
  });

  // ─── the 32-bit bit ops: GLSL ES 3.00 lacks bitCount/bitfieldReverse (an ES 3.10 feature), so both lower via
  //     an injected PRELUDE HELPER over uint + the bitwise operators (mirroring the _qslerp/_qmat prelude pattern) ───
  it('lowers countOneBits via an injected _countOneBits prelude helper (SWAR popcount over uint), TRUNCATING the input (not round)', () => {
    const { fn, host } = kernelOf('const a = f32(4, (i) => i)\ncomponent k(i) { return countOneBits(a[i]) }\nk');
    const glsl = emitGlsl(fn, gateKernel(fn, host).bindings, 'f32');
    expect(glsl).toContain('float _countOneBits(float x)');   // the helper is injected
    expect(glsl).toMatch(/_countOneBits\(/);                   // and called at the site
    expect(glsl).not.toContain('bitCount(');                   // never the ES-3.10-only builtin
    expect(glsl).toContain('uint v = uint(int(x));');          // truncate toward zero (== `>>>0`), NOT roundEven
    expect(glsl).not.toContain('roundEven(x)');                // never rounds the bit-op input
  });
  it('lowers reverseBits via an injected _reverseBits prelude helper (32-bit reverse over uint), TRUNCATING the input (not round)', () => {
    const { fn, host } = kernelOf('const a = f32(4, (i) => i)\ncomponent k(i) { return reverseBits(a[i]) }\nk');
    const glsl = emitGlsl(fn, gateKernel(fn, host).bindings, 'f32');
    expect(glsl).toContain('float _reverseBits(float x)');     // the helper is injected
    expect(glsl).toMatch(/_reverseBits\(/);                    // and called at the site
    expect(glsl).not.toContain('bitfieldReverse(');            // never the ES-3.10-only builtin
    expect(glsl).toContain('uint v = uint(int(x));');          // truncate toward zero (== `>>>0`), NOT roundEven
    expect(glsl).not.toContain('roundEven(x)');
  });
  it('injects a bit-op prelude helper ONLY when the body references it (none otherwise)', () => {
    const { fn, host } = kernelOf('const a = f32(4, (i) => i)\ncomponent k(i) { return a[i] + 1 }\nk');
    const glsl = emitGlsl(fn, gateKernel(fn, host).bindings, 'f32');
    expect(glsl).not.toContain('_countOneBits');
    expect(glsl).not.toContain('_reverseBits');
  });
});

describe('GLSL emitter — loud failure on an un-lowerable builtin (no silent-0 placeholder)', () => {
  it('throws when a head with no shader lowering reaches the emitter', () => {
    const { fn, host } = kernelOf('component k(i) { return (perspective(1, 1, 1, 2) * vec4(1, 0, 0, 0)).x } k');
    const { bindings } = gateKernel(fn, host, 1);
    expect(() => emitGlsl(fn, bindings, 'f32', 1)).toThrow(/no GLSL lowering for builtin 'perspective'/);
  });
});
