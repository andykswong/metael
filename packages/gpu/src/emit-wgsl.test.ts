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
  it('a single vecN return emits a valid unique temp + N flat writes (normal case still works)', () => {
    const { fn, host } = kernelOf(`const x = f32(12, (i) => i)\ncomponent k(i) { return vec3(x[i*3], x[i*3+1], x[i*3+2]) }\nk`);
    const wgsl = emitWgsl(fn, gateKernel(fn, host).bindings, 'f32', 3);
    const temps = [...wgsl.matchAll(/let (_r\d+) =/g)].map((m) => m[1]);
    expect(temps.length).toBe(1);
    const t = temps[0]!;
    expect(wgsl).toContain(`_out[_flat * 3u + 0u] = ${t}.x;`);
    expect(wgsl).toContain(`_out[_flat * 3u + 2u] = ${t}.z;`);
  });
});
