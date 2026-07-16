import { describe, it, expect } from 'vitest';
import { evaluateProgram, descriptorOf } from '@metael/lang';
import { PlainStorageHost, RecordingHostEnv } from './ports.ts';

const run = (src: string) => evaluateProgram(src, { host: new PlainStorageHost(), env: new RecordingHostEnv() });

describe('vec constructors + componentwise operators', () => {
  it('vec3(1,2,3) components via swizzle', () => {
    expect(run('vec3(1, 2, 3).x').value).toBe(1);
    expect(run('vec3(1, 2, 3).y').value).toBe(2);
    expect(run('vec3(1, 2, 3).z').value).toBe(3);
  });
  it('vec3(0) splats', () => {
    expect(run('vec3(0).x').value).toBe(0);
  });
  it('componentwise + - * /', () => {
    expect(run('(vec3(1,2,3) + vec3(4,5,6)).x').value).toBe(5);
    expect(run('(vec3(4,5,6) - vec3(1,2,3)).z').value).toBe(3);
    expect(run('(vec3(2,3,4) * vec3(2,2,2)).y').value).toBe(6);
  });
  it('vec * scalar and scalar * vec both scale (the 2*vec right-dispatch rule)', () => {
    expect(run('(vec3(1,2,3) * 2).z').value).toBe(6);
    expect(run('(2 * vec3(1,2,3)).x').value).toBe(2);
  });
  it('componentwise == via equals', () => {
    expect(run('vec3(1,2,3) == vec3(1,2,3)').value).toBe(true);
    expect(run('vec3(1,2,3) != vec3(9,9,9)').value).toBe(true);
  });
  it('unary -', () => {
    expect(run('(-vec3(1,2,3)).x').value).toBe(-1);
  });
  it('a multi-component swizzle returns a vec', () => {
    expect(run('vec3(1,2,3).xy.x').value).toBe(1);
    expect(run('vec3(1,2,3).xy.y').value).toBe(2);
  });
  it('an invalid swizzle is ML-LANG-UNKNOWN-MEMBER', () => {
    const r = run('vec3(1,2,3).q');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-UNKNOWN-MEMBER')).toBe(true);
  });
  it('vec + mat is a real "not defined" → ML-LANG-OP-UNSUPPORTED', () => {
    const r = run('vec3(1,2,3) + mat3()');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-OP-UNSUPPORTED')).toBe(true);
  });
  it('a vec is immutable — an index write is ML-LANG-IMMUTABLE', () => {
    const r = run('const v = vec3(1,2,3); v.x = 9; v.x');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-IMMUTABLE')).toBe(true);
    expect(r.value).toBe(1);
  });
});

describe('mat operators', () => {
  it('mat4() is identity', () => {
    expect(run('mat4() == mat4()').value).toBe(true);
  });
  it('mat * vec applies the matrix (identity → unchanged)', () => {
    expect(run('(mat3() * vec3(5,6,7)).y').value).toBe(6);
  });
  it('mat * mat is matmul (identity * identity = identity)', () => {
    expect(run('(mat3() * mat3()) == mat3()').value).toBe(true);
  });
});

describe('vec numeric builtins', () => {
  it('dot', () => { expect(run('dot(vec3(1,2,3), vec3(4,5,6))').value).toBe(32); });
  it('cross', () => { expect(run('cross(vec3(1,0,0), vec3(0,1,0)).z').value).toBe(1); });
  it('length', () => { expect(run('length(vec3(3,4,0))').value).toBe(5); });
  it('normalize', () => { expect(run('length(normalize(vec3(3,4,0)))').value).toBeCloseTo(1, 6); });
});

describe('vec/mat lowering', () => {
  it('vec3 lower is a value vecN with ops + swizzle members', () => {
    const v = run('vec3(1,2,3)').value;
    const low = descriptorOf(v)?.lower;
    expect(low?.shape).toBe('vecN');
    expect(low?.n).toBe(3);
    expect(low?.access).toBe('value');
    expect(low?.gpuStorable).toBe(true);
    expect(low?.ops?.['+']).toEqual({ kind: 'componentwise', op: 'add' });
    expect(low?.ops?.['*']).toBeDefined();
    expect(low?.ops?.['neg']).toEqual({ kind: 'unary', op: 'neg' });
    expect(low?.members).toEqual({ kind: 'swizzle', of: 'xyzw' });
  });
  it('mat3 lower is a matNxN value', () => {
    const m = run('mat3()').value;
    expect(descriptorOf(m)?.lower?.shape).toBe('matNxN');
    expect(descriptorOf(m)?.lower?.n).toBe(3);
  });
});

describe('vec/mat cross-type ops are OP-UNSUPPORTED, never a silent garbage value', () => {
  it('vec * mat (colliding shapes) is ML-LANG-OP-UNSUPPORTED, not a NaN matrix', () => {
    const r = run('vec3(1,2,3) * mat3()');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-OP-UNSUPPORTED')).toBe(true);
  });
  it('mat2 * vec4 (flat-length collision) is ML-LANG-OP-UNSUPPORTED, not a garbage vec4', () => {
    const r = run('mat2(1,2,3,4) * vec4(10,20,30,40)');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-OP-UNSUPPORTED')).toBe(true);
  });
  it('mat2 + vec4 / mat2 - vec4 / mat2 / vec4 are all OP-UNSUPPORTED', () => {
    for (const op of ['+', '-', '/']) {
      const r = run(`mat2(1,2,3,4) ${op} vec4(10,20,30,40)`);
      expect(r.diagnostics.some((d) => d.code === 'ML-LANG-OP-UNSUPPORTED')).toBe(true);
    }
  });
  it('vec2 * vec3 (different length) is OP-UNSUPPORTED', () => {
    const r = run('vec2(1,2) * vec3(1,2,3)');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-OP-UNSUPPORTED')).toBe(true);
  });
  it('mat3() * 2 (mat scale — undefined) is OP-UNSUPPORTED', () => {
    const r = run('mat3() * 2');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-OP-UNSUPPORTED')).toBe(true);
  });
  it('vec3 + 5 (vec + scalar — only * / scale) is OP-UNSUPPORTED', () => {
    const r = run('vec3(1,2,3) + 5');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-OP-UNSUPPORTED')).toBe(true);
  });
});

describe('non-identity matmul + matVec are correct row-major (not transposed)', () => {
  it('mat2(1,2,3,4) * vec2(5,6) = vec2(17, 39) — row-major, not the transpose [23,34]', () => {
    expect(run('(mat2(1,2,3,4) * vec2(5,6)).x').value).toBe(17);
    expect(run('(mat2(1,2,3,4) * vec2(5,6)).y').value).toBe(39);
  });
  it('mat2(1,2,3,4) * mat2(5,6,7,8) = row-major [[19,22],[43,50]]', () => {
    // Read the product back by applying it to the basis vectors: M*e0 = first column [19,43], M*e1 = [22,50].
    const col0x = run('((mat2(1,2,3,4) * mat2(5,6,7,8)) * vec2(1,0)).x').value;
    const col0y = run('((mat2(1,2,3,4) * mat2(5,6,7,8)) * vec2(1,0)).y').value;
    const col1x = run('((mat2(1,2,3,4) * mat2(5,6,7,8)) * vec2(0,1)).x').value;
    const col1y = run('((mat2(1,2,3,4) * mat2(5,6,7,8)) * vec2(0,1)).y').value;
    expect([col0x, col0y, col1x, col1y]).toEqual([19, 43, 22, 50]);
  });
  it('cross of a non-axis pair: cross(vec3(1,2,3),vec3(4,5,6)) = vec3(-3,6,-3)', () => {
    expect(run('cross(vec3(1,2,3),vec3(4,5,6)).x').value).toBe(-3);
    expect(run('cross(vec3(1,2,3),vec3(4,5,6)).y').value).toBe(6);
    expect(run('cross(vec3(1,2,3),vec3(4,5,6)).z').value).toBe(-3);
  });
  it('normalize of the zero vector is the zero vector, not NaN', () => {
    expect(run('normalize(vec3(0,0,0)).x').value).toBe(0);
    expect(run('normalize(vec3(0,0,0)).y').value).toBe(0);
  });
});

describe('vec numeric builtins reject non-vector args', () => {
  it('dot / length / cross / normalize on a number is ML-LANG-BUILTIN-ARG', () => {
    expect(run('dot(1, 2)').diagnostics.some((d) => d.code === 'ML-LANG-BUILTIN-ARG')).toBe(true);
    expect(run('length(5)').diagnostics.some((d) => d.code === 'ML-LANG-BUILTIN-ARG')).toBe(true);
    expect(run('cross(vec2(1,2), vec2(3,4))').diagnostics.some((d) => d.code === 'ML-LANG-BUILTIN-ARG')).toBe(true);
  });
  it('dot on a typed-array buffer (not a vecN) is ML-LANG-BUILTIN-ARG', () => {
    expect(run('dot(f32([1,2,3]), f32([4,5,6]))').diagnostics.some((d) => d.code === 'ML-LANG-BUILTIN-ARG')).toBe(true);
  });
});
