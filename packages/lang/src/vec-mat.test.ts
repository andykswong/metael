import { describe, it, expect } from 'vitest';
import { evaluateProgram, descriptorOf } from '@metael/lang';
import { PlainStorageHost, RecordingHostEnv } from './ports.ts';
import { makeVec, makeMat, vecStoreOf, descriptorOf as _d } from './custom-types.ts';
import { NOT_HANDLED as NOT_HANDLED_EXPORT } from './custom-types.ts';

const run = (src: string) => evaluateProgram(src, { host: new PlainStorageHost(), env: new RecordingHostEnv() });

function mul(a: object, b: object): object {
  const res = _d(a)!.binary!('*' as never, a, b);
  if (typeof res === 'symbol') throw new Error('ML-LANG-OP-UNSUPPORTED');
  return res as object;
}

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

describe('matmul: column-major, unified', () => {
  it('mat2 * vec2 is column-major', () => {
    // mat2(1,2,3,4) column-major = columns (1,2),(3,4) = matrix [[1,3],[2,4]]
    const m = makeMat([1, 2, 3, 4], 2, 2);
    const v = makeVec([5, 6]);
    const r = vecStoreOf(mul(m, v));
    expect(Array.from(r.c)).toEqual([23, 34]); // was (17,39) row-major
    expect(r.cols).toBe(1);
  });
  it('mat2 * mat2 is column-major', () => {
    const a = makeMat([1, 2, 3, 4], 2, 2);
    const r = vecStoreOf(mul(a, a));
    // column-major flat [7,10,15,22] == matrix rows [[7,15],[10,22]]
    expect(Array.from(r.c)).toEqual([7, 10, 15, 22]);
  });
  it('dimension mismatch (mat2 * vec3) is NOT_HANDLED (→ op-unsupported, becomes a gate reject in gpu)', () => {
    const m = makeMat([1, 2, 3, 4], 2, 2);
    const v = makeVec([1, 2, 3]);
    expect(() => mul(m, v)).toThrow(); // interpreter surfaces ML-LANG-OP-UNSUPPORTED
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
    expect(low?.rows).toBe(3);
    expect(low?.cols).toBe(1);
    expect(low?.access).toBe('value');
    expect(low?.gpuStorable).toBe(true);
    expect(low?.ops?.['+']).toEqual({ kind: 'componentwise', op: 'add' });
    expect(low?.ops?.['*']).toBeDefined();
    expect(low?.ops?.['neg']).toEqual({ kind: 'unary', op: 'neg' });
    expect(low?.members).toEqual({ kind: 'swizzle', of: 'xyzw' });
  });
  it('mat3 lower is a matMxN value', () => {
    const m = run('mat3()').value;
    expect(descriptorOf(m)?.lower?.shape).toBe('matMxN');
    expect(descriptorOf(m)?.lower?.rows).toBe(3);
    expect(descriptorOf(m)?.lower?.cols).toBe(3);
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
  it('mat3() * 2 scales the matrix componentwise (2*identity applied to a vec doubles it)', () => {
    const r = run('mat3() * 2');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-OP-UNSUPPORTED')).toBe(false);
    expect(run('((mat3() * 2) * vec3(5,6,7)).x').value).toBe(10);
    expect(run('((mat3() * 2) * vec3(5,6,7)).y').value).toBe(12);
    expect(run('((mat3() * 2) * vec3(5,6,7)).z').value).toBe(14);
  });
  it('vec3 + 5 (vec + scalar — only * / scale) is OP-UNSUPPORTED', () => {
    const r = run('vec3(1,2,3) + 5');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-OP-UNSUPPORTED')).toBe(true);
  });
});

describe('non-identity matmul + matVec are correct column-major (not transposed)', () => {
  it('mat2(1,2,3,4) * vec2(5,6) = vec2(23, 34) — column-major columns (1,2),(3,4)', () => {
    expect(run('(mat2(1,2,3,4) * vec2(5,6)).x').value).toBe(23);
    expect(run('(mat2(1,2,3,4) * vec2(5,6)).y').value).toBe(34);
  });
  it('mat2(1,2,3,4) * mat2(5,6,7,8) is column-major (columns [23,34] and [31,46])', () => {
    // Read the product back by applying it to the basis vectors: M*e0 = first column [23,34], M*e1 = [31,46].
    const col0x = run('((mat2(1,2,3,4) * mat2(5,6,7,8)) * vec2(1,0)).x').value;
    const col0y = run('((mat2(1,2,3,4) * mat2(5,6,7,8)) * vec2(1,0)).y').value;
    const col1x = run('((mat2(1,2,3,4) * mat2(5,6,7,8)) * vec2(0,1)).x').value;
    const col1y = run('((mat2(1,2,3,4) * mat2(5,6,7,8)) * vec2(0,1)).y').value;
    expect([col0x, col0y, col1x, col1y]).toEqual([23, 34, 31, 46]);
  });
  it('mat3(1..9) * vec3(1,2,3) = vec3(30,36,42) — non-symmetric mat3 pins column-major', () => {
    // mat3(1..9) column-major = columns (1,2,3),(4,5,6),(7,8,9) = rows [[1,4,7],[2,5,8],[3,6,9]].
    // ·(1,2,3): (1+8+21, 2+10+24, 3+12+27) = (30,36,42). Row-major fill would give (14,32,50).
    expect(run('(mat3(1,2,3,4,5,6,7,8,9) * vec3(1,2,3)).x').value).toBe(30);
    expect(run('(mat3(1,2,3,4,5,6,7,8,9) * vec3(1,2,3)).y').value).toBe(36);
    expect(run('(mat3(1,2,3,4,5,6,7,8,9) * vec3(1,2,3)).z').value).toBe(42);
  });
  it('mat4(1..16) * vec4(1,2,3,4) = vec4(90,100,110,120) — non-symmetric mat4 pins column-major', () => {
    // mat4(1..16) column-major = columns (1,2,3,4),(5,6,7,8),(9,10,11,12),(13,14,15,16)
    // = rows [[1,5,9,13],[2,6,10,14],[3,7,11,15],[4,8,12,16]]. ·(1,2,3,4) = (90,100,110,120).
    // Row-major fill would give (30,70,110,150) — differs in 3 of 4 components.
    expect(run('(mat4(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16) * vec4(1,2,3,4)).x').value).toBe(90);
    expect(run('(mat4(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16) * vec4(1,2,3,4)).y').value).toBe(100);
    expect(run('(mat4(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16) * vec4(1,2,3,4)).z').value).toBe(110);
    expect(run('(mat4(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16) * vec4(1,2,3,4)).w').value).toBe(120);
  });
  it('cross of a non-axis pair: cross(vec3(1,2,3),vec3(4,5,6)) = vec3(-3,6,-3)', () => {
    expect(run('cross(vec3(1,2,3),vec3(4,5,6)).x').value).toBe(-3);
    expect(run('cross(vec3(1,2,3),vec3(4,5,6)).y').value).toBe(6);
    expect(run('cross(vec3(1,2,3),vec3(4,5,6)).z').value).toBe(-3);
  });
  it('normalize of the zero vector is a NaN vector (matches native shader normalize(0))', () => {
    expect(Number.isNaN(run('normalize(vec3(0,0,0)).x').value as number)).toBe(true);
    expect(Number.isNaN(run('normalize(vec3(0,0,0)).y').value as number)).toBe(true);
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

describe('vec/mat store: rows/cols model (a vec is an n×1 matrix)', () => {
  it('a vec3 has rows=3, cols=1 and lower.shape "vecN"', () => {
    const v = makeVec([1, 2, 3]);
    const s = vecStoreOf(v);
    expect(s.rows).toBe(3);
    expect(s.cols).toBe(1);
    expect(descriptorOf(v)!.lower!.shape).toBe('vecN');
  });
  it('a square mat3 has rows=3, cols=3 and lower.shape "matMxN"', () => {
    const m = makeMat([1, 2, 3, 4, 5, 6, 7, 8, 9], 3, 3);
    const s = vecStoreOf(m);
    expect(s.rows).toBe(3);
    expect(s.cols).toBe(3);
    expect(descriptorOf(m)!.lower!.shape).toBe('matMxN');
  });
});

describe('vecLower carries rows/cols + element precision', () => {
  it('vec4 f32 lowering', () => {
    const v = makeVec([1, 2, 3, 4]);
    const low = descriptorOf(v)!.lower!;
    expect(low.shape).toBe('vecN');
    expect(low.rows).toBe(4);
    expect(low.cols).toBe(1);
    expect(low.element).toBe('f32');
  });
  it('non-square mat2x3 lowering (2 cols × 3 rows)', () => {
    const m = makeMat([1, 2, 3, 4, 5, 6], 3, 2); // rows=3, cols=2
    const low = descriptorOf(m)!.lower!;
    expect(low.shape).toBe('matMxN');
    expect(low.rows).toBe(3);
    expect(low.cols).toBe(2);
  });
});

describe('vec/mat neg/getMember/display via cols===1', () => {
  it('neg(mat2) negates componentwise, stays a mat', () => {
    const m = makeMat([1, -2, 3, -4], 2, 2);
    const n = vecStoreOf(descriptorOf(m)!.neg!(m));
    expect(Array.from(n.c)).toEqual([-1, 2, -3, 4]);
    expect(n.cols).toBe(2);
  });
  it('getMember swizzle is vec-only; a mat has no swizzle', () => {
    expect(descriptorOf(makeVec([9, 8, 7]))!.getMember!(makeVec([9, 8, 7]), 'y')).toBe(8);
    const m = makeMat([1, 2, 3, 4], 2, 2);
    expect(descriptorOf(m)!.getMember!(m, 'x')).toBe(NOT_HANDLED_EXPORT);
  });
  it('equal-flat-length, different-shape mats are not equal (shape-aware equals)', () => {
    const a = makeMat([1, 2, 3, 4, 5, 6], 3, 2); // mat2x3 (rows=3, cols=2)
    const b = makeMat([1, 2, 3, 4, 5, 6], 2, 3); // mat3x2 (rows=2, cols=3)
    expect(descriptorOf(a)!.equals!(a, b)).toBe(false);
    expect(descriptorOf(a)!.equals!(a, a)).toBe(true);
  });
});

describe('transpose', () => {
  it('transpose of a square mat2 swaps off-diagonal (column-major)', () => {
    // mat2(1,2,3,4) column-major = columns (1,2),(3,4) = matrix [[1,3],[2,4]]; transpose = [[1,2],[3,4]]
    // column-major flat of the transpose = columns (1,3),(2,4) = [1,3,2,4]
    const t = run('transpose(mat2(1,2,3,4))');
    const s = vecStoreOf(t.value);
    expect(s.rows).toBe(2); expect(s.cols).toBe(2);
    expect(Array.from(s.c)).toEqual([1, 3, 2, 4]);
  });
  it('transpose of a non-square mat2x3 yields a mat3x2 (column-major)', () => {
    // mat2x3 = 2 cols × 3 rows, flat [1,2,3,4,5,6] = columns (1,2,3),(4,5,6).
    // transpose = 3 cols × 2 rows; element (r,c)→(c,r). out[r*C+c] = in[c*R+r], R=3,C=2.
    // → columns of the result (each 2 rows): (1,4),(2,5),(3,6) = flat [1,4,2,5,3,6].
    const t = run('transpose(mat2x3(1,2,3,4,5,6))');
    const s = vecStoreOf(t.value);
    expect(s.rows).toBe(2); expect(s.cols).toBe(3);
    expect(Array.from(s.c)).toEqual([1, 4, 2, 5, 3, 6]);
  });
});

describe('non-square matrix constructors', () => {
  it('mat2x3 constructs a 2-col × 3-row matrix (column-major)', () => {
    const r = run('mat2x3(1,2,3,4,5,6)');
    const s = vecStoreOf(r.value);
    expect(s.cols).toBe(2); expect(s.rows).toBe(3);
    expect(Array.from(s.c)).toEqual([1, 2, 3, 4, 5, 6]);
  });
});

describe('determinant', () => {
  it('determinant of mat2 (column-major)', () => {
    // mat2(1,2,3,4) = [[1,3],[2,4]], det = 1*4 - 3*2 = -2
    expect(run('determinant(mat2(1,2,3,4))').value).toBeCloseTo(-2);
  });
  it('determinant of mat3 (column-major)', () => {
    // identity → 1
    expect(run('determinant(mat3(1,0,0, 0,1,0, 0,0,1))').value).toBeCloseTo(1);
    // a general 3×3: col-major [2,0,1, 1,3,0, 0,2,1] → matrix rows [[2,1,0],[0,3,2],[1,0,1]]
    // det = 2*(3*1-2*0) - 1*(0*1-2*1) + 0*(0*0-3*1) = 2*3 - 1*(-2) + 0 = 6+2 = 8
    expect(run('determinant(mat3(2,0,1, 1,3,0, 0,2,1))').value).toBeCloseTo(8);
  });
  it('determinant of mat4 identity is 1', () => {
    expect(run('determinant(mat4())').value).toBeCloseTo(1);
  });
  it('determinant of a non-square matrix is a builtin-arg error (interpreter)', () => {
    const res = run('determinant(mat2x3(1,2,3,4,5,6))');
    expect(res.diagnostics.some((d) => d.code === 'ML-LANG-BUILTIN-ARG')).toBe(true);
  });
});

describe('inverse', () => {
  it('inverse(M) · M ≈ identity (mat2)', () => {
    // A non-singular mat2. inverse(M) * M must round to the column-major identity [1,0,0,1].
    const prog = 'const M = mat2(4,2,7,6) const I = inverse(M) * M I';
    const s = vecStoreOf(run(prog).value);
    expect(s.rows).toBe(2); expect(s.cols).toBe(2);
    // `+ 0` normalizes a rounded −0 (from a tiny negative off-diagonal residual) to +0 so toEqual matches.
    expect(Array.from(s.c).map((x) => Math.round(x) + 0)).toEqual([1, 0, 0, 1]);
  });
  it('inverse(M) · M ≈ identity (mat3)', () => {
    // A non-singular mat3 (col-major). inverse(M) * M must round to the 3×3 identity.
    const prog = 'const M = mat3(2,0,1, 1,3,0, 0,2,1) const I = inverse(M) * M I';
    const s = vecStoreOf(run(prog).value);
    expect(s.rows).toBe(3); expect(s.cols).toBe(3);
    expect(Array.from(s.c).map((x) => Math.round(x) + 0)).toEqual([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  });
  it('inverse(M) · M ≈ identity (mat4)', () => {
    // A non-singular mat4 (col-major). inverse(M) * M must round to the 4×4 identity.
    const prog = 'const M = mat4(1,0,2,0, 0,1,0,3, 4,0,1,0, 0,5,0,1) const I = inverse(M) * M I';
    const s = vecStoreOf(run(prog).value);
    expect(s.rows).toBe(4); expect(s.cols).toBe(4);
    expect(Array.from(s.c).map((x) => Math.round(x) + 0)).toEqual([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
  });
  it('inverse of a non-square matrix is a builtin-arg error (interpreter)', () => {
    const res = run('inverse(mat2x3(1,2,3,4,5,6))');
    expect(res.diagnostics.some((d) => d.code === 'ML-LANG-BUILTIN-ARG')).toBe(true);
  });
});

describe('distance / reflect / refract / faceforward', () => {
  it('distance/reflect evaluate', () => {
    expect(run('distance(vec2(0,0), vec2(3,4))').value).toBeCloseTo(5);
    // reflect(I,N) = I - 2*dot(N,I)*N; reflect((1,-1),(0,1)) = (1,1)
    const rf = vecStoreOf(run('reflect(vec2(1,-1), vec2(0,1))').value);
    expect(Array.from(rf.c)).toEqual([1, 1]);
  });
  it('refract total-internal-reflection returns the zero vector', () => {
    // I=(1,0), N=(0,1): d = dot(I,N) = 0; k = 1 - eta²·(1 - 0) = 1 - 25 = -24 < 0 → TIR → zero vector
    const rr = vecStoreOf(run('refract(vec2(1,0), vec2(0,1), 5)').value);
    expect(Array.from(rr.c)).toEqual([0, 0]);
  });
  it('faceforward flips N to oppose I', () => {
    // dot(Nref,I) = dot((0,1),(0,1)) = 1 >= 0 → return -N = -(1,1) = (-1,-1)
    const ff = vecStoreOf(run('faceforward(vec2(1,1), vec2(0,1), vec2(0,1))').value);
    expect(Array.from(ff.c)).toEqual([-1, -1]);
    // dot(Nref,I) = dot((0,-1),(0,1)) = -1 < 0 → return N = (1,1)
    const ff2 = vecStoreOf(run('faceforward(vec2(1,1), vec2(0,1), vec2(0,-1))').value);
    expect(Array.from(ff2.c)).toEqual([1, 1]);
  });
});
