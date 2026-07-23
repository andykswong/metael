// The numeric builtin module — one Builtin per numeric name, injected by a consumer at evaluateProgram.
// Each `invoke` translates a call into the capability-context API: it evaluates args ONCE (binding each to
// a local, since ctx.evalArg re-evaluates), delegates the arithmetic to @metael/math core, boxes the result
// through the vec/mat/buffer descriptors, and applies the fail-loud domain-guard diagnostics the language
// surface expects (core returns raw NaN; the binding raises the loud diagnostic in scalar position only).
import type { BuiltinCtx, LowerElement } from '@metael/lang';
import { descriptorOf, isUserFn } from '@metael/lang';
import { defineBuiltin } from '@metael/lang/profile';
import type { BuiltinSpec, DefinedBuiltin } from '@metael/lang/profile';
import { makeVec, makeMat, identityMat, vecStoreOf } from './descriptors.ts';
import { makeTypedArray, BUFFER_KINDS } from './buffers.ts';
import * as core from '@metael/math';

// A typed-array construction cap: 2^24 elements. Over this, construction fails closed with ML-LANG-BUDGET.
const MAX_BUFFER_LENGTH = 16_777_216;

// ─────────────────────────────────────────── the numeric-builtin specs ───────────────────────────────────────────

/** The static capability specs for every numeric builtin this module dispatches (constructors, vec/mat/quat
 *  ops, transforms, scalar math, bit ops). Each `defineBuiltin` below co-locates its spec (read from here)
 *  with its invoke; `mathProfile` (in index.ts) republishes these specs for a classifier / language service.
 *  profile (`core`/`host`) + portability (`exact`/`gpu-tolerant`/`cpu-only`) classify cross-target
 *  reproducibility; a `lowerName` names the shader builtin a call lowers to when it differs from `name`. */
const MATH_SPECS: Readonly<Record<string, BuiltinSpec>> = {
  // --- typed-array constructors (custom-type protocol; buffers) ---
  f32:         { name: 'f32',         profile: 'core', portability: 'gpu-tolerant', takesClosure: true,  arity: [1, 2], doc: 'Builds a Float32Array of the given length, optionally filling each element from a closure.', params: [{ name: 'length', doc: 'the number of elements to allocate' }, { name: 'fill', optional: true, doc: 'a closure `(i) => value` computing element `i` (defaults to zeros)' }], returnDoc: 'a Float32Array of that length' },
  f64:         { name: 'f64',         profile: 'core', portability: 'exact',        takesClosure: true,  arity: [1, 2], doc: 'Builds a Float64Array of the given length, optionally filling each element from a closure.', params: [{ name: 'length', doc: 'the number of elements to allocate' }, { name: 'fill', optional: true, doc: 'a closure `(i) => value` computing element `i` (defaults to zeros)' }], returnDoc: 'a Float64Array of that length' },
  i32:         { name: 'i32',         profile: 'core', portability: 'exact',        takesClosure: true,  arity: [1, 2], doc: 'Builds an Int32Array of the given length, optionally filling each element from a closure.', params: [{ name: 'length', doc: 'the number of elements to allocate' }, { name: 'fill', optional: true, doc: 'a closure `(i) => value` computing element `i` (defaults to zeros)' }], returnDoc: 'an Int32Array of that length' },
  u32:         { name: 'u32',         profile: 'core', portability: 'exact',        takesClosure: true,  arity: [1, 2], doc: 'Builds a Uint32Array of the given length, optionally filling each element from a closure.', params: [{ name: 'length', doc: 'the number of elements to allocate' }, { name: 'fill', optional: true, doc: 'a closure `(i) => value` computing element `i` (defaults to zeros)' }], returnDoc: 'a Uint32Array of that length' },

  // --- vec/mat constructors ---
  vec2:        { name: 'vec2',        profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 2], doc: 'Builds a 2-component vector from its components, or from a single scalar broadcast to both.', params: [{ name: 'x', doc: 'the first component, or the scalar to broadcast to both' }, { name: 'y', optional: true, doc: 'the second component' }], returnDoc: 'a 2-component vector' },
  vec3:        { name: 'vec3',        profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 3], doc: 'Builds a 3-component vector from its components, or from a single scalar broadcast to all three.', params: [{ name: 'x', doc: 'the first component, or the scalar to broadcast to all three' }, { name: 'y', optional: true, doc: 'the second component' }, { name: 'z', optional: true, doc: 'the third component' }], returnDoc: 'a 3-component vector' },
  vec4:        { name: 'vec4',        profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 4], doc: 'Builds a 4-component vector from its components, or from a single scalar broadcast to all four.', params: [{ name: 'x', doc: 'the first component, or the scalar to broadcast to all four' }, { name: 'y', optional: true, doc: 'the second component' }, { name: 'z', optional: true, doc: 'the third component' }, { name: 'w', optional: true, doc: 'the fourth component' }], returnDoc: 'a 4-component vector' },
  mat2:        { name: 'mat2',        profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [0, 4], doc: 'Builds a 2×2 column-major matrix from its components, or the identity when called with no arguments.', params: [{ name: 'components', rest: true, doc: '4 column-major numbers, or 2 column vectors (empty for the identity)' }], returnDoc: 'a 2×2 matrix' },
  mat3:        { name: 'mat3',        profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [0, 9], doc: 'Builds a 3×3 column-major matrix from its components, or the identity when called with no arguments.', params: [{ name: 'components', rest: true, doc: '9 column-major numbers, or 3 column vectors (empty for the identity)' }], returnDoc: 'a 3×3 matrix' },
  mat4:        { name: 'mat4',        profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [0, 16], doc: 'Builds a 4×4 column-major matrix from its components, or the identity when called with no arguments.', params: [{ name: 'components', rest: true, doc: '16 column-major numbers, or 4 column vectors (empty for the identity)' }], returnDoc: 'a 4×4 matrix' },
  mat2x3:      { name: 'mat2x3',      profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [6, 6], doc: 'Builds a 2-column × 3-row matrix in column-major order.', params: [{ name: 'components', rest: true, doc: '6 column-major numbers, or 2 column vectors of 3 components each' }], returnDoc: 'a matrix of 2 columns × 3 rows' },
  mat2x4:      { name: 'mat2x4',      profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [8, 8], doc: 'Builds a 2-column × 4-row matrix in column-major order.', params: [{ name: 'components', rest: true, doc: '8 column-major numbers, or 2 column vectors of 4 components each' }], returnDoc: 'a matrix of 2 columns × 4 rows' },
  mat3x2:      { name: 'mat3x2',      profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [6, 6], doc: 'Builds a 3-column × 2-row matrix in column-major order.', params: [{ name: 'components', rest: true, doc: '6 column-major numbers, or 3 column vectors of 2 components each' }], returnDoc: 'a matrix of 3 columns × 2 rows' },
  mat3x4:      { name: 'mat3x4',      profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [12, 12], doc: 'Builds a 3-column × 4-row matrix in column-major order.', params: [{ name: 'components', rest: true, doc: '12 column-major numbers, or 3 column vectors of 4 components each' }], returnDoc: 'a matrix of 3 columns × 4 rows' },
  mat4x2:      { name: 'mat4x2',      profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [8, 8], doc: 'Builds a 4-column × 2-row matrix in column-major order.', params: [{ name: 'components', rest: true, doc: '8 column-major numbers, or 4 column vectors of 2 components each' }], returnDoc: 'a matrix of 4 columns × 2 rows' },
  mat4x3:      { name: 'mat4x3',      profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [12, 12], doc: 'Builds a 4-column × 3-row matrix in column-major order.', params: [{ name: 'components', rest: true, doc: '12 column-major numbers, or 4 column vectors of 3 components each' }], returnDoc: 'a matrix of 4 columns × 3 rows' },

  // --- vec/quat geometric + matrix ops ---
  dot:         { name: 'dot',         profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [2, 2], lowerName: 'dot', doc: 'The dot product of two vectors of equal length.', params: [{ name: 'a', doc: 'the first vector' }, { name: 'b', doc: 'the second vector (same length as `a`)' }], returnDoc: 'the scalar dot product' },
  cross:       { name: 'cross',       profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [2, 2], lowerName: 'cross', doc: 'The cross product of two 3-component vectors (perpendicular to both).', params: [{ name: 'a', doc: 'the first 3-component vector' }, { name: 'b', doc: 'the second 3-component vector' }], returnDoc: 'the 3-component cross-product vector' },
  normalize:   { name: 'normalize',   profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1], lowerName: 'normalize', doc: 'Scales a vector to unit length, preserving its direction.', params: [{ name: 'v', doc: 'the vector to normalize' }], returnDoc: 'the unit-length vector in the same direction' },
  length:      { name: 'length',      profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1], lowerName: 'length', doc: 'The Euclidean length (magnitude) of a vector.', params: [{ name: 'v', doc: 'the vector' }], returnDoc: 'the scalar length' },
  transpose:   { name: 'transpose',   profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1], lowerName: 'transpose', doc: 'The transpose of a matrix (rows and columns swapped).', params: [{ name: 'm', doc: 'the matrix to transpose' }], returnDoc: 'the transposed matrix' },
  determinant: { name: 'determinant', profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1], lowerName: 'determinant', doc: 'The determinant of a square matrix.', params: [{ name: 'm', doc: 'the square matrix' }], returnDoc: 'the scalar determinant' },
  // No lowerName: WGSL has no inverse() at all (hand-emitted per matrix size); GLSL has inverse() natively
  // (emitted by an explicit name override). The gate accepts it on portability 'gpu-tolerant'.
  inverse:     { name: 'inverse',     profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1], doc: 'The inverse of a square matrix.', params: [{ name: 'm', doc: 'the square matrix to invert' }], returnDoc: 'the inverse matrix' },
  distance:    { name: 'distance',    profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [2, 2], lowerName: 'distance', doc: 'The Euclidean distance between two points.', params: [{ name: 'a', doc: 'the first point (vector)' }, { name: 'b', doc: 'the second point (vector, same length as `a`)' }], returnDoc: 'the scalar distance between them' },
  reflect:     { name: 'reflect',     profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [2, 2], lowerName: 'reflect', doc: 'Reflects an incident vector about a surface normal.', params: [{ name: 'i', doc: 'the incident vector' }, { name: 'n', doc: 'the surface normal (unit vector)' }], returnDoc: 'the reflected vector' },
  refract:     { name: 'refract',     profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [3, 3], lowerName: 'refract', doc: 'Refracts an incident vector through a surface at a given index-of-refraction ratio.', params: [{ name: 'i', doc: 'the incident vector' }, { name: 'n', doc: 'the surface normal (unit vector)' }, { name: 'eta', doc: 'the ratio of indices of refraction' }], returnDoc: 'the refracted vector' },
  faceforward: { name: 'faceforward', profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [3, 3], doc: 'Orients a vector to face away from the incident direction (flips it if `dot(nref, i) >= 0`).', params: [{ name: 'n', doc: 'the vector to orient' }, { name: 'i', doc: 'the incident vector' }, { name: 'nref', doc: 'the reference normal' }], returnDoc: '`n`, flipped if needed to point away from `i`' },
  // Componentwise (Hadamard) matrix product. GLSL has matrixCompMult; WGSL has no equivalent. No lowerName
  // (no hand-emit yet); classified core/gpu-tolerant for a future codegen consumer.
  matrixCompMult: { name: 'matrixCompMult', profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [2, 2], doc: 'The componentwise (Hadamard) product of two same-shape matrices.', params: [{ name: 'a', doc: 'the first matrix' }, { name: 'b', doc: 'the second matrix (same shape as `a`)' }], returnDoc: 'the componentwise-product matrix' },

  // --- quaternions (vec4 layout (x,y,z,w) = imaginary xyz + real w; no distinct quat value type) ---
  // No lowerName: every q* op is HAND-EMITTED inline (or via a small prelude helper) on all targets —
  // there is no native quaternion type or builtin in WGSL/GLSL. The gate accepts them on 'gpu-tolerant'.
  qmul:       { name: 'qmul',       profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [2, 2], doc: 'The Hamilton product of two quaternions (composes their rotations).', params: [{ name: 'a', doc: 'the first quaternion (vec4 x,y,z,w)' }, { name: 'b', doc: 'the second quaternion (vec4 x,y,z,w)' }], returnDoc: 'the product quaternion (vec4)' },
  qconj:      { name: 'qconj',      profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1], doc: 'The conjugate of a quaternion (its imaginary part negated).', params: [{ name: 'q', doc: 'the quaternion (vec4 x,y,z,w)' }], returnDoc: 'the conjugate quaternion (vec4)' },
  qinvert:    { name: 'qinvert',    profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1], doc: 'The multiplicative inverse of a quaternion.', params: [{ name: 'q', doc: 'the quaternion (vec4 x,y,z,w)' }], returnDoc: 'the inverse quaternion (vec4)' },
  qaxisangle: { name: 'qaxisangle', profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [2, 2], doc: 'Builds the quaternion for a rotation of `angle` radians about `axis`.', params: [{ name: 'axis', doc: 'the rotation axis (vec3)' }, { name: 'angle', doc: 'the rotation angle in radians' }], returnDoc: 'the rotation quaternion (vec4)' },
  qrotate:    { name: 'qrotate',    profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [2, 2], doc: 'Rotates a 3-component vector by a quaternion.', params: [{ name: 'q', doc: 'the rotation quaternion (vec4 x,y,z,w)' }, { name: 'v', doc: 'the 3-component vector to rotate' }], returnDoc: 'the rotated 3-component vector' },
  qslerp:     { name: 'qslerp',     profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [3, 3], doc: 'The spherical linear interpolation between two quaternions.', params: [{ name: 'a', doc: 'the start quaternion (vec4)' }, { name: 'b', doc: 'the end quaternion (vec4)' }, { name: 't', doc: 'the interpolation factor in [0,1]' }], returnDoc: 'the interpolated quaternion (vec4)' },
  qmat:       { name: 'qmat',       profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1], doc: 'The 3×3 rotation matrix equivalent to a quaternion.', params: [{ name: 'q', doc: 'the quaternion (vec4 x,y,z,w)' }], returnDoc: 'the equivalent 3×3 rotation matrix' },

  // --- affine transform composition + camera projections (return matrices / a decomposed object) ---
  // decompose returns a heap object ({t,r,s}) so it is host/cpu-only; the rest return a mat and stay
  // core/gpu-tolerant.
  transformation: { name: 'transformation', profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [3, 3], doc: 'Composes a 4×4 affine matrix from translation, rotation, and scale (T·R·S).', params: [{ name: 't', doc: 'the translation (vec3)' }, { name: 'r', doc: 'the rotation quaternion (vec4)' }, { name: 's', doc: 'the scale (vec3)' }], returnDoc: 'the composed 4×4 affine matrix' },
  decompose:      { name: 'decompose',      profile: 'host', portability: 'cpu-only',     takesClosure: false, arity: [1, 1], doc: 'Decomposes a 4×4 affine matrix into its translation, rotation, and scale.', params: [{ name: 'm', doc: 'the 4×4 affine matrix to decompose' }], returnDoc: 'an object `{ t, r, s }` — translation (vec3), rotation (vec4), and scale (vec3)' },
  perspective:    { name: 'perspective',    profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [4, 4], doc: 'Builds a perspective projection matrix from field of view, aspect ratio, and clip planes.', params: [{ name: 'fovy', doc: 'the vertical field of view in radians' }, { name: 'aspect', doc: 'the viewport aspect ratio (width / height)' }, { name: 'near', doc: 'the near clip-plane distance' }, { name: 'far', doc: 'the far clip-plane distance' }], returnDoc: 'the 4×4 perspective projection matrix' },
  ortho:          { name: 'ortho',          profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [6, 6], doc: 'Builds an orthographic projection matrix from the clip-box bounds.', params: [{ name: 'left', doc: 'the left clip-box bound' }, { name: 'right', doc: 'the right clip-box bound' }, { name: 'bottom', doc: 'the bottom clip-box bound' }, { name: 'top', doc: 'the top clip-box bound' }, { name: 'near', doc: 'the near clip-box bound' }, { name: 'far', doc: 'the far clip-box bound' }], returnDoc: 'the 4×4 orthographic projection matrix' },
  lookAt:         { name: 'lookAt',         profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [3, 3], doc: 'Builds a view matrix for a camera positioned at `eye` looking toward `center`.', params: [{ name: 'eye', doc: 'the camera position (vec3)' }, { name: 'center', doc: 'the target point to look at (vec3)' }, { name: 'up', doc: 'the up direction (vec3)' }], returnDoc: 'the 4×4 view matrix' },

  // --- numeric core ---
  min:         { name: 'min',         profile: 'core', portability: 'exact',        takesClosure: false, arity: [2, 2], doc: 'The smaller of two values, componentwise over vectors.', params: [{ name: 'a', doc: 'the first value (scalar or vector)' }, { name: 'b', doc: 'the second value (scalar or vector)' }], returnDoc: 'the smaller value' },
  max:         { name: 'max',         profile: 'core', portability: 'exact',        takesClosure: false, arity: [2, 2], doc: 'The larger of two values, componentwise over vectors.', params: [{ name: 'a', doc: 'the first value (scalar or vector)' }, { name: 'b', doc: 'the second value (scalar or vector)' }], returnDoc: 'the larger value' },
  abs:         { name: 'abs',         profile: 'core', portability: 'exact',        takesClosure: false, arity: [1, 1], doc: 'The absolute value, componentwise over vectors.', params: [{ name: 'x', doc: 'the input value (scalar or vector)' }], returnDoc: 'the absolute value' },
  sign:        { name: 'sign',        profile: 'core', portability: 'exact',        takesClosure: false, arity: [1, 1], doc: 'The sign of a value (-1, 0, or 1), componentwise over vectors.', params: [{ name: 'x', doc: 'the input value (scalar or vector)' }], returnDoc: '-1, 0, or 1 by the sign of the input' },
  floor:       { name: 'floor',       profile: 'core', portability: 'exact',        takesClosure: false, arity: [1, 1], doc: 'The largest integer not greater than the value, componentwise over vectors.', params: [{ name: 'x', doc: 'the input value (scalar or vector)' }], returnDoc: 'the floored value' },
  ceil:        { name: 'ceil',        profile: 'core', portability: 'exact',        takesClosure: false, arity: [1, 1], doc: 'The smallest integer not less than the value, componentwise over vectors.', params: [{ name: 'x', doc: 'the input value (scalar or vector)' }], returnDoc: 'the ceiled value' },
  round:       { name: 'round',       profile: 'core', portability: 'exact',        takesClosure: false, arity: [1, 1], doc: 'The nearest integer to the value (ties to even), componentwise over vectors.', params: [{ name: 'x', doc: 'the input value (scalar or vector)' }], returnDoc: 'the rounded value' },
  clamp:       { name: 'clamp',       profile: 'core', portability: 'exact',        takesClosure: false, arity: [3, 3], doc: 'Constrains a value to a range, componentwise over vectors.', params: [{ name: 'x', doc: 'the value to constrain (scalar or vector)' }, { name: 'lo', doc: 'the lower bound' }, { name: 'hi', doc: 'the upper bound' }], returnDoc: '`x` constrained to `[lo, hi]`' },
  sqrt:        { name: 'sqrt',        profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1], lowerName: 'sqrt', doc: 'The non-negative square root, componentwise over vectors.', params: [{ name: 'x', doc: 'the input value (>= 0; scalar or vector)' }], returnDoc: 'the non-negative square root' },
  pow:         { name: 'pow',         profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [2, 2], lowerName: 'pow', doc: 'Raises a base to an exponent, componentwise over vectors.', params: [{ name: 'x', doc: 'the base (scalar or vector)' }, { name: 'y', doc: 'the exponent (scalar or vector)' }], returnDoc: '`x` raised to the power `y`' },
  // Floored modulo — sign follows the DIVISOR. No lowerName: it needs a bespoke emitter case per target.
  mod:         { name: 'mod',         profile: 'core', portability: 'exact',        takesClosure: false, arity: [2, 2], doc: 'The floored modulo `x - y*floor(x/y)` — the sign follows the divisor `y`; componentwise over vectors.', params: [{ name: 'x', doc: 'the dividend (scalar or vector)' }, { name: 'y', doc: 'the divisor (scalar or vector)' }], returnDoc: 'the floored remainder' },

  // --- transcendentals (native-lowerable to a shader builtin) ---
  sin:         { name: 'sin',         profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1], lowerName: 'sin', doc: 'The sine of an angle in radians, componentwise over vectors.', params: [{ name: 'x', doc: 'the angle in radians (scalar or vector)' }], returnDoc: 'the sine' },
  cos:         { name: 'cos',         profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1], lowerName: 'cos', doc: 'The cosine of an angle in radians, componentwise over vectors.', params: [{ name: 'x', doc: 'the angle in radians (scalar or vector)' }], returnDoc: 'the cosine' },
  tan:         { name: 'tan',         profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1], lowerName: 'tan', doc: 'The tangent of an angle in radians, componentwise over vectors.', params: [{ name: 'x', doc: 'the angle in radians (scalar or vector)' }], returnDoc: 'the tangent' },
  sinh:        { name: 'sinh',        profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1], lowerName: 'sinh', doc: 'The hyperbolic sine, componentwise over vectors.', params: [{ name: 'x', doc: 'the input value (scalar or vector)' }], returnDoc: 'the hyperbolic sine' },
  cosh:        { name: 'cosh',        profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1], lowerName: 'cosh', doc: 'The hyperbolic cosine, componentwise over vectors.', params: [{ name: 'x', doc: 'the input value (scalar or vector)' }], returnDoc: 'the hyperbolic cosine' },
  tanh:        { name: 'tanh',        profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1], lowerName: 'tanh', doc: 'The hyperbolic tangent, componentwise over vectors.', params: [{ name: 'x', doc: 'the input value (scalar or vector)' }], returnDoc: 'the hyperbolic tangent' },
  asin:        { name: 'asin',        profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1], lowerName: 'asin', doc: 'The arcsine in radians, componentwise over vectors.', params: [{ name: 'x', doc: 'the input value in [-1, 1] (scalar or vector)' }], returnDoc: 'the arcsine in radians' },
  acos:        { name: 'acos',        profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1], lowerName: 'acos', doc: 'The arccosine in radians, componentwise over vectors.', params: [{ name: 'x', doc: 'the input value in [-1, 1] (scalar or vector)' }], returnDoc: 'the arccosine in radians' },
  atan:        { name: 'atan',        profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1], lowerName: 'atan', doc: 'The arctangent in radians, componentwise over vectors.', params: [{ name: 'x', doc: 'the input value (scalar or vector)' }], returnDoc: 'the arctangent in radians' },
  atan2:       { name: 'atan2',       profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [2, 2], doc: 'The angle in radians of the point `(x, y)` from the positive x-axis (GLSL order y, x).', params: [{ name: 'y', doc: 'the y coordinate (scalar or vector)' }, { name: 'x', doc: 'the x coordinate (scalar or vector)' }], returnDoc: 'the angle in radians in (-π, π]' },
  // Inverse hyperbolics — a native shader builtin of the SAME name on both targets, so the emitter's generic
  // `lowerName` path handles them. Out-of-domain returns NaN on both interpreter + shader (NO fail-loud guard).
  asinh:       { name: 'asinh',       profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1], lowerName: 'asinh', doc: 'The inverse hyperbolic sine, componentwise over vectors.', params: [{ name: 'x', doc: 'the input value (scalar or vector)' }], returnDoc: 'the inverse hyperbolic sine' },
  acosh:       { name: 'acosh',       profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1], lowerName: 'acosh', doc: 'The inverse hyperbolic cosine, componentwise over vectors.', params: [{ name: 'x', doc: 'the input value (>= 1; scalar or vector)' }], returnDoc: 'the inverse hyperbolic cosine' },
  atanh:       { name: 'atanh',       profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1], lowerName: 'atanh', doc: 'The inverse hyperbolic tangent, componentwise over vectors.', params: [{ name: 'x', doc: 'the input value with |x| < 1 (scalar or vector)' }], returnDoc: 'the inverse hyperbolic tangent' },
  exp:         { name: 'exp',         profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1], lowerName: 'exp', doc: 'The base-e exponential `e^x`, componentwise over vectors.', params: [{ name: 'x', doc: 'the exponent (scalar or vector)' }], returnDoc: 'e raised to `x`' },
  exp2:        { name: 'exp2',        profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1], lowerName: 'exp2', doc: 'The base-2 exponential `2^x`, componentwise over vectors.', params: [{ name: 'x', doc: 'the exponent (scalar or vector)' }], returnDoc: '2 raised to `x`' },
  log:         { name: 'log',         profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1], lowerName: 'log', doc: 'The natural (base-e) logarithm, componentwise over vectors.', params: [{ name: 'x', doc: 'the input value (> 0; scalar or vector)' }], returnDoc: 'the natural logarithm' },
  log2:        { name: 'log2',        profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1], lowerName: 'log2', doc: 'The base-2 logarithm, componentwise over vectors.', params: [{ name: 'x', doc: 'the input value (> 0; scalar or vector)' }], returnDoc: 'the base-2 logarithm' },
  inverseSqrt: { name: 'inverseSqrt', profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1], doc: 'The reciprocal square root `1/sqrt(x)`, componentwise over vectors.', params: [{ name: 'x', doc: 'the input value (> 0; scalar or vector)' }], returnDoc: 'the reciprocal square root' },
  fract:       { name: 'fract',       profile: 'core', portability: 'exact',        takesClosure: false, arity: [1, 1], lowerName: 'fract', doc: 'The fractional part `x - floor(x)`, componentwise over vectors.', params: [{ name: 'x', doc: 'the input value (scalar or vector)' }], returnDoc: 'the fractional part in [0, 1)' },
  degrees:     { name: 'degrees',     profile: 'core', portability: 'exact',        takesClosure: false, arity: [1, 1], lowerName: 'degrees', doc: 'Converts radians to degrees, componentwise over vectors.', params: [{ name: 'radians', doc: 'the angle in radians (scalar or vector)' }], returnDoc: 'the angle in degrees' },
  radians:     { name: 'radians',     profile: 'core', portability: 'exact',        takesClosure: false, arity: [1, 1], lowerName: 'radians', doc: 'Converts degrees to radians, componentwise over vectors.', params: [{ name: 'degrees', doc: 'the angle in degrees (scalar or vector)' }], returnDoc: 'the angle in radians' },
  trunc:       { name: 'trunc',       profile: 'core', portability: 'exact',        takesClosure: false, arity: [1, 1], lowerName: 'trunc', doc: 'The integer part with the fraction discarded (toward zero), componentwise over vectors.', params: [{ name: 'x', doc: 'the input value (scalar or vector)' }], returnDoc: 'the truncated integer part' },
  step:        { name: 'step',        profile: 'core', portability: 'exact',        takesClosure: false, arity: [2, 2], lowerName: 'step', doc: 'A step function: 0 when `x < edge`, else 1; componentwise over vectors.', params: [{ name: 'edge', doc: 'the threshold (scalar or vector)' }, { name: 'x', doc: 'the value to test (scalar or vector)' }], returnDoc: '0 when `x < edge`, else 1' },
  mix:         { name: 'mix',         profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [3, 3], lowerName: 'mix', doc: 'Linearly interpolates between two values by `t`; componentwise over vectors.', params: [{ name: 'a', doc: 'the start value (scalar or vector)' }, { name: 'b', doc: 'the end value (scalar or vector)' }, { name: 't', doc: 'the interpolation factor (0 → `a`, 1 → `b`)' }], returnDoc: 'the interpolated value `a*(1-t) + b*t`' },
  smoothstep:  { name: 'smoothstep',  profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [3, 3], lowerName: 'smoothstep', doc: 'A smooth Hermite interpolation from 0 to 1 as `x` crosses `[e0, e1]`; componentwise over vectors.', params: [{ name: 'e0', doc: 'the lower edge' }, { name: 'e1', doc: 'the upper edge' }, { name: 'x', doc: 'the value to interpolate (scalar or vector)' }], returnDoc: 'a smooth 0→1 ramp of `x` across `[e0, e1]`' },

  // --- integer bit operations (32-bit unsigned; u32-reinterpret-of-f32 boundary; truncating input coercion) ---
  countOneBits: { name: 'countOneBits', profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1], doc: 'The population count (number of set bits) of a 32-bit integer, componentwise over vectors.', params: [{ name: 'x', doc: 'the 32-bit integer (scalar or vector)' }], returnDoc: 'the number of set bits' },
  reverseBits:  { name: 'reverseBits',  profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [1, 1], doc: 'Reverses the bit order of a 32-bit integer, componentwise over vectors.', params: [{ name: 'x', doc: 'the 32-bit integer (scalar or vector)' }], returnDoc: 'the integer with its bit order reversed' },
};

// ─────────────────────────────────────────── shared helpers ───────────────────────────────────────────

/** Numeric coercion mirroring the interpreter's toNum: number→v; boolean→0/1; trimmed string→Number|NaN;
 *  anything else→NaN. */
function toNum(v: unknown): number {
  if (typeof v === 'number') return v;
  if (typeof v === 'boolean') return v ? 1 : 0;
  if (typeof v === 'string') { const t = v.trim(); return t === '' ? NaN : Number(t); }
  return NaN;
}
/** Coerce an arg to a finite/coercible number, or null if not (a NaN or a non-coercible value). */
function num(v: unknown): number | null {
  if (typeof v === 'number') return Number.isNaN(v) ? null : v;
  const n = toNum(v);
  return Number.isNaN(n) ? null : n;
}
/** A strict numeric read: a real (non-NaN) number passes through, anything else → null (no coercion of
 *  strings/booleans). Used for the geometric/quaternion scalar slots that only ever accept a literal number. */
function strictNum(v: unknown): number | null {
  return (typeof v === 'number' && !Number.isNaN(v)) ? v : null;
}
/** The components of a value IFF it is a vecN (single-column), else null. */
function vecComps(v: unknown): number[] | null {
  const d = descriptorOf(v);
  if (!d || d.lower?.shape !== 'vecN') return null;
  return Array.from(vecStoreOf(v).c);
}
/** True iff the value is a vecN (single-column) custom value. */
function isVecValue(v: unknown): boolean {
  return descriptorOf(v)?.lower?.shape === 'vecN';
}
/** The result precision for a computed value: f64 if ANY vec/mat operand is f64, else f32. */
function elemOfArgs(args: readonly unknown[]): LowerElement {
  for (const a of args) if (descriptorOf(a)?.lower?.element === 'f64') return 'f64';
  return 'f32';
}
/** Read all supplied call args ONCE (ctx.evalArg re-evaluates), matching the old single-pass arg map. */
function evalAll(ctx: BuiltinCtx): unknown[] {
  const n = ctx.argCount();
  const out = new Array<unknown>(n);
  for (let i = 0; i < n; i++) out[i] = ctx.evalArg(i);
  return out;
}

// ─────────────────────────────────────────── typed-array constructors ───────────────────────────────────────────

function makeBufferBuiltin(spec: BuiltinSpec, kind: 'f32' | 'f64' | 'i32' | 'u32'): DefinedBuiltin {
  return defineBuiltin(spec,
    (ctx) => {
      ctx.tick();
      const spec = BUFFER_KINDS[kind];
      const a0 = ctx.evalArg(0);   // null when no arg (evalArg substitutes a null literal)
      let store: { [i: number]: number; length: number };
      if (typeof a0 === 'number') {
        const n = Math.floor(a0);
        if (!Number.isFinite(n) || n < 0) { ctx.error('ML-LANG-BUILTIN-ARG', `${kind}(n) — n must be a non-negative number`); return ctx.freeze([]); }
        if (n > MAX_BUFFER_LENGTH) { ctx.error('ML-LANG-BUDGET', `${kind}(${n}) exceeds the ${MAX_BUFFER_LENGTH}-element cap`); return ctx.freeze([]); }
        store = new spec.ctor(n);
        const genFn = ctx.argCount() > 1 ? ctx.evalArg(1) : undefined;
        if (genFn !== undefined) {
          if (typeof genFn !== 'function' && !isUserFn(genFn)) { ctx.error('ML-LANG-BUILTIN-ARG', `${kind}(n, fn) — the second argument must be a function`); return ctx.freeze([]); }
          for (let i = 0; i < n; i++) { ctx.tick(); const val = ctx.callClosure(genFn, [i]); store[i] = spec.coerce(typeof val === 'number' ? val : NaN); }
        }
      } else if (Array.isArray(a0)) {
        if (a0.length > MAX_BUFFER_LENGTH) { ctx.error('ML-LANG-BUDGET', `${kind}([…]) exceeds the ${MAX_BUFFER_LENGTH}-element cap`); return ctx.freeze([]); }
        store = new spec.ctor(a0.length);
        for (let i = 0; i < a0.length; i++) { ctx.tick(); const val = a0[i]; store[i] = spec.coerce(typeof val === 'number' ? val : NaN); }
      } else {
        ctx.error('ML-LANG-BUILTIN-ARG', `${kind}(n | […] | (n, fn)) — bad argument`);
        return ctx.freeze([]);
      }
      const gen = ctx.allocateGeneration();
      return makeTypedArray(kind, store, gen);
    },
  );
}

// ─────────────────────────────────────────── vec/mat constructors (with §GLSL/WGSL composition) ───────────────────────────────────────────

function makeVecBuiltin(spec: BuiltinSpec, head: 'vec2' | 'vec3' | 'vec4', N: number): DefinedBuiltin {
  return defineBuiltin(spec,
    (ctx) => {
      ctx.tick();
      const args = evalAll(ctx);
      // A single number splats to N copies (f32).
      if (args.length === 1 && typeof args[0] === 'number') return makeVec(new Array<number>(N).fill(args[0]));
      // Composition: each arg is a number or a vecM; flatten left-to-right; the total must be exactly N.
      const flat: number[] = [];
      let f64 = false;
      for (const a of args) {
        if (typeof a === 'number') { flat.push(a); continue; }
        if (isVecValue(a)) { const s = vecStoreOf(a); for (let i = 0; i < s.c.length; i++) flat.push(s.c[i] as number); if (s.element === 'f64') f64 = true; continue; }
        ctx.error('ML-LANG-BUILTIN-ARG', `${head}(...) — arguments must be numbers or vectors`); return ctx.freeze([]);
      }
      if (flat.length !== N) { ctx.error('ML-LANG-BUILTIN-ARG', `${head}(...) — components must total ${N}`); return ctx.freeze([]); }
      return makeVec(flat, f64 ? 'f64' : 'f32');
    },
  );
}

/** Build a matrix builtin for a rows×cols shape. Square shapes (rows===cols) additionally accept the
 *  zero-arg identity form. All shapes accept EITHER `rows*cols` numbers (column-major) OR exactly `cols`
 *  column vectors (each a vec of width `rows`). */
function makeMatBuiltin(spec: BuiltinSpec, head: string, rows: number, cols: number, allowIdentity: boolean): DefinedBuiltin {
  const total = rows * cols;
  return defineBuiltin(spec,
    (ctx) => {
      ctx.tick();
      const args = evalAll(ctx);
      if (args.length === 0 && allowIdentity) return makeMat(identityMat(rows), rows, cols);
      // Column-vector form: exactly `cols` args, each a vec of width `rows` (column-major layout).
      if (args.length === cols && args.every((a) => isVecValue(a) && vecStoreOf(a).rows === rows)) {
        const flat: number[] = [];
        let f64 = false;
        for (const a of args) { const s = vecStoreOf(a); for (let i = 0; i < s.c.length; i++) flat.push(s.c[i] as number); if (s.element === 'f64') f64 = true; }
        return makeMat(flat, rows, cols, f64 ? 'f64' : 'f32');
      }
      // All-numbers form (column-major).
      if (args.every((x) => typeof x === 'number') && args.length === total) return makeMat(args as number[], rows, cols);
      ctx.error('ML-LANG-BUILTIN-ARG', allowIdentity
        ? `${head}() (identity), ${head}(${total} numbers, column-major), or ${head}(${cols} column vectors)`
        : `${head}(${total} numbers, column-major) or ${head}(${cols} column vectors)`);
      return ctx.freeze([]);
    },
  );
}

// ─────────────────────────────────────────── matrix ops ───────────────────────────────────────────

const transposeBuiltin: DefinedBuiltin = defineBuiltin(MATH_SPECS.transpose!,
  (ctx) => {
    ctx.tick();
    const m = ctx.evalArg(0);
    const d = descriptorOf(m);
    if (!d || d.lower?.shape !== 'matMxN') { ctx.error('ML-LANG-BUILTIN-ARG', 'transpose(mat)'); return ctx.freeze([]); }
    const s = vecStoreOf(m);
    return makeMat(core.transpose(s.c as number[], s.rows, s.cols), s.cols, s.rows, s.element);
  },
);

const determinantBuiltin: DefinedBuiltin = defineBuiltin(MATH_SPECS.determinant!,
  (ctx) => {
    ctx.tick();
    const m = ctx.evalArg(0);
    const d = descriptorOf(m);
    const s = d?.lower?.shape === 'matMxN' ? vecStoreOf(m) : null;
    if (!s || s.rows !== s.cols) { ctx.error('ML-LANG-BUILTIN-ARG', 'determinant(square mat)'); return ctx.freeze([]); }
    return core.determinant(s.c as number[], s.rows);
  },
);

const inverseBuiltin: DefinedBuiltin = defineBuiltin(MATH_SPECS.inverse!,
  (ctx) => {
    ctx.tick();
    const m = ctx.evalArg(0);
    const d = descriptorOf(m);
    const s = d?.lower?.shape === 'matMxN' ? vecStoreOf(m) : null;
    if (!s || s.rows !== s.cols) { ctx.error('ML-LANG-BUILTIN-ARG', 'inverse(square mat)'); return ctx.freeze([]); }
    return makeMat(core.inverse(s.c as number[], s.rows), s.rows, s.rows, s.element);
  },
);

const matrixCompMultBuiltin: DefinedBuiltin = defineBuiltin(MATH_SPECS.matrixCompMult!,
  (ctx) => {
    ctx.tick();
    const a = ctx.evalArg(0); const b = ctx.evalArg(1);
    const da = descriptorOf(a); const db = descriptorOf(b);
    const sa = da?.lower?.shape === 'matMxN' ? vecStoreOf(a) : null;
    const sb = db?.lower?.shape === 'matMxN' ? vecStoreOf(b) : null;
    if (!sa || !sb || sa.rows !== sb.rows || sa.cols !== sb.cols) { ctx.error('ML-LANG-BUILTIN-ARG', 'matrixCompMult(mat, mat) — matrices must be the same shape'); return ctx.freeze([]); }
    return makeMat(core.matrixCompMult(sa.c as number[], sb.c as number[]), sa.rows, sa.cols, sa.element === 'f64' || sb.element === 'f64' ? 'f64' : 'f32');
  },
);

const qmatBuiltin: DefinedBuiltin = defineBuiltin(MATH_SPECS.qmat!,
  (ctx) => {
    ctx.tick();
    const q = ctx.evalArg(0);
    const d = descriptorOf(q);
    if (!d || d.lower?.shape !== 'vecN' || (d.lower.rows ?? 0) !== 4) { ctx.error('ML-LANG-BUILTIN-ARG', 'qmat(vec4 quaternion)'); return ctx.freeze([]); }
    const s = vecStoreOf(q);
    return makeMat(core.qmat(s.c as number[]), 3, 3, s.element);
  },
);

// ─────────────────────────────────────────── vec/quat geometric builtins ───────────────────────────────────────────

/** One builtin per geometric vec/quat op. Each reads its args once, validates vector shapes, delegates to
 *  core, and boxes a vec result (f64 iff any vec arg is f64). A bad-shape arg → ML-LANG-BUILTIN-ARG + []. */
const geometricBuiltins: DefinedBuiltin[] = (() => {
  const names = ['dot', 'cross', 'normalize', 'length', 'distance', 'reflect', 'refract', 'faceforward',
    'qmul', 'qconj', 'qinvert', 'qaxisangle', 'qrotate', 'qslerp'] as const;
  return names.map<DefinedBuiltin>((head) => defineBuiltin(MATH_SPECS[head]!,
    (ctx) => {
      ctx.tick();
      const args = evalAll(ctx);
      const elem = elemOfArgs(args);
      const badVec = (): unknown => { ctx.error('ML-LANG-BUILTIN-ARG', `${head}(vec…) — argument must be a vector`); return ctx.freeze([]); };
      switch (head) {
        case 'dot': { const x = vecComps(args[0]); const y = vecComps(args[1]); if (!x || !y || x.length !== y.length) return badVec(); return core.dot(x, y); }
        case 'cross': { const x = vecComps(args[0]); const y = vecComps(args[1]); if (!x || !y || x.length !== 3 || y.length !== 3) return badVec(); return makeVec(core.cross(x, y), elem); }
        case 'length': { const x = vecComps(args[0]); if (!x) return badVec(); return core.length(x); }
        case 'distance': { const x = vecComps(args[0]); const y = vecComps(args[1]); if (!x || !y || x.length !== y.length) return badVec(); return core.distance(x, y); }
        case 'reflect': { const I = vecComps(args[0]); const N = vecComps(args[1]); if (!I || !N || I.length !== N.length) return badVec(); return makeVec(core.reflect(I, N), elem); }
        case 'refract': { const I = vecComps(args[0]); const N = vecComps(args[1]); const eta = strictNum(args[2]); if (!I || !N || eta === null || I.length !== N.length) return badVec(); return makeVec(core.refract(I, N, eta), elem); }
        case 'faceforward': { const N = vecComps(args[0]); const I = vecComps(args[1]); const Nref = vecComps(args[2]); if (!N || !I || !Nref || N.length !== I.length || N.length !== Nref.length) return badVec(); return makeVec(core.faceforward(N, I, Nref), elem); }
        case 'qconj': { const q = vecComps(args[0]); if (!q || q.length !== 4) return badVec(); return makeVec(core.qconj(q), elem); }
        case 'qinvert': { const q = vecComps(args[0]); if (!q || q.length !== 4) return badVec(); return makeVec(core.qinvert(q), elem); }
        case 'qmul': { const A = vecComps(args[0]); const B = vecComps(args[1]); if (!A || A.length !== 4 || !B || B.length !== 4) return badVec(); return makeVec(core.qmul(A, B), elem); }
        case 'qaxisangle': { const ax = vecComps(args[0]); const ang = strictNum(args[1]); if (!ax || ax.length !== 3 || ang === null) return badVec(); return makeVec(core.qaxisangle(ax, ang), elem); }
        case 'qrotate': { const q = vecComps(args[0]); const v = vecComps(args[1]); if (!q || q.length !== 4 || !v || v.length !== 3) return badVec(); return makeVec(core.qrotate(q, v), elem); }
        case 'qslerp': { const A = vecComps(args[0]); const B = vecComps(args[1]); const t = strictNum(args[2]); if (!A || A.length !== 4 || !B || B.length !== 4 || t === null) return badVec(); return makeVec(core.qslerp(A, B, t), elem); }
        case 'normalize': default: { const x = vecComps(args[0]); if (!x) return badVec(); return makeVec(core.normalize(x), elem); }
      }
    },
  ));
})();

// ─────────────────────────────────────────── affine transform / camera builtins ───────────────────────────────────────────

const transformBuiltins: DefinedBuiltin[] = [
  defineBuiltin(MATH_SPECS.transformation!,
    (ctx) => {
      ctx.tick();
      const args = evalAll(ctx);
      const t = vecComps(args[0]); const r = vecComps(args[1]); const s = vecComps(args[2]);
      if (!t || t.length !== 3 || !r || r.length !== 4 || !s || s.length !== 3) { ctx.error('ML-LANG-BUILTIN-ARG', 'transformation(t:vec3, r:vec4, s:vec3)'); return ctx.freeze([]); }
      return makeMat(core.transformation(t, r, s), 4, 4, elemOfArgs(args));
    },
  ),
  defineBuiltin(MATH_SPECS.decompose!,
    (ctx) => {
      ctx.tick();
      const m = ctx.evalArg(0);
      const d = descriptorOf(m);
      const s = d?.lower?.shape === 'matMxN' ? vecStoreOf(m) : null;
      if (!s || s.rows !== 4 || s.cols !== 4) { ctx.error('ML-LANG-BUILTIN-ARG', 'decompose(mat4)'); return ctx.freeze([]); }
      const { t, r, sc } = ((): { t: number[]; r: number[]; sc: number[] } => { const dec = core.decompose(s.c as number[]); return { t: dec.t, r: dec.r, sc: dec.s }; })();
      return ctx.freeze({ t: makeVec(t, s.element), r: makeVec(r, s.element), s: makeVec(sc, s.element) });
    },
  ),
  defineBuiltin(MATH_SPECS.perspective!,
    (ctx) => {
      ctx.tick();
      const args = evalAll(ctx);
      const fovy = num(args[0]); const aspect = num(args[1]); const near = num(args[2]); const far = num(args[3]);
      if (fovy === null || aspect === null || near === null || far === null) { ctx.error('ML-LANG-BUILTIN-ARG', 'perspective(fovy, aspect, near, far) — numeric arguments'); return ctx.freeze([]); }
      return makeMat(core.perspective(fovy, aspect, near, far), 4, 4);
    },
  ),
  defineBuiltin(MATH_SPECS.ortho!,
    (ctx) => {
      ctx.tick();
      const args = evalAll(ctx);
      const xs = [num(args[0]), num(args[1]), num(args[2]), num(args[3]), num(args[4]), num(args[5])];
      if (xs.some((x) => x === null)) { ctx.error('ML-LANG-BUILTIN-ARG', 'ortho(left, right, bottom, top, near, far) — numeric arguments'); return ctx.freeze([]); }
      const [l, r, b, t, n, f] = xs as number[];
      return makeMat(core.ortho(l!, r!, b!, t!, n!, f!), 4, 4);
    },
  ),
  defineBuiltin(MATH_SPECS.lookAt!,
    (ctx) => {
      ctx.tick();
      const args = evalAll(ctx);
      const eye = vecComps(args[0]); const center = vecComps(args[1]); const up = vecComps(args[2]);
      if (!eye || eye.length !== 3 || !center || center.length !== 3 || !up || up.length !== 3) { ctx.error('ML-LANG-BUILTIN-ARG', 'lookAt(eye:vec3, center:vec3, up:vec3)'); return ctx.freeze([]); }
      return makeMat(core.lookAt(eye, center, up), 4, 4, elemOfArgs(args));
    },
  ),
];

// ─────────────────────────────────────────── scalar math (with GLSL componentwise-vec promotion) ───────────────────────────────────────────

/** The pure scalar math for the numeric builtins, over already-coerced numbers. Domain-guarded funcs
 *  (sqrt/asin/acos/log/log2/inverseSqrt) return raw NaN out-of-domain — the scalar-position fail-loud
 *  diagnostic is layered on in the invoke; the vec-componentwise path keeps the NaN component. */
const SCALAR: Readonly<Record<string, (xs: readonly number[]) => number>> = {
  min: (xs) => core.min(xs[0]!, xs[1]!),
  max: (xs) => core.max(xs[0]!, xs[1]!),
  abs: (xs) => core.abs(xs[0]!),
  sign: (xs) => core.sign(xs[0]!),
  floor: (xs) => core.floor(xs[0]!),
  ceil: (xs) => core.ceil(xs[0]!),
  round: (xs) => core.round(xs[0]!),
  clamp: (xs) => core.clamp(xs[0]!, xs[1]!, xs[2]!),
  sqrt: (xs) => core.sqrt(xs[0]!),
  pow: (xs) => core.pow(xs[0]!, xs[1]!),
  mod: (xs) => core.mod(xs[0]!, xs[1]!),
  sin: (xs) => core.sin(xs[0]!),
  cos: (xs) => core.cos(xs[0]!),
  tan: (xs) => core.tan(xs[0]!),
  sinh: (xs) => core.sinh(xs[0]!),
  cosh: (xs) => core.cosh(xs[0]!),
  tanh: (xs) => core.tanh(xs[0]!),
  asin: (xs) => core.asin(xs[0]!),
  acos: (xs) => core.acos(xs[0]!),
  atan: (xs) => core.atan(xs[0]!),
  atan2: (xs) => core.atan2(xs[0]!, xs[1]!),
  asinh: (xs) => core.asinh(xs[0]!),
  // acosh (domain x>=1) and atanh (domain |x|<1) return raw NaN out-of-domain — NO fail-loud guard (unlike
  // asin/acos/log below). This is deliberate: these lower to the native shader asinh/acosh/atanh, which also
  // yield NaN out-of-domain, so returning NaN here (not a loud diagnostic + 0) is what keeps the interpreter
  // oracle identical to the GPU result. asinh has no restricted domain at all.
  acosh: (xs) => core.acosh(xs[0]!),
  atanh: (xs) => core.atanh(xs[0]!),
  exp: (xs) => core.exp(xs[0]!),
  exp2: (xs) => core.exp2(xs[0]!),
  log: (xs) => core.log(xs[0]!),
  log2: (xs) => core.log2(xs[0]!),
  inverseSqrt: (xs) => core.inverseSqrt(xs[0]!),
  degrees: (xs) => core.degrees(xs[0]!),
  radians: (xs) => core.radians(xs[0]!),
  trunc: (xs) => core.trunc(xs[0]!),
  fract: (xs) => core.fract(xs[0]!),
  step: (xs) => core.step(xs[0]!, xs[1]!),
  mix: (xs) => core.mix(xs[0]!, xs[1]!, xs[2]!),
  smoothstep: (xs) => core.smoothstep(xs[0]!, xs[1]!, xs[2]!),
  countOneBits: (xs) => core.countOneBits(xs[0]!),
  reverseBits: (xs) => core.reverseBits(xs[0]!),
};

const SCALAR_NAMES = Object.keys(SCALAR);

function makeScalarBuiltin(spec: BuiltinSpec, head: string): DefinedBuiltin {
  const arity = spec.arity[0];
  const fn = SCALAR[head]!;
  return defineBuiltin(spec,
    (ctx) => {
      ctx.tick();
      const args = evalAll(ctx);
      const bad = (): unknown => { ctx.error('ML-LANG-BUILTIN-ARG', `${head}(number, …) — non-numeric argument`); return ctx.freeze([]); };
      const relevant = args.slice(0, arity);
      // Componentwise vec application (GLSL semantics): if any arity-relevant arg is a vecN, map the scalar
      // op over its components, broadcasting a plain scalar to every component. All vec args in that prefix
      // must share ONE width (a mismatch is fail-loud — the interpreter oracle never truncates). A NaN
      // component (out-of-domain, non-numeric broadcast) is KEPT — native shaders never abort the vector.
      const vecArgs = relevant.map(vecComps).filter((c): c is number[] => c !== null);
      if (vecArgs.length > 0) {
        const width = vecArgs[0]!.length;
        if (vecArgs.some((c) => c.length !== width)) { ctx.error('ML-LANG-BUILTIN-ARG', `${head}(vec…) — vector arguments must be the same width`); return ctx.freeze([]); }
        const scalarAt = (arg: unknown, i: number): number => { const c = vecComps(arg); return c ? (c[i] ?? NaN) : (num(arg) ?? NaN); };
        const out: number[] = [];
        for (let i = 0; i < width; i++) {
          const xs: number[] = [];
          for (let k = 0; k < arity; k++) xs.push(scalarAt(relevant[k], i));
          out.push(fn(xs));
        }
        return makeVec(out, elemOfArgs(relevant));
      }
      // Scalar path: coerce each needed arg (null → fail loud), intercept the domain-restricted funcs for
      // their specific diagnostics, then compute.
      const xs: number[] = [];
      for (let k = 0; k < arity; k++) { const v = num(args[k]); if (v === null) return bad(); xs.push(v); }
      if (head === 'sqrt' && xs[0]! < 0) { ctx.error('ML-LANG-BUILTIN-ARG', `sqrt(x) — x must be >= 0`); return ctx.freeze([]); }
      if (head === 'log' && xs[0]! <= 0) { ctx.error('ML-LANG-BUILTIN-ARG', `log(x) — x must be > 0`); return ctx.freeze([]); }
      if ((head === 'asin' || head === 'acos') && (xs[0]! < -1 || xs[0]! > 1)) { ctx.error('ML-LANG-BUILTIN-ARG', `${head}(x) — x must be in [-1, 1]`); return ctx.freeze([]); }
      if ((head === 'log2' || head === 'inverseSqrt') && xs[0]! <= 0) { ctx.error('ML-LANG-BUILTIN-ARG', `${head}(x) — x must be > 0`); return ctx.freeze([]); }
      return fn(xs);
    },
  );
}

// ─────────────────────────────────────────── the defined builtins ───────────────────────────────────────────

/** Every numeric builtin this module dispatches, each declared via `defineBuiltin` (spec + invoke together).
 *  Projected two ways in `index.ts`: `toBuiltinModule` → the runtime `MATH_BUILTINS`; `builtinSpecMap` →
 *  `mathProfile.builtins`. Order preserved for stable enumeration. */
export const MATH_DEFS: DefinedBuiltin[] = [
  makeBufferBuiltin(MATH_SPECS.f32!, 'f32'), makeBufferBuiltin(MATH_SPECS.f64!, 'f64'), makeBufferBuiltin(MATH_SPECS.i32!, 'i32'), makeBufferBuiltin(MATH_SPECS.u32!, 'u32'),
  makeVecBuiltin(MATH_SPECS.vec2!, 'vec2', 2), makeVecBuiltin(MATH_SPECS.vec3!, 'vec3', 3), makeVecBuiltin(MATH_SPECS.vec4!, 'vec4', 4),
  makeMatBuiltin(MATH_SPECS.mat2!, 'mat2', 2, 2, true), makeMatBuiltin(MATH_SPECS.mat3!, 'mat3', 3, 3, true), makeMatBuiltin(MATH_SPECS.mat4!, 'mat4', 4, 4, true),
  // Non-square matCxR: name matCxR → C columns × R rows, so [rows, cols] = [R, C].
  makeMatBuiltin(MATH_SPECS.mat2x3!, 'mat2x3', 3, 2, false), makeMatBuiltin(MATH_SPECS.mat2x4!, 'mat2x4', 4, 2, false), makeMatBuiltin(MATH_SPECS.mat3x2!, 'mat3x2', 2, 3, false),
  makeMatBuiltin(MATH_SPECS.mat3x4!, 'mat3x4', 4, 3, false), makeMatBuiltin(MATH_SPECS.mat4x2!, 'mat4x2', 2, 4, false), makeMatBuiltin(MATH_SPECS.mat4x3!, 'mat4x3', 3, 4, false),
  transposeBuiltin, determinantBuiltin, inverseBuiltin, matrixCompMultBuiltin, qmatBuiltin,
  ...geometricBuiltins,
  ...transformBuiltins,
  ...SCALAR_NAMES.map((head) => makeScalarBuiltin(MATH_SPECS[head]!, head)),
];
