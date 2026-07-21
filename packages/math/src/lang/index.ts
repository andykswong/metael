// @metael/math/lang — the boxed, protocol-wrapping numeric layer over the @metael/math core. It supplies
// the vec/mat/buffer INSTANCES (the custom-value builders + descriptors), the builtins CATALOG + static
// classifier, and the numeric builtin MODULE a consumer injects at evaluateProgram. Imports only
// @metael/lang (the descriptor protocol + registry seam) and @metael/math (the arithmetic core).
export { makeVec, makeMat, identityMat, vecStoreOf } from './descriptors.ts';
export { makeTypedArray, BUFFER_KINDS, TYPED_ARRAY_DESCRIPTORS } from './buffers.ts';
export type { BufferKind } from './buffers.ts';
export { BUILTINS, isBuiltin } from './registry-data.ts';
export type { BuiltinSpec, BuiltinProfile, Portability } from './registry-data.ts';
export { classifyProfile } from './classify.ts';
export type { ProfileResult } from './classify.ts';
export { MATH_BUILTINS } from './builtins.ts';
