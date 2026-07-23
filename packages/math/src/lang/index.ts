// @metael/math/lang — the boxed, protocol-wrapping numeric layer over the @metael/math core. It supplies
// the vec/mat/buffer INSTANCES (the custom-value builders + descriptors), the numeric builtin MODULE a
// consumer injects at evaluateProgram (MATH_BUILTINS), and the tooling PROFILE (mathProfile) a classifier /
// language service reads. Imports only @metael/lang (the descriptor protocol + profile seam) and
// @metael/math (the arithmetic core).
import { toBuiltinModule, builtinSpecMap, swizzleMembers } from '@metael/lang/profile';
import type { BuiltinModule } from '@metael/lang';
import type { Profile, TypeDescriptorMeta } from '@metael/lang/profile';
import { MATH_DEFS } from './builtins.ts';

export { makeVec, makeMat, identityMat, vecStoreOf } from './descriptors.ts';
export { makeTypedArray, BUFFER_KINDS, TYPED_ARRAY_DESCRIPTORS } from './buffers.ts';
export type { BufferKind } from './buffers.ts';

/** The numeric standard-library module: every numeric builtin (constructors, vec/mat/quat ops, transforms,
 *  scalar math, bit ops). A consumer injects it via `evaluateProgram(src, { …, builtins: [MATH_BUILTINS] })`. */
export const MATH_BUILTINS: BuiltinModule = toBuiltinModule(MATH_DEFS);

/** A static custom-type projection for an `n`-component vector: its swizzle members + its constructor name. */
const vecType = (n: number): TypeDescriptorMeta =>
  ({ name: `vec${n}`, members: swizzleMembers(n), constructors: [`vec${n}`], doc: `${n}-component vector` });

/** The math library's tooling profile — the static specs of its dispatched builtins (for a classifier / a
 *  language-service consumer) plus its vec custom-type projections. Composed with a host's own profile (and
 *  `coreIntrinsicsProfile`) by a consumer; math itself contributes no head vocabulary. */
export const mathProfile: Profile = {
  id: 'math',
  builtins: builtinSpecMap(MATH_DEFS),
  heads: new Map(),
  types: new Map([['vec2', vecType(2)], ['vec3', vecType(3)], ['vec4', vecType(4)]]),
};

/** True iff `name` is one of the numeric builtins this library dispatches. */
export const isMathBuiltin = (name: string): boolean => mathProfile.builtins.has(name);
