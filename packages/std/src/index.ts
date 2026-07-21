// @metael/std — the general-purpose standard library for the metael language kernel. It supplies the
// collection, string, and structural builtins as one injectable module (STD_BUILTINS): a consumer wires
// it in via evaluateProgram(src, { …, builtins: [STD_BUILTINS] }). Imports ONLY @metael/lang (the registry
// seam + the language's own truthiness/equality/stringify/forbidden-key primitives), so the standard
// library stays in lockstep with the language's semantics rather than re-deriving them.
import type { BuiltinModule } from '@metael/lang';
import { COLLECTION_BUILTINS } from './collections.ts';
import { STRING_BUILTINS } from './string.ts';
import { STRUCTURAL_BUILTINS } from './structural.ts';
import { RANDOM_BUILTINS } from './random.ts';
import { DATETIME_BUILTINS } from './datetime.ts';

export { COLLECTION_BUILTINS } from './collections.ts';
export { STRING_BUILTINS } from './string.ts';
export { STRUCTURAL_BUILTINS } from './structural.ts';
export { RANDOM_BUILTINS } from './random.ts';
export { DATETIME_BUILTINS } from './datetime.ts';
export { defaultCompare, stableSort } from './sort.ts';

/** The standard-library module a consumer injects at evaluateProgram: collection (map/filter/reduce/…),
 *  string (split/join/…/codePointAt), structural (keys/values/entries/object/has), random (rand — reads the
 *  seeded RNG the language owns), and datetime (now/monotonic — read the host's injected clock capability)
 *  builtins. NOTE: `range` is NOT here — it stays a language-kernel intrinsic (a bounded-loop primitive the
 *  compute-lowering gate + interpreter oracle depend on), dispatched by lang, not the standard library. */
export const STD_BUILTINS: BuiltinModule = {
  builtins: [...COLLECTION_BUILTINS, ...STRING_BUILTINS, ...STRUCTURAL_BUILTINS, ...RANDOM_BUILTINS, ...DATETIME_BUILTINS],
};
