import type { Profile } from './profile.ts';
import type { BuiltinSpec } from './types.ts';

/** The metadata for intrinsics the language kernel dispatches itself (not via an injected module).
 *  Today that is `range` (the bounded-loop primitive). A host composes this in so `range` is
 *  classifiable + completable even though no standard-library module contributes it. */
export const coreIntrinsicsProfile: Profile = {
  id: 'core-intrinsics',
  builtins: new Map<string, BuiltinSpec>([
    ['range', {
      name: 'range',
      profile: 'core',
      portability: 'exact',
      takesClosure: false,
      arity: [1, 1],
      doc: 'The integer sequence 0, 1, … n-1 — the bounded-loop primitive.',
      params: [{ name: 'n', doc: 'the exclusive upper bound (produces 0 … n-1)' }],
      returnDoc: 'an array of the integers from 0 to n-1',
    }],
  ]),
  heads: new Map(),
  types: new Map(),
};
