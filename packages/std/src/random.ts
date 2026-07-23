// Random builtin — `rand()` returns the next draw from the run's SEEDED pseudo-random stream. The language
// kernel owns the seed + the PRNG (threaded onto the Runner from EvalOptions.seed); this builtin only READS
// that stream via `ctx.rng()`, so `result = f(source, data, seed, …)` stays a language determinism guarantee
// rather than a per-domain re-implementation. `rand` is not shader-lowerable (it can't match the interpreter
// oracle deterministically), so a GPU kernel gate rejects it — hence it lives here, not as a numeric builtin.
import { defineBuiltin } from '@metael/lang/profile';
import type { DefinedBuiltin } from '@metael/lang/profile';

const randBuiltin: DefinedBuiltin = defineBuiltin(
  { name: 'rand', profile: 'core', portability: 'cpu-only', takesClosure: false, arity: [0, 0], doc: 'Draws the next value from the run’s seeded pseudo-random stream.', params: [], returnDoc: 'a pseudo-random number in [0, 1)' },
  // Tick the budget per call (so an unbounded loop of rand() fails closed with ML-LANG-BUDGET, never hangs),
  // then draw the next value from the seeded stream. NO argument evaluation — rand is nullary.
  (ctx) => { ctx.tick(); return ctx.rng(); },
);

/** The random builtin module: `rand()`, which returns the next draw from the run's SEEDED pseudo-random
 *  stream (owned by the language kernel via the run's seed). Reading a seeded stream keeps a run
 *  reproducible — the same source, data, and seed yield the same `rand()` sequence. Not shader-lowerable
 *  (it can't match the interpreter oracle deterministically on a GPU), so it lives here rather than as a
 *  numeric builtin. Inject via a run's builtin modules to expose `rand()`. */
export const RANDOM_BUILTINS: readonly DefinedBuiltin[] = [randBuiltin];
