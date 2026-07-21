// String builtins — a bridge between strings and arrays plus in-place string transforms. Each returns a
// NEW string (or a frozen array). A wrong-shape argument is a fail-loud ML-LANG-BUILTIN-ARG. `join` fails
// closed with ML-LANG-BUDGET before the result would cross the run's string-growth cap (mirroring the `+`
// operator), so a large collection of large strings can never build a string past the engine limit.
import type { Builtin, BuiltinSpec } from '@metael/lang';
import { strOf } from '@metael/lang';
import { badArg, toArray } from './helpers.ts';

const spec = (name: string, arity: readonly [number, number]): BuiltinSpec =>
  ({ name, profile: 'host', portability: 'cpu-only', takesClosure: false, arity });

const splitBuiltin: Builtin = {
  spec: spec('split', [2, 2]),
  invoke: (ctx) => {
    ctx.tick();
    const s = ctx.evalArg(0); const sep = ctx.evalArg(1);
    if (typeof s !== 'string' || typeof sep !== 'string') return badArg(ctx, `split(string, separator) — bad arguments`);
    // Charge proportional to the input BEFORE splitting: the output is at most one part per input
    // character, so this fails closed on a pathological input before the native split allocates.
    ctx.tick(s.length);
    const parts = sep === '' ? Array.from(s) : s.split(sep);
    return ctx.freeze(parts);
  },
};

const joinBuiltin: Builtin = {
  spec: spec('join', [2, 2]),
  invoke: (ctx) => {
    ctx.tick();
    const xs = toArray(ctx, ctx.evalArg(0)); const sep = ctx.evalArg(1);
    if (!xs || typeof sep !== 'string') return badArg(ctx, `join(array, separator) — bad arguments`);
    const parts: string[] = [];
    // Fail CLOSED before the result crosses the string cap (mirrors the `+` operator): a large
    // collection of large strings could otherwise build a string past the engine limit and throw
    // a raw RangeError. Treat the cap as a budget trip, before allocating the joined string.
    let total = 0;
    for (let i = 0; i < xs.length; i++) {
      ctx.tick();
      const s = strOf(xs[i]);
      total += s.length + (i > 0 ? sep.length : 0);
      if (total > ctx.maxStringLength) {
        ctx.error('ML-LANG-BUDGET', `string result would exceed the ${ctx.maxStringLength}-character limit`);
        return null;
      }
      parts.push(s);
    }
    return parts.join(sep);
  },
};

const charsBuiltin: Builtin = {
  spec: spec('chars', [1, 1]),
  invoke: (ctx) => {
    ctx.tick();
    const s = ctx.evalArg(0);
    if (typeof s !== 'string') return badArg(ctx, `chars(string) — bad argument`);
    // Charge proportional to the input BEFORE building the array (one entry per code point ≤ s.length).
    ctx.tick(s.length);
    const cs = Array.from(s);
    return ctx.freeze(cs);
  },
};

const toUpperCaseBuiltin: Builtin = {
  spec: spec('toUpperCase', [1, 1]),
  invoke: (ctx) => {
    ctx.tick();
    const s = ctx.evalArg(0);
    if (typeof s !== 'string') return badArg(ctx, `toUpperCase(string) — bad argument`);
    return s.toUpperCase();
  },
};

const toLowerCaseBuiltin: Builtin = {
  spec: spec('toLowerCase', [1, 1]),
  invoke: (ctx) => {
    ctx.tick();
    const s = ctx.evalArg(0);
    if (typeof s !== 'string') return badArg(ctx, `toLowerCase(string) — bad argument`);
    return s.toLowerCase();
  },
};

const trimBuiltin: Builtin = {
  spec: spec('trim', [1, 1]),
  invoke: (ctx) => {
    ctx.tick();
    const s = ctx.evalArg(0);
    if (typeof s !== 'string') return badArg(ctx, `trim(string) — bad argument`);
    return s.trim();
  },
};

const formatBuiltin: Builtin = {
  spec: spec('format', [2, 2]),
  invoke: (ctx) => {
    ctx.tick();
    const x = ctx.evalArg(0); const digits = ctx.evalArg(1);
    if (typeof x !== 'number' || Number.isNaN(x)) return badArg(ctx, `format(number, digits) — first argument must be a number`);
    if (typeof digits !== 'number' || !Number.isInteger(digits) || digits < 0 || digits > 100) return badArg(ctx, `format(number, digits) — digits must be an integer in [0, 100]`);
    return x.toFixed(digits);
  },
};

const codePointAtBuiltin: Builtin = {
  spec: spec('codePointAt', [2, 2]),
  invoke: (ctx) => {
    ctx.tick();
    const s = ctx.evalArg(0); const index = ctx.evalArg(1);
    if (typeof s !== 'string') return badArg(ctx, `codePointAt(string, index) — first argument must be a string`);
    if (typeof index !== 'number' || !Number.isInteger(index) || index < 0) return badArg(ctx, `codePointAt(string, index) — index must be a non-negative integer`);
    // String.prototype.codePointAt returns undefined for an out-of-range index — map that to the
    // language's null (never an undefined leaks into a DSL value).
    return s.codePointAt(index) ?? null;
  },
};

/** The string builtin module: `split`, `join`, `chars`, `toUpperCase`, `toLowerCase`, `trim`, `format`,
 *  and `codePointAt`. Bridges strings and arrays plus in-place string transforms; each returns a NEW
 *  string (or a frozen array). `join` fails closed with `ML-LANG-BUDGET` before its result would cross the
 *  run's string-growth cap (mirroring the `+` operator). Inject via a run's builtin modules to give a
 *  program string-processing vocabulary. */
export const STRING_BUILTINS: readonly Builtin[] = [
  splitBuiltin, joinBuiltin, charsBuiltin, toUpperCaseBuiltin, toLowerCaseBuiltin, trimBuiltin,
  formatBuiltin, codePointAtBuiltin,
];
