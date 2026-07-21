// Collection builtins — pure, intrinsic free functions that RETURN NEW frozen collections and never
// mutate an input. Each ticks the budget per call + per element so a large collection fails closed with
// ML-LANG-BUDGET. A callback may be an arrow OR a user `function`. A wrong-shape argument is a fail-loud
// ML-LANG-BUILTIN-ARG (never a throw) plus a safe empty result.
import type { Builtin, BuiltinSpec } from '@metael/lang';
import { truthy, looseEquals } from '@metael/lang';
import { asFn, badArg, toArray, toNum } from './helpers.ts';
import { defaultCompare, stableSort } from './sort.ts';

const spec = (name: string, takesClosure: boolean, arity: readonly [number, number]): BuiltinSpec =>
  ({ name, profile: 'host', portability: 'cpu-only', takesClosure, arity });

const mapBuiltin: Builtin = {
  spec: spec('map', true, [2, 2]),
  invoke: (ctx) => {
    ctx.tick();
    const xs = toArray(ctx, ctx.evalArg(0)); const fn = asFn(ctx, ctx.evalArg(1));
    if (!xs || !fn) return badArg(ctx, `map(array, fn) — bad arguments`);
    const out: unknown[] = []; xs.forEach((x, i) => { ctx.tick(); out.push(fn(x, i)); }); return ctx.freeze(out);
  },
};

const filterBuiltin: Builtin = {
  spec: spec('filter', true, [2, 2]),
  invoke: (ctx) => {
    ctx.tick();
    const xs = toArray(ctx, ctx.evalArg(0)); const fn = asFn(ctx, ctx.evalArg(1));
    if (!xs || !fn) return badArg(ctx, `filter(array, fn) — bad arguments`);
    const out: unknown[] = []; xs.forEach((x, i) => { ctx.tick(); if (truthy(fn(x, i))) out.push(x); }); return ctx.freeze(out);
  },
};

const reduceBuiltin: Builtin = {
  spec: spec('reduce', true, [3, 3]),
  invoke: (ctx) => {
    ctx.tick();
    const xs = toArray(ctx, ctx.evalArg(0)); const fn = asFn(ctx, ctx.evalArg(1)); const init = ctx.evalArg(2);
    if (!xs || !fn) return badArg(ctx, `reduce(array, fn, init) — bad arguments`);
    let acc = init; xs.forEach((x, i) => { ctx.tick(); acc = fn(acc, x, i); }); return acc;   // acc may be a scalar; a collection acc is already frozen by its own eval
  },
};

const someBuiltin: Builtin = {
  spec: spec('some', true, [2, 2]),
  invoke: (ctx) => {
    ctx.tick();
    const xs = toArray(ctx, ctx.evalArg(0)); const fn = asFn(ctx, ctx.evalArg(1));
    if (!xs || !fn) return badArg(ctx, `some(array, fn) — bad arguments`);
    for (let i = 0; i < xs.length; i++) { ctx.tick(); if (truthy(fn(xs[i], i))) return true; }
    return false;
  },
};

const everyBuiltin: Builtin = {
  spec: spec('every', true, [2, 2]),
  invoke: (ctx) => {
    ctx.tick();
    const xs = toArray(ctx, ctx.evalArg(0)); const fn = asFn(ctx, ctx.evalArg(1));
    if (!xs || !fn) return badArg(ctx, `every(array, fn) — bad arguments`);
    for (let i = 0; i < xs.length; i++) { ctx.tick(); if (!truthy(fn(xs[i], i))) return false; }
    return true;
  },
};

const findBuiltin: Builtin = {
  spec: spec('find', true, [2, 2]),
  invoke: (ctx) => {
    ctx.tick();
    const xs = toArray(ctx, ctx.evalArg(0)); const fn = asFn(ctx, ctx.evalArg(1));
    if (!xs || !fn) return badArg(ctx, `find(array, fn) — bad arguments`);
    for (let i = 0; i < xs.length; i++) { ctx.tick(); if (truthy(fn(xs[i], i))) return xs[i] ?? null; }
    return null;
  },
};

const findIndexBuiltin: Builtin = {
  spec: spec('findIndex', true, [2, 2]),
  invoke: (ctx) => {
    ctx.tick();
    const xs = toArray(ctx, ctx.evalArg(0)); const fn = asFn(ctx, ctx.evalArg(1));
    if (!xs || !fn) return badArg(ctx, `findIndex(array, fn) — bad arguments`);
    for (let i = 0; i < xs.length; i++) { ctx.tick(); if (truthy(fn(xs[i], i))) return i; }
    return -1;
  },
};

const includesBuiltin: Builtin = {
  spec: spec('includes', false, [2, 2]),
  invoke: (ctx) => {
    ctx.tick();
    const xs = toArray(ctx, ctx.evalArg(0)); const v = ctx.evalArg(1);
    if (!xs) return badArg(ctx, `includes(array, value) — bad argument`);
    for (const x of xs) { ctx.tick(); if (looseEquals(x, v)) return true; }
    return false;
  },
};

const sortBuiltin: Builtin = {
  spec: spec('sort', true, [1, 2]),
  invoke: (ctx) => {
    ctx.tick();
    const xs = toArray(ctx, ctx.evalArg(0));
    if (!xs) return badArg(ctx, `sort(array, comparator?) — bad argument`);
    const cmpArg = ctx.argCount() > 1 ? ctx.evalArg(1) : undefined;
    // Tick PER COMPARISON on the default path too — an O(n log n) sort of a large array must be
    // budget-charged or it bypasses the step + time guards (tick() is the only deadline check point).
    if (cmpArg === undefined) return ctx.freeze(stableSort(xs, (x, y) => { ctx.tick(); return defaultCompare(x, y); }));
    const cmpFn = asFn(ctx, cmpArg);
    if (!cmpFn) return badArg(ctx, `sort(array, comparator) — comparator is not callable`);
    let flagged = false;
    const cmp = (x: unknown, y: unknown): number => {
      ctx.tick();
      const res = cmpFn(x, y);
      if (typeof res !== 'number' || Number.isNaN(res)) {
        if (!flagged) { flagged = true; ctx.error('ML-LANG-BUILTIN-ARG', 'sort comparator must return a number'); }
        return 0;   // keep relative order on a bad return (stable)
      }
      return res;
    };
    return ctx.freeze(stableSort(xs, cmp));
  },
};

const sliceBuiltin: Builtin = {
  spec: spec('slice', false, [2, 3]),
  invoke: (ctx) => {
    ctx.tick();
    const src = ctx.evalArg(0);
    const startArg = ctx.argCount() > 1 ? ctx.evalArg(1) : undefined;
    const endArg = ctx.argCount() > 2 ? ctx.evalArg(2) : undefined;
    // A string arg returns a substring with the SAME start/end clamp semantics as the array case.
    const isString = typeof src === 'string';
    const xs = isString ? null : toArray(ctx, src);
    if (!isString && !xs) return badArg(ctx, `slice(array, start, end?) — bad argument`);
    const len = isString ? (src as string).length : xs!.length;
    const norm = (v: unknown, dflt: number): number => {
      if (v === undefined) return dflt;
      const n = Math.trunc(toNum(v));
      if (Number.isNaN(n)) return dflt;
      return n < 0 ? Math.max(len + n, 0) : Math.min(n, len);
    };
    const start = norm(startArg, 0);
    const end = norm(endArg, len);
    if (isString) {
      let out = '';
      for (let i = start; i < end; i++) { ctx.tick(); out += (src as string)[i]; }
      return out;
    }
    const out: unknown[] = [];
    for (let i = start; i < end; i++) { ctx.tick(); out.push(xs![i]); }
    return ctx.freeze(out);
  },
};

const reverseBuiltin: Builtin = {
  spec: spec('reverse', false, [1, 1]),
  invoke: (ctx) => {
    ctx.tick();
    const xs = toArray(ctx, ctx.evalArg(0));
    if (!xs) return badArg(ctx, `reverse(array) — bad argument`);
    const out: unknown[] = [];
    for (let i = xs.length - 1; i >= 0; i--) { ctx.tick(); out.push(xs[i]); }
    return ctx.freeze(out);
  },
};

/** The collection builtin module: `map`, `filter`, `reduce`, `some`, `every`, `find`, `findIndex`,
 *  `includes`, `sort`, `slice`, and `reverse`. Each is a pure, intrinsic-free function that returns a
 *  NEW frozen collection (or a scalar) and never mutates its input, budget-charging per call and per
 *  element so a large collection fails closed with `ML-LANG-BUDGET`. Inject via a run's builtin modules
 *  to give a program array-processing vocabulary. */
export const COLLECTION_BUILTINS: readonly Builtin[] = [
  mapBuiltin, filterBuiltin, reduceBuiltin, someBuiltin, everyBuiltin, findBuiltin, findIndexBuiltin,
  includesBuiltin, sortBuiltin, sliceBuiltin, reverseBuiltin,
];
