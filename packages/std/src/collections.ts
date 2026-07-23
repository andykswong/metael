// Collection builtins — pure, intrinsic free functions that RETURN NEW frozen collections and never
// mutate an input. Each ticks the budget per call + per element so a large collection fails closed with
// ML-LANG-BUDGET. A callback may be an arrow OR a user `function`. A wrong-shape argument is a fail-loud
// ML-LANG-BUILTIN-ARG (never a throw) plus a safe empty result.
import { defineBuiltin } from '@metael/lang/profile';
import type { BuiltinSpec, DefinedBuiltin, HeadParam } from '@metael/lang/profile';
import { truthy, looseEquals } from '@metael/lang';
import { asFn, badArg, toArray, toNum } from './helpers.ts';
import { defaultCompare, stableSort } from './sort.ts';

const spec = (name: string, takesClosure: boolean, arity: readonly [number, number], doc: string, params: readonly HeadParam[], returnDoc: string): BuiltinSpec =>
  ({ name, profile: 'host', portability: 'cpu-only', takesClosure, arity, doc, params, returnDoc });

const mapBuiltin: DefinedBuiltin = defineBuiltin(spec('map', true, [2, 2], 'Builds a new array by calling `fn` on each element of an array.', [{ name: 'items', doc: 'the array to map over' }, { name: 'fn', doc: 'a function (item, i) => value called on each element' }], 'a new array of the mapped values'), (ctx) => {
  ctx.tick();
  const xs = toArray(ctx, ctx.evalArg(0)); const fn = asFn(ctx, ctx.evalArg(1));
  if (!xs || !fn) return badArg(ctx, `map(array, fn) — bad arguments`);
  const out: unknown[] = []; xs.forEach((x, i) => { ctx.tick(); out.push(fn(x, i)); }); return ctx.freeze(out);
});

const filterBuiltin: DefinedBuiltin = defineBuiltin(spec('filter', true, [2, 2], 'Builds a new array of the elements of an array for which `predicate` returns a truthy value.', [{ name: 'items', doc: 'the array to filter' }, { name: 'predicate', doc: 'a function (item, i) => boolean; kept when truthy' }], 'a new filtered array'), (ctx) => {
  ctx.tick();
  const xs = toArray(ctx, ctx.evalArg(0)); const fn = asFn(ctx, ctx.evalArg(1));
  if (!xs || !fn) return badArg(ctx, `filter(array, fn) — bad arguments`);
  const out: unknown[] = []; xs.forEach((x, i) => { ctx.tick(); if (truthy(fn(x, i))) out.push(x); }); return ctx.freeze(out);
});

const reduceBuiltin: DefinedBuiltin = defineBuiltin(spec('reduce', true, [3, 3], 'Folds an array into a single accumulated value, starting from `initial`.', [{ name: 'items', doc: 'the array to fold' }, { name: 'fn', doc: 'a function (acc, item, i) => acc combining the accumulator with each element' }, { name: 'initial', doc: 'the starting accumulator value' }], 'the final accumulated value'), (ctx) => {
  ctx.tick();
  const xs = toArray(ctx, ctx.evalArg(0)); const fn = asFn(ctx, ctx.evalArg(1)); const init = ctx.evalArg(2);
  if (!xs || !fn) return badArg(ctx, `reduce(array, fn, init) — bad arguments`);
  let acc = init; xs.forEach((x, i) => { ctx.tick(); acc = fn(acc, x, i); }); return acc;   // acc may be a scalar; a collection acc is already frozen by its own eval
});

const someBuiltin: DefinedBuiltin = defineBuiltin(spec('some', true, [2, 2], 'Tests whether `predicate` returns a truthy value for at least one element of an array.', [{ name: 'items', doc: 'the array to test' }, { name: 'predicate', doc: 'a function (item, i) => boolean' }], 'a boolean: true if any element matches'), (ctx) => {
  ctx.tick();
  const xs = toArray(ctx, ctx.evalArg(0)); const fn = asFn(ctx, ctx.evalArg(1));
  if (!xs || !fn) return badArg(ctx, `some(array, fn) — bad arguments`);
  for (let i = 0; i < xs.length; i++) { ctx.tick(); if (truthy(fn(xs[i], i))) return true; }
  return false;
});

const everyBuiltin: DefinedBuiltin = defineBuiltin(spec('every', true, [2, 2], 'Tests whether `predicate` returns a truthy value for every element of an array.', [{ name: 'items', doc: 'the array to test' }, { name: 'predicate', doc: 'a function (item, i) => boolean' }], 'a boolean: true if all elements match'), (ctx) => {
  ctx.tick();
  const xs = toArray(ctx, ctx.evalArg(0)); const fn = asFn(ctx, ctx.evalArg(1));
  if (!xs || !fn) return badArg(ctx, `every(array, fn) — bad arguments`);
  for (let i = 0; i < xs.length; i++) { ctx.tick(); if (!truthy(fn(xs[i], i))) return false; }
  return true;
});

const findBuiltin: DefinedBuiltin = defineBuiltin(spec('find', true, [2, 2], 'Finds the first element of an array for which `predicate` returns a truthy value.', [{ name: 'items', doc: 'the array to search' }, { name: 'predicate', doc: 'a function (item, i) => boolean' }], 'the first matching item, or null if none matches'), (ctx) => {
  ctx.tick();
  const xs = toArray(ctx, ctx.evalArg(0)); const fn = asFn(ctx, ctx.evalArg(1));
  if (!xs || !fn) return badArg(ctx, `find(array, fn) — bad arguments`);
  for (let i = 0; i < xs.length; i++) { ctx.tick(); if (truthy(fn(xs[i], i))) return xs[i] ?? null; }
  return null;
});

const findIndexBuiltin: DefinedBuiltin = defineBuiltin(spec('findIndex', true, [2, 2], 'Finds the index of the first element of an array for which `predicate` returns a truthy value.', [{ name: 'items', doc: 'the array to search' }, { name: 'predicate', doc: 'a function (item, i) => boolean' }], 'the index of the first matching item, or -1 if none matches'), (ctx) => {
  ctx.tick();
  const xs = toArray(ctx, ctx.evalArg(0)); const fn = asFn(ctx, ctx.evalArg(1));
  if (!xs || !fn) return badArg(ctx, `findIndex(array, fn) — bad arguments`);
  for (let i = 0; i < xs.length; i++) { ctx.tick(); if (truthy(fn(xs[i], i))) return i; }
  return -1;
});

const includesBuiltin: DefinedBuiltin = defineBuiltin(spec('includes', false, [2, 2], 'Tests whether an array contains `value` (compared with loose equality).', [{ name: 'items', doc: 'the array to search' }, { name: 'value', doc: 'the value to look for' }], 'a boolean: true if the array contains the value'), (ctx) => {
  ctx.tick();
  const xs = toArray(ctx, ctx.evalArg(0)); const v = ctx.evalArg(1);
  if (!xs) return badArg(ctx, `includes(array, value) — bad argument`);
  for (const x of xs) { ctx.tick(); if (looseEquals(x, v)) return true; }
  return false;
});

const sortBuiltin: DefinedBuiltin = defineBuiltin(spec('sort', true, [1, 2], 'Builds a new array with the elements sorted in ascending order, or by an optional `compare` function.', [{ name: 'items', doc: 'the array to sort' }, { name: 'compare', optional: true, doc: 'an optional function (a, b) => number: negative if a sorts before b' }], 'a new sorted array (a stable sort; the input is not mutated)'), (ctx) => {
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
});

const sliceBuiltin: DefinedBuiltin = defineBuiltin(spec('slice', false, [2, 3], 'Extracts a shallow copy of an array or string from `start` up to (but not including) `end`; negative indices count from the end.', [{ name: 'items', doc: 'the array or string to slice' }, { name: 'start', doc: 'the start index (inclusive); negative counts from the end' }, { name: 'end', optional: true, doc: 'the end index (exclusive), defaulting to the length; negative counts from the end' }], 'a new array (or substring, if the input was a string)'), (ctx) => {
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
});

const reverseBuiltin: DefinedBuiltin = defineBuiltin(spec('reverse', false, [1, 1], 'Builds a new array with the elements of an array in reverse order.', [{ name: 'items', doc: 'the array to reverse' }], 'a new reversed array (the input is not mutated)'), (ctx) => {
  ctx.tick();
  const xs = toArray(ctx, ctx.evalArg(0));
  if (!xs) return badArg(ctx, `reverse(array) — bad argument`);
  const out: unknown[] = [];
  for (let i = xs.length - 1; i >= 0; i--) { ctx.tick(); out.push(xs[i]); }
  return ctx.freeze(out);
});

/** The collection builtin module: `map`, `filter`, `reduce`, `some`, `every`, `find`, `findIndex`,
 *  `includes`, `sort`, `slice`, and `reverse`. Each is a pure, intrinsic-free function that returns a
 *  NEW frozen collection (or a scalar) and never mutates its input, budget-charging per call and per
 *  element so a large collection fails closed with `ML-LANG-BUDGET`. Inject via a run's builtin modules
 *  to give a program array-processing vocabulary. */
export const COLLECTION_BUILTINS: readonly DefinedBuiltin[] = [
  mapBuiltin, filterBuiltin, reduceBuiltin, someBuiltin, everyBuiltin, findBuiltin, findIndexBuiltin,
  includesBuiltin, sortBuiltin, sliceBuiltin, reverseBuiltin,
];
