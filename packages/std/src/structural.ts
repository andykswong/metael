// Structural builtins — object⇄array decomposition/reconstruction plus own-property presence. Each
// returns a NEW frozen collection (or a boolean) and never mutates. A non-object argument is a fail-loud
// ML-LANG-BUILTIN-ARG. Object construction (`object`) and presence (`has`) are FORBIDDEN_KEYS-guarded, so
// a prototype-polluting key can never enter or be observed on a DSL object.
import type { Builtin, BuiltinSpec } from '@metael/lang';
import { FORBIDDEN_KEYS } from '@metael/lang';
import { badArg } from './helpers.ts';

const spec = (name: string, arity: readonly [number, number]): BuiltinSpec =>
  ({ name, profile: 'host', portability: 'cpu-only', takesClosure: false, arity });

/** A plain (non-array, non-null) object — the shape keys/values/entries/has accept. */
const isPlainObject = (o: unknown): o is Record<string, unknown> =>
  o !== null && typeof o === 'object' && !Array.isArray(o);

const keysBuiltin: Builtin = {
  spec: spec('keys', [1, 1]),
  invoke: (ctx) => {
    ctx.tick();
    const o = ctx.evalArg(0);
    if (!isPlainObject(o)) return badArg(ctx, `keys(object) — bad argument`);
    return ctx.freeze(Object.keys(o));
  },
};

const valuesBuiltin: Builtin = {
  spec: spec('values', [1, 1]),
  invoke: (ctx) => {
    ctx.tick();
    const o = ctx.evalArg(0);
    if (!isPlainObject(o)) return badArg(ctx, `values(object) — bad argument`);
    return ctx.freeze(Object.values(o));
  },
};

const entriesBuiltin: Builtin = {
  spec: spec('entries', [1, 1]),
  invoke: (ctx) => {
    ctx.tick();
    const o = ctx.evalArg(0);
    if (!isPlainObject(o)) return badArg(ctx, `entries(object) — bad argument`);
    return ctx.freeze(Object.entries(o).map(([k, v]) => [k, v]));
  },
};

const objectBuiltin: Builtin = {
  spec: spec('object', [1, 1]),
  invoke: (ctx) => {
    ctx.tick();
    const pairs = ctx.evalArg(0);
    if (!Array.isArray(pairs)) return badArg(ctx, `object(array of [key, value]) — bad argument`);
    // Null-prototype record: no inherited proto/constructor/toString (matches the object-literal builder).
    const out: Record<string, unknown> = Object.create(null);
    for (const p of pairs) { ctx.tick(); if (Array.isArray(p) && typeof p[0] === 'string' && !FORBIDDEN_KEYS.has(p[0])) out[p[0]] = p[1]; }
    return ctx.freeze(out);
  },
};

const hasBuiltin: Builtin = {
  spec: spec('has', [2, 2]),
  invoke: (ctx) => {
    ctx.tick();
    const o = ctx.evalArg(0); const key = ctx.evalArg(1);
    if (!isPlainObject(o)) return badArg(ctx, `has(object, key) — bad argument`);
    // A forbidden key is never observable: report absent without touching the prototype chain (so
    // `has(o, "__proto__")` is false, matching the object-literal / spread FORBIDDEN_KEYS guard).
    if (typeof key !== 'string' || FORBIDDEN_KEYS.has(key)) return false;
    return Object.prototype.hasOwnProperty.call(o, key);
  },
};

/** The structural builtin module: `keys`, `values`, `entries`, `object`, and `has` — object⇄array
 *  decomposition/reconstruction plus own-property presence. Each returns a NEW frozen collection (or a
 *  boolean) and never mutates. `object` and `has` are `FORBIDDEN_KEYS`-guarded, so a prototype-polluting
 *  key can never enter or be observed on a DSL object. Inject via a run's builtin modules to give a
 *  program object-reshaping vocabulary. */
export const STRUCTURAL_BUILTINS: readonly Builtin[] = [
  keysBuiltin, valuesBuiltin, entriesBuiltin, objectBuiltin, hasBuiltin,
];
