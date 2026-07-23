// Structural builtins — object⇄array decomposition/reconstruction plus own-property presence. Each
// returns a NEW frozen collection (or a boolean) and never mutates. A non-object argument is a fail-loud
// ML-LANG-BUILTIN-ARG. Object construction (`object`) and presence (`has`) are FORBIDDEN_KEYS-guarded, so
// a prototype-polluting key can never enter or be observed on a DSL object.
import { defineBuiltin } from '@metael/lang/profile';
import type { BuiltinSpec, DefinedBuiltin, HeadParam } from '@metael/lang/profile';
import { FORBIDDEN_KEYS } from '@metael/lang';
import { badArg } from './helpers.ts';

const spec = (name: string, arity: readonly [number, number], doc: string, params: readonly HeadParam[], returnDoc: string): BuiltinSpec =>
  ({ name, profile: 'host', portability: 'cpu-only', takesClosure: false, arity, doc, params, returnDoc });

/** A plain (non-array, non-null) object — the shape keys/values/entries/has accept. */
const isPlainObject = (o: unknown): o is Record<string, unknown> =>
  o !== null && typeof o === 'object' && !Array.isArray(o);

const keysBuiltin: DefinedBuiltin = defineBuiltin(spec('keys', [1, 1], 'Lists an object’s own property names.', [{ name: 'obj', doc: 'the object to read keys from' }], 'an array of the object’s own property-name strings'), (ctx) => {
  ctx.tick();
  const o = ctx.evalArg(0);
  if (!isPlainObject(o)) return badArg(ctx, `keys(object) — bad argument`);
  return ctx.freeze(Object.keys(o));
});

const valuesBuiltin: DefinedBuiltin = defineBuiltin(spec('values', [1, 1], 'Lists an object’s own property values.', [{ name: 'obj', doc: 'the object to read values from' }], 'an array of the object’s own property values'), (ctx) => {
  ctx.tick();
  const o = ctx.evalArg(0);
  if (!isPlainObject(o)) return badArg(ctx, `values(object) — bad argument`);
  return ctx.freeze(Object.values(o));
});

const entriesBuiltin: DefinedBuiltin = defineBuiltin(spec('entries', [1, 1], 'Lists an object’s own properties as `[key, value]` pairs.', [{ name: 'obj', doc: 'the object to read entries from' }], 'an array of [key, value] pair arrays'), (ctx) => {
  ctx.tick();
  const o = ctx.evalArg(0);
  if (!isPlainObject(o)) return badArg(ctx, `entries(object) — bad argument`);
  return ctx.freeze(Object.entries(o).map(([k, v]) => [k, v]));
});

const objectBuiltin: DefinedBuiltin = defineBuiltin(spec('object', [1, 1], 'Builds an object from an array of `[key, value]` pairs (the inverse of `entries`).', [{ name: 'entries', doc: 'an array of [key, value] pairs (the key must be a string)' }], 'a new object with those key/value properties'), (ctx) => {
  ctx.tick();
  const pairs = ctx.evalArg(0);
  if (!Array.isArray(pairs)) return badArg(ctx, `object(array of [key, value]) — bad argument`);
  // Null-prototype record: no inherited proto/constructor/toString (matches the object-literal builder).
  const out: Record<string, unknown> = Object.create(null);
  for (const p of pairs) { ctx.tick(); if (Array.isArray(p) && typeof p[0] === 'string' && !FORBIDDEN_KEYS.has(p[0])) out[p[0]] = p[1]; }
  return ctx.freeze(out);
});

const hasBuiltin: DefinedBuiltin = defineBuiltin(spec('has', [2, 2], 'Tests whether an object has its own property named `key`.', [{ name: 'obj', doc: 'the object to check' }, { name: 'key', doc: 'the property name to look for' }], 'a boolean: true if the object owns that property'), (ctx) => {
  ctx.tick();
  const o = ctx.evalArg(0); const key = ctx.evalArg(1);
  if (!isPlainObject(o)) return badArg(ctx, `has(object, key) — bad argument`);
  // A forbidden key is never observable: report absent without touching the prototype chain (so
  // `has(o, "__proto__")` is false, matching the object-literal / spread FORBIDDEN_KEYS guard).
  if (typeof key !== 'string' || FORBIDDEN_KEYS.has(key)) return false;
  return Object.prototype.hasOwnProperty.call(o, key);
});

/** The structural builtin module: `keys`, `values`, `entries`, `object`, and `has` — object⇄array
 *  decomposition/reconstruction plus own-property presence. Each returns a NEW frozen collection (or a
 *  boolean) and never mutates. `object` and `has` are `FORBIDDEN_KEYS`-guarded, so a prototype-polluting
 *  key can never enter or be observed on a DSL object. Inject via a run's builtin modules to give a
 *  program object-reshaping vocabulary. */
export const STRUCTURAL_BUILTINS: readonly DefinedBuiltin[] = [
  keysBuiltin, valuesBuiltin, entriesBuiltin, objectBuiltin, hasBuiltin,
];
