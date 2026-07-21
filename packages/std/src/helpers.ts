// Shared helpers for the standard-library builtins. Each builtin reaches the interpreter ONLY through the
// injected capability context (BuiltinCtx): it ticks the budget, raises diagnostics, invokes closures, and
// deep-freezes results through that channel — never through interpreter internals. These helpers translate
// the common shapes (a callback normalizer, a fail-loud bad-argument path, an iterable→array read) once so
// each builtin stays small.
import type { BuiltinCtx } from '@metael/lang';
import { descriptorOf, generationOf, isUserFn } from '@metael/lang';

/** Normalize a value that may be a callback into a uniform invoker, or null if it is not callable. A
 *  callback may be EITHER an arrow (a real JS closure) OR a user-declared `function` (a structured callable
 *  object); ctx.callClosure invokes both, so a named function works as a callback, not just an arrow. We
 *  decide callability up front here (callClosure returns null for a non-callable, indistinguishable from a
 *  legitimate null result), so a bad-argument diagnostic fires for a non-callable rather than a silent no-op. */
export function asFn(ctx: BuiltinCtx, v: unknown): ((...xs: unknown[]) => unknown) | null {
  if (typeof v === 'function' || isUserFn(v)) return (...xs: unknown[]) => ctx.callClosure(v, xs);
  return null;
}

/** Raise a fail-loud bad-argument diagnostic and return a safe, frozen empty array (never throws). */
export function badArg(ctx: BuiltinCtx, msg: string): unknown {
  ctx.error('ML-LANG-BUILTIN-ARG', msg);
  return ctx.freeze([]);
}

/** Coerce a value to an array the language can iterate: a plain array as-is, or a custom value whose
 *  descriptor exposes `iterate` (a typed array — the same seam for-of uses). Returns null for a non-iterable
 *  so the caller's bad-argument diagnostic fires. A whole-value read registers the value's generation
 *  dependency (like for-of / index / concat) so a reactive context re-runs on an in-place write. */
export function toArray(ctx: BuiltinCtx, xs: unknown): unknown[] | null {
  if (Array.isArray(xs)) return xs;
  const desc = descriptorOf(xs);
  const iter = desc?.iterate?.(xs);
  if (!iter) return null;
  const gen = generationOf(xs);
  if (gen !== undefined) ctx.readGeneration(gen);
  return Array.from(iter);
}

/** Numeric coercion mirroring the interpreter's toNum: number→v; boolean→0/1; trimmed string→Number|NaN;
 *  anything else→NaN. */
export function toNum(v: unknown): number {
  if (typeof v === 'number') return v;
  if (typeof v === 'boolean') return v ? 1 : 0;
  if (typeof v === 'string') { const t = v.trim(); return t === '' ? NaN : Number(t); }
  return NaN;
}
