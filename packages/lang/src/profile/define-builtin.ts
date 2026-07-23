import type { Builtin, BuiltinCtx, BuiltinModule } from '../registry.ts';
import type { Expr } from '../ast.ts';
import type { BuiltinSpec } from './types.ts';

/** A builtin declared with its metadata co-located: the pure {@link BuiltinSpec} a profile aggregates,
 *  plus the {@link Builtin.invoke} the interpreter runs. Projected two ways — {@link toBuiltinModule}
 *  for the runtime, {@link builtinSpecMap} for a profile — so a builtin is declared exactly once. */
export interface DefinedBuiltin {
  /** The pure metadata a profile publishes. */
  readonly spec: BuiltinSpec;
  /** The interpreter entry point (unevaluated arg expressions; pull via `ctx.evalArg`). */
  readonly invoke: (ctx: BuiltinCtx, argExprs: readonly Expr[]) => unknown;
}

/** Declare a builtin's spec + invoke together. */
export function defineBuiltin(spec: BuiltinSpec, invoke: DefinedBuiltin['invoke']): DefinedBuiltin {
  return { spec, invoke };
}

/** Project defined builtins into the runtime {@link BuiltinModule} the interpreter injects (name + invoke). */
export function toBuiltinModule(defs: readonly DefinedBuiltin[]): BuiltinModule {
  const builtins: Builtin[] = defs.map((d) => ({ name: d.spec.name, invoke: d.invoke }));
  return { builtins };
}

/** Project defined builtins into a name→{@link BuiltinSpec} map for a {@link Profile}. */
export function builtinSpecMap(defs: readonly DefinedBuiltin[]): ReadonlyMap<string, BuiltinSpec> {
  return new Map(defs.map((d) => [d.spec.name, d.spec] as const));
}
