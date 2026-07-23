import type { BuiltinSpec, HeadSpec, TypeDescriptorMeta } from './types.ts';

/** A named, composable bundle of a host environment's static vocabulary metadata — the tooling-layer
 *  analog of a runtime `HostEnvironment`. The language service is handed one (possibly composite)
 *  Profile and never has to know it was assembled from parts. */
export interface Profile {
  /** Identifier (e.g. `'vdom'`, or a composite like `'vdom+std'`). */
  readonly id: string;
  /** Builtin specs by name (aggregated from a package's builtins). */
  readonly builtins: ReadonlyMap<string, BuiltinSpec>;
  /** Head specs by name. For a closed vocabulary this is also the valid-name set. */
  readonly heads: ReadonlyMap<string, HeadSpec>;
  /** Custom-type projections by type name. */
  readonly types: ReadonlyMap<string, TypeDescriptorMeta>;
  /** True when the head set is OPEN (any lowercase tag valid, e.g. a DOM vocabulary). */
  readonly permissiveHeads?: boolean;
}

/** A {@link Profile} produced by {@link composeProfiles}, carrying the head/builtin/type names that
 *  appeared in more than one child (last-in-order wins at lookup, but the overlap is surfaced). */
export interface ComposedProfile extends Profile {
  /** Names present in more than one child map — a latent shadow, surfaced rather than hidden. */
  readonly collisions: readonly string[];
}

/** Keyed-union merge of one map kind across the children (last-in-order wins), recording every key that
 *  appeared in more than one child into `collisions`. */
function mergeMap<V>(children: readonly ReadonlyMap<string, V>[], collisions: Set<string>): Map<string, V> {
  const out = new Map<string, V>();
  for (const m of children) for (const [k, v] of m) { if (out.has(k)) collisions.add(k); out.set(k, v); }
  return out;
}

/** Merge N profiles: keyed union of builtins/heads/types (last wins), permissive if ANY child is,
 *  and every colliding name recorded in `collisions`. Associative; the result is itself a Profile. */
export function composeProfiles(...profiles: readonly Profile[]): ComposedProfile {
  const collisions = new Set<string>();
  const builtins = mergeMap(profiles.map((p) => p.builtins), collisions);
  const heads = mergeMap(profiles.map((p) => p.heads), collisions);
  const types = mergeMap(profiles.map((p) => p.types), collisions);
  return {
    id: profiles.map((p) => p.id).join('+') || 'empty',
    builtins, heads, types,
    permissiveHeads: profiles.some((p) => p.permissiveHeads === true),
    collisions: [...collisions],
  };
}
