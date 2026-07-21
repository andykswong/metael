import type { BindableHostEnv, HostEnvironment, ReactiveHost, Arg, HostValue, SourceSpan } from '@metael/lang';

/** A {@link BindableHostEnv} that also reports head-name collisions across the composed vocabularies. */
export interface ComposedHostEnv extends BindableHostEnv, Disposable {
  /** Head names declared in more than one child's `knownHeads` — first-in-array-order wins at dispatch,
   *  but an overlap is a latent shadow, so it is surfaced here rather than hidden. Empty when the union
   *  is permissive (some child declares no `knownHeads`). */
  readonly collisions: readonly string[];
}

/** Merge several single-vocabulary {@link HostEnvironment}s into one. `resolveCall` tries each env in
 *  array order and returns the first `{ handled: true }` (else `{ handled: false }`, so the derive still
 *  emits its wrapper fallback) — ARRAY ORDER IS THE PRIORITY. `bindHost` fans out to every child that has
 *  one (a stateless child with no `bindHost` is skipped); `[Symbol.dispose]` fans out to every child that
 *  is `Disposable`. `knownHeads` is the union of all children's — but only when EVERY child declares one;
 *  if any child is permissive (no `knownHeads`), the union is `undefined` so the composite stays permissive
 *  (matching the single-env "absent ⇒ permissive" contract). Head names in more than one child's
 *  `knownHeads` are surfaced on `collisions`. */
export function composeEnvs(envs: HostEnvironment[]): ComposedHostEnv {
  // Union knownHeads only if EVERY child declares one; else permissive (undefined). Track collisions.
  let knownHeads: Set<string> | undefined;
  const collisions: string[] = [];
  if (envs.length > 0 && envs.every((e) => e.knownHeads !== undefined)) {
    knownHeads = new Set();
    for (const e of envs) {
      for (const h of e.knownHeads!) {
        if (knownHeads.has(h)) {
          if (!collisions.includes(h)) collisions.push(h); // record each colliding head once, even on 3+-way overlap
        } else knownHeads.add(h);
      }
    }
  }

  return {
    resolveCall(head: string, key: string, args: Arg[], children: HostValue[], span: SourceSpan) {
      for (const e of envs) {
        const r = e.resolveCall(head, key, args, children, span);
        if (r.handled) return r;
      }
      return { handled: false };
    },
    bindHost(host: ReactiveHost): void {
      for (const e of envs) {
        const b = (e as Partial<BindableHostEnv>).bindHost;
        if (typeof b === 'function') b.call(e, host);
      }
    },
    [Symbol.dispose](): void {
      for (const e of envs) (e as Partial<Disposable>)[Symbol.dispose]?.();
    },
    knownHeads,
    collisions,
  };
}
