import { describe, it, expect } from 'vitest';
import { composeEnvs } from './index.ts';
import type { HostEnvironment, ReactiveHost, Arg, HostValue, SourceSpan } from '@metael/lang';

const span: SourceSpan = { start: 0, end: 0 };

// A minimal env that handles a fixed set of heads, records bindHost calls, and (optionally) is Disposable.
// `id` surfaces in each resolveCall result (`from: id`) so tests can assert WHICH env answered a shared head.
function makeEnv(heads: string[], opts: { known?: boolean; disposable?: boolean } = {}, id?: string) {
  const state = { bound: 0, disposed: 0 };
  const env: HostEnvironment & { bindHost(h: ReactiveHost): void; state: typeof state } = {
    state,
    resolveCall(head: string, key: string, _a: Arg[], _c: HostValue[], _s: SourceSpan) {
      return heads.includes(head) ? { handled: true as const, value: { head, key, from: id } } : { handled: false as const };
    },
    bindHost(_h: ReactiveHost) { state.bound++; },
    ...(opts.known ? { knownHeads: new Set(heads) } : {}),
  };
  if (opts.disposable) (env as unknown as Disposable)[Symbol.dispose] = () => { state.disposed++; };
  return env;
}

describe('composeEnvs', () => {
  it('resolveCall tries envs in array order — first handled wins', () => {
    const a = makeEnv(['x'], {}, 'a');
    const b = makeEnv(['x', 'y'], {}, 'b');
    const c = composeEnvs([a, b]);
    // Both `a` and `b` handle 'x'; the result must carry `from: 'a'` (the FIRST env), not 'b' — proving
    // array order is dispatch priority. A last-wins regression would surface `from: 'b'` here and fail.
    expect(c.resolveCall('x', 'k', [], [], span)).toEqual({ handled: true, value: { head: 'x', key: 'k', from: 'a' } });
    // 'y' only by `b`, 'z' by neither.
    expect(c.resolveCall('y', 'k', [], [], span)).toEqual({ handled: true, value: { head: 'y', key: 'k', from: 'b' } });
    expect(c.resolveCall('z', 'k', [], [], span)).toEqual({ handled: false });
  });

  it('bindHost fans out to every bindable child (skips a non-bindable one)', () => {
    const a = makeEnv(['x']);
    const plain: HostEnvironment = { resolveCall: () => ({ handled: false }) }; // no bindHost
    const c = composeEnvs([a, plain]);
    const host = {} as ReactiveHost;
    c.bindHost(host);   // must not throw despite `plain` having no bindHost
    expect(a.state.bound).toBe(1);
  });

  it('[Symbol.dispose] fans out only to disposable children', () => {
    const a = makeEnv(['x'], { disposable: true });
    const b = makeEnv(['y']);   // not disposable
    const c = composeEnvs([a, b]);
    c[Symbol.dispose]();
    expect(a.state.disposed).toBe(1);   // b simply skipped, no throw
  });

  it('knownHeads is the union when all children declare it; collisions are exposed', () => {
    const a = makeEnv(['x', 'shared'], { known: true });
    const b = makeEnv(['y', 'shared'], { known: true });
    const c = composeEnvs([a, b]);
    expect([...c.knownHeads!].sort()).toEqual(['shared', 'x', 'y']);
    expect(c.collisions).toEqual(['shared']);
  });

  it('a 3-way head overlap records the collision only once', () => {
    const a = makeEnv(['x', 'shared'], { known: true });
    const b = makeEnv(['y', 'shared'], { known: true });
    const d = makeEnv(['z', 'shared'], { known: true });
    const c = composeEnvs([a, b, d]);
    expect([...c.knownHeads!].sort()).toEqual(['shared', 'x', 'y', 'z']); // union still has every distinct head
    expect(c.collisions).toEqual(['shared']);                            // deduped, not ['shared', 'shared']
    expect(c.collisions.filter((h) => h === 'shared').length).toBe(1);
  });

  it('knownHeads is undefined (permissive) if ANY child is permissive', () => {
    const a = makeEnv(['x'], { known: true });
    const b = makeEnv(['y']);   // no knownHeads → permissive
    const c = composeEnvs([a, b]);
    expect(c.knownHeads).toBeUndefined();
    expect(c.collisions).toEqual([]);   // no union computed → no collisions reported
  });
});
