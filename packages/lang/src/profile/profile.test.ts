import { describe, it, expect } from 'vitest';
import { composeProfiles } from './index.ts';
import type { Profile } from './index.ts';

const P = (id: string, over: Partial<Profile>): Profile =>
  ({ id, builtins: new Map(), heads: new Map(), types: new Map(), ...over });

describe('composeProfiles', () => {
  it('unions builtins/heads/types by key', () => {
    const a = P('a', { builtins: new Map([['x', { name: 'x', profile: 'core', portability: 'exact', takesClosure: false, arity: [0, 0] }]]) });
    const b = P('b', { heads: new Map([['div', { name: 'div', params: [], arity: [0, 99], returns: 'node' }]]) });
    const c = composeProfiles(a, b);
    expect(c.builtins.has('x')).toBe(true);
    expect(c.heads.has('div')).toBe(true);
  });
  it('records a collision when a name appears in >1 child (last wins)', () => {
    const s1 = { name: 'x', profile: 'core', portability: 'exact', takesClosure: false, arity: [0, 0] } as const;
    const s2 = { name: 'x', profile: 'host', portability: 'cpu-only', takesClosure: false, arity: [0, 0] } as const;
    const c = composeProfiles(P('a', { builtins: new Map([['x', s1]]) }), P('b', { builtins: new Map([['x', s2]]) }));
    expect(c.collisions).toContain('x');
    expect(c.builtins.get('x')).toBe(s2); // last wins
  });
  it('is permissive if ANY child is permissive', () => {
    expect(composeProfiles(P('a', {}), P('b', { permissiveHeads: true })).permissiveHeads).toBe(true);
    expect(composeProfiles(P('a', {}), P('b', {})).permissiveHeads).toBe(false);
  });
  it('is associative on the merged key sets', () => {
    const a = P('a', { builtins: new Map([['a', { name: 'a', profile: 'core', portability: 'exact', takesClosure: false, arity: [0, 0] }]]) });
    const b = P('b', { builtins: new Map([['b', { name: 'b', profile: 'core', portability: 'exact', takesClosure: false, arity: [0, 0] }]]) });
    const c = P('c', { builtins: new Map([['c', { name: 'c', profile: 'core', portability: 'exact', takesClosure: false, arity: [0, 0] }]]) });
    const left = composeProfiles(composeProfiles(a, b), c);
    const right = composeProfiles(a, composeProfiles(b, c));
    expect([...left.builtins.keys()].sort()).toEqual([...right.builtins.keys()].sort());
  });
});
