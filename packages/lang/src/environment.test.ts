import { describe, it, expect } from 'vitest';
import { Environment } from './environment.ts';

describe('Environment (single-namespace, const/reactive metadata)', () => {
  it('defines and reads a binding in the current scope', () => {
    const env = new Environment();
    env.define('x', 1, { kind: 'const' });
    expect(env.has('x')).toBe(true);
    expect(env.get('x')).toBe(1);
  });

  it('reads from an enclosing scope', () => {
    const parent = new Environment();
    parent.define('x', 10, { kind: 'const' });
    const child = new Environment(parent);
    expect(child.get('x')).toBe(10);
  });

  it('assign updates the nearest enclosing binding, not a new local', () => {
    const parent = new Environment();
    parent.define('x', 1, { kind: 'let' });
    const child = new Environment(parent);
    expect(child.assign('x', 2)).toBe(true);
    expect(parent.get('x')).toBe(2);
    expect(child.hasOwn('x')).toBe(false);
  });

  it('assign to an unbound name returns false', () => {
    expect(new Environment().assign('nope', 1)).toBe(false);
  });

  it('a child define shadows the parent without mutating it', () => {
    const parent = new Environment();
    parent.define('x', 1, { kind: 'const' });
    const child = new Environment(parent);
    child.define('x', 99, { kind: 'const' });
    expect(child.get('x')).toBe(99);
    expect(parent.get('x')).toBe(1);
  });

  it('records binding metadata (const vs reactive let) for the evaluator to enforce', () => {
    const env = new Environment();
    env.define('c', 1, { kind: 'const' });
    env.define('r', 0, { kind: 'let', cell: 'CELL#0' });   // cell = opaque CellRef
    expect(env.meta('c')?.kind).toBe('const');
    expect(env.meta('r')?.kind).toBe('let');
  });

  it('a reactive let stores its opaque CellRef (value lives in the host cell, not here)', () => {
    const env = new Environment();
    env.define('r', 0, { kind: 'let', cell: 'CELL#7' });
    const m = env.meta('r');
    expect(m?.kind === 'let' && m.cell).toBe('CELL#7');
  });

  it('redeclaring an OWN binding is detectable (one JS namespace)', () => {
    const env = new Environment();
    env.define('x', 1, { kind: 'const' });
    expect(env.hasOwn('x')).toBe(true); // caller raises ML-LANG-REDECL on re-define
  });
});
