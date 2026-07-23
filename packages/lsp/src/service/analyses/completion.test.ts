import { describe, it, expect } from 'vitest';
import { LanguageService } from '../index.ts';
import { composeProfiles } from '@metael/lang/profile';
import { mathProfile } from '@metael/math/lang';
import { stdProfile } from '@metael/std';
import { vdomProfile } from '@metael/vdom/lang';

const compute = composeProfiles(mathProfile, stdProfile);

describe('completion', () => {
  it('offers a visible const binding by name', () => {
    const svc = new LanguageService();
    const src = 'const foo = 1\nconst bar = f';
    svc.openDocument('a', src, 1); svc.setProfile('a', compute);
    const items = svc.completion('a', src.length);
    expect(items.some((c) => c.label === 'foo')).toBe(true);
  });
  it('offers profile builtins (map from std, sqrt from math)', () => {
    const svc = new LanguageService();
    svc.openDocument('a', 'const x = ', 1); svc.setProfile('a', compute);
    const labels = svc.completion('a', 'const x = '.length).map((c) => c.label);
    expect(labels).toContain('map'); expect(labels).toContain('sqrt');
  });
  it('shows a builtin’s doc as its detail (not an "arity N..N" placeholder)', () => {
    const svc = new LanguageService();
    svc.openDocument('a', 'const x = ', 1); svc.setProfile('a', compute);
    const items = svc.completion('a', 'const x = '.length);
    const map = items.find((c) => c.label === 'map');
    expect(map?.detail).toBe(stdProfile.builtins.get('map')!.doc);
    expect(map?.detail).not.toMatch(/arity/);
    // The kernel-dispatched `range` intrinsic uses its real spec doc too.
    const range = items.find((c) => c.label === 'range');
    expect((range?.detail ?? '').length).toBeGreaterThan(0);
    expect(range?.detail).not.toMatch(/arity/);
  });
  it('offers vec3 members after a dot on a vec3-typed local', () => {
    const svc = new LanguageService();
    const src = 'const v = vec3(1,2,3)\nconst a = v.';
    svc.openDocument('a', src, 1); svc.setProfile('a', compute);
    const items = svc.completion('a', src.length);
    expect(items.some((c) => c.label === 'x' && c.kind === 'member')).toBe(true);
    expect(items.some((c) => c.label === 'xyz')).toBe(true);
  });
  it('offers heads (any tag) under a permissive vdom profile in child position', () => {
    const svc = new LanguageService();
    svc.openDocument('a', 'component App() { di }', 1); svc.setProfile('a', vdomProfile);
    const items = svc.completion('a', 'component App() { di'.length);
    expect(items.some((c) => c.label === 'div' && c.kind === 'head')).toBe(true);
  });

  it('offers language keywords at a statement-start position', () => {
    const svc = new LanguageService();
    const src = 'const foo = 1\n';
    svc.openDocument('a', src, 1); svc.setProfile('a', compute);
    const items = svc.completion('a', src.length);
    expect(items.some((c) => c.label === 'const' && c.kind === 'keyword')).toBe(true);
    expect(items.some((c) => c.label === 'if' && c.kind === 'keyword')).toBe(true);
    expect(items.some((c) => c.label === 'for' && c.kind === 'keyword')).toBe(true);
  });

  it('ranks the closest near-miss of the partial word to the front (didYouMean)', () => {
    const svc = new LanguageService();
    // `stdProfile` alone: `map` is the UNIQUE Levenshtein-1 near-miss of `maq`
    // (math's `max` would tie, so it is deliberately excluded from this profile).
    const src = 'const x = maq';
    svc.openDocument('a', src, 1); svc.setProfile('a', stdProfile);
    const items = svc.completion('a', src.length);
    const labels = items.map((c) => c.label);
    // Present …
    expect(labels).toContain('map');
    // … and re-ordered to the front (a stable move, not a filter).
    expect(labels[0]).toBe('map');
    // … while every other candidate is preserved (didYouMean re-orders, never drops).
    expect(labels).toContain('filter');
    expect(labels).toContain('range');
  });

  it('never leaks keywords into member context (members-only after a dot)', () => {
    const svc = new LanguageService();
    const src = 'const v = vec3(1,2,3)\nconst a = v.';
    svc.openDocument('a', src, 1); svc.setProfile('a', compute);
    const items = svc.completion('a', src.length);
    expect(items.length).toBeGreaterThan(0);
    expect(items.every((c) => c.kind === 'member')).toBe(true);
    expect(items.some((c) => c.kind === 'keyword')).toBe(false);
  });
});
