import { describe, it, expect } from 'vitest';
import type { BuiltinProfile, Portability } from '@metael/lang/profile';
import { stdProfile, STD_BUILTINS } from './index.ts';

const VALID_PROFILES: BuiltinProfile[] = ['core', 'host'];
const VALID_PORTABILITY: Portability[] = ['exact', 'gpu-tolerant', 'cpu-only'];

describe('stdProfile', () => {
  it('publishes a spec for every dispatched builtin (same name set as STD_BUILTINS)', () => {
    const moduleNames = new Set(STD_BUILTINS.builtins.map((b) => b.name));
    const profileNames = new Set(stdProfile.builtins.keys());
    expect(profileNames).toEqual(moduleNames);
  });
  it('classifies map as a host builtin', () => {
    expect(stdProfile.builtins.get('map')!.profile).toBe('host');
  });

  // The same catalog-wide cross-field invariants as mathProfile, over the std builtins map (the
  // source of truth post-migration). std has no buffer constructors, so the takesClosure⇒host rule
  // applies to ALL closure-taking builtins (map/filter/reduce/…). `defineBuiltin` validates nothing,
  // so these guard against a contradictory hand-transcribed spec shipping green.
  describe('every builtin spec is internally consistent', () => {
    it('profile and portability are valid union members', () => {
      for (const [name, spec] of stdProfile.builtins) {
        expect(VALID_PROFILES, name).toContain(spec.profile);
        expect(VALID_PORTABILITY, name).toContain(spec.portability);
      }
    });
    it('arity min never exceeds max', () => {
      for (const [name, spec] of stdProfile.builtins) {
        expect(spec.arity[0], name).toBeLessThanOrEqual(spec.arity[1]);
      }
    });
    it('a host builtin is always cpu-only', () => {
      for (const [name, spec] of stdProfile.builtins) {
        if (spec.profile === 'host') expect(spec.portability, name).toBe('cpu-only');
      }
    });
    it('a closure-taking builtin is host (std has no buffer constructors)', () => {
      for (const [name, spec] of stdProfile.builtins) {
        if (spec.takesClosure) expect(spec.profile, name).toBe('host');
      }
    });
    it('the map key equals the spec name field', () => {
      for (const [key, spec] of stdProfile.builtins) {
        expect(spec.name).toBe(key);
      }
    });
  });

  // Editor tooling (hover / signature help / completion detail) reads these two fields, so every
  // builtin must carry a non-empty doc and a params list whose length matches its declared max arity
  // (each call position is a named parameter).
  describe('every builtin carries tooling metadata (doc + params)', () => {
    it('every builtin has a non-empty doc', () => {
      for (const [name, spec] of stdProfile.builtins) {
        expect(typeof spec.doc, name).toBe('string');
        expect((spec.doc ?? '').length, name).toBeGreaterThan(0);
      }
    });
    it('params length matches the declared max arity', () => {
      for (const [name, spec] of stdProfile.builtins) {
        expect(spec.params, name).toBeDefined();
        expect(spec.params!.length, name).toBe(spec.arity[1]);
      }
    });
    it('every builtin has a non-empty returnDoc (the hover `Returns …` line)', () => {
      for (const [name, spec] of stdProfile.builtins) {
        expect(typeof spec.returnDoc, name).toBe('string');
        expect((spec.returnDoc ?? '').length, name).toBeGreaterThan(0);
      }
    });
    it('every declared param carries a non-empty doc (the hover per-arg list)', () => {
      for (const [name, spec] of stdProfile.builtins) {
        for (const p of spec.params ?? []) {
          expect(typeof p.doc, `${name}(${p.name})`).toBe('string');
          expect((p.doc ?? '').length, `${name}(${p.name})`).toBeGreaterThan(0);
        }
      }
    });
    it('map documents its two parameters (items, fn)', () => {
      const map = stdProfile.builtins.get('map')!;
      expect(map.doc).toBeTruthy();
      expect(map.params?.length).toBe(2);
      expect(map.params?.[0]?.name).toBe('items');
      expect(map.params?.[1]?.name).toBe('fn');
    });
  });
});
