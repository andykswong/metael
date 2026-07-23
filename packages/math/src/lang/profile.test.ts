import { describe, it, expect } from 'vitest';
import { mathProfile, MATH_BUILTINS, makeVec } from './index.ts';
import { swizzleMembers } from '@metael/lang/profile';
import type { BuiltinProfile, Portability } from '@metael/lang/profile';
import { descriptorOf } from '@metael/lang';

// The buffer-constructor builtin names — the only closure-taking specs allowed to be non-`host` (a
// typed-array constructor takes a fill closure but stays `core`). Every OTHER closure-taking spec must
// be `host`.
const BUFFER_CTORS = new Set(['f32', 'f64', 'i32', 'u32']);
const VALID_PROFILES: BuiltinProfile[] = ['core', 'host'];
const VALID_PORTABILITY: Portability[] = ['exact', 'gpu-tolerant', 'cpu-only'];

describe('mathProfile', () => {
  it('publishes a spec for every dispatched math builtin', () => {
    expect(new Set(mathProfile.builtins.keys())).toEqual(new Set(MATH_BUILTINS.builtins.map((b) => b.name)));
  });

  // Catalog-wide cross-field invariants over every migrated BuiltinSpec. `defineBuiltin` does no
  // validation and profile/portability are independent unions, so a contradictory spec (e.g. a `host`
  // builtin marked gpu-tolerant, or a closure-taking non-host non-buffer builtin) would otherwise ship
  // green. These pin the whole map; a single offending spec fails the relevant assertion by name.
  describe('every builtin spec is internally consistent', () => {
    it('profile and portability are valid union members', () => {
      for (const [name, spec] of mathProfile.builtins) {
        expect(VALID_PROFILES, name).toContain(spec.profile);
        expect(VALID_PORTABILITY, name).toContain(spec.portability);
      }
    });
    it('arity min never exceeds max', () => {
      for (const [name, spec] of mathProfile.builtins) {
        expect(spec.arity[0], name).toBeLessThanOrEqual(spec.arity[1]);
      }
    });
    it('a host builtin is always cpu-only (cannot reproduce on a restricted target)', () => {
      for (const [name, spec] of mathProfile.builtins) {
        if (spec.profile === 'host') expect(spec.portability, name).toBe('cpu-only');
      }
    });
    it('a closure-taking builtin is host unless it is a buffer constructor', () => {
      for (const [name, spec] of mathProfile.builtins) {
        if (spec.takesClosure && !BUFFER_CTORS.has(spec.name)) expect(spec.profile, name).toBe('host');
      }
    });
    it('the map key equals the spec name field', () => {
      for (const [key, spec] of mathProfile.builtins) {
        expect(spec.name).toBe(key);
      }
    });
  });
  it('projects vec3 members (x/y/z + swizzles) and its constructor', () => {
    const t = mathProfile.types.get('vec3')!;
    expect(t.constructors).toContain('vec3');
    expect(t.members.some((m) => m.name === 'xyz')).toBe(true);
  });
  it('sqrt is gpu-tolerant', () => {
    expect(mathProfile.builtins.get('sqrt')!.portability).toBe('gpu-tolerant');
  });
  it('carries a doc + named params for hover/signature (dot has two)', () => {
    const dot = mathProfile.builtins.get('dot')!;
    expect(dot.doc).toBeTruthy();
    expect(dot.params?.length).toBe(2);
  });
  it('documents every dispatched builtin with a doc + params', () => {
    for (const [name, spec] of mathProfile.builtins) {
      expect(spec.doc, name).toBeTruthy();
      expect(spec.params, name).toBeDefined();
    }
  });
  // A coverage guard for the editor hover card, which renders `name(args)` + doc + a `  arg — doc` line
  // per param + a `Returns <returnDoc>.` line. Every builtin must carry a returnDoc AND a doc on EVERY
  // param, so a future un-documented arg / return fails CI rather than shipping a hover with a blank slot.
  it('documents a returnDoc + a doc for every param of every builtin', () => {
    for (const [name, spec] of mathProfile.builtins) {
      expect(typeof spec.returnDoc, `${name} returnDoc`).toBe('string');
      expect((spec.returnDoc ?? '').trim().length, `${name} returnDoc`).toBeGreaterThan(0);
      for (const p of spec.params ?? []) {
        expect(typeof p.doc, `${name}.${p.name} doc`).toBe('string');
        expect((p.doc ?? '').trim().length, `${name}.${p.name} doc`).toBeGreaterThan(0);
      }
    }
  });
  it('every generated vec3 component name resolves via the real descriptor getMember', () => {
    const v = makeVec([1, 2, 3], 'f32');
    const d = descriptorOf(v)!;
    for (const m of swizzleMembers(3).filter((x) => x.kind === 'component')) {
      expect(d.getMember!(v, m.name)).not.toBeUndefined();
    }
  });
});
