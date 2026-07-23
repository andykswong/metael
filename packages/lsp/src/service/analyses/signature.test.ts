import { describe, it, expect } from 'vitest';
import { LanguageService } from '../index.ts';
import { composeProfiles } from '@metael/lang/profile';
import type { Profile } from '@metael/lang/profile';
import { mathProfile } from '@metael/math/lang';
import { stdProfile } from '@metael/std';

const p = composeProfiles(mathProfile, stdProfile);

/** Open a document under the composed math+std profile and request signature help at `offset`
 *  (defaulting to end-of-source). */
function sigAt(src: string, offset = src.length) {
  const svc = new LanguageService();
  svc.openDocument('a', src, 1); svc.setProfile('a', p);
  return svc.signatureHelp('a', offset);
}

/** A profile carrying one builtin `blend` with named params, for the param-label assertion. */
const richProfile: Profile = {
  id: 'rich',
  heads: new Map(),
  types: new Map(),
  builtins: new Map([
    ['blend', {
      name: 'blend', profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [2, 3],
      doc: 'Mixes two values by an optional weight.',
      params: [{ name: 'a' }, { name: 'b' }, { name: 'weight', optional: true, doc: 'blend factor in [0,1]' }],
    }],
  ]),
};

describe('signatureHelp', () => {
  it('labels the enclosing call and reports activeParam 0 at the open paren', () => {
    const src = 'const y = sqrt(2)';
    const sig = sigAt(src, src.indexOf('(') + 1);
    expect(sig).not.toBeNull();
    expect(sig!.label).toContain('sqrt');
    expect(sig!.activeParam).toBe(0);
  });

  it('reports activeParam 0 at a bare open paren with no argument yet typed', () => {
    const src = 'const y = sqrt(';
    const sig = sigAt(src);
    expect(sig).not.toBeNull();
    expect(sig!.label).toContain('sqrt');
    expect(sig!.activeParam).toBe(0);
  });

  it('advances activeParam past a comma in a closed two-arg call', () => {
    const src = 'const r = dot(a, b)';
    const sig = sigAt(src, src.indexOf(',') + 1);
    expect(sig).not.toBeNull();
    expect(sig!.label).toContain('dot');
    expect(sig!.params.length).toBe(2);
    expect(sig!.activeParam).toBe(1);
  });

  it('gives signature help for an UNCLOSED call mid-typing (no trailing arg yet)', () => {
    const src = 'const r = dot(a, ';
    const sig = sigAt(src);
    expect(sig).not.toBeNull();
    expect(sig!.label).toContain('dot');
    expect(sig!.activeParam).toBe(1);
  });

  it('gives signature help when the cursor sits before a not-yet-typed second arg', () => {
    // Cursor just past the comma+space, before the still-empty second slot's closing paren.
    const src = 'const r = dot(a, )';
    const sig = sigAt(src, src.indexOf(',') + 2);
    expect(sig).not.toBeNull();
    expect(sig!.activeParam).toBe(1);
  });

  it("does not let a nested call's comma leak into the outer activeParam", () => {
    // dot(dot(a, b), <cursor> — the inner `a, b` comma is nested, so the OUTER active param is 1.
    const src = 'const r = dot(dot(a, b), ';
    const sig = sigAt(src);
    expect(sig).not.toBeNull();
    expect(sig!.label).toContain('dot');
    expect(sig!.activeParam).toBe(1);
  });

  it("reports the INNER call's activeParam when the cursor is inside a nested call", () => {
    const src = 'const r = dot(dot(a, b), c)';
    const sig = sigAt(src, 'const r = dot(dot(a,'.length);
    expect(sig).not.toBeNull();
    expect(sig!.activeParam).toBe(1);
  });

  it('returns null when the offset is not inside any call', () => {
    const src = 'const y = 1';
    expect(sigAt(src, src.indexOf('1'))).toBeNull();
  });

  it('returns null inside a grouping paren with no callee ident before it', () => {
    expect(sigAt('const r = (a + ')).toBeNull();
  });

  it('returns null for a call whose callee resolves to neither head nor builtin', () => {
    const src = 'const y = mystery(1)';
    expect(sigAt(src, src.indexOf('(') + 1)).toBeNull();
  });

  it("uses a builtin's declared param names (not synthesized argN) when the spec carries params", () => {
    const src = 'const y = blend(1, ';
    const svc = new LanguageService();
    svc.openDocument('a', src, 1); svc.setProfile('a', richProfile);
    const sig = svc.signatureHelp('a', src.length);
    expect(sig).not.toBeNull();
    expect(sig!.params.map((x) => x.label)).toEqual(['a', 'b', 'weight']);
    expect(sig!.label).toBe('blend(a, b, weight)');
    // The optional param carries its doc through.
    expect(sig!.params[2]!.doc).toBe('blend factor in [0,1]');
  });
});
