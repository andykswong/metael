import { describe, it, expect } from 'vitest';
import { swizzleMembers } from './index.ts';

describe('swizzleMembers', () => {
  it('emits single components for a vec3 (x,y,z) but not w', () => {
    const names = swizzleMembers(3).map((m) => m.name);
    expect(names).toContain('x'); expect(names).toContain('y'); expect(names).toContain('z');
    expect(names).not.toContain('w');
  });
  it('emits 2- and 3-length swizzles within range', () => {
    const names = new Set(swizzleMembers(3).map((m) => m.name));
    expect(names.has('xy')).toBe(true); expect(names.has('xyz')).toBe(true);
    expect(names.has('xw')).toBe(false); // w out of range for vec3
  });
  it('tags every entry kind:"swizzle" (len>1) or "component" (len 1)', () => {
    for (const m of swizzleMembers(4)) expect(m.kind).toBe(m.name.length === 1 ? 'component' : 'swizzle');
  });
  it('a vec2 has no length-3 swizzles', () => {
    expect(swizzleMembers(2).some((m) => m.name.length === 3)).toBe(false);
  });
});
