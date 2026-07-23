// packages/lang/src/registry.test.ts
import { describe, it, expect } from 'vitest';
import type { Builtin, BuiltinModule } from './registry.ts';
import { buildRegistry } from './registry.ts';

describe('builtin registry', () => {
  it('builds a name→Builtin map from modules, later modules win on name clash', () => {
    const a: Builtin = { name: 'x', invoke: () => 1 };
    const b: Builtin = { name: 'x', invoke: () => 2 };
    const modA: BuiltinModule = { builtins: [a] };
    const modB: BuiltinModule = { builtins: [b] };
    const reg = buildRegistry([modA, modB]);
    expect(reg.get('x')?.invoke({} as never, [])).toBe(2);
    expect(reg.has('x')).toBe(true);
    expect(reg.get('nope')).toBeUndefined();
  });

  it('buildRegistry keys the dispatch map on Builtin.name', () => {
    const a: Builtin = { name: 'foo', invoke: () => 1 };
    const b: Builtin = { name: 'bar', invoke: () => 2 };
    const reg = buildRegistry([{ builtins: [a, b] }]);
    expect(reg.get('foo')).toBe(a);
    expect(reg.get('bar')).toBe(b);
    expect(reg.size).toBe(2);
  });
});
