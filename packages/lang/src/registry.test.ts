// packages/lang/src/registry.test.ts
import { describe, it, expect } from 'vitest';
import type { Builtin, BuiltinModule } from './registry.ts';
import { buildRegistry } from './registry.ts';

describe('builtin registry', () => {
  it('builds a name→Builtin map from modules, later modules win on name clash', () => {
    const a: Builtin = { spec: { name: 'x', profile: 'core', portability: 'exact', takesClosure: false, arity: [0, 0] }, invoke: () => 1 };
    const b: Builtin = { spec: { name: 'x', profile: 'core', portability: 'exact', takesClosure: false, arity: [0, 0] }, invoke: () => 2 };
    const modA: BuiltinModule = { builtins: [a] };
    const modB: BuiltinModule = { builtins: [b] };
    const reg = buildRegistry([modA, modB]);
    expect(reg.get('x')?.invoke({} as never, [])).toBe(2);
    expect(reg.has('x')).toBe(true);
    expect(reg.get('nope')).toBeUndefined();
  });
});
