import { describe, it, expect } from 'vitest';
import { buildRegistry } from '@metael/lang';
import type { BuiltinCtx } from '@metael/lang';
import { defineBuiltin, toBuiltinModule, builtinSpecMap } from './index.ts';

const noop = (_ctx: BuiltinCtx) => 0;

describe('defineBuiltin', () => {
  it('projects a spec+invoke into a runtime Builtin (name only) and a spec map', () => {
    const d = defineBuiltin({ name: 'sq', profile: 'core', portability: 'exact', takesClosure: false, arity: [1, 1] }, noop);
    expect(d.spec.name).toBe('sq');
    const mod = toBuiltinModule([d]);
    expect(mod.builtins[0]!.name).toBe('sq');
    expect('spec' in mod.builtins[0]!).toBe(false); // runtime Builtin carries no spec
    const reg = buildRegistry([mod]);
    expect(reg.get('sq')!.invoke).toBe(noop);
    const specs = builtinSpecMap([d]);
    expect(specs.get('sq')!.portability).toBe('exact');
  });
});
