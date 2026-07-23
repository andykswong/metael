// packages/lang/src/registry-default.test.ts
import { describe, it, expect } from 'vitest';
import { evaluateProgram } from './evaluate.ts';
import { PlainStorageHost, RecordingHostEnv } from './ports.ts';
import type { BuiltinModule } from './registry.ts';
import { MATH_BUILTINS } from '@metael/math/lang';
const H = () => ({ host: new PlainStorageHost(), env: new RecordingHostEnv() });

describe('lang ships an empty registry; builtins are injected', () => {
  it('an injected module resolves', () => {
    const mod: BuiltinModule = { builtins: [{
      name: 'inc',
      invoke: (ctx) => (ctx.evalArg(0) as number) + 1,
    }] };
    expect(evaluateProgram('inc(41)', { ...H(), builtins: [mod] }).value).toBe(42);
  });
  it('WITHOUT the module, that same name is unknown (proves lang privileges no builtin)', () => {
    // NB: `inc` is not a lang builtin. This asserts the injection is what supplies it — not lang.
    const res = evaluateProgram('inc(41)', H());
    expect(res.diagnostics.some((d) => d.code === 'ML-LANG-UNKNOWN-CALL')).toBe(true);
  });
});

// The empty-kernel invariant, stated over a real math head (`dot`/`vec2`). These names are
// household-name numeric builtins that a naive kernel would privilege — proving lang no longer
// does. The negative case is the whole point of the refactor: with nothing injected, a numeric
// call fails closed exactly like any other unknown head.
describe('the empty-kernel invariant: even a household numeric head is not privileged', () => {
  const SRC = 'dot(vec2(1,2),vec2(3,4))';

  it('with NO builtins injected, dot/vec2 fail closed (ML-LANG-UNKNOWN-CALL)', () => {
    const res = evaluateProgram(SRC, H());
    expect(res.value).toBeNull();
    expect(res.diagnostics.some((d) => d.code === 'ML-LANG-UNKNOWN-CALL')).toBe(true);
  });

  it('with the math module injected, the same source computes 1*3 + 2*4 = 11', () => {
    const res = evaluateProgram(SRC, { ...H(), builtins: [MATH_BUILTINS] });
    expect(res.diagnostics).toEqual([]);
    expect(res.value).toBe(11);
  });
});
