import { describe, it, expect } from 'vitest';
import { evaluateProgram } from './evaluate.ts';
import { PlainStorageHost, RecordingHostEnv } from './ports.ts';
import type { BuiltinModule } from './registry.ts';

const host = () => new PlainStorageHost();

describe('registry dispatch via ctx', () => {
  it('a registered builtin is invoked with lazy args + tick + freeze', () => {
    const calls: number[] = [];
    const mod: BuiltinModule = {
      builtins: [{
        name: 'twice',
        invoke: (ctx, argExprs) => { ctx.tick(); calls.push(argExprs.length); return (ctx.evalArg(0) as number) * 2; },
      }],
    };
    const res = evaluateProgram('twice(21)', { host: host(), env: new RecordingHostEnv(), builtins: [mod] });
    expect(res.value).toBe(42);
    expect(calls).toEqual([1]);
  });

  it('an unregistered unbound head still fails loud (ML-LANG-UNKNOWN-CALL)', () => {
    const res = evaluateProgram('nope(1)', { host: host(), env: new RecordingHostEnv() });
    expect(res.diagnostics.some((d) => d.code === 'ML-LANG-UNKNOWN-CALL')).toBe(true);
  });
});
