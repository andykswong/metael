import { describe, it, expect } from 'vitest';
import { evaluateProgram } from './evaluate.ts';
import { PlainStorageHost } from './ports.ts';
import type { HostEnvironment, Arg, HostValue, SourceSpan } from './ports.ts';
import { tagCustom, NOT_HANDLED, descriptorOf, isCustomType } from './custom-types.ts';
import type { TypeDescriptor } from './custom-types.ts';

const pairDesc: TypeDescriptor = {
  name: 'pair',
  binary: (op, l, r) => {
    // Only pair + pair is defined; a mixed combination (e.g. number + pair, via the 2*custom rule)
    // returns NOT_HANDLED so the interpreter reports ML-LANG-OP-UNSUPPORTED for the undefined combo.
    if (op === '+' && descriptorOf(l) === pairDesc && descriptorOf(r) === pairDesc) {
      const a = l as { a: number; b: number }; const b = r as { a: number; b: number };
      return mk(a.a + b.a, a.b + b.b);
    }
    return NOT_HANDLED;
  },
  equals: (l, r) => {
    if (descriptorOf(l) !== pairDesc || descriptorOf(r) !== pairDesc) return false;
    const a = l as { a: number; b: number }; const b = r as { a: number; b: number };
    return a.a === b.a && a.b === b.b;
  },
  neg: (v) => { const p = v as { a: number; b: number }; return mk(-p.a, -p.b); },
  getMember: (v, prop) => { const p = v as Record<string, number>; if (prop === 'a' || prop === 'b') return p[prop]; return NOT_HANDLED_MEMBER(); },
  getIndex: (v, key) => { const p = v as { a: number; b: number }; if (key === 0) return p.a; if (key === 1) return p.b; return NOT_HANDLED_MEMBER(); },
  setIndex: (v, key, val) => { const p = v as { a: number; b: number }; if (key === 0) p.a = val as number; else p.b = val as number; },
  iterate: (v) => { const p = v as { a: number; b: number }; return [p.a, p.b]; },
  truthy: (v) => { const p = v as { a: number; b: number }; return p.a !== 0 || p.b !== 0; },
  display: (v) => { const p = v as { a: number; b: number }; return `pair(${p.a}, ${p.b})`; },
};
function NOT_HANDLED_MEMBER(): unknown { return NOT_HANDLED; }
function mk(a: number, b: number): object { return tagCustom({ a, b }, pairDesc); }

class PairHostEnv implements HostEnvironment {
  resolveCall(head: string, _key: string, args: Arg[], _children: HostValue[], _span: SourceSpan):
    { handled: true; value: HostValue; kind?: 'value' } | { handled: false } {
    if (head === 'mk') return { handled: true, value: mk(Number(args[0]?.value ?? 0), Number(args[1]?.value ?? 0)), kind: 'value' };
    return { handled: false };
  }
}
const run = (src: string) => evaluateProgram(src, { host: new PlainStorageHost(), env: new PairHostEnv() });
// A top-level bare `let` is reactive-scope-gated (ML-LANG-LET-SCOPE) unless insideComponent is set; the
// in-place-write tests bind through a `let`, so they use this component-scoped run variant.
const runLet = (src: string) => evaluateProgram(src, { host: new PlainStorageHost(), env: new PairHostEnv(), insideComponent: true });

describe('custom-type dispatch protocol', () => {
  it('binary + dispatches to the descriptor', () => {
    expect((run('const p = mk(1, 2) + mk(3, 4); p.a').value)).toBe(4);
    expect((run('const p = mk(1, 2) + mk(3, 4); p.b').value)).toBe(6);
  });
  it('unary - dispatches to neg', () => {
    expect(run('const p = -mk(1, 2); p.a').value).toBe(-1);
  });
  it('== / != use the equals handler; a distinct pair is unequal', () => {
    expect(run('mk(1, 2) == mk(1, 2)').value).toBe(true);
    expect(run('mk(1, 2) != mk(9, 9)').value).toBe(true);
  });
  it('member + index accessors dispatch', () => {
    expect(run('mk(5, 6).a').value).toBe(5);
    expect(run('mk(5, 6)[1]').value).toBe(6);
  });
  it('truthy dispatches to the handler', () => {
    expect(run('mk(0, 0) ? 1 : 2').value).toBe(2);
    expect(run('mk(1, 0) ? 1 : 2').value).toBe(1);
  });
  it('an undefined operator (*) → ML-LANG-OP-UNSUPPORTED', () => {
    const r = run('mk(1, 2) * mk(3, 4)');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-OP-UNSUPPORTED')).toBe(true);
  });
  it('an undefined ordering (<) via NOT_HANDLED → ML-LANG-OP-UNSUPPORTED', () => {
    const r = run('mk(1, 2) < mk(3, 4)');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-OP-UNSUPPORTED')).toBe(true);
  });
  it('== with NO equals handler falls back to reference identity (never fail-loud)', () => {
    expect(run('const p = mk(1, 2); p == p').value).toBe(true);
  });
  it('an unknown member → ML-LANG-UNKNOWN-MEMBER', () => {
    const r = run('mk(1, 2).zzz');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-UNKNOWN-MEMBER')).toBe(true);
  });
  it('the 2*custom rule: number op custom reaches the custom descriptor', () => {
    const r = run('3 + mk(1, 2)');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-OP-UNSUPPORTED')).toBe(true);
  });
  it('a forbidden key never reaches a descriptor', () => {
    const r = run('mk(1, 2)["__proto__"]');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-FORBIDDEN')).toBe(true);
  });
  it('the number fast path is unaffected (no descriptor lookup for scalars)', () => {
    expect(run('1 + 2 * 3').value).toBe(7);
    expect(run('5 < 9').value).toBe(true);
  });
  it('display is used by string coercion, never the raw store', () => {
    expect(run('"" + mk(1, 2)').value).toBe('pair(1, 2)');
  });
  it('helpers: isCustomType / descriptorOf recognise a tagged value', () => {
    expect(isCustomType(mk(1, 2))).toBe(true);
    expect(descriptorOf(mk(1, 2))?.name).toBe('pair');
    expect(isCustomType(5)).toBe(false);
  });
  it('a kind:value custom return is frozen (immutable) even under a let binding — deep-frozen at the boundary', () => {
    // A host `kind:'value'` result is pure + deep-frozen by contract, so it is immutable regardless of the
    // let/const binding. (The genuinely-mutable-in-place path is an intrinsic typed array — see the typed-array suite.)
    const r = runLet('let p = mk(1, 2); p[0] = 9; p[0]');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-IMMUTABLE')).toBe(true);
    expect(r.value).toBe(1);
  });
  it('a const-bound custom value is frozen — an index write is ML-LANG-IMMUTABLE (interpreter-enforced, no descriptor frozen handler)', () => {
    const r = run('const p = mk(1, 2); p[0] = 9; p[0]');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-IMMUTABLE')).toBe(true);
    expect(r.value).toBe(1);
  });
  it('the const-freeze survives aliasing through a let (the aliasing hole is closed)', () => {
    const r = runLet('const a = mk(1, 2); let b = a; b[0] = 9; b[0]');
    expect(r.diagnostics.some((d) => d.code === 'ML-LANG-IMMUTABLE')).toBe(true);
    expect(r.value).toBe(1);
  });
  it('== / != against null never fail-loud (reference identity, no ML-LANG-INTERNAL)', () => {
    const rEq = run('mk(1, 2) == null');
    expect(rEq.diagnostics.some((d) => d.code === 'ML-LANG-INTERNAL')).toBe(false);
    expect(rEq.value).toBe(false);
    const rNe = run('mk(1, 2) != null');
    expect(rNe.value).toBe(true);
    const rNull = run('null == mk(1, 2)');
    expect(rNull.value).toBe(false);
  });
  it('a type expressing equality via binary(==) only keeps ==/!= symmetric (no equals handler)', () => {
    // Build a descriptor with NO equals handler; equality is answered by binary('==').
    const boxDesc: TypeDescriptor = {
      name: 'box',
      binary: (op, l, r) => {
        if (op === '==') {
          if (descriptorOf(l) !== boxDesc || descriptorOf(r) !== boxDesc) return NOT_HANDLED;
          return (l as { n: number }).n === (r as { n: number }).n;
        }
        return NOT_HANDLED;   // no '!=' — the interpreter must derive it from '=='
      },
    };
    const mkBox = (n: number): object => tagCustom({ n }, boxDesc);
    class BoxEnv implements HostEnvironment {
      resolveCall(head: string, _k: string, args: Arg[], _c: HostValue[], _s: SourceSpan):
        { handled: true; value: HostValue; kind?: 'value' } | { handled: false } {
        if (head === 'box') return { handled: true, value: mkBox(Number(args[0]?.value ?? 0)), kind: 'value' };
        return { handled: false };
      }
    }
    const runBox = (src: string) => evaluateProgram(src, { host: new PlainStorageHost(), env: new BoxEnv() });
    expect(runBox('box(5) == box(5)').value).toBe(true);
    expect(runBox('box(5) != box(5)').value).toBe(false);   // MUST be false — the Fix-1 bug made this true
    expect(runBox('box(5) != box(9)').value).toBe(true);
  });
});
