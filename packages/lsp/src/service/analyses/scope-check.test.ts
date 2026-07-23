import { describe, it, expect } from 'vitest';
import { evaluateProgram, PlainStorageHost, RecordingHostEnv } from '@metael/lang';
import { composeProfiles, coreIntrinsicsProfile } from '@metael/lang/profile';
import type { Profile } from '@metael/lang/profile';
import { mathProfile, MATH_BUILTINS } from '@metael/math/lang';
import { stdProfile, STD_BUILTINS } from '@metael/std';
import { vdomProfile } from '@metael/vdom/lang';
import { LanguageService } from '../index.ts';

/** A composed "known name" profile: math + std builtins plus the core intrinsics, used for the cases
 *  that lean on a builtin/type being recognised (so it is not wrongly flagged as an undeclared var). */
const compute: Profile = composeProfiles(mathProfile, stdProfile, coreIntrinsicsProfile);

/** Open `src` under `profile` and return its diagnostics. */
function diag(src: string, profile: Profile = compute): readonly { code: string; span: { start: number; end: number } }[] {
  const svc = new LanguageService();
  svc.openDocument('a', src, 1);
  svc.setProfile('a', profile);
  return svc.diagnostics('a');
}

/** The undeclared-var / redecl codes the scope-check pass adds, filtered from the merged diagnostics. */
const scopeCodes = (src: string, profile: Profile = compute): string[] =>
  diag(src, profile).map((d) => d.code).filter((c) => c === 'ML-LANG-UNKNOWN-VAR' || c === 'ML-LANG-REDECL');

/** Whether `evaluateProgram` emits an UNKNOWN-VAR for `src` (the cross-check oracle: our static pass must
 *  never flag an undeclared var the interpreter would not). */
function evalEmitsUnknownVar(src: string, data?: unknown): boolean {
  const opts = { host: new PlainStorageHost(), env: new RecordingHostEnv(), builtins: [MATH_BUILTINS, STD_BUILTINS] };
  const { diagnostics } = data === undefined
    ? evaluateProgram(src, opts)
    : evaluateProgram(src, { ...opts, data });
  return diagnostics.some((d) => d.code === 'ML-LANG-UNKNOWN-VAR');
}

describe('scope-check — undeclared value-reads (ML-LANG-UNKNOWN-VAR)', () => {
  it('flags an undeclared value-read', () => {
    const codes = scopeCodes('const y = x + 1');
    expect(codes.filter((c) => c === 'ML-LANG-UNKNOWN-VAR')).toHaveLength(1);
    const d = diag('const y = x + 1').find((x) => x.code === 'ML-LANG-UNKNOWN-VAR')!;
    // The span covers the `x` at offset 10.
    expect(d.span.start).toBe(10);
    expect(d.span.end).toBe(11);
  });

  it('does NOT flag a visible const (x is in scope)', () => {
    expect(scopeCodes('const x = 1\nconst y = x + 1')).toEqual([]);
  });

  it('does NOT flag a builtin/call head or an arrow param (map + v)', () => {
    expect(scopeCodes('const xs = [1,2,3]\nmap(xs, (v) => v * 2)')).toEqual([]);
  });

  it('does NOT flag a math builtin used as a call head (sqrt)', () => {
    expect(scopeCodes('sqrt(2)')).toEqual([]);
  });

  it('does NOT flag the implicit `data` root binding', () => {
    expect(scopeCodes('data.items')).toEqual([]);
  });

  it('does NOT flag a member property (v visible; x is a member string)', () => {
    expect(scopeCodes('const v = vec3(1,2,3)\nv.x')).toEqual([]);
  });

  it('does NOT flag a for-binding used in the loop body', () => {
    expect(scopeCodes('const xs = [1,2,3]\nfor (const item of xs) { div(item) }', vdomProfile)).toEqual([]);
  });

  it('does NOT flag a let read+write inside an arrow within a component', () => {
    const src = 'component App() { let n = 0\nbutton({ onClick: () => { n = n + 1 } }, "+") }';
    expect(scopeCodes(src, vdomProfile)).toEqual([]);
  });

  it('does NOT flag a lowercase call head under a permissive vdom profile', () => {
    expect(scopeCodes('component App() { customtag("x") }', vdomProfile)).toEqual([]);
  });

  it('cross-check: the interpreter agrees no UNKNOWN-VAR for the negative sources', () => {
    expect(evalEmitsUnknownVar('const xs = [1,2,3]\nmap(xs, (v) => v * 2)')).toBe(false);
    expect(evalEmitsUnknownVar('sqrt(2)')).toBe(false);
    expect(evalEmitsUnknownVar('data.items', { items: [1, 2] })).toBe(false);
    expect(evalEmitsUnknownVar('const v = vec3(1,2,3)\nv.x')).toBe(false);
    expect(evalEmitsUnknownVar('const x = 1\nconst y = x + 1')).toBe(false);
  });
});

describe('scope-check — block-scope redeclaration (ML-LANG-REDECL)', () => {
  it('flags a block-scope redecl the parser misses (component body)', () => {
    const src = 'component App() { const a = 1\nconst a = 2\ndiv(a) }';
    const codes = diag(src, vdomProfile).map((d) => d.code);
    expect(codes.filter((c) => c === 'ML-LANG-REDECL')).toHaveLength(1);
  });

  it('does NOT double-report a top-level redecl (parser already emits one)', () => {
    // The parser emits exactly one ML-LANG-REDECL for a top-level double-declaration; our pass must add none.
    const codes = diag('const a = 1\nconst a = 2').map((d) => d.code);
    expect(codes.filter((c) => c === 'ML-LANG-REDECL')).toHaveLength(1);
  });

  it('does NOT flag a nested-if declaration sharing a name with the enclosing block', () => {
    // ScopeModel flattens the if-block binding into the component-body scope span, but the evaluator
    // gives the if-block its own frame — so this is NOT a redecl. (Interpreter: no ML-LANG-REDECL.)
    const src = 'component App() { const a = 1\nif (true) { const a = 2 }\ndiv(a) }';
    const codes = diag(src, vdomProfile).map((d) => d.code);
    expect(codes.filter((c) => c === 'ML-LANG-REDECL')).toHaveLength(0);
  });

  it('does NOT flag same-named consts in two sibling if-blocks (separate frames)', () => {
    const src = 'component App() { if (true) { const a = 1 }\nif (true) { const a = 2 }\ndiv(a) }';
    const codes = diag(src, vdomProfile).map((d) => d.code);
    expect(codes.filter((c) => c === 'ML-LANG-REDECL')).toHaveLength(0);
  });
});
