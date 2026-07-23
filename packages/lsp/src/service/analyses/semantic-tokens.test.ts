import { describe, it, expect } from 'vitest';
import { LanguageService } from '../index.ts';
import { composeProfiles } from '@metael/lang/profile';
import { mathProfile } from '@metael/math/lang';
import { stdProfile } from '@metael/std';
import { vdomProfile } from '@metael/vdom/lang';
import type { SvcToken, SvcTokenKind } from '../results.ts';

const p = composeProfiles(mathProfile, stdProfile, vdomProfile);

/** The kind of the token whose span begins at `start`, or `undefined` if none begins there. */
function kindAt(tokens: readonly SvcToken[], start: number): SvcTokenKind | undefined {
  return tokens.find((t) => t.span.start === start)?.kind;
}

describe('semanticTokens', () => {
  it('classifies keyword, builtin, head, and local-variable idents', () => {
    const svc = new LanguageService();
    const src = 'component App() {\n  const items = map(xs, f)\n  div { }\n}';
    svc.openDocument('a', src, 1); svc.setProfile('a', p);
    const toks = svc.semanticTokens('a');
    expect(toks.length).toBeGreaterThan(0);

    expect(kindAt(toks, src.indexOf('component'))).toBe('keyword');
    expect(kindAt(toks, src.indexOf('const'))).toBe('keyword');
    expect(kindAt(toks, src.indexOf('map'))).toBe('builtin');
    expect(kindAt(toks, src.indexOf('div'))).toBe('head');
    expect(kindAt(toks, src.indexOf('items'))).toBe('variable');
  });

  it('classifies a component declaration name as a function', () => {
    const svc = new LanguageService();
    const src = 'component App() { }';
    svc.openDocument('a', src, 1); svc.setProfile('a', p);
    const toks = svc.semanticTokens('a');
    expect(kindAt(toks, src.indexOf('App'))).toBe('function');
  });

  it('colours a local binding that shadows a profile head as the binding, not the head', () => {
    // `a` is a vdom head (hyperlink); a `component a` declaration shadows it, matching the evaluator
    // (local bindings resolve before injected heads), so the reference must colour as the component.
    const svc = new LanguageService();
    const src = 'component a() {\n  a\n}';
    svc.openDocument('a', src, 1); svc.setProfile('a', p);
    const toks = svc.semanticTokens('a');
    // The declaration name and the body reference both resolve to the binding, not the `a` head.
    expect(kindAt(toks, src.indexOf('a'))).toBe('function');
    expect(kindAt(toks, src.lastIndexOf('a'))).toBe('function');
  });

  it('colours a param that shadows a profile builtin as the parameter, not the builtin', () => {
    // `map` is a std builtin; a `map` parameter shadows it inside the function body.
    const svc = new LanguageService();
    const src = 'function f(map) {\n  map\n}';
    svc.openDocument('a', src, 1); svc.setProfile('a', p);
    const toks = svc.semanticTokens('a');
    expect(kindAt(toks, src.indexOf('map'))).toBe('parameter');
    expect(kindAt(toks, src.lastIndexOf('map'))).toBe('parameter');
  });

  it('classifies string, number, operator, and punctuation tokens by type', () => {
    const svc = new LanguageService();
    const src = 'const x = 1 + "hi"';
    svc.openDocument('a', src, 1); svc.setProfile('a', p);
    const toks = svc.semanticTokens('a');
    expect(kindAt(toks, src.indexOf('1'))).toBe('number');
    expect(kindAt(toks, src.indexOf('+'))).toBe('operator');
    expect(kindAt(toks, src.indexOf('"hi"'))).toBe('string');
    expect(kindAt(toks, src.indexOf('='))).toBe('operator');
  });

  it('returns an empty list when no profile is set', () => {
    const svc = new LanguageService();
    svc.openDocument('a', 'const x = 1', 1);
    expect(svc.semanticTokens('a')).toEqual([]);
  });

  it('classifies a `//` line comment as a comment token over its span', () => {
    const svc = new LanguageService();
    const src = 'const x = 1 // note';
    svc.openDocument('a', src, 1); svc.setProfile('a', p);
    const toks = svc.semanticTokens('a');
    const commentStart = src.indexOf('//');
    const comment = toks.find((t) => t.span.start === commentStart);
    expect(comment?.kind).toBe('comment');
    expect(comment?.span).toEqual({ start: commentStart, end: src.length });
  });
});
