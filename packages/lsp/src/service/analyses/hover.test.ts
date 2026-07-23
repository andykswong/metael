import { describe, it, expect } from 'vitest';
import { LanguageService } from '../index.ts';
import { composeProfiles } from '@metael/lang/profile';
import type { Profile } from '@metael/lang/profile';
import { mathProfile } from '@metael/math/lang';
import { stdProfile } from '@metael/std';

const p = composeProfiles(mathProfile, stdProfile);

/** A profile carrying one builtin `blend` with a doc + named params, for the rich-card assertions. */
const richProfile: Profile = {
  id: 'rich',
  heads: new Map(),
  types: new Map(),
  builtins: new Map([
    ['blend', {
      name: 'blend', profile: 'core', portability: 'gpu-tolerant', takesClosure: false, arity: [2, 3],
      doc: 'Mixes two values by an optional weight.',
      returnDoc: 'the blended value',
      params: [
        { name: 'a', doc: 'the first value' },
        { name: 'b', doc: 'the second value' },
        { name: 'weight', optional: true, doc: 'blend factor in [0,1]' },
      ],
    }],
  ]),
};

/** A profile carrying one `'cpu-only'` builtin, for the portability-prefix + returns assertions. */
const cpuOnlyProfile: Profile = {
  id: 'cpu-only',
  heads: new Map(),
  types: new Map(),
  builtins: new Map([
    ['join', {
      name: 'join', profile: 'host', portability: 'cpu-only', takesClosure: false, arity: [2, 2],
      doc: 'Joins an array of items into a single string.',
      returnDoc: 'the joined string',
      params: [
        { name: 'items', doc: 'the array to join' },
        { name: 'separator', doc: 'the text placed between items' },
      ],
    }],
  ]),
};

/** A profile carrying one `'exact'` builtin, for asserting NO portability prefix is rendered. */
const exactProfile: Profile = {
  id: 'exact',
  heads: new Map(),
  types: new Map(),
  builtins: new Map([
    ['abs', {
      name: 'abs', profile: 'core', portability: 'exact', takesClosure: false, arity: [1, 1],
      doc: 'The absolute value of a number.', params: [{ name: 'x', doc: 'the input' }],
    }],
  ]),
};

/** A profile carrying one head with documented params + a returnDoc, for the head-card assertions. */
const headProfile: Profile = {
  id: 'head',
  heads: new Map([
    ['gpu', {
      name: 'gpu', arity: [1, 2], returns: 'value',
      doc: 'Dispatch a map kernel over its inputs.',
      returnDoc: 'a reactive result handle',
      params: [
        { name: 'kernel', doc: 'the map component to run' },
        { name: 'cfg', optional: true, doc: 'the dispatch configuration' },
      ],
    }],
  ]),
  types: new Map(),
  builtins: new Map(),
};

/** A profile whose sole head is `a` (a hyperlink), for the local-binding-shadows-a-head assertions. */
const aHeadProfile: Profile = {
  id: 'a-head',
  heads: new Map([
    ['a', {
      name: 'a', arity: [0, Infinity], returns: 'node',
      doc: 'A hyperlink element.',
      returnDoc: 'a hyperlink node',
      params: [{ name: 'props', optional: true, doc: 'the attributes' }],
    }],
  ]),
  types: new Map(),
  builtins: new Map(),
};

describe('hover', () => {
  it('shows a builtin card for sqrt', () => {
    const svc = new LanguageService(); const src = 'const y = sqrt(2)';
    svc.openDocument('a', src, 1); svc.setProfile('a', p);
    const h = svc.hover('a', src.indexOf('sqrt') + 1);
    expect(h).not.toBeNull();
    expect(h!.markdown).toContain('sqrt');
  });
  it('renders a builtin doc + a param-named signature when the spec carries them', () => {
    const svc = new LanguageService(); const src = 'const y = blend(1, 2)';
    svc.openDocument('a', src, 1); svc.setProfile('a', richProfile);
    const h = svc.hover('a', src.indexOf('blend') + 1);
    expect(h).not.toBeNull();
    // A named signature, not the bare `blend(…)` placeholder.
    expect(h!.markdown).toContain('blend(a, b, weight)');
    expect(h!.markdown).not.toContain('blend(…)');
    // A gpu-tolerant builtin gets the compact portability prefix on its description.
    expect(h!.markdown).toContain('(gpu-tolerant) Mixes two values by an optional weight.');
    // The documented params render as an indented `  name — doc` list.
    expect(h!.markdown).toContain('  a — the first value');
    expect(h!.markdown).toContain('  b — the second value');
    expect(h!.markdown).toContain('  weight — blend factor in [0,1]');
    // The returnDoc renders as a `Returns …` line.
    expect(h!.markdown).toContain('Returns the blended value.');
    // No trace of the old classification line.
    expect(h!.markdown).not.toContain('builtin ·');
    expect(h!.markdown).not.toContain('arity');
  });
  it('renders a cpu-only builtin with a portability prefix, per-arg list, and a Returns line', () => {
    const svc = new LanguageService(); const src = 'const y = join([], ", ")';
    svc.openDocument('a', src, 1); svc.setProfile('a', cpuOnlyProfile);
    const h = svc.hover('a', src.indexOf('join') + 1);
    expect(h).not.toBeNull();
    expect(h!.markdown).toContain('join(items, separator)');
    expect(h!.markdown).toContain('(cpu-only) Joins an array of items into a single string.');
    expect(h!.markdown).toContain('  items — the array to join');
    expect(h!.markdown).toContain('  separator — the text placed between items');
    expect(h!.markdown).toContain('Returns the joined string.');
    // The old middle classification line is gone.
    expect(h!.markdown).not.toContain('builtin · portability');
    expect(h!.markdown).not.toContain('arity 2');
  });
  it('renders NO portability prefix for an exact builtin', () => {
    const svc = new LanguageService(); const src = 'const y = abs(-1)';
    svc.openDocument('a', src, 1); svc.setProfile('a', exactProfile);
    const h = svc.hover('a', src.indexOf('abs') + 1);
    expect(h).not.toBeNull();
    expect(h!.markdown).toContain('The absolute value of a number.');
    // `'exact'` carries no prefix — the description is bare.
    expect(h!.markdown).not.toContain('(exact)');
  });
  it('renders a nullary builtin as `name()`, not the `(…)` placeholder', () => {
    // A declared-but-empty params list is still a declaration → an explicit empty arg list, not `(…)`.
    const nullaryProfile: Profile = {
      id: 'nullary', heads: new Map(), types: new Map(),
      builtins: new Map([['tick', {
        name: 'tick', profile: 'host', portability: 'cpu-only', takesClosure: false, arity: [0, 0],
        doc: 'The current tick.', params: [],
      }]]),
    };
    const svc = new LanguageService(); const src = 'const y = tick()';
    svc.openDocument('a', src, 1); svc.setProfile('a', nullaryProfile);
    const h = svc.hover('a', src.indexOf('tick') + 1);
    expect(h).not.toBeNull();
    expect(h!.markdown).toContain('tick()');
    expect(h!.markdown).not.toContain('tick(…)');
  });
  it('renders a head card with a per-arg list + a Returns line, and no `Head · returns`', () => {
    const svc = new LanguageService(); const src = 'gpu(k)';
    svc.openDocument('a', src, 1); svc.setProfile('a', headProfile);
    const h = svc.hover('a', src.indexOf('gpu') + 1);
    expect(h).not.toBeNull();
    expect(h!.markdown).toContain('gpu(kernel, cfg)');
    expect(h!.markdown).toContain('Dispatch a map kernel over its inputs.');
    expect(h!.markdown).toContain('  kernel — the map component to run');
    expect(h!.markdown).toContain('  cfg — the dispatch configuration');
    expect(h!.markdown).toContain('Returns a reactive result handle.');
    // The old classification line is gone.
    expect(h!.markdown).not.toContain('Head · returns');
  });
  it('shows the binding card, not the head card, when a local binding shadows a same-named head', () => {
    // `a` is a head in this profile, but a `component a` binding shadows it — the evaluator resolves a
    // local binding before an injected head, so hover must show the component, not the hyperlink head.
    const svc = new LanguageService(); const src = 'component a() {}';
    svc.openDocument('a', src, 1); svc.setProfile('a', aHeadProfile);
    const h = svc.hover('a', src.indexOf('a'));
    expect(h).not.toBeNull();
    expect(h!.markdown).toContain('component a(');
    // None of the head card leaks through.
    expect(h!.markdown).not.toContain('A hyperlink element.');
    expect(h!.markdown).not.toContain('hyperlink node');
  });
  it('renders a component/function binding card with its parameter signature', () => {
    const svc = new LanguageService(); const src = 'component KPI(label, value) {}';
    svc.openDocument('a', src, 1); svc.setProfile('a', p);
    const h = svc.hover('a', src.indexOf('KPI') + 1);
    expect(h).not.toBeNull();
    expect(h!.markdown).toContain('component KPI(label, value)');

    const svc2 = new LanguageService(); const src2 = 'function fib(n) {}';
    svc2.openDocument('b', src2, 1); svc2.setProfile('b', p);
    const h2 = svc2.hover('b', src2.indexOf('fib') + 1);
    expect(h2).not.toBeNull();
    expect(h2!.markdown).toContain('function fib(n)');

    // A zero-parameter component still shows the empty parens.
    const svc3 = new LanguageService(); const src3 = 'component App() {}';
    svc3.openDocument('c', src3, 1); svc3.setProfile('c', p);
    const h3 = svc3.hover('c', src3.indexOf('App') + 1);
    expect(h3).not.toBeNull();
    expect(h3!.markdown).toContain('component App()');
  });
  it('renders a non-function binding card bare (no parens)', () => {
    const svc = new LanguageService(); const src = 'const xs = [1]';
    svc.openDocument('a', src, 1); svc.setProfile('a', p);
    const h = svc.hover('a', src.indexOf('xs') + 1);
    expect(h).not.toBeNull();
    expect(h!.markdown).toContain('const xs');
    expect(h!.markdown).not.toContain('const xs(');
  });
  it('renders a param binding card bare, and resolves the innermost shadow', () => {
    // A top-level `const x` shadowed by a `function f(x)` param: the inner `x` resolves to the param.
    const svc = new LanguageService(); const src = 'const x = 1\nfunction f(x) {\n  x\n}';
    svc.openDocument('a', src, 1); svc.setProfile('a', p);
    const innerUse = svc.hover('a', src.lastIndexOf('x'));
    expect(innerUse).not.toBeNull();
    expect(innerUse!.markdown).toContain('param x');
    expect(innerUse!.markdown).not.toContain('const x');
    // The top-level declaration still resolves to the const.
    const outer = svc.hover('a', src.indexOf('x'));
    expect(outer).not.toBeNull();
    expect(outer!.markdown).toContain('const x');
  });
  it('returns null in whitespace', () => {
    const svc = new LanguageService(); svc.openDocument('a', '   ', 1); svc.setProfile('a', p);
    expect(svc.hover('a', 1)).toBeNull();
  });
});
