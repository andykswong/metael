import { describe, it, expect } from 'vitest';
import { Document, ScopeModel } from './index.ts';

/** Build a ScopeModel over a fresh document and return the visible binding names at `offset`. */
function namesAt(src: string, offset: number): string[] {
  return new ScopeModel(new Document(src, 1)).visibleAt(offset).map((b) => b.name);
}

describe('ScopeModel', () => {
  it('makes a top-level const visible across the whole program', () => {
    const src = 'const x = 1\nconst y = 2';
    expect(namesAt(src, 0)).toContain('x');
    expect(namesAt(src, src.length)).toContain('x');
    expect(namesAt(src, src.length)).toContain('y');
  });

  it('scopes a function param to the body — visible inside, not at a following sibling', () => {
    const src = 'function f(a) {\n  const b = a\n}\nconst z = 1';
    const insideBody = src.indexOf('const b') + 2; // within the function body
    const atSibling = src.indexOf('const z') + 2; // after the function, outside its scope
    expect(namesAt(src, insideBody)).toContain('a');
    expect(namesAt(src, insideBody)).toContain('f');
    expect(namesAt(src, atSibling)).not.toContain('a');
    expect(namesAt(src, atSibling)).toContain('f');
    expect(namesAt(src, atSibling)).toContain('z');
  });

  it('scopes a for-of binding to the loop body', () => {
    const src = 'for (item of items) {\n  const q = item\n}';
    const insideBody = src.indexOf('const q') + 2;
    expect(namesAt(src, insideBody)).toContain('item');
    expect(namesAt(src, insideBody)).toContain('q');
  });

  it('scopes an arrow param buried in a call to the arrow body', () => {
    const src = 'const r = items.map(x => x)';
    const insideArrow = src.indexOf('=> x') + 4; // within the arrow body
    const visInside = namesAt(src, insideArrow);
    expect(visInside).toContain('x');
    expect(new ScopeModel(new Document(src, 1)).visibleAt(insideArrow).find((b) => b.name === 'x')?.kind).toBe('param');
    expect(namesAt(src, 0)).not.toContain('x'); // not visible outside the arrow
  });

  it('stays usable on a partial parse without throwing', () => {
    const src = 'function f(a) { const b = ';
    const model = new ScopeModel(new Document(src, 1));
    expect(() => model.visibleAt(10)).not.toThrow();
    expect(() => model.allBindings()).not.toThrow();
  });

  it('keeps a component param + local visible on a blank line before the closing brace', () => {
    const src = 'component App() {\n  const total = 5\n\n}';
    const closeBrace = src.lastIndexOf('}');
    const blankLine = closeBrace - 1; // the blank line's newline, before the `}`
    expect(namesAt(src, blankLine)).toContain('total');
    expect(namesAt(src, closeBrace)).toContain('total'); // inclusive of the `}` offset itself
  });

  it('keeps a function param and local visible on a blank line before the closing brace', () => {
    const src = 'function f(xyz) {\n  const a = 1\n\n}';
    const closeBrace = src.lastIndexOf('}');
    const blankLine = closeBrace - 1;
    expect(namesAt(src, blankLine)).toContain('xyz'); // param
    expect(namesAt(src, blankLine)).toContain('a'); // local
    expect(namesAt(src, closeBrace)).toContain('xyz');
    expect(namesAt(src, closeBrace)).toContain('a');
  });

  it('keeps a for-of body binding visible on a blank line before the closing brace', () => {
    const src = 'for (item of items) {\n  const q = item\n\n}';
    const closeBrace = src.lastIndexOf('}');
    const blankLine = closeBrace - 1;
    expect(namesAt(src, blankLine)).toContain('item');
    expect(namesAt(src, blankLine)).toContain('q');
    expect(namesAt(src, closeBrace)).toContain('item');
  });

  it('keeps a block-bodied arrow param and local visible on a blank line before the closing brace', () => {
    const src = 'const f = (n) => {\n  const b = n\n\n}';
    const closeBrace = src.lastIndexOf('}');
    const blankLine = closeBrace - 1;
    expect(namesAt(src, blankLine)).toContain('n'); // param
    expect(namesAt(src, blankLine)).toContain('b'); // local
    expect(namesAt(src, closeBrace)).toContain('n');
  });

  it('does not leak a body binding past the closing brace to a sibling', () => {
    const src = 'function f(a) {\n  const b = 1\n\n}\nconst z = 1';
    const atSibling = src.indexOf('const z') + 2;
    expect(namesAt(src, atSibling)).not.toContain('a');
    expect(namesAt(src, atSibling)).not.toContain('b');
    expect(namesAt(src, atSibling)).toContain('f');
    expect(namesAt(src, atSibling)).toContain('z');
  });
});
