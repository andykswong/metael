import { describe, it, expect } from 'vitest';
import { styleObjectToCss, applyAttr } from './patch.ts';

describe('styleObjectToCss — serialize a style object to a CSS declaration string', () => {
  it('camelCases property names to kebab-case', () => {
    expect(styleObjectToCss({ backgroundColor: 'red', fontSize: '12px' }))
      .toBe('background-color: red; font-size: 12px');
  });
  it('leaves already-kebab and custom (--var) properties as-is', () => {
    expect(styleObjectToCss({ color: 'red', '--brand': '#f0f' }))
      .toBe('color: red; --brand: #f0f');
  });
  it('coerces non-string values with String()', () => {
    expect(styleObjectToCss({ opacity: 0.5, zIndex: 3 }))
      .toBe('opacity: 0.5; z-index: 3');
  });
  it('drops null / undefined / false entries (and true, which has no CSS meaning)', () => {
    expect(styleObjectToCss({ color: 'red', border: null, outline: undefined, margin: false, padding: true }))
      .toBe('color: red');
  });
  it('an empty (or all-dropped) object serializes to the empty string', () => {
    expect(styleObjectToCss({})).toBe('');
    expect(styleObjectToCss({ a: null, b: undefined })).toBe('');
  });
});

describe('applyAttr — object style value serializes; scalar path unchanged', () => {
  function makeEl(): Element & { _get(k: string): string | undefined } {
    const attrs = new Map<string, string>();
    return {
      setAttribute: (k: string, v: string) => { attrs.set(k, v); },
      removeAttribute: (k: string) => { attrs.delete(k); },
      _get: (k: string) => attrs.get(k),
    } as unknown as Element & { _get(k: string): string | undefined };
  }

  it('an object style value is serialized to CSS text (not [object Object])', () => {
    const el = makeEl();
    applyAttr(el, 'style', { color: 'red', fontSize: '12px' });
    expect(el._get('style')).toBe('color: red; font-size: 12px');
  });
  it('an empty-serializing object style REMOVES the attribute', () => {
    const el = makeEl();
    applyAttr(el, 'style', { color: 'red' });
    applyAttr(el, 'style', {});
    expect(el._get('style')).toBeUndefined();
  });
  it('a STRING style value is still set verbatim (scalar path unchanged)', () => {
    const el = makeEl();
    applyAttr(el, 'style', 'color: blue');
    expect(el._get('style')).toBe('color: blue');
  });
});
