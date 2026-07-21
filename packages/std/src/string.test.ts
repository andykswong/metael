import { describe, it, expect } from 'vitest';
import { evaluateProgram, PlainStorageHost, RecordingHostEnv } from '@metael/lang';
import { STD_BUILTINS } from './index.ts';

const run = (src: string, data?: unknown) =>
  evaluateProgram(src, { data, host: new PlainStorageHost(), env: new RecordingHostEnv(), builtins: [STD_BUILTINS] });

describe('std string builtins', () => {
  it('split splits on a separator; an empty separator yields code points', () => {
    expect(run('split("a,b,c", ",")').value).toEqual(['a', 'b', 'c']);
    expect(run('split("abc", "")').value).toEqual(['a', 'b', 'c']);
  });
  it('join concatenates with a separator', () => {
    expect(run('join(["a", "b", "c"], "-")').value).toBe('a-b-c');
    expect(run('join([1, 2, 3], ", ")').value).toBe('1, 2, 3');
  });
  it('chars returns an array of code points', () => {
    expect(run('chars("abc")').value).toEqual(['a', 'b', 'c']);
  });
  it('toUpperCase / toLowerCase / trim transform strings', () => {
    expect(run('toUpperCase("abc")').value).toBe('ABC');
    expect(run('toLowerCase("ABC")').value).toBe('abc');
    expect(run('trim("  hi  ")').value).toBe('hi');
  });
  it('format renders a fixed-point string', () => {
    expect(run('format(3.14159, 2)').value).toBe('3.14');
    expect(run('format(1, 0)').value).toBe('1');
  });
  it('format rejects a non-number / bad digits with ML-LANG-BUILTIN-ARG', () => {
    expect(run('format("x", 2)').diagnostics.some((d) => d.code === 'ML-LANG-BUILTIN-ARG')).toBe(true);
    expect(run('format(1, -1)').diagnostics.some((d) => d.code === 'ML-LANG-BUILTIN-ARG')).toBe(true);
  });

  // --- NEW: codePointAt ---
  it('codePointAt returns the code point at an index (String.prototype.codePointAt)', () => {
    expect(run('codePointAt("ABC", 0)').value).toBe(65);
    expect(run('codePointAt("ABC", 2)').value).toBe(67);
  });
  it('codePointAt of an astral character returns the full code point', () => {
    // "😀" is U+1F600 (128512), a surrogate pair — codePointAt(0) yields the whole code point.
    expect(run('codePointAt("😀", 0)').value).toBe(128512);
  });
  it('codePointAt out of range returns null (undefined mapped to null)', () => {
    expect(run('codePointAt("A", 5)').value).toBe(null);
  });
  it('codePointAt of a non-string is fail-loud ML-LANG-BUILTIN-ARG', () => {
    expect(run('codePointAt(5, 0)').diagnostics.some((d) => d.code === 'ML-LANG-BUILTIN-ARG')).toBe(true);
  });

  // --- NEW: string slice ---
  it('slice on a string returns a substring with array-style clamp semantics', () => {
    expect(run('slice("hello", 1, 3)').value).toBe('el');
    expect(run('slice("hello", -2)').value).toBe('lo');
    expect(run('slice("hello", 1)').value).toBe('ello');
  });
});
