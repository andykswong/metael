import { describe, it, expect } from 'vitest';
import { parseProgram, printProgram } from '@metael/lang';
import { LanguageService } from '../index.ts';
import { stdProfile } from '@metael/std';

/** Open a document under stdProfile and return its format edits. */
function formatOf(src: string) {
  const svc = new LanguageService();
  svc.openDocument('a', src, 1);
  svc.setProfile('a', stdProfile);
  return svc.format('a');
}

describe('format', () => {
  it('reprints via the canonical printer as one whole-document edit that re-parses clean', () => {
    const src = 'const  x=1';
    const edits = formatOf(src);
    expect(edits.length).toBe(1);
    const edit = edits[0]!;
    // A single edit replacing the entire source.
    expect(edit.span).toEqual({ start: 0, end: src.length });
    // Its text is exactly the canonical print of the parsed AST...
    const expected = printProgram(parseProgram(src).program);
    expect(edit.newText).toBe(expected);
    // ...and that text is real, clean source (the printer's conservation law).
    expect(parseProgram(edit.newText).diagnostics).toEqual([]);
  });

  it('returns [] when the text is already canonical (no-op)', () => {
    expect(formatOf('const x = 1')).toEqual([]);
  });

  it('returns [] for a program with a parse error (never reprints a broken AST)', () => {
    expect(formatOf('const x = (')).toEqual([]);
  });
});
