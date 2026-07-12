import { describe, it, expect } from 'vitest';
import type { Diagnostic } from '@metael/lang';
import { dedupeDiagnostics, diagnosticView, spanToLineCol } from './diagnostics.ts';

const d = (code: string, start?: number, end?: number): Diagnostic =>
  start === undefined ? { code, message: code } : { code, message: code, span: { start, end: end ?? start } };

describe('dedupeDiagnostics', () => {
  it('collapses identical (code, span) cascades to one', () => {
    const diags = [d('ML-LANG-PARSE', 16, 17), d('ML-LANG-PARSE', 16, 17), d('ML-LANG-PARSE', 16, 17)];
    expect(dedupeDiagnostics(diags)).toHaveLength(1);
  });
  it('keeps distinct spans and distinct codes', () => {
    const diags = [d('ML-LANG-PARSE', 16, 17), d('ML-LANG-PARSE', 20, 21), d('ML-LANG-LEX', 16, 17)];
    expect(dedupeDiagnostics(diags)).toHaveLength(3);
  });
  it('dedupes span-less diagnostics by code', () => {
    expect(dedupeDiagnostics([d('ML-LANG-NO-ENTRY'), d('ML-LANG-NO-ENTRY')])).toHaveLength(1);
  });
});

describe('diagnosticView', () => {
  it('caps and reports overflow', () => {
    const many = Array.from({ length: 8 }, (_, i) => d('ML-LANG-PARSE', i, i + 1));
    const v = diagnosticView(many, 5);
    expect(v.shown).toHaveLength(5);
    expect(v.overflow).toBe(3);
    expect(v.total).toBe(8);
  });
});

describe('spanToLineCol', () => {
  it('maps offsets to 1-based line/col', () => {
    const src = 'ab\ncde\nf';
    expect(spanToLineCol(src, 0)).toEqual({ line: 1, col: 1 });
    expect(spanToLineCol(src, 3)).toEqual({ line: 2, col: 1 }); // 'c'
    expect(spanToLineCol(src, 7)).toEqual({ line: 3, col: 1 }); // 'f'
  });
  it('clamps out-of-range offsets', () => {
    expect(spanToLineCol('abc', 99)).toEqual({ line: 1, col: 4 });
  });
});
