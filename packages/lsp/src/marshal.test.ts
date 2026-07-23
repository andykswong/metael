import { describe, it, expect } from 'vitest';
import { LineIndex } from '@metael/lsp/service';
import {
  spanToRange,
  toDiagnostic,
  encodeSemanticTokens,
  toSelectionRange,
} from './marshal.ts';

describe('marshal', () => {
  it('converts an offset span to an LSP Range via LineIndex', () => {
    const li = new LineIndex('ab\ncd');
    expect(spanToRange({ start: 3, end: 5 }, li)).toEqual({ start: { line: 1, character: 0 }, end: { line: 1, character: 2 } });
  });

  it('maps Svc severity to LSP DiagnosticSeverity', () => {
    const li = new LineIndex('abc');
    const d = toDiagnostic({ span: { start: 0, end: 3 }, severity: 'error', code: 'ML-LANG-PARSE', message: 'x' }, li);
    expect(d.severity).toBe(1); // DiagnosticSeverity.Error
    expect(d.code).toBe('ML-LANG-PARSE');
  });

  it('encodes a two-token doc as the LSP delta integer array', () => {
    // Doc 'ab\ncd': token A = keyword at [0,2) on line 0; token B = variable at [3,5) on line 1.
    const li = new LineIndex('ab\ncd');
    const legend = ['keyword', 'variable'];
    const tokens = [
      { span: { start: 3, end: 5 }, kind: 'variable' as const }, // deliberately out of order — encoder must sort
      { span: { start: 0, end: 2 }, kind: 'keyword' as const },
    ];
    const out = encodeSemanticTokens(tokens, li, legend);
    // [deltaLine, deltaStartChar, length, tokenType, tokenModifiers] per token, previous-relative:
    //   A: (0,0), len 2, type keyword=0 → [0,0,2,0,0]
    //   B: line 1 vs prev 0 → deltaLine 1, deltaStartChar = startChar 0, len 2, type variable=1 → [1,0,2,1,0]
    expect(out.data).toEqual([0, 0, 2, 0, 0, 1, 0, 2, 1, 0]);
  });

  it('drops tokens whose kind is absent from the legend', () => {
    const li = new LineIndex('abcd');
    const out = encodeSemanticTokens(
      [
        { span: { start: 0, end: 2 }, kind: 'keyword' as const },
        { span: { start: 2, end: 4 }, kind: 'comment' as const }, // not in legend
      ],
      li,
      ['keyword'],
    );
    expect(out.data).toEqual([0, 0, 2, 0, 0]);
  });

  it('builds a narrowest-first SelectionRange parent chain that widens outward', () => {
    // ranges are widening: narrowest first, widest last.
    const li = new LineIndex('abcde');
    const head = toSelectionRange({ ranges: [{ start: 2, end: 3 }, { start: 1, end: 4 }, { start: 0, end: 5 }] }, li);
    expect(head.range).toEqual({ start: { line: 0, character: 2 }, end: { line: 0, character: 3 } }); // narrowest
    expect(head.parent?.range).toEqual({ start: { line: 0, character: 1 }, end: { line: 0, character: 4 } }); // wider
    expect(head.parent?.parent?.range).toEqual({ start: { line: 0, character: 0 }, end: { line: 0, character: 5 } }); // widest
    expect(head.parent?.parent?.parent).toBeUndefined(); // outermost has no parent
  });
});
