import { describe, it, expect } from 'vitest';
import { makeDiagnostic, type Diagnostic, type SourceSpan } from './diagnostics.ts';

describe('diagnostics', () => {
  it('makeDiagnostic carries a ML-* code, message, and span', () => {
    const span: SourceSpan = { start: 0, end: 3 };
    const d: Diagnostic = makeDiagnostic('ML-LANG-PARSE', 'unexpected token', span);
    expect(d.code).toBe('ML-LANG-PARSE');
    expect(d.span).toEqual(span);
  });
});
