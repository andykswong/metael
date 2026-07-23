import type { Document } from '../document.ts';
import type { SvcDiagnostic, SvcSpan } from '../results.ts';

/** Surface the document's parse diagnostics — which already include the lexer's — as offset-based Svc
 *  records. Span-less diagnostics (e.g. budget) get a whole-document range; severity is derived from the
 *  ML-* code family. */
export function computeDiagnostics(doc: Document): readonly SvcDiagnostic[] {
  const whole: SvcSpan = { start: 0, end: doc.text.length };
  return doc.parse.diagnostics.map((d) => ({
    span: d.span ?? whole,
    severity: d.code.includes('PROFILE') ? 'info' : 'error',
    code: d.code,
    message: d.message,
  }));
}
