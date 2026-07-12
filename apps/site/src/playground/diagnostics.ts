// Diagnostics presentation helpers. One author error can cascade into several diagnostics that share a
// span (e.g. repeated ML-LANG-PARSE at the same offset), so dedupe by (code, span) and cap the visible
// list. spanToLineCol maps a UTF-16 offset to 1-based {line, col} for the error list + the overlay squiggle.
import type { Diagnostic } from '@metael/lang';

const DEFAULT_CAP = 5;

/** Dedupe by (code, span.start, span.end). Span-less diagnostics dedupe by code alone. Order preserved. */
export function dedupeDiagnostics(diags: readonly Diagnostic[]): Diagnostic[] {
  const seen = new Set<string>();
  const out: Diagnostic[] = [];
  for (const d of diags) {
    const k = d.span ? `${d.code}@${d.span.start}:${d.span.end}` : `${d.code}@nospan`;
    if (seen.has(k)) continue;
    seen.add(k);
    out.push(d);
  }
  return out;
}

export interface DiagnosticView {
  readonly shown: Diagnostic[];
  readonly overflow: number;   // how many were hidden past the cap
  readonly total: number;      // deduped total
}

/** Dedupe + cap for display. `overflow` drives an "and N more" line. */
export function diagnosticView(diags: readonly Diagnostic[], cap: number = DEFAULT_CAP): DiagnosticView {
  const deduped = dedupeDiagnostics(diags);
  return { shown: deduped.slice(0, cap), overflow: Math.max(0, deduped.length - cap), total: deduped.length };
}

/** 1-based line & column for a UTF-16 offset into `src`. Offsets past the end clamp to the last position. */
export function spanToLineCol(src: string, offset: number): { line: number; col: number } {
  const clamped = Math.max(0, Math.min(offset, src.length));
  let line = 1;
  let lastNewline = -1;
  for (let i = 0; i < clamped; i++) {
    if (src.charCodeAt(i) === 10 /* \n */) { line++; lastNewline = i; }
  }
  return { line, col: clamped - lastNewline };
}
