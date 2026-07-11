// Grammar-independent diagnostic types.
export interface SourceSpan { readonly start: number; readonly end: number }
export type LiteralValue = number | string | boolean | null;

export interface Diagnostic {
  readonly code: string;      // ML-LANG-*, ML-IR-* …
  readonly message: string;
  readonly span?: SourceSpan;
}

export function makeDiagnostic(code: string, message: string, span?: SourceSpan): Diagnostic {
  return span ? { code, message, span } : { code, message };
}
