// Grammar-independent diagnostic types.

/**
 * A half-open range of the source text a {@link Diagnostic} refers to, as character offsets into the
 * original program string. Attached to AST nodes during parsing and carried through to any diagnostic
 * raised about that node, so a consumer can map an error back to the exact source location.
 */
export interface SourceSpan {
  /** Inclusive start offset (0-based character index into the source). */
  readonly start: number;
  /** Exclusive end offset (the character index one past the last character of the span). */
  readonly end: number;
}

/** The set of primitive literal values the language admits directly in source: a number, a string, a
 *  boolean, or `null`. */
export type LiteralValue = number | string | boolean | null;

/**
 * A structured, non-throwing report of something that happened during parsing or evaluation. The
 * language kernel is total: instead of raising, it collects diagnostics and continues, so a single
 * malformed construct never aborts the whole run.
 */
export interface Diagnostic {
  /** A stable machine-readable code identifying the diagnostic category (an `ML-*` identifier, e.g.
   *  `ML-LANG-UNKNOWN-CALL` or `ML-LANG-BUDGET`), suitable for programmatic handling. */
  readonly code: string;
  /** A human-readable description of what went wrong, in the language's own terms. */
  readonly message: string;
  /** The source range this diagnostic refers to ({@link SourceSpan}), when a location is known. Omitted
   *  for diagnostics that are not tied to a specific span (e.g. a whole-run budget exhaustion). */
  readonly span?: SourceSpan;
}

/**
 * Construct a {@link Diagnostic}, omitting the `span` field entirely when no location is supplied.
 *
 * @param code - the stable `ML-*` category code.
 * @param message - the human-readable description.
 * @param span - the source range the diagnostic refers to; omitted for location-less diagnostics.
 * @returns a frozen-shape diagnostic record carrying `span` only when one was provided.
 */
export function makeDiagnostic(code: string, message: string, span?: SourceSpan): Diagnostic {
  return span ? { code, message, span } : { code, message };
}
