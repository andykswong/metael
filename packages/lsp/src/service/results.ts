// The offset-based result records every analysis returns. Positions are char offsets (UTF-16 code-unit
// indices into the source string) or `{ start, end }` offset spans — no protocol types appear here, so the
// analysis engine stays protocol-free and the shell maps these to the wire types in one place.

/** A half-open range of char offsets `[start, end)` into a document's source string. */
export interface SvcSpan {
  /** The offset of the first char in the range (UTF-16 code-unit index, inclusive). */
  readonly start: number;
  /** The offset one past the last char in the range (UTF-16 code-unit index, exclusive). */
  readonly end: number;
}

/** How severe a diagnostic is, from most to least serious: a hard error, a warning, informational, or a hint. */
export type SvcSeverity = 'error' | 'warning' | 'info' | 'hint';

/** A single problem reported for a document — an error, warning, info note, or hint at a source span. */
export interface SvcDiagnostic {
  /** The source range the diagnostic applies to. */
  readonly span: SvcSpan;
  /** How severe the problem is. */
  readonly severity: SvcSeverity;
  /** A stable machine-readable identifier for the diagnostic (for filtering, docs, and quick-fix routing). */
  readonly code: string;
  /** The human-readable description of the problem. */
  readonly message: string;
}

/** The semantic category of a highlighted token, used to drive syntax colouring. */
export type SvcTokenKind =
  | 'keyword'
  | 'string'
  | 'number'
  | 'variable'
  | 'function'
  | 'parameter'
  | 'head'
  | 'builtin'
  | 'type'
  | 'operator'
  | 'comment'
  | 'punctuation';

/** One semantically classified token span, carrying its kind plus any modifiers for finer styling. */
export interface SvcToken {
  /** The source range the token covers. */
  readonly span: SvcSpan;
  /** The token's semantic category. */
  readonly kind: SvcTokenKind;
  /** Optional finer-grained styling flags (e.g. declaration, readonly) layered on top of the kind. */
  readonly modifiers?: readonly string[];
}

/** The semantic category of a completion candidate, used to pick its icon and ranking. */
export type SvcCompletionKind =
  | 'keyword'
  | 'variable'
  | 'function'
  | 'parameter'
  | 'head'
  | 'builtin'
  | 'member'
  | 'type';

/** A single completion candidate offered at a cursor position. */
export interface SvcCompletion {
  /** The text shown in the completion list and, by default, inserted when accepted. */
  readonly label: string;
  /** The candidate's semantic category. */
  readonly kind: SvcCompletionKind;
  /** Optional short detail shown beside the label (e.g. a type or signature summary). */
  readonly detail?: string;
  /** Optional longer documentation (markdown) shown in the expanded detail panel. */
  readonly doc?: string;
  /** Optional text to insert instead of `label` (e.g. a snippet or call template). */
  readonly insert?: string;
}

/** The hover content for a source position — markdown rendered over the span it describes. */
export interface SvcHover {
  /** The source range the hover describes. */
  readonly span: SvcSpan;
  /** The hover content, formatted as markdown. */
  readonly markdown: string;
}

/** One parameter within a signature, for signature-help display. */
export interface SvcParam {
  /** The parameter's label as it appears in the signature string. */
  readonly label: string;
  /** Optional documentation (markdown) for this parameter. */
  readonly doc?: string;
}

/** Signature help for a call — the rendered signature, its parameters, and which one is active. */
export interface SvcSignature {
  /** The full signature rendered as a single line (e.g. `fn(a, b)`). */
  readonly label: string;
  /** The parameters in order, aligned with substrings of `label`. */
  readonly params: readonly SvcParam[];
  /** The 0-based index of the parameter the cursor is currently within. */
  readonly activeParam: number;
}

/** A foldable region of the document, expressed as an offset range. */
export interface SvcFold {
  /** The offset of the first char of the foldable region. */
  readonly start: number;
  /** The offset one past the last char of the foldable region. */
  readonly end: number;
}

/** The set of selection ranges for one requested offset, ordered innermost-first and widening outward. */
export interface SvcSelection {
  /** The nested ranges from the tightest enclosing span to the widest (outermost-last). */
  readonly ranges: readonly SvcSpan[];
}

/** A single text edit — replace the text in `span` with `newText`. */
export interface SvcEdit {
  /** The source range to replace. */
  readonly span: SvcSpan;
  /** The replacement text. */
  readonly newText: string;
}

/** A code lens over a span, reporting whether the covered construct is lowerable and why (or why not). */
export interface SvcLens {
  /** The source range the lens annotates. */
  readonly span: SvcSpan;
  /** The label shown for the lens. */
  readonly label: string;
  /** Whether the covered construct can be lowered. */
  readonly lowerable: boolean;
  /** Optional explanations backing the `lowerable` verdict (e.g. what blocks lowering). */
  readonly reasons?: readonly string[];
}
