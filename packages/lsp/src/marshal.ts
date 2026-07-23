// The sole offset↔Position + offset-based-result↔wire-type boundary. Every analysis result is offset-based
// and protocol-free; these pure converters map each one to its `vscode-languageserver-protocol` wire type,
// using a document's `LineIndex` to turn char offsets into 0-based UTF-16 `{ line, character }` positions.
// This file sits at `src/` (the protocol shell), so it may legitimately import the wire types the analysis
// engine under `src/service/` must not.

import { DiagnosticSeverity, CompletionItemKind } from 'vscode-languageserver-protocol';
import type {
  Diagnostic,
  CompletionItem,
  Hover,
  Range,
  Position,
  TextEdit,
  MarkupContent,
  SignatureHelp,
  SignatureInformation,
  ParameterInformation,
  FoldingRange,
  SelectionRange,
  SemanticTokens,
} from 'vscode-languageserver-protocol';
import type {
  LineIndex,
  SvcSpan,
  SvcDiagnostic,
  SvcCompletion,
  SvcHover,
  SvcEdit,
  SvcSeverity,
  SvcCompletionKind,
  SvcSignature,
  SvcFold,
  SvcSelection,
  SvcToken,
} from './service/index.ts';

/** An offset span → an LSP Range using the document's LineIndex. */
export function spanToRange(span: SvcSpan, li: LineIndex): Range {
  return { start: li.offsetToLineCol(span.start), end: li.offsetToLineCol(span.end) };
}

/** An LSP Position → a char offset. */
export function positionToOffset(pos: Position, li: LineIndex): number {
  return li.lineColToOffset(pos);
}

/** Maps each offset-based severity to its LSP `DiagnosticSeverity` numeric code. */
const SEVERITY: Record<SvcSeverity, DiagnosticSeverity> = {
  error: DiagnosticSeverity.Error,
  warning: DiagnosticSeverity.Warning,
  info: DiagnosticSeverity.Information,
  hint: DiagnosticSeverity.Hint,
};

/** An SvcDiagnostic → an LSP Diagnostic (tagged with the `metael` source). */
export function toDiagnostic(d: SvcDiagnostic, li: LineIndex): Diagnostic {
  return { range: spanToRange(d.span, li), severity: SEVERITY[d.severity], code: d.code, message: d.message, source: 'metael' };
}

/** Maps each offset-based completion kind to the LSP `CompletionItemKind` that picks its icon and ranking. */
const COMPLETION_KIND: Record<SvcCompletionKind, CompletionItemKind> = {
  keyword: CompletionItemKind.Keyword,
  variable: CompletionItemKind.Variable,
  function: CompletionItemKind.Function,
  parameter: CompletionItemKind.Variable,
  head: CompletionItemKind.Constructor,
  builtin: CompletionItemKind.Function,
  member: CompletionItemKind.Field,
  type: CompletionItemKind.Class,
};

/** An SvcCompletion → an LSP CompletionItem. */
export function toCompletionItem(c: SvcCompletion): CompletionItem {
  return { label: c.label, kind: COMPLETION_KIND[c.kind], detail: c.detail, documentation: c.doc, insertText: c.insert };
}

/** An SvcHover → an LSP Hover rendered as markdown over the described span. */
export function toHover(h: SvcHover, li: LineIndex): Hover {
  const contents: MarkupContent = { kind: 'markdown', value: h.markdown };
  return { contents, range: spanToRange(h.span, li) };
}

/** An SvcEdit → an LSP TextEdit. */
export function toTextEdit(e: SvcEdit, li: LineIndex): TextEdit {
  return { range: spanToRange(e.span, li), newText: e.newText };
}

/** An SvcSignature → an LSP SignatureHelp holding one signature with the active parameter selected. */
export function toSignatureHelp(s: SvcSignature): SignatureHelp {
  const parameters: ParameterInformation[] = s.params.map((p) => ({ label: p.label, documentation: p.doc }));
  const sig: SignatureInformation = { label: s.label, parameters, activeParameter: s.activeParam };
  return { signatures: [sig], activeSignature: 0, activeParameter: s.activeParam };
}

/** An SvcFold → a line-based LSP FoldingRange (start/end lines of the foldable region). */
export function toFoldingRange(f: SvcFold, li: LineIndex): FoldingRange {
  return { startLine: li.offsetToLineCol(f.start).line, endLine: li.offsetToLineCol(f.end).line };
}

/** An empty/zero LSP Range at the document origin, used when there is nothing to select. */
const EMPTY_RANGE: Range = { start: { line: 0, character: 0 }, end: { line: 0, character: 0 } };

/** An SvcSelection → the head of an LSP SelectionRange chain: the narrowest range, whose `parent` links
 *  widen outward. `sel.ranges` is narrowest-first/widest-last, so folding from the widest inward yields a
 *  chain whose head is the narrowest and whose outermost link (the widest) has no `parent`. */
export function toSelectionRange(sel: SvcSelection, li: LineIndex): SelectionRange {
  const head = sel.ranges.reduceRight<SelectionRange | undefined>(
    (parent, span) => (parent === undefined ? { range: spanToRange(span, li) } : { range: spanToRange(span, li), parent }),
    undefined,
  );
  return head ?? { range: EMPTY_RANGE };
}

/** Encode classified tokens as an LSP `SemanticTokens` full result — the standard 5-integer-per-token delta
 *  array `[deltaLine, deltaStartChar, length, tokenType, tokenModifiers]`, each relative to the previous
 *  token. Tokens are sorted by (line, startChar); the `tokenType` is `legend.indexOf(kind)`, and a token
 *  whose kind is absent from the legend is dropped. Modifiers are not encoded (always 0). */
export function encodeSemanticTokens(tokens: readonly SvcToken[], li: LineIndex, legend: readonly string[]): SemanticTokens {
  const positioned = tokens
    .map((t) => {
      const start = li.offsetToLineCol(t.span.start);
      const end = li.offsetToLineCol(t.span.end);
      // Length in UTF-16 units: the on-line char delta for single-line tokens, else the raw offset span.
      const length = end.line === start.line ? end.character - start.character : t.span.end - t.span.start;
      return { line: start.line, char: start.character, length, type: legend.indexOf(t.kind) };
    })
    .filter((t) => t.type >= 0)
    .sort((a, b) => (a.line - b.line) || (a.char - b.char));

  const data: number[] = [];
  let prevLine = 0;
  let prevChar = 0;
  for (const t of positioned) {
    const deltaLine = t.line - prevLine;
    const deltaChar = deltaLine === 0 ? t.char - prevChar : t.char;
    data.push(deltaLine, deltaChar, t.length, t.type, 0);
    prevLine = t.line;
    prevChar = t.char;
  }
  return { data };
}
