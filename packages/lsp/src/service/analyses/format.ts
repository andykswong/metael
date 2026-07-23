import { printProgram } from '@metael/lang';
import type { Document } from '../document.ts';
import type { SvcEdit } from '../results.ts';

/**
 * Whole-document format via the language's canonical printer.
 *
 * @remarks
 * Reprints {@link Document.parse}'s AST with `printProgram` and returns a single edit replacing the
 * entire source `[0, text.length)`. Two guards keep the result safe and quiet:
 *
 * 1. **No reprint of a broken AST.** When the parse carries any `ML-LANG-PARSE`/`ML-LANG-LEX`
 *    diagnostic the best-effort AST is incomplete, so formatting would silently drop or mangle source;
 *    the function returns `[]` instead.
 * 2. **No-op suppression.** When the reprint already equals the current text there is nothing to change,
 *    so it returns `[]` rather than a churn edit.
 *
 * The printer's conservation law (`parseProgram(printProgram(ast))` re-parses to the same stripped AST)
 * makes the single whole-document replacement safe. Pure and total.
 */
export function computeFormat(doc: Document): readonly SvcEdit[] {
  if (doc.parse.diagnostics.some((d) => d.code.startsWith('ML-LANG-PARSE') || d.code.startsWith('ML-LANG-LEX'))) return [];
  const newText = printProgram(doc.parse.program);
  if (newText === doc.text) return [];
  return [{ span: { start: 0, end: doc.text.length }, newText }];
}
