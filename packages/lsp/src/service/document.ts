import { lex, parseProgram } from '@metael/lang';
import type { LexResult, ParseProgramResult } from '@metael/lang';
import { LineIndex } from './line-index.ts';

/** One open source document at a given version. Lexing + parsing (both total in `@metael/lang`) run
 *  lazily and are memoized for the document's lifetime; an edit produces a fresh Document via `update`. */
export class Document {
  /** The document text at this version. */
  readonly text: string;
  /** The monotonic version supplied by the caller. */
  readonly version: number;
  /** Offset↔line/col mapper for this version. */
  readonly lineIndex: LineIndex;
  private _lex: LexResult | undefined;
  private _parse: ParseProgramResult | undefined;

  /** Build a document for `text` at `version`; lexing + parsing are deferred until first accessed. */
  constructor(text: string, version: number) {
    this.text = text;
    this.version = version;
    this.lineIndex = new LineIndex(text);
  }

  /** The (memoized) token stream plus any scan diagnostics. */
  get lex(): LexResult {
    return (this._lex ??= lex(this.text));
  }

  /** The (memoized) best-effort AST plus any lex/parse diagnostics. */
  get parse(): ParseProgramResult {
    return (this._parse ??= parseProgram(this.text));
  }

  /** Produce a fresh Document for an edited text + new version. */
  update(text: string, version: number): Document {
    return new Document(text, version);
  }
}
