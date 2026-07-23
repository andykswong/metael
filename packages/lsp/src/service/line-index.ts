/** A 0-based line + UTF-16 `character` position (the LSP protocol's default position encoding). */
export interface LineCol {
  /** The 0-based line number. */
  readonly line: number;
  /** The 0-based UTF-16 code-unit offset from the start of the line. */
  readonly character: number;
}

/** Precomputed line boundaries for a document version, converting between a char offset (a UTF-16 code
 *  unit index into the source string) and an LSP-style `{ line, character }`. Built once per version. */
export class LineIndex {
  /** `lineStarts[i]` is the offset of the first char of line `i`. */
  private readonly lineStarts: readonly number[];
  private readonly length: number;

  /** Build the line-boundary table for a document's full source `text`. */
  constructor(text: string) {
    this.length = text.length;
    const starts = [0];
    for (let i = 0; i < text.length; i++) if (text.charCodeAt(i) === 10 /* \n */) starts.push(i + 1);
    this.lineStarts = starts;
  }

  /** Convert a char offset to a 0-based line/character; offsets past the end clamp to the last position. */
  offsetToLineCol(offset: number): LineCol {
    const o = Math.max(0, Math.min(offset, this.length));
    // binary search for the greatest lineStart <= o
    let lo = 0, hi = this.lineStarts.length - 1;
    while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (this.lineStarts[mid]! <= o) lo = mid; else hi = mid - 1; }
    return { line: lo, character: o - this.lineStarts[lo]! };
  }

  /** Convert a 0-based line/character back to a char offset (clamped to the document). */
  lineColToOffset(pos: LineCol): number {
    const line = Math.max(0, Math.min(pos.line, this.lineStarts.length - 1));
    const base = this.lineStarts[line]!;
    const next = line + 1 < this.lineStarts.length ? this.lineStarts[line + 1]! : this.length + 1;
    return Math.min(base + Math.max(0, pos.character), next - 1, this.length);
  }
}
