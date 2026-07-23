import { describe, it, expect } from 'vitest';
import { StringStream } from '@codemirror/language';
import { metaelStreamParser } from './cold-highlight.ts';

// Drive the cold-start `StreamParser` over one line the way the CodeMirror `StreamLanguage` runtime does:
// a fresh StringStream per line, repeated `token()` calls until end-of-line, each returning the tag for the
// consumed run. Returns `[text, tag]` segments so a test can assert what colour each run paints in.
function segmentsOf(line: string): Array<[string, string | null]> {
  const state = metaelStreamParser.startState();
  const stream = new StringStream(line, 2, 2);
  const segments: Array<[string, string | null]> = [];
  let guard = 0;
  while (!stream.eol()) {
    const before = stream.pos;
    const tag = metaelStreamParser.token(stream, state);
    // The tokenizer MUST always advance to avoid an infinite loop.
    expect(stream.pos).toBeGreaterThan(before);
    segments.push([line.slice(before, stream.pos), tag]);
    if (++guard > line.length + 5) throw new Error('tokenizer did not advance');
  }
  return segments;
}

describe('cold-highlight stream parser', () => {
  it('colours a trailing `// comment` after tokens with the comment tag', () => {
    const segs = segmentsOf('let x = 1 // note');
    const comment = segs.find(([text]) => text.startsWith('//'));
    expect(comment).toEqual(['// note', 'comment']);
    // The code before the comment is still classified (the keyword `let` is a keyword).
    expect(segs.find(([text]) => text === 'let')?.[1]).toBe('keyword');
  });

  it('colours a full-line `// comment` with the comment tag', () => {
    const segs = segmentsOf('// a whole line');
    expect(segs).toContainEqual(['// a whole line', 'comment']);
    // Nothing on the line is coloured as anything but a comment (no stray token tags).
    expect(segs.filter(([, tag]) => tag !== null && tag !== 'comment')).toEqual([]);
  });

  it('leaves a comment-free line uncoloured in its gaps and never returns the comment tag', () => {
    const segs = segmentsOf('let x = 1');
    expect(segs.some(([, tag]) => tag === 'comment')).toBe(false);
    expect(segs.find(([text]) => text === 'let')?.[1]).toBe('keyword');
  });
});
