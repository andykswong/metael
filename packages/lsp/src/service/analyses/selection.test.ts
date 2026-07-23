import { describe, it, expect } from 'vitest';
import { LanguageService } from '../index.ts';
import { stdProfile } from '@metael/std';

/** Open a document under stdProfile and return the selection ranges at the given offsets. */
function selectAt(src: string, offsets: readonly number[]) {
  const svc = new LanguageService();
  svc.openDocument('a', src, 1);
  svc.setProfile('a', stdProfile);
  return svc.selectionRanges('a', offsets);
}

describe('selectionRanges', () => {
  it('widens from an inner expression outward through its enclosing spans', () => {
    const src = 'const total = 1 + 2 * 3';
    const offset = src.indexOf('2');
    const [sel] = selectAt(src, [offset]);
    expect(sel).toBeDefined();
    expect(sel!.ranges.length).toBeGreaterThanOrEqual(2);
    // Every range contains the offset, and each successive range strictly contains the previous.
    for (const r of sel!.ranges) {
      expect(r.start).toBeLessThanOrEqual(offset);
      expect(r.end).toBeGreaterThanOrEqual(offset);
    }
    for (let i = 1; i < sel!.ranges.length; i++) {
      const prev = sel!.ranges[i - 1]!;
      const cur = sel!.ranges[i]!;
      expect(cur.start).toBeLessThanOrEqual(prev.start);
      expect(cur.end).toBeGreaterThanOrEqual(prev.end);
      expect(cur.end - cur.start).toBeGreaterThan(prev.end - prev.start);
    }
    // The widest range is the whole document.
    const widest = sel!.ranges[sel!.ranges.length - 1]!;
    expect(widest.start).toBe(0);
    expect(widest.end).toBe(src.length);
  });

  it('returns one SvcSelection per requested offset, in order', () => {
    const src = 'const a = 1\nconst b = 2';
    const sels = selectAt(src, [src.indexOf('1'), src.indexOf('2')]);
    expect(sels.length).toBe(2);
    // Each offset's innermost range hugs the offset it was asked about.
    expect(sels[0]!.ranges[0]!.start).toBeLessThanOrEqual(src.indexOf('1'));
    expect(sels[1]!.ranges[0]!.start).toBeLessThanOrEqual(src.indexOf('2'));
    expect(sels[1]!.ranges[0]!.start).toBeGreaterThan(sels[0]!.ranges[0]!.end - 1);
  });
});
