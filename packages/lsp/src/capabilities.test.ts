import { describe, it, expect } from 'vitest';
import type { SvcTokenKind } from '@metael/lsp/service';
import { TOKEN_LEGEND } from './capabilities.ts';

// A TS union has no runtime reflection, so we enumerate the SvcTokenKind members as a literal here.
// This literal MUST mirror the `SvcTokenKind` union in `service/results.ts` — its ORDER is the wire
// contract (a client decodes each token's numeric type by indexing TOKEN_LEGEND, so index === kind).
// The `satisfies readonly SvcTokenKind[]` on TOKEN_LEGEND already rejects an entry that isn't a valid
// kind at compile time; these tests catch a kind that the union GREW but the legend never received, plus
// any reordering that would silently miscolour every token past the swapped index.
const EXPECTED_KINDS: SvcTokenKind[] = [
  'keyword',
  'string',
  'number',
  'variable',
  'function',
  'parameter',
  'head',
  'builtin',
  'type',
  'operator',
  'comment',
  'punctuation',
];

describe('TOKEN_LEGEND ↔ SvcTokenKind', () => {
  it('has the same count as the SvcTokenKind union', () => {
    expect(TOKEN_LEGEND.length).toBe(EXPECTED_KINDS.length);
  });

  it('covers exactly the SvcTokenKind set (no missing, no extra)', () => {
    expect(new Set(TOKEN_LEGEND)).toEqual(new Set(EXPECTED_KINDS));
  });

  it('preserves the wire-contract order (index === encoded token type)', () => {
    expect([...TOKEN_LEGEND]).toEqual(EXPECTED_KINDS);
  });

  it('has no duplicate entries (each index maps to a distinct kind)', () => {
    expect(new Set(TOKEN_LEGEND).size).toBe(TOKEN_LEGEND.length);
  });
});
