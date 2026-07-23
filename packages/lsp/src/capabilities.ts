// The server's advertised feature set + the semantic-token legend. This file sits at `src/` (the protocol
// shell), so it legitimately imports the wire types the analysis engine under `src/service/` must not.
import type { ServerCapabilities } from 'vscode-languageserver-protocol';
import { TextDocumentSyncKind } from 'vscode-languageserver-protocol';
import type { SvcTokenKind } from './service/index.ts';

/** The fixed semantic-token type legend (index = the encoded token type). Order is the wire contract: it
 *  MUST match the `SvcTokenKind` order the encoder maps against, since a client decodes each token's numeric
 *  type by indexing this array. The `satisfies readonly SvcTokenKind[]` makes an entry that ISN'T an
 *  `SvcTokenKind` (a typo/renamed/extra kind) a compile error; a MISSING kind (union grew) is caught by
 *  the exact-set assertion in `capabilities.test.ts`. */
export const TOKEN_LEGEND = [
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
] as const satisfies readonly SvcTokenKind[];

/** The exact set of features this server advertises: full-text sync, completion (triggered on `.`/`(`),
 *  hover, signature help (triggered on `(`/`,`), folding, selection ranges, whole-document formatting, and
 *  full-document semantic tokens keyed to {@link TOKEN_LEGEND}. */
export const CAPABILITIES: ServerCapabilities = {
  textDocumentSync: TextDocumentSyncKind.Full,
  completionProvider: { triggerCharacters: ['.', '('], resolveProvider: false },
  hoverProvider: true,
  signatureHelpProvider: { triggerCharacters: ['(', ','] },
  foldingRangeProvider: true,
  selectionRangeProvider: true,
  documentFormattingProvider: true,
  semanticTokensProvider: { legend: { tokenTypes: [...TOKEN_LEGEND], tokenModifiers: [] }, full: true },
};
