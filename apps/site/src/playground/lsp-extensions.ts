// The CodeMirror 6 extension set that turns an `LspClient` into live language intelligence in the editor:
// squiggles (lint), autocomplete, hover cards, semantic-token colouring, a capability-lens gutter, and a
// lightweight signature-help hint — plus the instant cold-start highlighter so colouring never lags the
// worker. Everything routes through the client's thin promise methods (which already reverse LSP ranges to
// CM6 `{from,to}` offsets), so this module owns only the CM6 wiring, no protocol.
import { EditorView, Decoration, ViewPlugin, hoverTooltip, gutter, GutterMarker } from '@codemirror/view';
import type { DecorationSet, Tooltip, TooltipView, ViewUpdate } from '@codemirror/view';
import { StateField, StateEffect, RangeSet } from '@codemirror/state';
import type { Extension, Range } from '@codemirror/state';
import { linter, setDiagnostics, lintGutter } from '@codemirror/lint';
import type { Diagnostic } from '@codemirror/lint';
import { autocompletion } from '@codemirror/autocomplete';
import type { CompletionContext, CompletionResult, Completion } from '@codemirror/autocomplete';
import { CompletionItemKind } from 'vscode-languageserver-protocol';
import type { LspClient, ClientDiagnostic, ClientSemanticToken, ClientLens } from './lsp-client.ts';
import { coldHighlight } from './cold-highlight.ts';

/** How long to wait after a doc change before re-fetching tokens/lenses — pure throttling, so a burst of
 *  keystrokes triggers one fetch rather than one per keystroke. Correctness no longer depends on this delay:
 *  the worker's document is synced SYNCHRONOUSLY on every change (see `documentSyncPlugin`) before the timer
 *  elapses, so a debounced fetch always queries the up-to-date text — short enough to still feel live. */
const FETCH_DEBOUNCE_MS = 120;

/** The CM6 lint severity string for an LSP numeric severity (1 = error … 4 = hint). */
function severityOf(n: number): Diagnostic['severity'] {
  switch (n) {
    case 2: return 'warning';
    case 3: return 'info';
    case 4: return 'hint';
    default: return 'error';
  }
}

// ── Hover / completion cards: a structured, syntax-coloured renderer ─────────────────────────────────────
// The server emits a light markdown card (a fenced signature, a portability-prefixed description, an indented
// `  name — doc` per-arg list, and a `Returns …` line). We parse it into distinct parts and build a separate
// styled DOM node per part so the card reads like an IDE hover (title / metadata / prose / args / returns),
// instead of one flat monochrome blob. HARD invariant: every text run enters the DOM via `textContent` /
// `createTextNode`, NEVER `innerHTML` — arbitrary user source flows near here, so nothing is ever parsed as
// markup. Inline `` `code` `` / `**bold**` are rendered by splitting the run and styling child spans, each
// still text-only. The parser is total: an unrecognised shape falls through to plain description lines, so it
// never throws and never drops content.

/** The parts parsed out of a hover-card markdown string. Every field degrades: a line matching no known
 *  shape lands in `descLines`, so nothing is lost even for a card shape the parser does not fully recognise. */
interface HoverParts {
  /** The fenced signature block (usually one line, e.g. `join(items, separator)`), or `null` when absent. */
  signature: string | null;
  /** A portability marker (`cpu-only` / `gpu-tolerant`) stripped from the head of the first description line. */
  portability: string | null;
  /** Description prose lines (portability marker already stripped from the first), each may carry inline runs. */
  descLines: string[];
  /** The documented parameters, in call order. */
  params: { name: string; doc: string }[];
  /** The text following `Returns ` on the trailing returns line (period kept), or `null`. */
  returns: string | null;
}

const PARAM_LINE = /^\s{2,}(.+?) — (.+)$/;        // `  name — doc` — two-space indent, em-dash (U+2014) split
const RETURNS_LINE = /^Returns (.+)$/;            // `Returns <text>.`
const PORTABILITY_PREFIX = /^\(([a-z][a-z0-9-]*)\)\s+/; // leading `(cpu-only) ` / `(gpu-tolerant) ` metadata

/** Parse the light hover markdown into styled-render parts. Total and injection-agnostic: it only splits
 *  text on line/marker boundaries, never interprets HTML. Any line matching no known shape falls through to
 *  `descLines`, so content is never dropped. */
function parseHoverCard(md: string): HoverParts {
  const parts: HoverParts = { signature: null, portability: null, descLines: [], params: [], returns: null };
  const sigLines: string[] = [];
  let inFence = false;
  for (const line of md.split('\n')) {
    if (/^\s*```/.test(line)) { inFence = !inFence; continue; }   // fence marker — toggle, never rendered
    if (inFence) { sigLines.push(line); continue; }
    if (line.trim() === '') continue;
    const pm = PARAM_LINE.exec(line);
    if (pm) { parts.params.push({ name: pm[1]!, doc: pm[2]! }); continue; }
    const rm = RETURNS_LINE.exec(line);
    if (rm && parts.returns === null) { parts.returns = rm[1]!; continue; }
    // A description / other line. Strip a leading portability marker from the FIRST such line only.
    if (parts.descLines.length === 0 && parts.portability === null) {
      const port = PORTABILITY_PREFIX.exec(line);
      if (port) { parts.portability = port[1]!; parts.descLines.push(line.slice(port[0].length)); continue; }
    }
    parts.descLines.push(line);
  }
  parts.signature = sigLines.join('\n').trim() || null;
  return parts;
}

/** Append `text` to `parent`, rendering inline `` `code` `` as a code-coloured span and `**bold**` as bold —
 *  by SPLITTING the string and appending one text node per plain run (never `innerHTML`), so any HTML in the
 *  source becomes literal text, not markup. */
function appendInline(parent: HTMLElement, text: string): void {
  const re = /`([^`]+)`|\*\*([^*]+?)\*\*/g;
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) parent.appendChild(document.createTextNode(text.slice(last, m.index)));
    if (m[1] !== undefined) {
      const code = document.createElement('span');
      code.className = 'cm-hover-code';
      code.textContent = m[1];
      parent.appendChild(code);
    } else {
      const bold = document.createElement('span');
      bold.style.fontWeight = '600';
      bold.textContent = m[2]!;
      parent.appendChild(bold);
    }
    last = m.index + m[0].length;
  }
  if (last < text.length) parent.appendChild(document.createTextNode(text.slice(last)));
}

/** Build a structured, syntax-coloured hover card from the server's light markdown: a monospace signature
 *  title, a metadata portability badge, prose description (with inline code/bold), an indented per-arg list,
 *  and a labelled Returns row — each a distinct styled node for IDE-like hierarchy. Injection-safe: every
 *  text run enters via `textContent` / `createTextNode`, never `innerHTML`. Degrades gracefully — an
 *  unrecognised shape renders its lines as plain description prose; it never throws and never drops content.
 *  Reused (with a lighter class) for the completion info panel, whose markdown is just a description line. */
export function renderHoverCard(md: string, cls = 'cm-hover-doc'): HTMLElement {
  const parts = parseHoverCard(md);
  const root = document.createElement('div');
  root.className = cls;

  if (parts.signature !== null) {
    const sig = document.createElement('div');
    sig.className = 'cm-hover-sig';
    sig.textContent = parts.signature;
    root.appendChild(sig);
  }

  parts.descLines.forEach((text, i) => {
    const desc = document.createElement('div');
    desc.className = 'cm-hover-desc';
    if (i === 0 && parts.portability !== null) {   // the badge prefixes only the first description line
      const badge = document.createElement('span');
      badge.className = 'cm-hover-portability';
      badge.textContent = parts.portability;
      desc.appendChild(badge);
    }
    appendInline(desc, text);
    root.appendChild(desc);
  });

  if (parts.params.length > 0) {
    const list = document.createElement('ul');
    list.className = 'cm-hover-params';
    for (const p of parts.params) {
      const li = document.createElement('li');
      const name = document.createElement('span');
      name.className = 'cm-hover-param-name';
      name.textContent = p.name;
      li.appendChild(name);
      li.appendChild(document.createTextNode(' — '));
      const doc = document.createElement('span');
      doc.className = 'cm-hover-param-doc';
      appendInline(doc, p.doc);
      li.appendChild(doc);
      list.appendChild(li);
    }
    root.appendChild(list);
  }

  if (parts.returns !== null) {
    const ret = document.createElement('div');
    ret.className = 'cm-hover-returns';
    const label = document.createElement('span');
    label.className = 'cm-hover-returns-label';
    label.textContent = 'Returns';
    ret.appendChild(label);
    ret.appendChild(document.createTextNode(' '));
    appendInline(ret, parts.returns);
    root.appendChild(ret);
  }

  return root;
}

/** Map an LSP `CompletionItemKind` to a CM6 completion `type` (drives the picker icon). */
function completionType(kind: number | undefined): string | undefined {
  switch (kind) {
    case CompletionItemKind.Function:
    case CompletionItemKind.Method:
      return 'function';
    case CompletionItemKind.Keyword:
      return 'keyword';
    case CompletionItemKind.Variable:
      return 'variable';
    case CompletionItemKind.Field:
    case CompletionItemKind.Property:
      return 'property';
    case CompletionItemKind.Constant:
      return 'constant';
    case CompletionItemKind.Class:
    case CompletionItemKind.Struct:
    case CompletionItemKind.Interface:
      return 'type';
    default:
      return undefined;
  }
}

/** Extract the plain-text body of an LSP `documentation` field (a string or a `MarkupContent`). */
function docText(documentation: unknown): string | undefined {
  if (typeof documentation === 'string') return documentation || undefined;
  if (documentation && typeof documentation === 'object' && 'value' in documentation) {
    const v = (documentation as { value: unknown }).value;
    return typeof v === 'string' && v ? v : undefined;
  }
  return undefined;
}

// ── Diagnostics: pushed by the worker, cached, replayed to the lint layer ────────────────────────────────
// The worker publishes diagnostics asynchronously (the analysis round-trip already happened). We cache the
// latest per document so CM6's `linter` source can return it synchronously for the gutter/panel, and the push
// handler ALSO dispatches `setDiagnostics` directly (see below) so the squiggles appear regardless of CM's
// idle-lint scheduler. This is the clean push-based pattern — no analysis request on each idle.
class DiagnosticStore {
  private latest: readonly ClientDiagnostic[] = [];
  set(diags: readonly ClientDiagnostic[]): void { this.latest = diags; }
  get(): readonly ClientDiagnostic[] { return this.latest; }
}

/** Convert the client's diagnostics to CM6's shape, clamping each span into `[0, docLength]` and forcing a
 *  non-empty range so the squiggle is visible even for a zero-width diagnostic at EOF. */
function toCmDiagnostics(diags: readonly ClientDiagnostic[], docLength: number): Diagnostic[] {
  return diags.map((d) => {
    const from = Math.min(d.from, docLength);
    const to = Math.min(Math.max(d.to, from + 1), docLength);
    return {
      from,
      to,
      severity: severityOf(d.severity),
      message: d.message,
      source: d.code != null ? String(d.code) : undefined,
    };
  });
}

// ── Semantic tokens: a doc-change-driven decoration layer ────────────────────────────────────────────────
const setSemanticTokens = StateEffect.define<DecorationSet>();

const semanticTokenField = StateField.define<DecorationSet>({
  create: () => Decoration.none,
  update(deco, tr) {
    // Map existing marks through the change first so they stay glued to their text until fresh tokens land;
    // a `setSemanticTokens` effect then replaces the whole set with the new response.
    let next = deco.map(tr.changes);
    for (const e of tr.effects) if (e.is(setSemanticTokens)) next = e.value;
    return next;
  },
  provide: (f) => EditorView.decorations.from(f),
});

/** Build a decoration set from decoded semantic tokens. Each mark is `cmt-<kind>` so the theme colours it. */
function tokensToDecorations(tokens: readonly ClientSemanticToken[], docLength: number): DecorationSet {
  const ranges: Range<Decoration>[] = [];
  for (const t of tokens) {
    // Drop empty or out-of-range spans (a stale response for older/longer text).
    if (t.from >= t.to || t.to > docLength) continue;
    ranges.push(Decoration.mark({ class: `cmt-${t.kind}` }).range(t.from, t.to));
  }
  return RangeSet.of(ranges, true);   // sort: marks may be unordered after filtering
}

// ── Capability-lens gutter ───────────────────────────────────────────────────────────────────────────────
class LensMarker extends GutterMarker {
  constructor(private readonly lens: ClientLens) { super(); }
  override eq(other: GutterMarker): boolean {
    return other instanceof LensMarker
      && other.lens.label === this.lens.label
      && other.lens.lowerable === this.lens.lowerable;
  }
  override toDOM(): Node {
    const el = document.createElement('span');
    el.className = `cm-lens ${this.lens.lowerable ? 'cm-lens-ok' : 'cm-lens-no'}`;
    el.textContent = this.lens.lowerable ? '●' : '○';   // ● lowerable, ○ not
    const why = this.lens.reasons.length ? `\n${this.lens.reasons.join('\n')}` : '';
    el.title = this.lens.lowerable ? this.lens.label : `${this.lens.label}${why}`;
    return el;
  }
}

const setLenses = StateEffect.define<RangeSet<GutterMarker>>();

const lensField = StateField.define<RangeSet<GutterMarker>>({
  create: () => RangeSet.empty,
  update(set, tr) {
    let next = set.map(tr.changes);
    for (const e of tr.effects) if (e.is(setLenses)) next = e.value;
    return next;
  },
});

// ── The theme: colours for the semantic-token marks + the lens gutter + the signature hint ───────────────
// Self-contained so the module ships its own styling (no dependency on where the app's CSS lives). Colours
// mirror the site's warm-phosphor syntax palette used by the static overlay in `styles.css`.
const lspTheme = EditorView.theme({
  '.cmt-keyword': { color: '#e8a33d', fontWeight: '500' },
  '.cmt-string': { color: '#9dd6a8' },
  '.cmt-number': { color: '#7fb7e6' },
  '.cmt-variable': { color: '#e9e4d8' },
  '.cmt-function': { color: '#e8a33d' },
  '.cmt-parameter': { color: '#f3c987', fontStyle: 'italic' },
  '.cmt-head': { color: '#6fb3d9', fontWeight: '500' },
  '.cmt-builtin': { color: '#e08fb4' },
  '.cmt-type': { color: '#7fb7e6' },
  '.cmt-operator': { color: '#e08fb4' },
  '.cmt-comment': { color: '#5b6377', fontStyle: 'italic' },
  '.cmt-punctuation': { color: '#7b8398' },
  '.cm-lens-gutter': { width: '1.2em', textAlign: 'center' },
  '.cm-lens': { fontSize: '0.72em', cursor: 'default' },
  '.cm-lens-ok': { color: '#7ec48b' },
  '.cm-lens-no': { color: '#e8a33d' },
  '.cm-signature-hint': {
    position: 'absolute', bottom: '4px', left: '8px', right: '8px',
    padding: '2px 8px', font: '12px var(--mono, monospace)',
    background: 'rgba(10,13,19,0.95)', color: '#e9e4d8',
    border: '1px solid #23293a', borderRadius: '6px', pointerEvents: 'none', zIndex: '5',
  },

  // ── Floating layers (autocomplete popup, hover card, lint tooltip) ────────────────────────────
  // These reparent under a tooltip host that CodeMirror tags with THIS editor's theme scope class, so
  // the rules stay scoped to this instance. Without them the popups inherit the page's sans-serif and
  // show a transparent, borderless box. Palette mirrors the dark well (ink-2 panel + cool hairline).
  '.cm-tooltip': {
    background: '#10141d', color: '#e9e4d8',
    border: '1px solid #2f3852', borderRadius: '9px',
    boxShadow: '0 10px 34px -12px rgba(0,0,0,0.75)',
    fontFamily: "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
    fontSize: '12.5px',
  },
  '.cm-tooltip.cm-tooltip-autocomplete': { padding: '0' },
  '.cm-tooltip.cm-tooltip-autocomplete > ul': {
    fontFamily: "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
    fontSize: '12.5px', maxHeight: '16em',
  },
  '.cm-tooltip.cm-tooltip-autocomplete > ul > li': {
    padding: '3px 10px 3px 6px', lineHeight: '1.5', color: '#e9e4d8',
  },
  '.cm-tooltip-autocomplete ul li[aria-selected]': {
    background: 'rgba(232,163,61,0.22)', color: '#f6efe2',
  },
  '.cm-completionLabel': { color: '#e9e4d8' },
  '.cm-completionMatchedText': { color: '#e8a33d', textDecoration: 'none', fontWeight: '600' },
  '.cm-completionDetail': { color: '#8b93a7', fontStyle: 'normal', marginLeft: '0.6em' },
  '.cm-completionIcon': { color: '#8b93a7', opacity: '0.85', paddingRight: '0.55em' },
  '.cm-tooltip.cm-completionInfo': {
    background: '#10141d', color: '#cfd3dd',
    border: '1px solid #2f3852', borderRadius: '9px', padding: '7px 10px',
    fontFamily: "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
    fontSize: '12px', lineHeight: '1.5', maxWidth: '22em',
  },
  // ── Structured hover / completion card ────────────────────────────────────────────────────────
  // A distinct styled node per part (signature title / portability badge / prose / arg list / returns)
  // for IDE-like visual hierarchy over the previously flat monochrome text.
  '.cm-hover-doc, .cm-completionInfo-doc': {
    fontFamily: "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
    fontSize: '12px', lineHeight: '1.5', color: '#cfd3dd', maxWidth: '30em', padding: '1px 1px',
  },
  // The signature line — the card title: code-coloured, a touch larger/bolder, with a hairline underline.
  '.cm-hover-sig': {
    color: '#6fb3d9', fontWeight: '600', fontSize: '13px', whiteSpace: 'pre-wrap',
    paddingBottom: '5px', marginBottom: '5px', borderBottom: '1px solid rgba(47,56,82,0.7)',
  },
  // Portability metadata as a small dim pill, set apart from the description prose.
  '.cm-hover-portability': {
    display: 'inline-block', color: '#8b93a7', fontSize: '10.5px', fontWeight: '600',
    letterSpacing: '0.02em', background: 'rgba(139,147,167,0.12)',
    border: '1px solid rgba(139,147,167,0.28)', borderRadius: '5px',
    padding: '0 5px', marginRight: '6px', verticalAlign: '1px',
  },
  '.cm-hover-desc': { color: '#cfd3dd', whiteSpace: 'pre-wrap', margin: '0 0 2px' },
  // Inline `code` runs in a green code colour, mirroring the string token.
  '.cm-hover-code': { color: '#9dd6a8' },
  // The per-arg list: real indented rows, param name in the parameter colour.
  '.cm-hover-params': {
    listStyle: 'none', margin: '5px 0 0', padding: '0',
    borderTop: '1px solid rgba(47,56,82,0.5)', paddingTop: '5px',
  },
  '.cm-hover-params li': { margin: '1px 0', lineHeight: '1.5', color: '#cfd3dd' },
  '.cm-hover-param-name': { color: '#f3c987', fontStyle: 'italic' },
  '.cm-hover-param-doc': { color: '#cfd3dd' },
  // The Returns row on its own line with a dim label and a little breathing room above.
  '.cm-hover-returns': { color: '#cfd3dd', marginTop: '5px' },
  '.cm-hover-returns-label': { color: '#8b93a7', fontWeight: '600', marginRight: '4px' },
  '.cm-tooltip.cm-tooltip-hover': { padding: '7px 10px' },
  // Lint severity colours matching the palette (error rose, warning amber).
  '.cm-diagnostic': {
    fontFamily: "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
    fontSize: '12px', padding: '3px 8px 3px 6px',
  },
  '.cm-diagnostic-error': { borderLeft: '3px solid #ff6b6b' },
  '.cm-diagnostic-warning': { borderLeft: '3px solid #e8a33d' },
});

/** Build the full CM6 extension set driving the editor's language intelligence from `client` for `uri`: the
 *  document-sync notifier, cold-start highlighter, semantic tokens, lint (+ gutter), autocomplete, hover, the
 *  capability-lens gutter, and a light signature-help hint. Fully self-contained — the caller only opens the
 *  document once (`didOpen`, at `openVersion`) and reconfigures the editor's compartment with this set; from
 *  there THIS module owns keeping the worker's text in sync, firing `didChange` synchronously on every edit
 *  (see `documentSyncPlugin`) so every debounced token/lens/lint fetch queries the up-to-date document.
 *
 *  @param openVersion the version the caller passed to `didOpen`; the sync plugin's monotonic counter starts
 *    just after it, so the first edit is `openVersion + 1`. */
export function lspExtensions(client: LspClient, uri: string, openVersion = 1): Extension[] {
  const store = new DiagnosticStore();

  // The single writer of the worker's document text after `didOpen`. It fires `didChange` SYNCHRONOUSLY on
  // every editor doc change — a cheap fire-and-forget JSON-RPC notification — BEFORE any of the debounced
  // token/lens/lint fetches below run. Because JSON-RPC preserves notification/request order, a subsequent
  // `semanticTokens`/`completion`/etc. request is therefore always answered against the just-synced text.
  // This closes the state-vs-painted-DOM race where a token fetch on its own timer could query the worker
  // before a separately-timed document sync had sent the new text, re-marking a stale (now-shorter) span.
  // Owning the version counter here (rather than in the caller) keeps a single monotonic sequence — no two
  // writers fighting over version numbers.
  let version = openVersion;
  const documentSyncPlugin = ViewPlugin.fromClass(
    class {
      update(u: ViewUpdate): void {
        if (u.docChanged) client.didChange(uri, u.state.doc.toString(), ++version);
      }
    },
  );

  // The worker's push is the source of truth for squiggles. It arrives asynchronously, AFTER CM6's own idle
  // lint run has already fired (so `forceLinting` would no-op — its plugin's pending flag is cleared). We
  // therefore make the push authoritative: cache the latest AND dispatch `setDiagnostics` directly so the
  // marks appear regardless of the lint scheduler. The `linter` source below still serves CM's idle re-lint
  // and gutter, returning the same cache. Owned by a ViewPlugin for the `view`; the client exposes no
  // listener-removal, so the callback is guarded by a `disposed` flag rather than unsubscribed on destroy.
  const diagnosticsPlugin = ViewPlugin.fromClass(
    class {
      private disposed = false;
      constructor(private readonly view: EditorView) {
        client.onDiagnostics((u, diags) => {
          if (this.disposed || u !== uri) return;
          store.set(diags);
          const cm = toCmDiagnostics(diags, this.view.state.doc.length);
          this.view.dispatch(setDiagnostics(this.view.state, cm));
        });
      }
      destroy(): void { this.disposed = true; }
    },
  );

  // Semantic tokens + lenses share the same debounced, staleness-guarded refresh shape: fetch on init + on
  // each doc change (debounced only to throttle a keystroke burst — `documentSyncPlugin` has already synced
  // the worker for this change, so the fetch is guaranteed to query current text), and drop a response a
  // newer edit has superseded.
  const semanticTokenPlugin = ViewPlugin.fromClass(
    class {
      private seq = 0;
      private timer: ReturnType<typeof setTimeout> | null = null;
      constructor(private readonly view: EditorView) { this.queue(); }
      update(u: ViewUpdate): void { if (u.docChanged) this.queue(); }
      destroy(): void { if (this.timer) clearTimeout(this.timer); }
      private queue(): void {
        if (this.timer) clearTimeout(this.timer);
        this.timer = setTimeout(() => void this.refresh(), FETCH_DEBOUNCE_MS);
      }
      private async refresh(): Promise<void> {
        const mine = ++this.seq;
        // A disposed client rejects its pending request during teardown — expected, so swallow it.
        const tokens = await client.semanticTokens(uri).catch(() => null);
        if (tokens === null || mine !== this.seq) return;   // superseded, or the client was disposed
        const deco = tokensToDecorations(tokens, this.view.state.doc.length);
        this.view.dispatch({ effects: setSemanticTokens.of(deco) });
      }
    },
  );

  const lensPlugin = ViewPlugin.fromClass(
    class {
      private seq = 0;
      private timer: ReturnType<typeof setTimeout> | null = null;
      constructor(private readonly view: EditorView) { this.queue(); }
      update(u: ViewUpdate): void { if (u.docChanged) this.queue(); }
      destroy(): void { if (this.timer) clearTimeout(this.timer); }
      private queue(): void {
        if (this.timer) clearTimeout(this.timer);
        this.timer = setTimeout(() => void this.refresh(), FETCH_DEBOUNCE_MS);
      }
      private async refresh(): Promise<void> {
        const mine = ++this.seq;
        const lenses = await client.capabilityLens(uri).catch(() => null);
        if (lenses === null || mine !== this.seq) return;   // superseded, or the client was disposed
        const len = this.view.state.doc.length;
        const ranges: Range<GutterMarker>[] = [];
        for (const l of lenses) if (l.from <= len) ranges.push(new LensMarker(l).range(l.from));
        this.view.dispatch({ effects: setLenses.of(RangeSet.of(ranges, true)) });
      }
    },
  );

  // A minimal signature hint: when the caret sits just after a `(` or `,`, show the active signature label in
  // a self-dismissing strip. Instance-scoped (own DOM + timer, cleaned up in destroy) so it can't leak or
  // destabilise the load-bearing lint/complete/hover/token paths. Deliberately not a full parameter widget.
  const signaturePlugin = ViewPlugin.fromClass(
    class {
      private dom: HTMLElement | null = null;
      private timer: ReturnType<typeof setTimeout> | null = null;
      private seq = 0;
      constructor(private readonly view: EditorView) {}
      update(u: ViewUpdate): void {
        if (!u.docChanged && !u.selectionSet) return;
        const pos = this.view.state.selection.main.head;
        const before = this.view.state.doc.sliceString(Math.max(0, pos - 1), pos);
        if (before !== '(' && before !== ',') { this.seq++; this.hide(); return; }
        const mine = ++this.seq;
        client.signatureHelp(uri, pos).then((sig) => {
          if (mine !== this.seq || !sig || sig.signatures.length === 0) return;
          const active = sig.signatures[sig.activeSignature ?? 0] ?? sig.signatures[0]!;
          this.show(active.label);
        }, () => { /* client disposed / request superseded — ignore */ });
      }
      private show(label: string): void {
        if (!this.dom) {
          this.dom = document.createElement('div');
          this.dom.className = 'cm-signature-hint';
          this.dom.setAttribute('role', 'status');
          this.view.dom.appendChild(this.dom);
        }
        this.dom.textContent = label;
        this.dom.style.display = '';
        if (this.timer) clearTimeout(this.timer);
        this.timer = setTimeout(() => this.hide(), 2500);
      }
      private hide(): void {
        if (this.timer) { clearTimeout(this.timer); this.timer = null; }
        if (this.dom) this.dom.style.display = 'none';
      }
      destroy(): void {
        if (this.timer) clearTimeout(this.timer);
        this.dom?.remove();
      }
    },
  );

  return [
    documentSyncPlugin,
    ...coldHighlight(),
    lspTheme,
    semanticTokenField,
    semanticTokenPlugin,
    diagnosticsPlugin,
    lintGutter(),
    // The idle-lint source returns the same cached diagnostics (the push above is authoritative for
    // freshness). Kept so CM's lint gutter + panel + F8 navigation see the diagnostics and a first idle lint
    // still runs if a push somehow beat the plugin's install.
    linter((view) => toCmDiagnostics(store.get(), view.state.doc.length), { delay: 100 }),
    autocompletion({
      override: [
        async (ctx: CompletionContext): Promise<CompletionResult | null> => {
          const items = await client.completion(uri, ctx.pos).catch(() => []);
          if (items.length === 0 && !ctx.explicit) return null;
          const options: Completion[] = items.map((it) => {
            const info = docText(it.documentation);
            return {
              label: it.label,
              detail: it.detail,
              type: completionType(it.kind),
              apply: it.insertText ?? it.label,
              info: info ? (): HTMLElement => renderHoverCard(info, 'cm-completionInfo-doc') : undefined,
            };
          });
          const word = ctx.matchBefore(/[\w$]*/);
          // `validFor` is the CM6 completion-source contract: while the text between `from` and the cursor
          // still matches this RegExp (an identifier run), CM keeps THIS result and re-filters it client-side
          // instead of re-invoking the async source per keystroke. That stabilizes the apply range (`from`/`to`
          // stay computed for the text that was current when this result was produced) — without it, a slower
          // request for an earlier, shorter cursor position can resolve AFTER the current one, leaving CM
          // holding a result whose range is stale, so a later accept/click lands the caret mid-word. The class
          // matches the `from` anchor above, and it serves the member case too: right after a `.` the partial
          // member word is the empty string (which matches), and it stays an identifier run as the user types.
          return { from: word?.from ?? ctx.pos, options, validFor: /^[\w$]*$/ };
        },
      ],
    }),
    hoverTooltip(async (_view, pos): Promise<Tooltip | null> => {
      const h = await client.hover(uri, pos).catch(() => null);
      if (!h) return null;
      return {
        pos: h.from,
        end: h.to,
        create: (): TooltipView => ({ dom: renderHoverCard(h.markdown, 'cm-hover-doc') }),
      };
    }),
    lensField,
    lensPlugin,
    gutter({
      class: 'cm-lens-gutter',
      markers: (view) => view.state.field(lensField, false) ?? RangeSet.empty,
    }),
    signaturePlugin,
  ];
}
