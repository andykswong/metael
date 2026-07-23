import { describe, it, expect, beforeEach } from 'vitest';
import { EditorView } from '@codemirror/view';
import { acceptCompletion, startCompletion, currentCompletions } from '@codemirror/autocomplete';
import { createPlayground } from './create.ts';

// The RENDERED-DECORATION regression guard for the post-completion caret desync. Accepting `span` after
// typing `sp` puts the correct text in the doc AND the state cursor at offset 4 — earlier tests asserted only
// that (and passed). But the SEMANTIC-TOKEN decoration could still be a stale mark covering `[0,2]` (the two
// chars that were current when the token fetch last queried the worker), which — after the accept's own token
// fetch lands ~a debounce later — SHRINKS the rendered word to `<span class="cmt-*">sp</span>` + a bare `an`.
// A `Decoration.mark` boundary mid-word desyncs CodeMirror's PAINTED caret to that boundary (offset 2) even
// though state + the DOM Selection API report 4.
//
// The staleness had a root cause: the worker's document sync and the token fetch ran on two independent,
// unordered timers, so a fetch could query the worker BEFORE the matching text sync landed and re-mark the
// short span — and once installed, the split PERSISTED (every later fetch raced the same way). The fix makes
// the sync synchronous per doc-change (ordered before any fetch), so the fetch triggered by the accept always
// sees `span` and the mark covers the whole word.
//
// This asserts the decoration is NOT split AFTER the accept's token fetch has settled — the window the
// pre-fix bug lived in. It FAILS against the pre-fix code (the split mark lands ~130ms after accept and never
// heals) and passes once the sync is ordered.

let container: HTMLElement;
beforeEach(() => {
  container = document.createElement('div');
  document.body.appendChild(container);
});

/** The live CodeMirror view behind the playground's editor. */
function viewOf(root: HTMLElement): EditorView {
  return EditorView.findFromDOM(root.querySelector('.cm-editor') as HTMLElement)!;
}

/** Replace the whole doc through a real (non-programmatic) transaction, then clear it. */
function clearDoc(view: EditorView): void {
  view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: '' } });
}

/** Type `text` one char at a time as genuine `input.type` transactions (what real typing dispatches). */
function typeChars(view: EditorView, text: string): void {
  for (const ch of text) {
    const at = view.state.selection.main.head;
    view.dispatch({ changes: { from: at, insert: ch }, selection: { anchor: at + 1 }, userEvent: 'input.type' });
  }
}

/** Deadline-poll a predicate (no fixed sleeps); resolves true once it holds, false at the deadline. */
async function until(pred: () => boolean, ms = 4000): Promise<boolean> {
  const deadline = Date.now() + ms;
  while (!pred() && Date.now() < deadline) await new Promise((r) => setTimeout(r, 15));
  return pred();
}

const sleep = (ms: number): Promise<void> => new Promise((r) => setTimeout(r, ms));

/** Mount a UI-target playground (where `span` is a known head), FOCUS the editor, and clear the doc.
 *  Focus + one microtask first so the focus-driven selectionchange settles before any programmatic edit. */
async function mountUi(): Promise<{ pg: ReturnType<typeof createPlayground>; view: EditorView }> {
  const pg = createPlayground(container, { defaultExampleId: 'counter' });
  pg.runNow();
  const view = viewOf(pg.root);
  view.focus();
  await new Promise((r) => setTimeout(r, 0));
  clearDoc(view);
  return { pg, view };
}

/** Type `partial`, sync the worker (runNow), open completion, and poll until a `span` candidate is present. */
async function openWithSpanCandidate(pg: ReturnType<typeof createPlayground>, view: EditorView, partial: string): Promise<void> {
  typeChars(view, partial);
  pg.runNow();
  startCompletion(view);
  const ok = await until(() => currentCompletions(view.state).some((c) => c.label === 'span'));
  expect(ok).toBe(true);
}

/** The semantic-token elements (`.cmt-*` marks) rendered in the editor's content, with their visible text.
 *  Distinct from the cold-highlighter's generated `ͼ*` classes — those are NOT `.cmt-*`. */
function semanticTokenTexts(root: HTMLElement): string[] {
  const content = root.querySelector('.cm-content') as HTMLElement;
  return Array.from(content.querySelectorAll('[class*="cmt-"]'))
    .map((e) => e.textContent ?? '')
    .filter((t) => t.length > 0);
}

/** True once at least one semantic-token mark has been painted AND none of them is a proper prefix of `span`
 *  (`sp`/`spa`) — i.e. the word is not split at a mark boundary. Used both as the settle condition and the
 *  final assertion: the pre-fix split lands ~130ms after the accept and never heals, so a poll that first
 *  waits for a mark to appear then requires no-split is RED pre-fix and GREEN post-fix. */
function markCoversWholeWord(root: HTMLElement): boolean {
  const marks = semanticTokenTexts(root);
  if (marks.length === 0) return false;                        // no semantic mark yet — keep waiting
  return marks.includes('span') && !marks.some((t) => t === 'sp' || t === 'spa');
}

describe('post-completion semantic-token decoration (Chromium)', () => {
  it('does not leave a split token mark mid-word after accepting a completion', async () => {
    const { pg, view } = await mountUi();
    await openWithSpanCandidate(pg, view, 'sp');

    // Past CM's interactionDelay (~75ms): acceptCompletion is a no-op if fired too soon after the popup opens.
    await sleep(200);
    expect(acceptCompletion(view)).toBe(true);

    // The state contract earlier tests already cover — assert it still holds.
    expect(view.state.doc.toString()).toBe('span');
    expect(view.state.selection.main.head).toBe(4);

    // The rendered-decoration contract (the gap that let the caret desync ship). The accept is a doc change,
    // so it triggers a fresh token fetch on the plugin's debounce. Pre-fix, that fetch (unordered vs the
    // worker's text sync) re-marks the stale `[0,2]` span ~130ms later and the word STAYS split forever.
    // Post-fix, the synchronous sync means the fetch sees `span` → one mark over the whole word.
    //
    // Poll until a mark that covers the whole word appears (deadline, no fixed sleeps beyond the interaction
    // delay above). Pre-fix this never becomes true — a `sp`/`spa`-only mark is what settles — so the poll
    // exhausts its deadline and the assertion fails: the RED this test is designed to catch.
    const whole = await until(() => markCoversWholeWord(pg.root));
    expect(whole).toBe(true);

    // And it must STAY whole: re-check after a further beat so a late stale fetch can't heal-then-resplit.
    await sleep(200);
    const marks = semanticTokenTexts(pg.root);
    expect(marks).toContain('span');
    expect(marks.some((t) => t === 'sp' || t === 'spa')).toBe(false);

    pg[Symbol.dispose]();
  });

  it('does not split the mark when the completion is applied by a real popup-row mousedown (owner repro)', async () => {
    const { pg, view } = await mountUi();
    await openWithSpanCandidate(pg, view, 'sp');

    await sleep(200);
    const rowReady = await until(() =>
      Array.from(pg.root.querySelectorAll('.cm-tooltip-autocomplete li')).some((li) =>
        (li.textContent ?? '').startsWith('span'),
      ),
    );
    expect(rowReady).toBe(true);
    const row = Array.from(pg.root.querySelectorAll('.cm-tooltip-autocomplete li')).find((li) =>
      (li.textContent ?? '').startsWith('span'),
    ) as HTMLElement;
    row.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, cancelable: true }));

    expect(await until(() => view.state.doc.toString() === 'span')).toBe(true);
    expect(view.state.selection.main.head).toBe(4);

    const whole = await until(() => markCoversWholeWord(pg.root));
    expect(whole).toBe(true);

    await sleep(200);
    const marks = semanticTokenTexts(pg.root);
    expect(marks).toContain('span');
    expect(marks.some((t) => t === 'sp' || t === 'spa')).toBe(false);

    pg[Symbol.dispose]();
  });
});
