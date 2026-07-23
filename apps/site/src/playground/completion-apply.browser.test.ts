import { describe, it, expect, beforeEach } from 'vitest';
import { EditorView } from '@codemirror/view';
import { acceptCompletion, startCompletion, currentCompletions } from '@codemirror/autocomplete';
import { createPlayground } from './create.ts';

// The autocomplete APPLY path in a real browser — the regression guard for the stale-range race. The
// completion source returns `validFor` so CodeMirror keeps the FIRST async result and re-filters it
// client-side as the user types, instead of re-invoking the worker per keystroke (where a slower request
// for an earlier, shorter cursor position could resolve AFTER the current one, leaving CM holding a result
// whose `from`/`to` were computed for shorter text → a later accept/click lands the caret mid-word).
//
// Two things are proven here:
//  1. RESULT REUSE (the root-cause guard, and the assertion that FAILS without `validFor`): after one more
//     matching keystroke, the SAME result is reused synchronously — no re-query, dialog stays live. Without
//     `validFor`, `checkValid` fails, the source drops to pending, and the dialog is disabled synchronously
//     (so `currentCompletions` is empty on the very next tick), which is exactly the re-query behaviour that
//     opens the out-of-order-response race.
//  2. APPLY CORRECTNESS (the symptom guard): accepting `span` — via `acceptCompletion` AND via a real
//     mousedown on the popup row (the owner's exact repro) — replaces the partial and puts the caret at the
//     END of the inserted word, not mid-word.

let container: HTMLElement;
beforeEach(() => {
  container = document.createElement('div');
  document.body.appendChild(container);
});

/** The live CodeMirror view behind the playground's editor. */
function viewOf(root: HTMLElement): EditorView {
  return EditorView.findFromDOM(root.querySelector('.cm-editor') as HTMLElement)!;
}

/** Replace the whole doc through a real (non-programmatic) transaction, then clear it — leaves an empty doc
 *  with the change pipeline (onChange → schedule) having fired exactly as user editing would. */
function clearDoc(view: EditorView): void {
  view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: '' } });
}

/** Type `text` one character at a time as genuine `input.type` transactions (what real typing dispatches),
 *  so CodeMirror's activate-on-typing + the completion state machine see each keystroke individually. */
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
 *  Focus happens BEFORE any programmatic edit and yields one microtask: CodeMirror's DOMObserver turns the
 *  contenteditable's `selectionchange` (fired async by focusing) into a `select` transaction, and if that
 *  lands AFTER our programmatic typing it clobbers the caret back to 0. Focusing first (and letting that
 *  settle) keeps the DOM selection glued to state on every subsequent keystroke dispatch. */
async function mountUi(): Promise<{ pg: ReturnType<typeof createPlayground>; view: EditorView }> {
  const pg = createPlayground(container, { defaultExampleId: 'counter' });
  pg.runNow();
  const view = viewOf(pg.root);
  view.focus();
  await new Promise((r) => setTimeout(r, 0));    // let the focus-driven selectionchange settle first
  clearDoc(view);
  return { pg, view };
}

/** Type `partial`, force the worker into sync (runNow bypasses the change debounce so its analysis reflects
 *  the current buffer), open completion, and poll until a `span` candidate is present. */
async function openWithSpanCandidate(pg: ReturnType<typeof createPlayground>, view: EditorView, partial: string): Promise<void> {
  typeChars(view, partial);
  pg.runNow();                                   // sync the LSP worker to the current buffer before querying
  startCompletion(view);
  const ok = await until(() => currentCompletions(view.state).some((c) => c.label === 'span'));
  expect(ok).toBe(true);
}

describe('autocomplete apply (Chromium)', () => {
  it('reuses the SAME result on the next matching keystroke — no re-query (the validFor guard)', async () => {
    const { pg, view } = await mountUi();
    await openWithSpanCandidate(pg, view, 'sp');
    expect(currentCompletions(view.state).some((c) => c.label === 'span')).toBe(true);

    // Type one MORE matching identifier char. With `validFor: /^[\w$]*$/`, CodeMirror re-filters the existing
    // result client-side and keeps the dialog live — SYNCHRONOUSLY, in the same tick as this transaction, so
    // no worker round-trip (which is inherently async) can have run. Without `validFor` the source would drop
    // to pending and the dialog would be disabled here, making currentCompletions empty on this very tick —
    // the exact re-query behaviour that lets an out-of-order response install a stale apply range.
    typeChars(view, 'a');                         // 'sp' → 'spa' (still a prefix of 'span')
    const reused = currentCompletions(view.state);
    expect(reused.some((c) => c.label === 'span')).toBe(true);   // reused without a worker round-trip

    pg[Symbol.dispose]();
  });

  it('accepting the completion replaces the partial and puts the caret at the end of the word', async () => {
    const { pg, view } = await mountUi();
    await openWithSpanCandidate(pg, view, 'sp');

    // Past CM's interactionDelay (~75ms): acceptCompletion is a no-op if fired too soon after the popup opens.
    await sleep(200);
    const accepted = acceptCompletion(view);
    expect(accepted).toBe(true);

    // The doc is the full word and the caret sits AFTER the last char (position === word length) — not the
    // stale-range symptom (caret left after 'p').
    expect(view.state.doc.toString()).toBe('span');
    expect(view.state.selection.main.head).toBe(4);

    pg[Symbol.dispose]();
  });

  it('a real mousedown on the popup row applies span with the caret at the end (owner repro)', async () => {
    const { pg, view } = await mountUi();
    await openWithSpanCandidate(pg, view, 'sp');

    // The rendered option rows carry the same result CM is holding; click the `span` row exactly as the owner
    // did. CodeMirror's mousedown handler applies the selected option's `apply` string over the result's
    // (now range-stable) from/to.
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

    pg[Symbol.dispose]();
  });
});
