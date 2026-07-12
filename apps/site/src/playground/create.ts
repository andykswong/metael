// createPlayground wires the pieces into the CodePen loop: editor -> debounce -> runTarget -> preview +
// diagnostics, plus target selection, example loading, and URL-share. The tool shell is host TS; the
// UI-target PREVIEW is a real @metael/vdom mount (the dogfood). Policy pinned by the design: keep the
// last-good preview on any diagnostic (never blank mid-keystroke) — only swap the preview when a run is
// diagnostic-free.
import { mount } from '@metael/vdom';
import type { VDomHandle } from '@metael/vdom';
import { el, clear } from '../ui.ts';
import { createEditor } from './editor.ts';
import { runTarget } from './targets.ts';
import { diagnosticView, spanToLineCol } from './diagnostics.ts';
import { encodeState, type ShareState } from './share.ts';
import { EXAMPLES, exampleById, defaultExampleForTarget, DEFAULT_EXAMPLE_ID, type Target, type Example } from './examples.ts';

export interface PlaygroundOptions {
  compact?: boolean;             // hero embed: hide the example picker + share bar
  defaultExampleId?: string;
  initialState?: ShareState;     // e.g. decoded from the URL fragment
  debounceMs?: number;
  openHref?: string;             // compact embeds: render an "Open the full editor ↗" link in the toolbar
}

export interface PlaygroundHandle {
  readonly root: HTMLElement;
  /** Current shareable state (source + target + data/seed). */
  getState(): ShareState;
  /** Force a run now (bypasses debounce) — used by tests. */
  runNow(): void;
  destroy(): void;
}

const DEBOUNCE_MS = 180;

export function createPlayground(container: Element, opts: PlaygroundOptions = {}): PlaygroundHandle {
  const first: Example = (opts.initialState && stateAsExample(opts.initialState))
    ?? exampleById(opts.defaultExampleId ?? DEFAULT_EXAMPLE_ID)
    ?? EXAMPLES[0]!;

  let target: Target = first.target;
  let data: unknown = first.data;
  let seed: number | undefined = opts.initialState?.seed;
  // Remember the last example id loaded per target, so toggling UI↔Compute and back restores the example you
  // were on (not the target default). Seeded with the initial example for its target.
  const lastExampleByTarget: Partial<Record<Target, string>> = { [first.target]: first.id };

  const editor = createEditor(first.source);
  const previewHost = el('div', { class: 'pg-preview-host' });
  const computeView = el('pre', { class: 'pg-compute' });
  // The error surface is a live region so a screen-reader user is told their program failed without a focus
  // move (the banner is an assertive alert; the diagnostics list updates politely).
  const banner = el('div', { class: 'pg-banner', hidden: 'hidden', role: 'alert' });
  const diagList = el('ul', { class: 'pg-diags', 'aria-live': 'polite' });

  const targetSel = el('select', { class: 'pg-target', 'aria-label': 'Run target' }) as HTMLSelectElement;
  targetSel.append(new Option('UI', 'ui'), new Option('Compute', 'compute'));
  targetSel.value = target;

  // aria-live so the transient "Copied!" label change is announced (a focused button whose text silently
  // mutates is not reliably re-announced by screen readers).
  const shareBtn = el('button', { class: 'pg-share', type: 'button', 'aria-live': 'polite' }, ['Share']);

  // Layout: [editor | preview], with a toolbar (example picker + a blurb line + share) and a diagnostics strip.
  const blurbEl = el('span', { class: 'pg-blurb' });
  const toolbar = el('div', { class: 'pg-toolbar' }, [targetSel]);
  let picker: HTMLSelectElement | null = null;   // present only in the non-compact shell; kept in sync on load
  if (!opts.compact) {
    picker = el('select', { class: 'pg-examples', 'aria-label': 'Load example' }) as HTMLSelectElement;
    for (const e of EXAMPLES) {
      const opt = new Option(`${e.label} (${e.target})`, e.id);
      opt.title = e.blurb;                       // hover tooltip = what the example demonstrates
      picker.append(opt);
    }
    picker.value = first.id;
    picker.addEventListener('change', () => loadExample(picker!.value));
    toolbar.append(picker, shareBtn);
    blurbEl.textContent = first.blurb;
  } else if (opts.openHref) {
    // Compact embed (e.g. the landing hero): a far-right link into the full playground on the toolbar row,
    // mirroring where Share sits in the full playground. Same-site → (not ↗).
    toolbar.append(el('a', { class: 'pg-open', href: opts.openHref }, ['Open in Playground →']));
  }
  const previewPane = el('div', { class: 'pg-preview' }, [previewHost, computeView]);
  const root = el('div', { class: `pg-root${opts.compact ? ' pg-compact' : ''}` }, [
    toolbar,
    ...(opts.compact ? [] : [blurbEl]),
    el('div', { class: 'pg-panes' }, [el('div', { class: 'pg-editor' }, [editor.root]), previewPane]),
    banner,
    diagList,
  ]);
  container.append(root);

  let liveHandle: VDomHandle | null = null;   // the currently-mounted UI preview (real @metael/vdom app)
  let timer: ReturnType<typeof setTimeout> | null = null;

  function currentSource(): string { return editor.getValue(); }

  function render(): void {
    const source = currentSource();
    // STEP 1 — a cheap PROBE run first, only for its diagnostics (never touches the visible preview):
    //   • compute: evaluateProgram is pure — the probe IS the result (reused on success).
    //   • ui: mount HEADLESS (container undefined) — no DOM, no delegation, just diagnostics.
    // This lets us keep the last-good preview on error WITHOUT ever tearing down or detaching it, and
    // avoids the "mount into a throwaway node then move it" trap: mount() attaches event delegation to the
    // exact container it's given, so a moved subtree would strand its click listener. We instead mount the
    // real preview DIRECTLY into the persistent previewHost only once the source is known-clean.
    const probe = runTarget(target, source, undefined, { data, seed });
    const view = diagnosticView(probe.diagnostics);

    if (probe.diagnostics.length === 0) {
      if (target === 'ui') {
        if (liveHandle) liveHandle.unmount();      // unmount() clears previewHost.textContent for us
        clear(previewHost);                        // belt-and-suspenders (first run: nothing to clear)
        liveHandle = mount(source, previewHost, { data, seed });   // mount INTO the persistent host → delegation is correct
        computeView.hidden = true; previewHost.hidden = false;
      } else {
        if (liveHandle) { liveHandle.unmount(); liveHandle = null; }
        computeView.textContent = probe.kind === 'compute' ? probe.text : '';
        computeView.hidden = false; previewHost.hidden = true;
      }
      banner.hidden = true; root.classList.remove('pg-error');
    } else {
      // Failure — keep the last-good preview; just show a non-blocking banner. The headless probe created
      // no DOM/handle to clean up.
      banner.hidden = false;
      banner.textContent = `${view.total} problem${view.total === 1 ? '' : 's'} — showing last working preview`;
      root.classList.add('pg-error');
    }

    // Always refresh the diagnostics list (deduped + capped).
    clear(diagList);
    for (const d of view.shown) {
      const where = d.span ? ` (line ${spanToLineCol(source, d.span.start).line})` : '';
      diagList.append(el('li', { class: 'pg-diag' }, [`${d.code}: ${d.message}${where}`]));
    }
    if (view.overflow > 0) diagList.append(el('li', { class: 'pg-diag pg-diag-more' }, [`…and ${view.overflow} more`]));
  }

  function schedule(): void {
    if (timer) clearTimeout(timer);
    timer = setTimeout(render, opts.debounceMs ?? DEBOUNCE_MS);
  }

  function loadExample(id: string): void {
    const e = exampleById(id);
    if (!e) return;
    loadExampleObj(e);
  }

  /** Load a concrete example: sets target/data/seed + source, syncs the target select + the picker, renders. */
  function loadExampleObj(e: Example): void {
    target = e.target; data = e.data; seed = undefined;
    if (e.id !== '__shared') lastExampleByTarget[target] = e.id;   // remember, per target, for a later switch-back
    targetSel.value = target;
    if (picker) picker.value = e.id;   // keep the picker in sync (e.g. when a target switch loads a new example)
    blurbEl.textContent = e.blurb;
    editor.setValue(e.source);
    render();
  }

  /** Switching the run target restores the example you last had on that target (or its default the first time).
   *  A target switch is an EXAMPLE switch — running a UI component through the compute backend would just
   *  yield null, and vice versa. */
  function switchTarget(next: Target): void {
    const remembered = lastExampleByTarget[next];
    const e = (remembered ? exampleById(remembered) : undefined) ?? defaultExampleForTarget(next);
    loadExampleObj(e);
  }

  function stateAsExample(s: ShareState): Example {
    return { id: '__shared', label: 'Shared', blurb: 'a shared snippet', target: s.target, source: s.source, data: s.data };
  }

  editor.onChange(schedule);
  targetSel.addEventListener('change', () => switchTarget(targetSel.value as Target));
  shareBtn.addEventListener('click', async () => {
    const frag = await encodeState(getState());
    const url = `${location.origin}${location.pathname}#${frag}`;
    history.replaceState(null, '', url);
    try { await navigator.clipboard?.writeText(url); shareBtn.textContent = 'Copied!'; setTimeout(() => { shareBtn.textContent = 'Share'; }, 1200); }
    catch { /* clipboard denied — the URL bar still updated */ }
  });

  function getState(): ShareState { return { source: currentSource(), target, data, seed }; }

  render();

  return {
    root,
    getState,
    runNow: () => { if (timer) clearTimeout(timer); render(); },
    destroy: () => { if (timer) clearTimeout(timer); if (liveHandle) liveHandle.unmount(); container.removeChild(root); },
  };
}
