// createPlayground wires the pieces into the CodePen loop: editor -> debounce -> runTarget -> preview +
// diagnostics, plus target selection, example loading, and URL-share. The tool shell is host TS; the
// UI-target PREVIEW is a real @metael/vdom mount (the dogfood). Policy pinned by the design: keep the
// last-good preview on any diagnostic (never blank mid-keystroke) — only swap the preview when a run is
// diagnostic-free.
import type { VDomHandle } from '@metael/vdom/lang';
import type { Diagnostic } from '@metael/lang';
import { el, clear } from '../ui.ts';
import { createEditor } from './editor.ts';
import { runTarget, runComputeSettled } from './targets.ts';
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
  [Symbol.dispose](): void;
}

const DEBOUNCE_MS = 180;

// A compute source "uses a gpu head" if it calls gpu/gpuReduce/gpuHistogram. The plain compute probe env
// (RecordingHostEnv) does not know those heads — it would report a spurious ML-LANG-UNKNOWN-CALL — so a
// gpu-head compute program is routed through the gpu-aware, async-settled compute path (runComputeSettled)
// instead of the synchronous probe. No existing non-gpu compute example matches this, and every 'gpu'-target
// example wraps its gpu call inside a `component Story()` (a ui/gpu mount), so this cleanly selects only the
// unified gpu-on-compute case.
function usesGpuHead(source: string): boolean {
  return /\bgpu(Reduce|Histogram)?\s*\(/.test(source);
}

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
  targetSel.append(new Option('UI', 'ui'), new Option('Compute', 'compute'), new Option('GPU', 'gpu'));
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
  let renderSeq = 0;                          // bumped per render(); guards a stale mount's late gpu callback

  function currentSource(): string { return editor.getValue(); }

  // Render the deduped/capped diagnostics list for `source`. Split out so a LATE gpu reason (surfaced after
  // the initial mount, when a kernel guarded behind an async-settled resource finally derives non-core) can
  // re-render the list + flip the banner without re-running the whole loop. `extra` are diagnostics beyond
  // the run's own (the late gpu reasons); they are merged + deduped by (code, message).
  function paintDiagnostics(source: string, base: Diagnostic[], extra: Diagnostic[]): void {
    const seen = new Set<string>();
    const merged: Diagnostic[] = [];
    for (const d of [...base, ...extra]) { const k = `${d.code} ${d.message}`; if (!seen.has(k)) { seen.add(k); merged.push(d); } }
    const view = diagnosticView(merged);
    if (view.total > 0) {
      banner.hidden = false;
      banner.textContent = `${view.total} problem${view.total === 1 ? '' : 's'} — showing last working preview`;
      root.classList.add('pg-error');
    } else {
      banner.hidden = true; root.classList.remove('pg-error');
    }
    clear(diagList);
    for (const d of view.shown) {
      const where = d.span ? ` (line ${spanToLineCol(source, d.span.start).line})` : '';
      diagList.append(el('li', { class: 'pg-diag' }, [`${d.code}: ${d.message}${where}`]));
    }
    if (view.overflow > 0) diagList.append(el('li', { class: 'pg-diag pg-diag-more' }, [`…and ${view.overflow} more`]));
  }

  function render(): void {
    const source = currentSource();
    const seq = ++renderSeq;   // this render's identity — a later render invalidates a stale gpu callback
    // A LATE gpu reason (a kernel guarded behind an async-settled resource that finally derives non-core,
    // after the live mount returns) is folded into the panel here. Ignored if a newer render has started,
    // and always re-merged with THIS run's diagnostics so it never erases a parse/lang error.
    let lateGpu: Diagnostic[] = [];
    const onGpuIssues = (diags: Diagnostic[]): void => {
      if (seq !== renderSeq) return;   // a newer render owns the panel now
      lateGpu = diags;
      paintDiagnostics(source, liveDiags, lateGpu);
    };
    // A gpu-head compute program takes the DOM-free, gpu-aware, ASYNC-SETTLED compute path — NOT the
    // synchronous probe below. The probe's env (RecordingHostEnv) does not know the `gpu` head, so its
    // diagnostics would carry a spurious unknown-`gpu` error AND its value would be the pending frame. We
    // hide the preview + show the compute view synchronously, then await the settle and paint ITS text +
    // diagnostics (guarded by the outer `seq` so a slow settle from an old edit never overwrites a newer
    // render). Return before the synchronous probe/paint so the wrong diagnostics never reach the panel.
    if (target === 'compute' && usesGpuHead(source)) {
      if (liveHandle) { liveHandle.unmount(); liveHandle = null; }
      computeView.hidden = false; previewHost.hidden = true;
      void runComputeSettled(source, { data, seed }).then((out) => {
        if (seq !== renderSeq) return;   // a newer render superseded this one — drop
        computeView.textContent = out.text;
        paintDiagnostics(source, out.diagnostics, lateGpu);
      }).catch((err: unknown) => {
        // runComputeSettled can reject (the runtime's converge guard throws on arbitrary pasted source).
        // Surface it as a diagnostic instead of letting it become an unhandledrejection. Same staleness
        // guard: a newer render owns the panel, so a stale settle's failure is dropped silently.
        if (seq !== renderSeq) return;
        const message = err instanceof Error ? err.message : String(err);
        paintDiagnostics(source, [{ code: 'MLGPU-COMPUTE-FAILED', message }], lateGpu);
      });
      return;
    }
    // STEP 1 — a cheap PROBE run first, only for its diagnostics (never touches the visible preview):
    //   • compute: evaluateProgram is pure — the probe IS the result (reused on success).
    //   • ui: mount HEADLESS (container undefined) — no DOM, no delegation, just diagnostics.
    // This lets us keep the last-good preview on error WITHOUT ever tearing down or detaching it, and
    // avoids the "mount into a throwaway node then move it" trap: mount() attaches event delegation to the
    // exact container it's given, so a moved subtree would strand its click listener. We instead mount the
    // real preview DIRECTLY into the persistent previewHost only once the source is known-clean.
    const probe = runTarget(target, source, undefined, { data, seed });
    // The probe is a HEADLESS mount for a 'ui'/'gpu' target — it spins up a real host (and, for 'gpu', a
    // GpuEngine that may acquire a WebGPU device + run a throwaway dispatch). Tear it down immediately once
    // its diagnostics are read, so the probe never leaks an engine/device (unmount() disposes a disposable env).
    if (probe.kind === 'ui') probe.handle.unmount();

    let liveDiags = probe.diagnostics;   // the diagnostics the visible run reports (probe, unless it's clean)
    if (probe.diagnostics.length === 0) {
      if (target === 'ui' || target === 'gpu') {
        // 'ui' and 'gpu' both mount a live vdom preview (a 'gpu' run is a ui run whose app calls the gpu
        // head via the composite env). Mount through runTarget so the correct env is chosen per target.
        if (liveHandle) liveHandle.unmount();      // unmount() clears previewHost.textContent for us
        clear(previewHost);                        // belt-and-suspenders (first run: nothing to clear)
        // onGpuIssues is honored only for a 'gpu' run (a ui run ignores it) — it surfaces a LATE non-core
        // reason from an async-guarded kernel. The synchronous reasons are already in run.diagnostics.
        const run = runTarget(target, source, previewHost, { data, seed, onGpuIssues });   // mount INTO the persistent host → delegation is correct
        liveHandle = run.kind === 'ui' ? run.handle : null;
        liveDiags = run.diagnostics;
        computeView.hidden = true; previewHost.hidden = false;
      } else {
        if (liveHandle) { liveHandle.unmount(); liveHandle = null; }
        computeView.textContent = probe.kind === 'compute' ? probe.text : '';
        computeView.hidden = false; previewHost.hidden = true;
      }
    }
    // else: keep the last-good preview; paintDiagnostics shows the banner. The headless probe created no
    // DOM/handle to clean up.

    paintDiagnostics(source, liveDiags, lateGpu);
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
    [Symbol.dispose]: () => { if (timer) clearTimeout(timer); if (liveHandle) liveHandle.unmount(); container.removeChild(root); },
  };
}
