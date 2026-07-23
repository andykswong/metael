// A CodeMirror 6 editor exposing the narrow EditorHandle seam the playground consumes. The LSP-backed
// extensions (lint/autocomplete/hover/semantic tokens) are added by the client wiring; this module owns
// only the view + the seam. Programmatic setValue is flagged so it never fires onChange.
import { EditorState, Compartment } from '@codemirror/state';
import { EditorView, keymap } from '@codemirror/view';
import { defaultKeymap, history, historyKeymap } from '@codemirror/commands';
import { el } from '../ui.ts';

export interface EditorHandle {
  readonly root: HTMLElement;
  readonly view: EditorView;              // exposed so the client wiring can reconfigure extensions
  readonly extensions: Compartment;       // a compartment the client fills with LSP extensions
  getValue(): string;
  setValue(text: string): void;
  onChange(cb: (value: string) => void): void;
  destroy(): void;
}

export function createEditor(initial: string): EditorHandle {
  const root = el('div', { class: 'ed-root' });
  let changeCb: ((value: string) => void) | null = null;
  let programmatic = false;                // true while setValue drives the doc → suppress onChange

  const lspExtensions = new Compartment();
  const updateListener = EditorView.updateListener.of((u) => {
    if (u.docChanged && !programmatic) changeCb?.(u.state.doc.toString());
  });

  const view = new EditorView({
    parent: root,
    state: EditorState.create({
      doc: initial,
      extensions: [
        history(),
        keymap.of([...defaultKeymap, ...historyKeymap]),
        updateListener,
        // Give the contenteditable an accessible name so a screen-reader announces the edit box.
        EditorView.contentAttributes.of({ 'aria-label': 'metael source editor' }),
        lspExtensions.of([]),
      ],
    }),
  });

  return {
    root, view, extensions: lspExtensions,
    getValue: () => view.state.doc.toString(),
    setValue: (text: string) => {
      programmatic = true;
      view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: text } });
      programmatic = false;
    },
    onChange: (cb) => { changeCb = cb; },
    destroy: () => view.destroy(),
  };
}
