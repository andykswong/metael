// Highlight-over-textarea editor: a transparent <textarea> (real caret, native undo/IME) layered over a
// <pre> that renders lex()-classified segments. The textarea is UNCONTROLLED (the DOM holds the text) so
// the caret never jumps; on input we re-render the highlight overlay and notify the owner. The overlay also
// hosts squiggle underlines the owner positions (via the returned API) for diagnostic spans.
import { tokensToSegments } from './highlight.ts';
import { el } from '../ui.ts';

export interface EditorHandle {
  readonly root: HTMLElement;
  getValue(): string;
  setValue(text: string): void;
  onChange(cb: (value: string) => void): void;
}

export function createEditor(initial: string): EditorHandle {
  const highlightLayer = el('pre', { class: 'ed-highlight', 'aria-hidden': 'true' });
  const textarea = el('textarea', {
    class: 'ed-input', spellcheck: 'false', autocapitalize: 'off', autocomplete: 'off', wrap: 'off',
    'aria-label': 'metael source editor',
  }) as HTMLTextAreaElement;
  const root = el('div', { class: 'ed-root' }, [highlightLayer, textarea]);

  let changeCb: ((value: string) => void) | null = null;

  function renderHighlight(text: string): void {
    highlightLayer.textContent = '';
    for (const seg of tokensToSegments(text)) {
      highlightLayer.append(el('span', { class: `tok-${seg.kind}` }, [seg.text]));
    }
    // A trailing newline needs a spacer so the <pre> height matches the textarea's scrollHeight.
    if (text.endsWith('\n') || text === '') highlightLayer.append(document.createTextNode('\u200B'));
  }

  textarea.value = initial;
  renderHighlight(initial);

  textarea.addEventListener('input', () => {
    renderHighlight(textarea.value);
    changeCb?.(textarea.value);
  });
  // Keep the overlay scroll-synced with the textarea.
  textarea.addEventListener('scroll', () => {
    highlightLayer.scrollTop = textarea.scrollTop;
    highlightLayer.scrollLeft = textarea.scrollLeft;
  });

  return {
    root,
    getValue: () => textarea.value,
    setValue: (text: string) => { textarea.value = text; renderHighlight(text); },
    onChange: (cb) => { changeCb = cb; },
  };
}
