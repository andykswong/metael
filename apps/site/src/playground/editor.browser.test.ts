import { describe, it, expect, beforeEach } from 'vitest';
import { createEditor } from './editor.ts';

let container: HTMLElement;
beforeEach(() => { container = document.createElement('div'); document.body.appendChild(container); });

describe('createEditor (real DOM)', () => {
  it('renders the initial text into both the textarea and the highlight overlay', () => {
    const ed = createEditor('let n = 0');
    container.appendChild(ed.root);
    const ta = container.querySelector('textarea') as HTMLTextAreaElement;
    const pre = container.querySelector('pre.ed-highlight')!;
    expect(ta.value).toBe('let n = 0');
    expect(pre.textContent).toContain('let n = 0');
    expect(pre.querySelector('.tok-keyword')?.textContent).toBe('let');
  });

  it('fires onChange and re-highlights on input', () => {
    const ed = createEditor('');
    container.appendChild(ed.root);
    const ta = container.querySelector('textarea') as HTMLTextAreaElement;
    let seen = '';
    ed.onChange((v) => { seen = v; });
    ta.value = 'const x = 1';
    ta.dispatchEvent(new Event('input', { bubbles: true }));
    expect(seen).toBe('const x = 1');
    expect(container.querySelector('pre.ed-highlight')!.querySelector('.tok-keyword')?.textContent).toBe('const');
  });

  it('setValue updates both layers without an input event', () => {
    const ed = createEditor('a');
    container.appendChild(ed.root);
    ed.setValue('component Story() {}');
    const ta = container.querySelector('textarea') as HTMLTextAreaElement;
    expect(ta.value).toBe('component Story() {}');
    expect(container.querySelector('pre.ed-highlight')!.textContent).toContain('component');
  });

  it('the textarea has an accessible name + the highlight overlay is hidden from AT', () => {
    const ed = createEditor('let n = 0');
    container.appendChild(ed.root);
    const ta = container.querySelector('textarea') as HTMLTextAreaElement;
    expect(ta.getAttribute('aria-label')).toBe('metael source editor');
    // the decorative highlight <pre> must not be double-read by a screen reader
    expect(container.querySelector('pre.ed-highlight')!.getAttribute('aria-hidden')).toBe('true');
  });
});
