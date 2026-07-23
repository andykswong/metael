import { describe, it, expect } from 'vitest';
import { createEditor } from './editor.ts';

describe('CodeMirror editor handle (Chromium)', () => {
  it('preserves the EditorHandle seam: getValue/setValue/onChange', () => {
    const ed = createEditor('const x = 1');
    document.body.appendChild(ed.root);
    expect(ed.getValue()).toBe('const x = 1');
    ed.setValue('const y = 2');
    expect(ed.getValue()).toBe('const y = 2');
  });
  it('programmatic setValue does NOT fire onChange (loadExampleObj relies on this)', () => {
    const ed = createEditor('a');
    document.body.appendChild(ed.root);
    let fired = 0; ed.onChange(() => { fired++; });
    ed.setValue('b');
    expect(fired).toBe(0);
  });
  it('exposes an accessible name on the editable content (screen-reader label)', () => {
    const ed = createEditor('x');
    document.body.appendChild(ed.root);
    expect(ed.root.querySelector('.cm-content')?.getAttribute('aria-label')).toBe('metael source editor');
  });
});
