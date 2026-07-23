import { describe, it, expect } from 'vitest';
import { renderHoverCard } from './lsp-extensions.ts';

// The structured hover-card renderer, exercised directly (deterministic, not flaky like a real pointer
// hover): feed it the exact light markdown the LSP server emits and assert it builds DISTINCT styled nodes
// — a signature title, an arg list with coloured param names, a returns row — instead of one flat blob.
// Also a real injection-safety regression guard: markup in the source must become literal text, never DOM.

/** The canonical builtin card the server emits for `join` (fenced signature + portability-prefixed
 *  description + a `  name — doc` per-arg list + a trailing `Returns …` line). */
const JOIN_CARD = [
  '```metael',
  'join(items, separator)',
  '```',
  '(cpu-only) Joins an array of items into a single string, placing `separator` between each element.',
  '  items — the array to join (each element is coerced to a string)',
  '  separator — the string inserted between elements',
  'Returns the joined string.',
].join('\n');

describe('renderHoverCard (Chromium)', () => {
  it('builds a distinct signature title node carrying the signature line', () => {
    const el = renderHoverCard(JOIN_CARD);
    const sig = el.querySelector('.cm-hover-sig');
    expect(sig).not.toBeNull();
    expect(sig!.textContent).toBe('join(items, separator)');
  });

  it('renders the portability marker as a separate metadata badge, not sentence prose', () => {
    const el = renderHoverCard(JOIN_CARD);
    const badge = el.querySelector('.cm-hover-portability');
    expect(badge).not.toBeNull();
    expect(badge!.textContent).toBe('cpu-only');
    // The badge parenthesis text is NOT left in the description prose.
    const desc = el.querySelector('.cm-hover-desc');
    expect(desc!.textContent).not.toContain('(cpu-only)');
    expect(desc!.textContent).toContain('Joins an array of items into a single string');
  });

  it('renders inline `code` runs in a distinct code span, not literal backticks', () => {
    const el = renderHoverCard(JOIN_CARD);
    const code = el.querySelector('.cm-hover-code');
    expect(code).not.toBeNull();
    expect(code!.textContent).toBe('separator');
    // No literal backtick survives into the rendered text.
    expect(el.textContent).not.toContain('`');
  });

  it('renders each documented param as a list row with a coloured name and its doc', () => {
    const el = renderHoverCard(JOIN_CARD);
    const names = Array.from(el.querySelectorAll('.cm-hover-param-name')).map((n) => n.textContent);
    expect(names).toEqual(['items', 'separator']);
    const list = el.querySelector('.cm-hover-params');
    expect(list).not.toBeNull();
    expect(list!.querySelectorAll('li').length).toBe(2);
    // The doc text of the first param survives beside its name.
    expect(el.querySelector('.cm-hover-params li')!.textContent).toContain('the array to join');
  });

  it('renders the Returns line as its own row with a dim label', () => {
    const el = renderHoverCard(JOIN_CARD);
    const ret = el.querySelector('.cm-hover-returns');
    expect(ret).not.toBeNull();
    expect(ret!.querySelector('.cm-hover-returns-label')!.textContent).toBe('Returns');
    expect(ret!.textContent).toContain('the joined string.');
  });

  it('degrades gracefully for a simpler card shape (a local binding) — no throw, content kept', () => {
    const el = renderHoverCard('```metael\nconst total\n```');
    expect(el.querySelector('.cm-hover-sig')!.textContent).toBe('const total');
    // No params / returns nodes for a shape that has none.
    expect(el.querySelector('.cm-hover-params')).toBeNull();
    expect(el.querySelector('.cm-hover-returns')).toBeNull();
  });

  it('renders a plain description (the completion-info shape) as readable prose', () => {
    const el = renderHoverCard('Joins an array of items into a single string.', 'cm-completionInfo-doc');
    expect(el.className).toBe('cm-completionInfo-doc');
    expect(el.querySelector('.cm-hover-desc')!.textContent).toContain('Joins an array of items');
  });

  it('is injection-safe: HTML in the source becomes literal text, never a DOM element', () => {
    const evil = [
      '```metael',
      'evil(x)',
      '```',
      'A description with <img src=x onerror=alert(1)> markup and a `<script>` run.',
      '  x — the <b>bold</b> input',
      'Returns <svg onload=alert(2)>.',
    ].join('\n');
    const el = renderHoverCard(evil);
    // NO markup element was created anywhere in the produced DOM.
    expect(el.querySelector('img')).toBeNull();
    expect(el.querySelector('script')).toBeNull();
    expect(el.querySelector('b')).toBeNull();
    expect(el.querySelector('svg')).toBeNull();
    // The angle-bracket text survives verbatim as literal content (proving it was set via textContent).
    expect(el.textContent).toContain('<img src=x onerror=alert(1)>');
    expect(el.textContent).toContain('<b>bold</b>');
    expect(el.textContent).toContain('<svg onload=alert(2)>');
  });
});
