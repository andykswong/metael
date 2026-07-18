import { describe, it, expect } from 'vitest';
import { escapeText, safeAttrName, safeAttrValue } from './sanitize.ts';

describe('escapeText — for the HTML-string (static-render) path only, NOT live text nodes', () => {
  it('escapes the five HTML-significant chars', () => {
    expect(escapeText('<script>alert(1)</script>')).toBe('&lt;script&gt;alert(1)&lt;/script&gt;');
    expect(escapeText(`a & b "c" 'd'`)).toBe('a &amp; b &quot;c&quot; &#39;d&#39;');
  });
  it('coerces non-strings', () => {
    expect(escapeText(42 as unknown as string)).toBe('42');
    expect(escapeText(null as unknown as string)).toBe('');
  });
});

describe('safeAttrName — block event-handler + raw-HTML attribute names', () => {
  it('rejects on* handler attribute names', () => {
    expect(safeAttrName('onclick')).toBe(false);
    expect(safeAttrName('onerror')).toBe(false);
  });
  it('rejects the raw-HTML sink', () => {
    expect(safeAttrName('innerHTML')).toBe(false);
    expect(safeAttrName('dangerouslySetInnerHTML')).toBe(false);
  });
  it('rejects srcdoc (an iframe raw-HTML-document sink)', () => {
    expect(safeAttrName('srcdoc')).toBe(false);
    expect(safeAttrName('srcDoc')).toBe(false);
  });
  it('allows ordinary attribute names', () => {
    for (const n of ['class', 'id', 'data-key', 'aria-label', 'href', 'value']) expect(safeAttrName(n)).toBe(true);
  });
});

describe('safeAttrValue — block javascript:/data:/vbscript: on URL attributes', () => {
  it('blocks dangerous schemes on href/src (case + whitespace insensitive)', () => {
    expect(safeAttrValue('href', 'javascript:alert(1)')).toBe(null);
    expect(safeAttrValue('href', '  JaVaScRiPt:alert(1)')).toBe(null);
    expect(safeAttrValue('src', 'data:text/html,<script>')).toBe(null);
  });
  it('blocks schemes obfuscated with embedded tab/newline/CR (browsers strip these before scheme resolution)', () => {
    expect(safeAttrValue('href', 'java\tscript:alert(1)')).toBe(null);
    expect(safeAttrValue('href', 'java\nscript:alert(1)')).toBe(null);
    expect(safeAttrValue('href', 'java\rscript:alert(1)')).toBe(null);
    expect(safeAttrValue('href', 'jav\ta\nscript\r:alert(1)')).toBe(null);
  });
  it('blocks schemes hidden behind leading control chars', () => {
    expect(safeAttrValue('href', '\x01javascript:alert(1)')).toBe(null);
    expect(safeAttrValue('href', '\x00javascript:alert(1)')).toBe(null);
    expect(safeAttrValue('src', '\x1fdata:text/html,<script>')).toBe(null);
  });
  it('allows safe URLs and returns them UNCHANGED (does not mutate the value)', () => {
    for (const u of ['https://example.com', '/rel/path', '#anchor', 'mailto:a@b.com']) expect(safeAttrValue('href', u)).toBe(u);
    // a legitimate value that merely contains tab/newline is returned verbatim (only the scheme test is normalized)
    expect(safeAttrValue('href', '/path?q=a\tb')).toBe('/path?q=a\tb');
  });
  it('does not URL-filter non-URL attributes', () => {
    expect(safeAttrValue('class', 'javascript:looking-name')).toBe('javascript:looking-name');
  });
});
