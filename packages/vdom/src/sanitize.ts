// Output safety at the DOM-patch boundary. metael is eval-free (user CODE can't run), but a vnode renders
// to real DOM, so this is ordinary templating XSS discipline: forbid handler/raw-HTML attribute names +
// block dangerous URL schemes on URL-bearing attributes. escapeText is provided for a future HTML-string
// (static-render) path; the LIVE-DOM patcher writes text via a Text node (which does not parse HTML) so it
// does NOT escape — escaping a text node would double-encode + corrupt visible text.

const ESCAPE: Record<string, string> = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };

/** Escape the five HTML-significant characters — for building an HTML STRING (static render), not for a
 *  live text node (a Text node is XSS-safe by construction). */
export function escapeText(v: unknown): string {
  if (v === null || v === undefined) return '';
  const s = typeof v === 'string' ? v : String(v);
  return s.replace(/[&<>"']/g, (c) => ESCAPE[c]!);
}

// Raw-HTML sinks that must never be set as attributes. `srcdoc` parses its value as a full HTML document
// (iframe), so like innerHTML it is a script-execution sink with no scheme-check remedy — forbid the name.
const FORBIDDEN_ATTR = new Set(['innerhtml', 'outerhtml', 'dangerouslysetinnerhtml', 'srcdoc']);
/** True if `name` is safe to set as a DOM attribute. Rejects on* handler names + raw-HTML sinks. */
export function safeAttrName(name: string): boolean {
  const n = name.toLowerCase();
  if (n.startsWith('on')) return false;
  if (FORBIDDEN_ATTR.has(n)) return false;
  return true;
}

const URL_ATTRS = new Set(['href', 'src', 'xlink:href', 'action', 'formaction', 'poster']);
const BLOCKED_SCHEME = /^(javascript|data|vbscript):/i;
/** For a URL attribute, return null (drop) if the value uses a blocked scheme; else return it unchanged.
 *  Non-URL attributes pass through.
 *
 *  The scheme test runs against a NORMALIZED copy of the value, mirroring how a browser resolves a URL:
 *  ASCII tab/LF/CR are stripped from ANYWHERE (a browser removes them before scheme resolution, so
 *  `java\tscript:` becomes `javascript:`), and leading C0 control chars + spaces are trimmed. Without this,
 *  those characters slip a blocked scheme past a naive `^\s*` guard. The ORIGINAL (unmodified) value is
 *  returned when allowed — normalization only gates the decision, it never mutates a legitimate value. */
export function safeAttrValue(name: string, value: string): string | null {
  if (URL_ATTRS.has(name.toLowerCase())) {
    if (BLOCKED_SCHEME.test(normalizeUrlForSchemeTest(value))) return null;
  }
  return value;
}

/** Normalize a URL the way a browser does before resolving its scheme: strip ASCII tab/LF/CR from anywhere
 *  (a browser removes them everywhere in a URL, so `java\tscript:` reads as `javascript:`), then trim any
 *  leading C0 control chars + space. Used ONLY to decide whether to block — never to mutate a kept value.
 *  Leading controls are trimmed by code point (not a control-char regex literal, which lint forbids). */
function normalizeUrlForSchemeTest(value: string): string {
  const stripped = value.replace(/[\t\n\r]/g, '');
  let i = 0;
  while (i < stripped.length && stripped.charCodeAt(i) <= 0x20) i++;
  return stripped.slice(i);
}
