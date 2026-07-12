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

const FORBIDDEN_ATTR = new Set(['innerhtml', 'outerhtml', 'dangerouslysetinnerhtml']);
/** True if `name` is safe to set as a DOM attribute. Rejects on* handler names + raw-HTML sinks. */
export function safeAttrName(name: string): boolean {
  const n = name.toLowerCase();
  if (n.startsWith('on')) return false;
  if (FORBIDDEN_ATTR.has(n)) return false;
  return true;
}

const URL_ATTRS = new Set(['href', 'src', 'xlink:href', 'action', 'formaction', 'poster']);
const BLOCKED_SCHEME = /^\s*(javascript|data|vbscript):/i;
/** For a URL attribute, return null (drop) if the value uses a blocked scheme; else return it unchanged.
 *  Non-URL attributes pass through. */
export function safeAttrValue(name: string, value: string): string | null {
  if (URL_ATTRS.has(name.toLowerCase()) && BLOCKED_SCHEME.test(value)) return null;
  return value;
}
