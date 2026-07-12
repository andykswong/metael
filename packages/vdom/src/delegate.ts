// Event delegation: one root listener per DOM event walks from the event target up to the nearest ancestor
// whose data-key owns a handler for that event, and dispatches through the handler registry. Delegation
// (not per-node listeners) is what lets handlers survive keyed reconciliation — a handler is keyed by node
// key in the registry, not bound to a specific DOM element that a move/re-create would replace.

export const EVENT_MAP: Record<string, string> = {
  click: 'onClick', input: 'onInput', change: 'onChange', keydown: 'onKeyDown', submit: 'onSubmit',
};

/** Pure ancestor-walk resolution (DOM-free, unit-testable): the nearest chain entry that owns a handler. */
export function resolveHandlerKey<T>(chain: readonly T[], has: (c: T) => boolean, keyOf: (c: T) => string): string | null {
  for (const c of chain) if (has(c)) return keyOf(c);
  return null;
}

/** Attach one delegated listener per event to `root`. On an event, walk from e.target up to `root`,
 *  resolve the nearest data-key with a registered handler, and invoke it via `dispatch` (which wraps the
 *  call in the runtime change() boundary). Returns a detach function. The registry map is read live, so a
 *  re-derive that swaps handlers needs no re-attach. */
export function attachDelegation(
  root: Element,
  registry: Map<string, (arg: unknown) => void>,
  dispatch: (fn: (arg: unknown) => void, ev: Event) => void,
): () => void {
  const listeners: [string, EventListener][] = [];
  for (const [domEvent, handlerName] of Object.entries(EVENT_MAP)) {
    const listener: EventListener = (ev) => {
      let el: Element | null = ev.target as Element | null;
      while (el && el !== root) {
        const key = el.getAttribute?.('data-key');
        if (key) { const fn = registry.get(`${key}:${handlerName}`); if (fn) { dispatch(fn, ev); return; } }
        el = el.parentElement;
      }
    };
    root.addEventListener(domEvent, listener);
    listeners.push([domEvent, listener]);
  }
  return () => { for (const [e, l] of listeners) root.removeEventListener(e, l); };
}
