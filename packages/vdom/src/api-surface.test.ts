import { describe, it, expect } from 'vitest';
import * as vdom from './index.ts';

describe('@metael/vdom public API surface', () => {
  it('exports the API-first core (render/h) with no DSL binding on the core barrel', () => {
    expect(typeof vdom.render).toBe('function');      // API-first driver
    expect(typeof vdom.h).toBe('function');           // API-first builder
    expect(vdom.Fragment).toBe('');                   // FRAGMENT sentinel re-exported as Fragment
    // The DSL binding lives in @metael/vdom/lang — the core barrel carries no mount/renderSource.
    expect((vdom as Record<string, unknown>).mount).toBeUndefined();
    expect((vdom as Record<string, unknown>).renderSource).toBeUndefined();
  });
});
