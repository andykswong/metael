import { describe, it, expect } from 'vitest';
import * as vdom from './index.ts';

describe('@metael/vdom public API surface', () => {
  it('exports the API-first seam alongside the DSL mount()', () => {
    expect(typeof vdom.mount).toBe('function');       // DSL path unchanged
    expect(typeof vdom.h).toBe('function');           // API-first builder
    expect(typeof vdom.render).toBe('function');      // API-first driver
    expect(vdom.Fragment).toBe('');                   // FRAGMENT sentinel re-exported as Fragment
  });
});
