import { describe, it, expect, beforeEach } from 'vitest';
import { renderSource } from '@metael/vdom/lang';
import { LANDING_SOURCE, HEADER_SOURCE } from './landing-source.ts';

let container: HTMLElement;
beforeEach(() => { container = document.createElement('div'); document.body.appendChild(container); });

describe('landing metael sources render cleanly', () => {
  it('LANDING_SOURCE derives with zero diagnostics and exposes a fillable #hero-slot', () => {
    const h = renderSource(LANDING_SOURCE, container, {});
    expect(h.diagnostics).toEqual([]);
    expect(container.querySelector('header.ln-header')).not.toBeNull();
    expect(container.querySelector('main.ln-root')).not.toBeNull();
    const slot = container.querySelector('#hero-slot') as HTMLElement;
    expect(slot).not.toBeNull();
    expect(slot.childNodes.length).toBe(0);   // empty slot for the host playground
    // a host-injected child into a static (never-re-rendered) mount survives
    const inj = document.createElement('div'); inj.className = 'injected';
    slot.appendChild(inj);
    expect(container.querySelector('#hero-slot .injected')).not.toBeNull();
  });

  it('HEADER_SOURCE (used on the playground page) renders the shared nav', () => {
    const h = renderSource(HEADER_SOURCE, container, {});
    expect(h.diagnostics).toEqual([]);
    const links = Array.from(container.querySelectorAll('.ln-nav-link')).map((a) => a.textContent);
    expect(links).toEqual(['Playground', 'GitHub ↗', 'API docs ↗']);
    expect(container.querySelector('.ln-wordmark')!.textContent).toBe('metael');
  });
});
