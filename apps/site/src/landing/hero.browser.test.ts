import { describe, it, expect, beforeEach } from 'vitest';
import { renderLanding } from './hero.ts';

let container: HTMLElement;
beforeEach(() => { container = document.createElement('div'); document.body.appendChild(container); });

describe('renderLanding (real DOM) — a metael-rendered page', () => {
  it('renders the hero title + the pitch sections (via @metael/vdom)', () => {
    renderLanding(container);
    expect(container.querySelector('.ln-title')!.textContent).toContain('reactive language');
    expect(container.querySelector('.ln-what')).not.toBeNull();   // "What is metael" above Why
    expect(container.querySelector('.ln-why')).not.toBeNull();
  });

  it('shows the dogfood callout + footer (the page is itself a metael program)', () => {
    renderLanding(container);
    // the dogfood pill says the page is written in metael + links to the "How it's built" section
    expect(container.querySelector('.ln-dogfood')!.textContent!.toLowerCase()).toContain('metael');
    expect((container.querySelector('.ln-dogfood') as HTMLAnchorElement).getAttribute('href')).toBe('#how-built');
    // the footer has copyright + MIT license link (no more "Built with metael" note — the source viewer section replaced it)
    expect(container.querySelector('.ln-footer-copy')!.textContent).toContain('2026');
    expect((container.querySelector('.ln-footer-link') as HTMLAnchorElement).textContent).toBe('MIT License');
    // the "How it's built" section shows the page's own source, syntax-highlighted
    expect(container.querySelector('#source-slot .ln-source-code')).not.toBeNull();
    expect(container.querySelector('#source-slot .tok-keyword')).not.toBeNull();   // highlighted
  });

  it('embeds a LIVE playground in the hero (a real @metael/vdom mount inside the metael-rendered slot)', () => {
    renderLanding(container);
    const slot = container.querySelector('#hero-slot')!;
    const preview = slot.querySelector('.pg-preview-host')!;
    expect(preview.querySelector('.counter')).not.toBeNull();
    expect(preview.querySelector('.count')!.textContent).toBe('0');
    // the counter buttons are "-", then the count span, then "+"; click "+" and the count updates live
    const plus = Array.from(preview.querySelectorAll('button')).find((b) => b.textContent === '+') as HTMLButtonElement;
    plus.click();
    expect(preview.querySelector('.count')!.textContent).toBe('1');  // the hero demo actually works
  });

  it('the hero playground hides the example picker + share, and shows the toolbar open-editor link (compact)', () => {
    renderLanding(container);
    const slot = container.querySelector('#hero-slot')!;
    expect(slot.querySelector('.pg-examples')).toBeNull();
    expect(slot.querySelector('.pg-share')).toBeNull();
    const open = slot.querySelector('a.pg-open') as HTMLAnchorElement;   // far-right toolbar action
    expect(open.getAttribute('href')).toBe('/play.html');
    expect(open.textContent).toBe('Open in Playground →');   // consistent CTA, same-site arrow
    // the caption label is retained above the frame (no longer a link — the action moved into the toolbar)
    expect(container.querySelector('.ln-hero-caption-label')!.textContent).toContain('Live playground');
    // the old bottom CTA button is gone
    expect(container.querySelector('.ln-cta-btn')).toBeNull();
  });

  it('has a header nav with Playground, GitHub, and API-docs links', () => {
    renderLanding(container);
    const links = Array.from(container.querySelectorAll('.ln-nav-link')) as HTMLAnchorElement[];
    const byText = (t: string) => links.find((a) => a.textContent!.startsWith(t));
    expect(byText('Playground')!.getAttribute('href')).toBe('/play.html');
    expect(byText('GitHub')!.getAttribute('href')).toBe('https://github.com/andykswong/metael');
    expect(byText('API docs')!.getAttribute('href')).toBe('/api/index.html');
    // external links open in a new tab safely
    expect(byText('GitHub')!.getAttribute('rel')).toBe('noopener');
  });

  it('orders Why-metael bullets by importance and drops the "heads" jargon', () => {
    renderLanding(container);
    const items = Array.from(container.querySelectorAll('.ln-why li')).map((li) => li.textContent!);
    expect(items[0]).toContain('Bring your own vocabulary');   // the core reason leads
    expect(items[1]).toContain('Eval-free');
    // the old jargon "register your own heads" must be gone
    expect(items.join(' ')).not.toContain('heads');
  });
});
