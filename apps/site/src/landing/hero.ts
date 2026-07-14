// The landing page. The static chrome (header + hero + What/Why + "How it's built" + footer) is authored as a
// metael program (landing-source.ts) and rendered by @metael/vdom — so the page IS a metael program. The host
// fills two slots after mount: the live playground (#hero-slot) and the syntax-highlighted source viewer
// (#source-slot). Both are capabilities the sandboxed language leaves a "hole" for — the playground because it
// needs host APIs, the source viewer as a deliberate demo of "bring your own vocabulary" (the host registers
// the rendering of a slot the language defines).
import { mount } from '@metael/vdom';
import { el } from '../ui.ts';
import { createPlayground } from '../playground/create.ts';
import { tokensToSegments } from '../playground/highlight.ts';
import { LANDING_SOURCE } from './landing-source.ts';

export function renderLanding(root: Element): void {
  // Mount the static chrome as a real metael program.
  mount(LANDING_SOURCE, root, {});

  // Fill the playground slot.
  const playSlot = root.querySelector('#hero-slot');
  if (playSlot) createPlayground(playSlot, { compact: true, defaultExampleId: 'counter', openHref: 'play.html' });

  // Fill the source-viewer slot: syntax-highlight the LANDING_SOURCE (the page's own code) using the same
  // lex()-driven highlighter the editor uses — so "this page shows its own source" is literally true.
  const sourceSlot = root.querySelector('#source-slot');
  if (sourceSlot) {
    const pre = el('pre', { class: 'ln-source-code' });
    for (const seg of tokensToSegments(LANDING_SOURCE)) {
      pre.append(el('span', { class: `tok-${seg.kind}` }, [seg.text]));
    }
    sourceSlot.appendChild(pre);
  }
}
