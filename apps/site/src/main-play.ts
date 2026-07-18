import 'disposablestack/auto';   // Symbol.dispose + DisposableStack polyfill for Safari (no-op where native)
import './styles.css';
import { mount } from '@metael/vdom';
import { createPlayground } from './playground/create.ts';
import { decodeState } from './playground/share.ts';
import { HEADER_SOURCE } from './landing/landing-source.ts';

async function boot(): Promise<void> {
  const app = document.getElementById('app');
  if (!app) return;
  // The same metael-rendered header as the landing (dogfood), then the playground below it.
  mount(HEADER_SOURCE, app, {});
  const playHost = document.createElement('div');
  playHost.className = 'pg-page';
  app.appendChild(playHost);
  const fragment = location.hash.slice(1);
  const decoded = fragment ? await decodeState(fragment) : null;
  createPlayground(playHost, { initialState: decoded && decoded.ok ? decoded.state : undefined });
}
void boot();
