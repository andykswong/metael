import 'disposablestack/auto';   // Symbol.dispose + DisposableStack polyfill for Safari (no-op where native)
import './styles.css';
import { renderLanding } from './landing/hero.ts';

const app = document.getElementById('app');
if (app) renderLanding(app);
