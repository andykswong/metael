import './styles.css';
import { renderLanding } from './landing/hero.ts';

const app = document.getElementById('app');
if (app) renderLanding(app);
