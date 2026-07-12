// Dev-only demo harness: mounts the example components into a page for a human visual sign-off. NOT part of
// the public API (excluded from the barrel). Run via a scratch HTML page during development.
import { mount, type VDomHandle, type MountOptions } from './mount.ts';
import { COUNTER, TODO, FORM } from './examples.ts';

export function mountDemos(root: Element): VDomHandle[] {
  const doc = root.ownerDocument!;
  const specs: Array<[string, string, MountOptions]> = [
    ['counter', COUNTER, {}],
    ['todo', TODO, { data: [{ id: 0, label: 'first' }, { id: 1, label: 'second' }], reactiveData: true }],
    ['form', FORM, {}],
  ];
  const handles: VDomHandle[] = [];
  for (const [label, src, opts] of specs) {
    const section = doc.createElement('section');
    const h3 = doc.createElement('h3'); h3.textContent = label; section.appendChild(h3);
    const mountPoint = doc.createElement('div'); section.appendChild(mountPoint);
    root.appendChild(section);
    handles.push(mount(src, mountPoint, opts));
  }
  return handles;
}
