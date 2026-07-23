import type { Profile, HeadSpec } from '@metael/lang/profile';

// Every DOM-element head shares one call shape: an optional leading PROPS object (attributes + event
// handlers) followed by any number of CHILD nodes (or a string child). `returnDoc` may be tailored per
// tag, but every tag carries the props + children params so a hover card lists both arguments.
const tag = (name: string, doc: string, returnDoc = 'a DOM element node'): HeadSpec => ({
  name,
  params: [
    { name: 'props', optional: true, doc: 'an object of attributes + event handlers (e.g. { class, onClick })' },
    { name: 'children', rest: true, doc: 'child nodes or text' },
  ],
  arity: [0, Infinity],
  returns: 'node',
  doc,
  returnDoc,
  takesChildren: true,
});

const COMMON_TAGS: readonly HeadSpec[] = [
  tag('div', 'A generic block container.'),
  tag('span', 'A generic inline container.'),
  tag('button', 'A clickable button (e.g. { onClick }).', 'a button node'),
  tag('input', 'A form input control (e.g. { value, type, onInput }).', 'a form input node'),
  tag('p', 'A paragraph.'),
  tag('ul', 'An unordered list.'), tag('li', 'A list item.'),
  tag('a', 'A hyperlink (set the target with { href }).', 'a hyperlink node'),
  tag('h1', 'A top-level heading.'),
];

/** The vdom tooling profile: an OPEN head set (any lowercase tag is a valid element), with documented
 *  specs for common tags. No builtins or custom types. */
export const vdomProfile: Profile = {
  id: 'vdom',
  builtins: new Map(),
  heads: new Map(COMMON_TAGS.map((h) => [h.name, h])),
  types: new Map(),
  permissiveHeads: true,
};
