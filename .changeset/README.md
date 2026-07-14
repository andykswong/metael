# Changesets

This folder holds [changesets](https://github.com/changesets/changesets) — one Markdown file per
pending change, describing which `@metael/*` packages it affects and at what semver bump. The three
publishable packages (`@metael/{lang,runtime,vdom}`) are versioned in **lockstep** (`fixed`), so any
bump moves all three together; `@metael/site` is ignored (private).

- Add a changeset for a change:  `npm run changeset`  (pick the bump level + write a summary)
- Preview the pending release:    `npx changeset status`
- Versions + changelogs are applied by CI (`npm run version`) via a "Version Packages" PR; merging
  that PR publishes to npm with provenance (`npm run release`). See `.github/workflows/release.yaml`.
