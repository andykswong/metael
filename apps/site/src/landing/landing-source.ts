// The landing page's STATIC CHROME, authored as a metael program — so the page you are reading is itself a
// metael program rendered by @metael/vdom (the dogfood, made literal). The one part that cannot be metael is
// the interactive playground: a sandboxed eval-free language can't reach the compiler/clipboard/URL, so the
// host mounts the real playground into the `#hero-slot` div this program leaves empty (an injected child of a
// static, never-re-rendered mount survives — verified). renderLanding (hero.ts) mounts this, then fills the
// slot. Copy names ONLY metael (clean-room). Keep this in sync with styles.css class names.
export const REPO_URL = 'https://github.com/andykswong/metael';

// The shared top header, as a metael snippet reused verbatim by the landing (below) and the playground page
// (main-play.ts mounts HEADER_SOURCE) — the same nav on both, both real metael programs.
const HEADER = `header({ class: "ln-header" }) {
    a({ class: "ln-wordmark", href: "/" }, "metael")
    nav({ class: "ln-nav" }) {
      a({ class: "ln-nav-link", href: "/play.html" }, "Playground")
      a({ class: "ln-nav-link", href: "${REPO_URL}", target: "_blank", rel: "noopener" }, "GitHub ↗")
      a({ class: "ln-nav-link", href: "/api/index.html", target: "_blank", rel: "noopener" }, "API docs ↗")
    }
  }`;

/** A standalone metael program that renders just the shared header — mounted on the playground page. */
export const HEADER_SOURCE = `component Story() {\n  ${HEADER}\n}`;

export const LANDING_SOURCE = `component Story() {
  ${HEADER}
  main({ class: "ln-root" }) {
    section({ class: "ln-hero" }) {
      a({ class: "ln-dogfood", href: "#how-built" }, "▚ This page is written in metael. See how it is built →")
      h1({ class: "ln-title" }) {
        span({ class: "ln-title-line" }, "One eval-free reactive language.")
        span({ class: "ln-title-line" }, "Many targets.")
      }
      p({ class: "ln-sub" }, "Write a component, see live DOM. Write a pure program, see its value. Bring your own vocabulary. No eval — so it is safe to run anything, right here.")
      figure({ class: "ln-hero-frame" }) {
        figcaption({ class: "ln-hero-caption" }) {
          span({ class: "ln-hero-caption-label" }, "Live playground — edit and run it right here")
        }
        div({ class: "ln-hero-playground", id: "hero-slot" })
      }
    }
    section({ class: "ln-section ln-what" }) {
      h2({}, "What is metael")
      ul {
        li({}, "A tiny language kernel — a legible JS/ES-syntax surface run by an eval-free tree-walking interpreter.")
        li({}, "A fine-grained reactive runtime — signals, memos and effects with a synchronous change() boundary.")
        li({}, "A host-injection seam — you supply the vocabulary and what it renders to; metael declares, composes, resolves and reacts.")
        li({}, "For builders of DSLs, creative-coding tools and reactive UIs who don't want to hand-roll a language kernel.")
      }
    }
    section({ class: "ln-section ln-why" }) {
      h2({}, "Why metael")
      ul {
        li({}, "Bring your own vocabulary — the grammar privileges no domain; you define the nouns (the elements) and verbs (the operations) it builds from.")
        li({}, "Eval-free & safe — no eval/new Function; budgeted so runaway loops fail closed.")
        li({}, "Deterministic — result = f(source, data, seed, state); machine-verifiable output.")
        li({}, "Editable AST — a serializable reactive-component tree, not opaque bytecode.")
      }
    }
    section({ class: "ln-section ln-built" }) {
      h2({ id: "how-built" }, "How this page is built")
      p({ class: "ln-built-lead" }, "The header, hero, and these sections are one metael program rendered by @metael/vdom — here is its source. (The code viewer and live editor are host-rendered components the program leaves a slot for.)")
      div({ class: "ln-source", id: "source-slot" })
    }
    footer({ class: "ln-footer" }) {
      span({ class: "ln-footer-copy" }, "© 2026 Andy Wong")
      a(
        {
          class: "ln-footer-link",
          href: "${REPO_URL}/blob/main/LICENSE",
          target: "_blank",
          rel: "noopener"
        },
        "MIT License"
      )
    }
  }
}`;
