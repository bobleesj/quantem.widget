#!/usr/bin/env python3
"""Guard the docs mobile hamburger against sphinx theme drift.

Both sphinx-book-theme and pydata-sphinx-theme wire the sidebar drawer to
``document.querySelector('.primary-toggle')`` — the FIRST matching element.
pydata-sphinx-theme >= 0.17 renders its own extra (display: none) header
button with that class ahead of the visible sphinx-book-theme hamburger, so
every handler binds to the invisible button and the visible hamburger goes
dead (found live on 2026-07-28: the phone nav could not be opened at all).

This check fails the build when a built page carries more than one
``primary-toggle`` button, which is exactly the state where the wrong button
receives the handlers. Run after ``jupyter-book build docs``:

    python scripts/check_docs_nav_toggle.py [docs/_build/html]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

TOGGLE_RE = re.compile(r"<(?:button|label)[^>]*class=\"[^\"]*\bprimary-toggle\b", re.I)


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/_build/html")
    pages = [p for p in root.rglob("*.html") if "_sources" not in p.parts]
    if not pages:
        print(f"check_docs_nav_toggle: no built pages under {root}", file=sys.stderr)
        return 1
    bad = []
    for page in pages:
        count = len(TOGGLE_RE.findall(page.read_text(encoding="utf-8", errors="replace")))
        if count > 1:
            bad.append((page.relative_to(root), count))
    if bad:
        print(
            "check_docs_nav_toggle: FAIL — pages with more than one "
            "primary-toggle button (theme drift: handlers bind to the first, "
            "hidden one and the visible hamburger goes dead). Pin "
            "sphinx-book-theme / pydata-sphinx-theme in docs/requirements.txt "
            "to a pair that renders a single toggle:",
            file=sys.stderr,
        )
        for rel, count in bad[:20]:
            print(f"  {rel}: {count} primary-toggle buttons", file=sys.stderr)
        return 1
    print(f"check_docs_nav_toggle: OK — {len(pages)} pages, one primary-toggle each")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
