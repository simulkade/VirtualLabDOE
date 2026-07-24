#!/usr/bin/env bash
# Local build-and-release of the monograph.
#
#   bash scripts/release.sh             # tag = v<version> from pyproject.toml
#   bash scripts/release.sh v0.2.0      # explicit tag
#   REGEN=0 bash scripts/release.sh     # reuse existing figures (skip regeneration)
#
# Builds two self-contained, downloadable assets and attaches them to a GitHub
# Release (created if absent, otherwise its assets are refreshed):
#
#   build/release/monograph.pdf   - the typeset PDF (pdflatex/latexmk)
#   build/release/monograph.html  - ONE self-contained web page: every figure
#                                   and the stylesheet are embedded as data URIs,
#                                   so the file renders on its own after download.
#                                   (Equations are rendered client-side via the
#                                   MathJax CDN, so viewing math needs a network.)
#
# Requires: pdflatex + latexmk, pandoc (>= 2.19 for --embed-resources), and an
# authenticated `gh` (run `gh auth status`).
set -euo pipefail

DOC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT_DIR="$(cd "$DOC_DIR/.." && pwd)"
REL_DIR="$DOC_DIR/build/release"
cd "$DOC_DIR"
mkdir -p "$REL_DIR"

VERSION="$(grep -iE '^version' "$ROOT_DIR/pyproject.toml" | head -1 | sed -E 's/.*"([^"]+)".*/\1/')"
TAG="${1:-v$VERSION}"

# --- 1. figures (shared by both outputs) -------------------------------------
if [[ "${REGEN:-1}" != "0" ]]; then
  echo ">> Generating figures from the package ..."
  PY="${PYTHON:-}"
  if [[ -z "$PY" ]]; then
    if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then PY="$ROOT_DIR/.venv/bin/python"; else PY="python3"; fi
  fi
  "$PY" "$DOC_DIR/scripts/make_figures.py"            || echo "!! figure generation failed; continuing"
  "$PY" "$DOC_DIR/scripts/make_foundations_figures.py" || echo "!! foundations figure generation failed; continuing"
fi

# --- 2. PDF ------------------------------------------------------------------
echo ">> Building PDF ..."
REGEN=0 bash "$DOC_DIR/scripts/build_pdf.sh"
cp -f "$DOC_DIR/build/monograph.pdf" "$REL_DIR/monograph.pdf"

# --- 3. self-contained HTML --------------------------------------------------
echo ">> Building self-contained HTML ..."
MATHJAX_URL="${MATHJAX_URL:-https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js}"
pandoc main.tex \
  --from=latex --to=html5 \
  --standalone --embed-resources \
  --toc --toc-depth=2 --number-sections \
  --mathjax="$MATHJAX_URL" \
  --metadata title="Virtual Lab DOE — A Monograph" \
  --resource-path=".:figures" \
  --css="scripts/monograph.css" \
  --output="$REL_DIR/monograph.html"

# Pandoc's default html5 template injects a <script> from polyfill.io, a domain
# that was taken over in 2024 and no longer serves a valid certificate. MathJax
# 3 needs no polyfill on any modern browser, so strip that line from the output.
sed -i '/polyfill\.io/d' "$REL_DIR/monograph.html"

# --- 4. GitHub release -------------------------------------------------------
TITLE="Virtual Lab DOE Monograph $TAG"
NOTES=$(cat <<EOF
Downloadable monograph build.

- **monograph.pdf** — the typeset PDF.
- **monograph.html** — a single self-contained web page (figures and styling embedded; equations rendered via MathJax).

Every figure is generated directly from the package source.
EOF
)

echo ">> Publishing release $TAG ..."
if gh release view "$TAG" >/dev/null 2>&1; then
  gh release upload "$TAG" "$REL_DIR/monograph.pdf" "$REL_DIR/monograph.html" --clobber
else
  gh release create "$TAG" "$REL_DIR/monograph.pdf" "$REL_DIR/monograph.html" \
    --title "$TITLE" --notes "$NOTES"
fi
echo ">> Done: release $TAG carries monograph.pdf + monograph.html"

# --- 5. Publish the HTML page to GitHub Pages (gh-pages branch) ---------------
# The release asset is served as an attachment (it downloads); GitHub Pages
# serves the same page as text/html so it opens in the browser. We keep a single
# orphan commit on gh-pages (force-pushed each release) to avoid repo bloat.
if [[ "${PAGES:-1}" != "0" ]]; then
  echo ">> Publishing HTML to GitHub Pages (gh-pages) ..."
  WT="$(mktemp -d)"
  git -C "$ROOT_DIR" worktree add --force --detach "$WT" >/dev/null
  (
    cd "$WT"
    git checkout --orphan gh-pages-tmp >/dev/null 2>&1
    git rm -rf . >/dev/null 2>&1 || true
    cp -f "$REL_DIR/monograph.html" index.html
    touch .nojekyll   # skip Jekyll so the raw HTML is served verbatim
    git add -A
    git commit -q -m "Publish monograph HTML ($TAG)"
    git push -q --force origin gh-pages-tmp:gh-pages
  )
  git -C "$ROOT_DIR" worktree remove --force "$WT"
  git -C "$ROOT_DIR" branch -D gh-pages-tmp >/dev/null 2>&1 || true
  echo ">> Pages updated: https://simulkade.github.io/VirtualLabDOE/"
fi
