#!/usr/bin/env bash
# Convert the monograph to a single self-contained HTML page with Pandoc.
#
#   bash scripts/build_html.sh            # build/html/monograph.html
#   REGEN=0 bash scripts/build_html.sh    # skip figure regeneration
#
# Math is rendered client-side with MathJax; the numbered sections, figures,
# cross-references, and the titled call-out boxes (styled via monograph.css)
# all carry over from the LaTeX source.
set -euo pipefail

DOC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$DOC_DIR/build/html"
cd "$DOC_DIR"
mkdir -p "$OUT_DIR"

# --- 1. (Re)generate figures from the package, unless REGEN=0 ----------------
if [[ "${REGEN:-1}" != "0" ]]; then
  echo ">> Generating figures from the package ..."
  PY="${PYTHON:-}"
  if [[ -z "$PY" ]]; then
    if [[ -x "$DOC_DIR/../.venv/bin/python" ]]; then PY="$DOC_DIR/../.venv/bin/python";
    else PY="python3"; fi
  fi
  "$PY" "$DOC_DIR/scripts/make_figures.py" || \
    echo "!! figure generation failed; continuing (figures may be stale/missing)"
  "$PY" "$DOC_DIR/scripts/make_foundations_figures.py" || \
    echo "!! foundations figure generation failed; continuing"
fi

# --- 2. Stage assets next to the HTML ----------------------------------------
# Figures are referenced by bare filename in the LaTeX source, so copy them
# (and the stylesheet) flat into the output directory.  This keeps the build
# fully offline; MathJax is loaded from a CDN by the browser at view time.
cp -f "$DOC_DIR"/figures/*.png "$OUT_DIR"/ 2>/dev/null || \
  echo "!! no figures to copy (run make_figures first)"
cp -f "$DOC_DIR/scripts/monograph.css" "$OUT_DIR/monograph.css"

MATHJAX_URL="${MATHJAX_URL:-https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js}"

# --- 3. Convert with Pandoc --------------------------------------------------
echo ">> Converting to HTML with Pandoc ..."
pandoc main.tex \
  --from=latex \
  --to=html5 \
  --standalone \
  --toc --toc-depth=2 \
  --number-sections \
  --mathjax="$MATHJAX_URL" \
  --metadata title="VlabDOE — A Monograph" \
  --resource-path=".:figures" \
  --css="monograph.css" \
  --output="$OUT_DIR/monograph.html"

# Pandoc's default html5 template injects a <script> from polyfill.io, a domain
# that was taken over in 2024 and no longer serves a valid certificate. MathJax
# 3 needs no polyfill on any modern browser, so strip that line from the output.
sed -i '/polyfill\.io/d' "$OUT_DIR/monograph.html"

echo ">> Done: $OUT_DIR/monograph.html  (figures + monograph.css alongside it)"
