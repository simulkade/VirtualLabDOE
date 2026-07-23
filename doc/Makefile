# Makefile for the VlabDOE monograph.
#
#   make            # figures + PDF + HTML
#   make pdf        # build/monograph.pdf
#   make html       # build/html/monograph.html
#   make figures    # (re)generate figures from the package
#   make clean      # remove LaTeX aux files
#   make distclean  # remove the whole build/ tree
#
# Override the interpreter with:  make PYTHON=/path/to/python

PYTHON ?= $(shell [ -x ../.venv/bin/python ] && echo ../.venv/bin/python || echo python3)
BUILD  := build

.PHONY: all pdf html figures clean distclean

all: pdf html

figures:
	$(PYTHON) scripts/make_figures.py
	$(PYTHON) scripts/make_foundations_figures.py

pdf: figures
	REGEN=0 bash scripts/build_pdf.sh

html: figures
	REGEN=0 bash scripts/build_html.sh

clean:
	latexmk -C -output-directory=$(BUILD) main.tex 2>/dev/null || true
	rm -f $(BUILD)/*.aux $(BUILD)/*.log $(BUILD)/*.out $(BUILD)/*.toc \
	      $(BUILD)/*.lof $(BUILD)/*.fls $(BUILD)/*.fdb_latexmk

distclean:
	rm -rf $(BUILD)
