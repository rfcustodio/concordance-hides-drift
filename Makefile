# Reproduce every number and figure of the manuscript.
PY ?= python3 -I
.PHONY: all test results figures clean
all: test results figures
test:
	$(PY) -m pytest -q tests
results:
	$(PY) experiments/run_all.py --out results
	$(PY) experiments/postprocess.py results
	$(PY) scripts/validate_label.py examples/dynamic_ai_safety_label.example.json
figures:
	cd figures && for f in standalone_*.tex; do pdflatex -interaction=nonstopmode -halt-on-error $$f >/dev/null || exit 1; done
	cd figures && pdflatex -interaction=nonstopmode -halt-on-error all_figures.tex >/dev/null
clean:
	rm -f figures/*.aux figures/*.log figures/*.pdf
