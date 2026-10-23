.PHONY: all test fulltest fixtures e0 paper
all:
	bash build.sh
test:
	python3 -m pytest tests -q
fulltest:
	CGMY_FULL_TESTS=1 python3 -m pytest tests -q
fixtures:
	python3 scripts/build_fixtures.py
e0:
	python3 scripts/run_E0.py --nproc 2
paper:
	cd manuscript_v9 && pdflatex -interaction=nonstopmode main_v9 && pdflatex -interaction=nonstopmode main_v9 && pdflatex -interaction=nonstopmode main_v9
