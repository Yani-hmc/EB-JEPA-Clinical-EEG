all: report.pdf beamer.pdf

report.pdf: report.tex figures/comparison_dual.png figures/ablation_what_moved.png \
            figures/encoder_comparison.png figures/vicreg_fix.png
	pdflatex -interaction=nonstopmode report.tex
	pdflatex -interaction=nonstopmode report.tex

beamer.pdf: beamer.tex figures/ablation_what_moved.png figures/vicreg_fix.png \
            figures/encoder_comparison.png
	pdflatex -interaction=nonstopmode beamer.tex
	pdflatex -interaction=nonstopmode beamer.tex

figures/%.png:
	cd figures && python3 make_figures.py

illustrations/cave_allegory.pdf: illustrations/cave_allegory.tex
	cd illustrations && pdflatex -interaction=nonstopmode cave_allegory.tex

clean:
	rm -f *.aux *.log *.out *.toc *.nav *.snm *.vrb *.fls *.fdb_latexmk
