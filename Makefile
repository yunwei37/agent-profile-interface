TEX = aiops26-template
PDF = $(TEX).pdf

all: $(PDF)

$(PDF): $(TEX).tex $(TEX).bib
	pdflatex $(TEX)
	bibtex $(TEX)
	pdflatex $(TEX)
	pdflatex $(TEX)

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fdb_latexmk *.fls *.synctex.gz

distclean: clean
	rm -f $(PDF)

.PHONY: all clean distclean
