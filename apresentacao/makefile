# Project configs
BRIEF  := main
OUTPUT := ine5426_AL_AS1

TARGETS := $(BRIEF)

# Compiler configs
ifndef LATEX
	LATEX  := xelatex
endif
BIBTEX := bibtex

# Etc.
OBJS   := $(wildcard **/*.tex) $(wildcard *.tex)

# Rules
.PHONY: clean clean-pdf all

all: $(TARGETS)

main:  $(BRIEF).pdf

%.pdf: $(OBJS)
	$(LATEX) --shell-escape $(basename $@)
	#$(LATEX) --shell-escape $(basename $@)

	cp $@ $(OUTPUT).pdf

clean:
	rm -f *.{aux,bbl,blg,log,nav,snm,out,toc}
	rm -rf _minted-*

clean-pdf:
	rm $(BRIEF).pdf
