TOP_DIR := $(shell pwd)

SCRIPT_DIR := script
BUILD_DIR := build
BACKUP_DIR := backups
LIB_DIR := lib
FIGURES_DIR := build/figures
DEPS_DIR := .dependencies
GNUPLOT_TERMINAL := epslatex color solid
GNUPLOT := gnuplot
FIG2DEV := fig2dev
INKSCAPE = inkscape
OCTAVE := octave
MATLAB := matlab
PYTHON := python3
MATPLOTLIB := $(PYTHON) $(SCRIPT_DIR)/domatplotlib.py --backend agg
JULIA := julia
PDFLATEX_FLAGS :=
DEFAULT_FIGURE_TYPE := paper

# Set defaults for using backed up files vs. regenerating them. Can be
# overridden locally by changing them in a file called Makefile.options.
USE_MAT_BACKUPS := True
USE_PICKLE_BACKUPS := True
USE_DIA_BACKUPS := False

## We don't need figure windows popping up.
OCTAVE_FLAGS := --silent \
                --path "$(TOP_DIR)/$(LIB_DIR)" \
                --eval "set(0,'DefaultFigureVisible','off'); cd $(BUILD_DIR);"
GS_GREY_FLAGS := -sDEVICE=pdfwrite -sColorConversionStrategy=Gray -dProcessColorModel=/DeviceGray \
                -dCompatibilityLevel=1.4 -dNOPAUSE -dBATCH -dQUIET
EPSTOPDF := epstopdf
PDFVIEWER := xpdf

TEXSUFFIXES := .pdf .aux .log .bbl .blg .nav .out .snm .toc .synctex.gz \
    .lof .lot -blx.bib .run.xml .fls

# We keep these separate so that we can change figure font depending on whether
# we are making the article or the talk. Note that the nomenclature is kind of
# confusing here: in the subfolders, we split paper/presentation/poster, whereas
# these variables are article/talk/poster, so keep that in mind.
ARTICLE_SRC := paper.tex resultssummary.tex nnrobustness.tex
TALK_SRC := 
POSTER_SRC := 

PAPER_BIB := 
PAPER_MENDELEYBIB := 

## TEX_SRC. These are .tex files included in the main document that are not generated
## by other programs.

TEX_SRC := 

# FIG_SRC.  These are figures that can be described completely using fig
## commands and that require no external data.

FIG_SRC := 

## INKSCAPE_SRC. These are figures defined by an Inkscape .svg file

INKSCAPE_SRC := 

## GNUPLOT_SRC. These are Gnuplots scripts that produce .tex and .eps files,
## which are eventually included in LaTeX documents. Any data files used by
## these figures can be specified in their headers.

GNUPLOT_SRC :=

## EXTRA_EPS.  These are standalone PostScript figures that have no
## corresponding source files. They get converted to PDFs and put in the
## build/figures directory.

EXTRA_EPS := 

## EXTRA_INCLUDED_FILES.  These are files that are included directly in Latex
## documents and require no extra processing. They are simply symlinked in the
## build directory so that Latex can find them. Note that you can achieve the
## same effect by placing these files in the texmf folder.

EXTRA_INCLUDED_FILES := 

## STANDALONE_TEX_SRC. These are .tex files that are compiled on their own into
## pdfs which are then included within the paper. This is useful, e.g., for
## precompiling TIKZ pictures and then sending them as pdfs to people that don't
## have TIKZ.

STANDALONE_TEX_SRC := 

## DIA_SRC files are made in dia. They are exported as eps.

DIA_SRC := cstrs_with_flash.dia cdu.dia

## PY_MAT_SRC, OCT_MAT_SRC. These are .py and .m files that create .mat files
## to be shared to make plots. Dependencies should be specified for any subsequent
## files that use these data files. See README.md in $(DEPS_DIR) for more
## details. These .mat files can also be backed up in $(BACKUPS_DIR).
## PY_PICKLE_SRC, are the pickle files that create .pickle files. These
## .pickle files can also be backed up in $(BACKUP_DIR)

PY_MAT_SRC :=
OCT_MAT_SRC := 
MATLAB_MAT_SRC := 
PY_PICKLE_SRC := di_parameters.py di_mpc.py di_us.py di_satdlqr.py \
                 di_short_horizon.py di_train.py di_neural_network.py \
				 cstrs_parameters.py cstrs_mpc.py cstrs_us.py cstrs_satdlqr.py \
                 cstrs_short_horizon.py cstrs_train.py cstrs_neural_network.py \
				 cdu_parameters.py cdu_mpc.py cdu_us.py cdu_satdlqr.py \
				 cdu_short_horizon.py cdu_train.py cdu_neural_network.py

## JL_MAT_SRC. These are .jl files on which you run Julia to produce a .mat file.
## Note that Julia's .mat files are in an HDF5 format, so you will need to use
## different means to open them in Octave/Python.

JL_MAT_SRC :=

## OCT_DAT_SRC. Same as OCT_MAT_SRC, except that a .dat file is produced instead of
## a .mat file, and the .dat files cannot be backed up in $(BACKUPS_DIR).
 
OCT_DAT_SRC := 

## PY_PLOT files are .py files that create .pdf plots.

PY_PLOT := di_comparision_plots.py cstrs_comparision_plots.py \
		   cdu_comparision_plots.py

## PY_MOVIE files are .py files that create movies as a .pdf document.

PY_MOVIE := 

## OCT_PLOT files are .m files that create .pdf plots.

OCT_PLOT := 

## OCT_TABLE files are .m files that create .tex tables. PY_TABLE are .py files
## that create .tex tables.

OCT_TABLE := 
PY_TABLE := 

## SCRATCH_FILES are temporary files that are created during various script executions.
## These are removed by target "realclean"

SCRATCH_FILES := 

## HGIGNORE files are extra files to be included in .hgignore. Wildcards are allowed.

HGIGNORE := 

## ************************************************************************************************
## End of user specified files.
## ************************************************************************************************

# Append prefixes and everything.
ARTICLE_PDF := $(ARTICLE_SRC:.tex=.pdf)
ARTICLE_GREY_SRC := $(addprefix $(BUILD_DIR)/, $(ARTICLE_SRC:.tex=-grey.tex))
ARTICLE_GREY_PDF := $(ARTICLE_SRC:.tex=-grey.pdf)
TALK_PDF := $(TALK_SRC:.tex=.pdf)
HANDOUT_PDF := $(TALK_SRC:.tex=-handout.pdf)
HANDOUT_SRC := $(addprefix $(BUILD_DIR)/, $(TALK_SRC:.tex=-handout.tex))
FOURUP_PDF := $(TALK_SRC:.tex=-4up.pdf)
POSTER_PDF := $(POSTER_SRC:.tex=.pdf)

ARTICLE_BUILD_PDF := $(addprefix $(BUILD_DIR)/, $(ARTICLE_PDF))
ARTICLE_GREY_BUILD_PDF := $(addprefix $(BUILD_DIR)/, $(ARTICLE_GREY_PDF))
TALK_BUILD_PDF := $(addprefix $(BUILD_DIR)/, $(TALK_PDF))
HANDOUT_BUILD_PDF := $(addprefix $(BUILD_DIR)/, $(HANDOUT_PDF))
POSTER_BUILD_PDF := $(addprefix $(BUILD_DIR)/, $(POSTER_PDF))

PAPER_BUILD_PDF := $(ARTICLE_BUILD_PDF) $(ARTICLE_GREY_BUILD_PDF) \
                   $(TALK_BUILD_PDF) $(HANDOUT_BUILD_PDF) $(POSTER_BUILD_PDF)

PAPER_SRC := $(ARTICLE_SRC) $(TALK_SRC) $(POSTER_SRC)
PAPER_PDF  := $(notdir $(PAPER_BUILD_PDF))

PAPER_XBIB := $(PAPER_SRC:.tex=.xbib)
PAPER_BUILD_XBIB := $(addprefix $(BUILD_DIR)/, $(PAPER_XBIB))

PAPER_DIST_ZIP := $(PAPER_SRC:.tex=.zip)

PAPER_MISC := $(foreach paper, $(basename $(PAPER_BUILD_PDF)), $(addprefix $(paper), $(TEXSUFFIXES)))
PAPER_MISCBIB := $(filter %.bbl %.blg %.aux, $(PAPER_MISC)) ## Grab bibtex-related intermediates.

LOCAL_TEXMF_FILES := $(filter-out %README, $(wildcard texmf/*))

# Set dependency names.
PAPER_MENDELEYBIB_CLEANED := $(addprefix $(BUILD_DIR)/, $(PAPER_MENDELEYBIB:.mendeleybib=.bib))

# Gnuplot files. This one is tricky because a single run of gnuplot produces both
# a .tex and a .eps file. We spoof Make by telling it gnuplot produces a .gnuplot
# file, and that .gnuplot file then produces the .tex and .eps. Note that since
# the .gnuplot file (stored in GP_FLAG) never actually exists, it's marked as
# .SECONDARY below.
GP_SRC := $(GNUPLOT_SRC)
GP_FLAG := $(addprefix $(BUILD_DIR)/, $(GP_SRC:.gp=.gnuplot))
GP_TEX := $(addprefix $(BUILD_DIR)/, $(GP_SRC:.gp=.tex))
GP_EPS := $(addprefix $(BUILD_DIR)/, $(GP_SRC:.gp=.eps))

GP_PDF := $(addprefix $(FIGURES_DIR)/, $(GP_SRC:.gp=.pdf))

FIG_TEX := $(addprefix $(BUILD_DIR)/, $(FIG_SRC:.fig=.tex))
FIG_EPS := $(addprefix $(BUILD_DIR)/, $(FIG_SRC:.fig=.eps))
FIG_PDF := $(addprefix $(FIGURES_DIR)/, $(FIG_SRC:.fig=.pdf))

# Inkscape is similar to Gnuplot in that it produces multiple output from a
# single call. Thus, we have to use the same flag procedure.
INKSCAPE_FLAG := $(addprefix $(BUILD_DIR)/, $(INKSCAPE_SRC:.svg=.inkscape))
INKSCAPE_PDF := $(addprefix $(BUILD_DIR)/, $(INKSCAPE_SRC:.svg=.pdf))
INKSCAPE_TEX := $(addprefix $(BUILD_DIR)/, $(INKSCAPE_SRC:.svg=.tex))

DIA_EPS := $(addprefix $(BUILD_DIR)/, $(DIA_SRC:.dia=.eps))
DIA_PDF := $(addprefix $(FIGURES_DIR)/, $(DIA_SRC:.dia=.pdf))
BACKUP_DIA := $(addprefix $(BACKUP_DIR)/, $(DIA_SRC:.dia=.eps))
DIA_PDF_GREY := $(addprefix $(FIGURES_DIR)/grey/, $(DIA_SRC:.dia=.pdf))

OCT_PLOT_PDF := $(addprefix $(BUILD_DIR)/, $(OCT_PLOT:.m=.pdf))

OCT_TABLE_TEX := $(addprefix $(BUILD_DIR)/, $(OCT_TABLE:.m=.tex))
PY_TABLE_TEX := $(addprefix $(BUILD_DIR)/,  $(PY_TABLE:.py=.tex))
TABLE_TEX := $(OCT_TABLE_TEX) $(PY_TABLE_TEX)

# For Python plots, we have four versions: one for the paper, one for the talk,
# one for the poster, and one greyscale (for the paper).
PY_PLOT_PDF := $(addprefix $(FIGURES_DIR)/paper/, $(PY_PLOT:.py=.pdf))
PY_PLOT_TALK_PDF := $(addprefix $(FIGURES_DIR)/presentation/, $(PY_PLOT:.py=.pdf))
PY_MOVIE_PDF := $(addprefix $(FIGURES_DIR)/presentation/, $(PY_MOVIE:.py=.pdf))
PY_MOVIE_TIMELINE := $(addprefix $(BUILD_DIR)/, $(PY_MOVIE:.py=.timeline))
PY_PLOT_GREY_PDF := $(addprefix $(FIGURES_DIR)/grey/, $(PY_PLOT:.py=.pdf))
PY_PLOT_POSTER_PDF := $(addprefix $(FIGURES_DIR)/poster/, $(PY_PLOT:.py=.pdf))
OCT_PLOT_PDF := $(addprefix $(FIGURES_DIR)/, $(OCT_PLOT:.m=.pdf))

# Target and intermediates for standalone tex pdfs.
STANDALONE_TEX_LINKS := $(addprefix $(BUILD_DIR)/, $(STANDALONE_TEX_SRC))
STANDALONE_TEX_PDF := $(addprefix $(FIGURES_DIR)/, $(STANDALONE_TEX_SRC:.tex=.pdf))
STANDALONE_TEX_MISC := $(foreach p, $(basename $(STANDALONE_TEX_LINKS)), $(addprefix $(p), $(TEXSUFFIXES)))

# Create a target to make a particular figure in the root directory.
PY_PLOT_LOCAL_PDF := $(PY_PLOT:.py=.pdf)

# Things for scripts that create shared mat files. 
PY_MAT := $(addprefix $(BUILD_DIR)/, $(PY_MAT_SRC:.py=.mat))
OCT_MAT := $(addprefix $(BUILD_DIR)/, $(OCT_MAT_SRC:.m=.mat))
JL_MAT := $(addprefix $(BUILD_DIR)/, $(JL_MAT_SRC:.jl=.mat))
MATLAB_MAT := $(addprefix $(BUILD_DIR)/, $(MATLAB_MAT_SRC:.m=.mat))
PY_PICKLE := $(addprefix $(BUILD_DIR)/, $(PY_PICKLE_SRC:.py=.pickle))

BACKUP_MAT := $(addprefix $(BACKUP_DIR)/, $(PY_MAT_SRC:.py=.mat) $(OCT_MAT_SRC:.m=.mat) $(JL_MAT_SRC:.jl=.mat) $(MATLAB_MAT_SRC:.m=.mat))
BACKUP_PICKLE := $(addprefix $(BACKUP_DIR)/, $(PY_PICKLE_SRC:.py=.pickle))

# Octave-produced .dat files. These are not backed up.
OCT_DAT := $(addprefix $(BUILD_DIR)/, $(OCT_DAT_SRC:.m=.dat))

# Stuff for tex files.
TEX_SRC_LINKS := $(addprefix $(BUILD_DIR)/, $(TEX_SRC) $(PAPER_SRC) $(PAPER_BIB))
LOCAL_TEXMF_LINKS := $(addprefix $(BUILD_DIR)/, $(notdir $(LOCAL_TEXMF_FILES)))
EXTRA_INCLUDED_LINKS := $(addprefix $(BUILD_DIR)/, $(EXTRA_INCLUDED_FILES) $(EXTRA_EPS))
TEX_LINKS := $(TEX_SRC_LINKS) $(LOCAL_TEXMF_LINKS) $(EXTRA_INCLUDED_LINKS)

# Some things for standalone eps files. Not sure who really uses these any more.
EXTRA_EPS_PDF := $(addprefix $(FIGURES_DIR)/, $(EXTRA_EPS:.eps=.pdf))

# Make database as JSON used for searching dependencies.
MAKEFILE_JSON := $(DEPS_DIR)/Makefile.json

## The parts of the figures we create.  The PDF parts will be
## generated automatically by the rules in epstopdf.sty.  They are
## listed here so we can use them in the clean rules. THESE THINGS DO NOT HAVE
## RULES TO BUILD ANYTHING AND ARE ONLY USED IN clean RULES.
##
## Note that all files in PDF_PARTS are made from an eps file. pdf files made
## directly should not be included here, but should be dependencies for paper.pdf.

TEX_PARTS := $(FIG_TEX) $(GP_TEX) $(TABLE_TEX)
EPS_PARTS := $(FIG_EPS) $(GP_EPS) $(DIA_EPS)
PDF_PARTS := $(FIG_PDF) $(GP_PDF) $(DIA_PDF) $(EXTRA_EPS_PDF)

## Here we take care of script files that may depend on other mat files. Any
## script that does calculations is included here, and a corresponding .deps file
## is created in $(DEPS_DIR)
AUTODEPENDENCIES_SRC := $(PY_MAT_SRC) $(OCT_MAT_SRC) $(PY_PLOT) $(OCT_PLOT) \
                        $(OCT_DAT_SRC) $(PY_MOVIE) $(PY_TABLE) $(OCT_TABLE) \
                        $(JL_MAT_SRC) $(GNUPLOT_SRC) $(PAPER_SRC) $(MATLAB_MAT_SRC) \
                        $(PY_PICKLE_SRC)
AUTODEPENDENCIES := $(addprefix $(DEPS_DIR)/, $(addsuffix .dep, $(AUTODEPENDENCIES_SRC)))
-include $(AUTODEPENDENCIES)

## The first target is the one that will be made by default (by typing
## "make" with no target arguments).  The traditional name for the default
## target is "all". We also provide a "current" goal that users can edit in
## case they're only interested in a subset of files for the time being.

.DEFAULT_GOAL := paper.pdf

all: $(ARTICLE_PDF) $(TALK_PDF) $(POSTER_PDF)
.PHONY: all

# Disable implicit rules.
.SUFFIXES :

matbackups: $(BACKUP_MAT)
.PHONY: matbackups

diabackups: $(BACKUP_DIA)
.PHONY: diabackups

picklebackups: $(BACKUP_PICKLE)
.PHONY: picklebackups

install: .hgignore
	$(PYTHON) $(SCRIPT_DIR)/INSTALL.py
.PHONY: install
forceinstall: .hgignore
	$(PYTHON) $(SCRIPT_DIR)/INSTALL.py --force
.PHONY: forceinstall

# Extra figure/bibliography dependencies.
$(ARTICLE_BUILD_PDF) : $(PY_PLOT_PDF) $(PAPER_MENDELEYBIB_CLEANED)
$(ARTICLE_GREY_BUILD_PDF) : $(PY_PLOT_GREY_PDF) $(PAPER_MENDELEYBIB_CLEANED) $(DIA_PDF_GREY)
$(TALK_BUILD_PDF) $(HANDOUT_BUILD_PDF) : $(PY_PLOT_TALK_PDF) $(PAPER_MENDELEYBIB_CLEANED)
$(TALK_BUILD_PDF) : $(PY_MOVIE_PDF) $(PY_MOVIE_TIMELINE)
$(POSTER_BUILD_PDF) : $(PY_PLOT_POSTER_PDF) $(PAPER_MENDELEYBIB_CLEANED)

$(HANDOUT_SRC) : $(BUILD_DIR)/%-handout.tex : %.tex
	@echo Building $@.
	@echo "\PassOptionsToClass{handout}{beamer}" > $@
	@cat $*.tex >> $@

$(FOURUP_PDF) : %-4up.pdf : $(BUILD_DIR)/%-handout.pdf
	@echo Building $@.
	@pdfjam --quiet --letterpaper --scale 0.95 --landscape --nup 2x2 --delta "5mm 5mm" -o $@ $<

$(PAPER_BUILD_PDF) : %.pdf : %.tex $(TEX_PARTS) $(PDF_PARTS) $(OCT_PLOT_PDF) $(TEX_LINKS) $(PAPER_BIB) $(INKSCAPE_PDF) $(INKSCAPE_TEX) $(STANDALONE_TEX_PDF)
	@echo Building $@.
	@./$(SCRIPT_DIR)/latex2pdf.py $(PDFLATEX_FLAGS) --dir $(BUILD_DIR) $<

$(PAPER_PDF) : %.pdf : $(BUILD_DIR)/%.pdf
	@echo "Copying $<"
	@cp $< $@

grey : $(ARTICLE_GREY_PDF)
.PHONY: grey

paper : $(ARTICLE_PDF)
.PHONY: paper

poster : $(POSTER_PDF)
.PHONY: poster

presentation : $(TALK_PDF)
.PHONY: presentation

# Cleanup rules.
CLEAN_FILES := \
    $(TEX_PARTS) $(EPS_PARTS) $(PDF_PARTS) $(OCT_PLOT_PDF) \
    $(TEX_LINKS) $(PAPER_MENDELEYBIB_CLEANED) \
    $(SCRATCH_FILES) $(addprefix $(BUILD_DIR)/, $(SCRATCH_FILES)) \
    $(PAPER_MISC) $(PAPER_PDF) $(PY_PLOT_PDF) $(PY_PLOT_GREY_PDF) \
    $(PY_PLOT_TALK_PDF) $(PY_PLOT_POSTER_PDF) $(HANDOUT_SRC) \
    $(STANDALONE_TEX_PDF) $(STANDALONE_TEX_MISC) $(STANDALONE_TEX_LINKS) \
    $(MARKDOWN_DOCX) $(MAKEFILE_JSON) $(PAPER_DIST_ZIP) \
    $(PAPER_XBIB) $(PAPER_BUILD_XBIB) \

REALCLEAN_FILES := \
    .hgignore $(OCT_DAT) $(CUSTOM_DAT) $(OCT_MAT) $(PY_MAT) $(JL_MAT) \
    $(AUTODEPENDENCIES) $(PY_PLOT_LOCAL_PDF) $(MATLAB_MAT) $(LIB_DIR)/**.pyc \
    $(PY_PICKLE)

clean:
	@echo "Cleaning up."
	@rm -f $(CLEAN_FILES)
.PHONY: clean

realclean: clean
	@rm -f $(REALCLEAN_FILES)
.PHONY: realclean

# Rule for .hgignore.
HGIGNORE_DEFAULT := Makefile.options
HGIGNORE_ALL := $(HGIGNORE) $(HGIGNORE_DEFAULT) $(CLEAN_FILES) $(REALCLEAN_FILES)
.hgignore :
	@echo "Making $@"
	@rm -f $@
	@echo "# Automatically generated by Make. Add files to HGIGNORE variable in Makefile instead." > $@
	@echo "syntax: glob" >> $@
	@echo "$(HGIGNORE_ALL)" | tr ' ' '\n' >> $@
	@chmod a-w $@
.PHONY : .hgignore

## Handle automatic dependency generation.
$(AUTODEPENDENCIES) : $(DEPS_DIR)/%.dep : % $(SCRIPT_DIR)/getdependencies.py
	@echo Reading dependencies from $<.
	@./$(SCRIPT_DIR)/getdependencies.py --build $(BUILD_DIR) --output $@ $<

## Links for various tex files.
define do-symlink
ln -srf $< $@
endef

$(TEX_SRC_LINKS) : $(BUILD_DIR)/% : %
	@$(do-symlink)

$(LOCAL_TEXMF_LINKS) : $(BUILD_DIR)/% : texmf/%
	@$(do-symlink)

$(EXTRA_INCLUDED_LINKS) : $(BUILD_DIR)/% : %
	@$(do-symlink)

$(STANDALONE_TEX_LINKS) : $(BUILD_DIR)/% : %
	@$(do-symlink)

## We specify greyscale vs. color code in the source using a "toggle" (from the
## etoolbox package) named greyscale. We assume that it is set to false by
## default, and if we want the greyscale version, then we use sed to create a
## copy of the .tex file with the toggle changed.
$(ARTICLE_GREY_SRC) : $(BUILD_DIR)/%-grey.tex : %.tex
	@sed -r 's/^\\settoggle\{greyscale\}\{false\}/\\settoggle\{greyscale\}\{true\}/g' $< > $@

## How to generate epslatex (.eps + .tex) files from a .fig file.
##
## By using a pattern rule like
##
##   $(FIG_TEX) : %.tex : %.fig
##
## instead of simply
##
##   %.tex : %.fig
##
## we limit the application of this rule to those files in the $(FIG_TEX) list.

$(FIG_TEX) : $(BUILD_DIR)/%.tex : %.fig
	@echo making $@
	@$(FIG2DEV) -L pstex_t -p $* $< > $@.t
	@mv $@.t $@

$(FIG_EPS) : $(BUILD_DIR)/%.eps : %.fig
	@echo making $@
	@$(FIG2DEV) -L pstex $< > $@.t
	@mv $@.t $@

## Rule for standalone tex pdfs.
$(STANDALONE_TEX_PDF) : $(FIGURES_DIR)/%.pdf : $(BUILD_DIR)/%.tex $(LOCAL_TEXMF_LINKS)
	@echo Building $@.
	@./$(SCRIPT_DIR)/latex2pdf.py $(PDFLATEX_FLAGS) --dir $(BUILD_DIR) $<
	@mv $(BUILD_DIR)/$*.pdf $@

## Rules for making data files with Octave and Python. Note that if these
## commands fail, we need to delete the resulting data file, as it could be
## corrupt. One way to do this is with the ".DELETE_ON_ERROR" make special
## variable. However, we choose to not use this approach because it also
## deletes files if the user interrupts the script, which we don't want.

define do-octave-command
echo making $@
$(OCTAVE) $(OCTAVE_FLAGS) --eval "source('$(TOP_DIR)/$<')" || { echo "***Octave error"; rm -f $@; false; }
endef

define do-python-command
echo making $@
$(PYTHON) $(SCRIPT_DIR)/dopython.py --dir "$(BUILD_DIR)" --path "$(LIB_DIR)" "$(TOP_DIR)/$<" || \
    { echo "***Python error"; rm -f $@; false; }
endef

define do-julia-command
echo making $@
cd $(BUILD_DIR) && JULIA_LOAD_PATH="$(JULIA_LOAD_PATH)$(TOP_DIR)/$(LIB_DIR):" $(JULIA) ../$<
endef

define do-matlab-command
echo making $@
cd $(BUILD_DIR) && $(MATLAB) -nosplash -nodesktop -r "addpath('..'); addpath('../$(LIB_DIR)'); $(basename $<); quit"
endef

# Define a command to grab a file from the backup directory.
define do-use-backup-command
if [ -f $(BACKUP_DIR)/$(@F) ]; then \
    echo +-Copying $(@F) from $(BACKUP_DIR); \
    cp $(BACKUP_DIR)/$(@F) $@; \
    touch $@; \
else \
    echo ***ERROR-$(BACKUP_DIR)/$(@F) not found!; \
    false; \
fi
endef

## Read options file.
-include Makefile.options

# Define commands to make mat files.
ifeq ($(USE_MAT_BACKUPS), True)
    define do-octave-mat
    $(do-use-backup-command)
    endef
    
    define do-python-mat
    $(do-use-backup-command)
    endef
    
    define do-julia-mat
    $(do-use-backup-command)
    endef

    define do-matlab-mat
    $(do-use-backup-command)
    endef
else
    define do-octave-mat
    $(do-octave-command)
    endef

    define do-python-mat
    $(do-python-command)
    endef
    
    define do-julia-mat
    $(do-julia-command)
    endef

    define do-matlab-mat
    $(do-matlab-command)
    endef
endif

# Define commands to make pdfs from dia files.
ifeq ($(USE_DIA_BACKUPS), True)
    define do-dia-eps
    $(do-use-backup-command)
    endef
else
    define do-dia-eps
    @echo making $@
	@dia --filter=eps-pango --export=$@ $<
    endef
endif

# Define commands to make pickle files from .py files
ifeq ($(USE_PICKLE_BACKUPS), True)
    define do-python-pickle
    $(do-use-backup-command)
    endef
else
    define do-python-pickle
    $(do-python-command)
    endef
endif

## Rules for creating shared mat files with Octave, Python, or Julia.
$(PY_MAT) : $(BUILD_DIR)/%.mat : %.py
	@$(do-python-mat)

$(OCT_MAT) : $(BUILD_DIR)/%.mat : %.m
	@$(do-octave-mat)

$(JL_MAT) : $(BUILD_DIR)/%.mat : %.jl
	@$(do-julia-mat)

$(MATLAB_MAT) : $(BUILD_DIR)/%.mat : %.m
	@$(do-matlab-mat)

$(PY_PICKLE) : $(BUILD_DIR)/%.pickle : %.py
	@$(do-python-pickle)

## Rule for generating .dat files  from .m files using Octave.

$(OCT_DAT) : $(BUILD_DIR)/%.dat : %.m
	@$(do-octave-command)

## Rules for making epslatex (.eps + .tex) files from a .gp file and possibly a
## .dat file created from an Octave .m file. Note that we have to use a dummy
## .SECONDARY file so that gnuplot is run only once.

define CHECKEXIST
if [ ! -e "$@" ]; then \
    echo "!!! File '$@' does not exist!"; \
    echo "!!! Should have been made by recipe '$<'!"; \
    false; \
fi
endef

$(GP_EPS) : %.eps : %.gnuplot
	@$(CHECKEXIST)
$(GP_TEX) : %.tex : %.gnuplot
	@$(CHECKEXIST)
ifdef GP_FLAG
    .SECONDARY : $(GP_FLAG)
endif

$(GP_FLAG) : $(BUILD_DIR)/%.gnuplot : %.gp
	@echo making $*.tex and $*.eps
	@./$(SCRIPT_DIR)/dognuplot.py --build $(BUILD_DIR) --terminal "$(GNUPLOT_TERMINAL)" $<

## Note that we want to remove the corresponding .bbl, .blg, and .bst file to make
## sure latex rebuilds them to reflect changes.
$(PAPER_MENDELEYBIB_CLEANED) : $(BUILD_DIR)/%.bib : %.mendeleybib
	@echo making $@
	@./$(SCRIPT_DIR)/mendeleyexport.py --out $@ $<
	@rm -f $(BUILD_DIR)/$*.bst
	@rm -f $(PAPER_MISCBIB)

## Rule for generating .pdf and .tex files from .svg files. A bit tricky since
## we basically need a pattern rule with multiple targets. To spoof this
## behavior, we use .SECONDARY files *.inkscape (stored in INKSCAPE_FLAG,
## although they never actually get created).

$(INKSCAPE_PDF) : %.pdf : %.inkscape
	@$(CHECKEXIST)
$(INKSCAPE_TEX) : %.tex : %.inkscape
	@$(CHECKEXIST)
ifdef INKSCAPE_FLAG
    .SECONDARY : $(INKSCAPE_FLAG)
endif

$(INKSCAPE_FLAG) : $(BUILD_DIR)/%.inkscape : %.svg
	@echo making $*.tex and $*.pdf
	$(INKSCAPE) --export-pdf=$(BUILD_DIR)/$*.pdf --export-latex $<
	mv $(BUILD_DIR)/$*.pdf_tex $(BUILD_DIR)/$*.tex

## Rule for .dia --> .eps --> .pdf

$(DIA_EPS) : $(BUILD_DIR)/%.eps : %.dia
	@$(do-dia-eps)

$(DIA_PDF_GREY) : $(FIGURES_DIR)/grey/%.pdf : $(FIGURES_DIR)/%.pdf
	@echo making $@
	@gs -sOutputFile=$@ $(GS_GREY_FLAGS) $<

## Rule for generating .pdf files from .eps files

$(PDF_PARTS) : $(FIGURES_DIR)/%.pdf : $(BUILD_DIR)/%.eps
	@echo making $@ from $<
	@$(EPSTOPDF) $< --outfile=$@.t
	@mv $@.t $@

# ****************
# Matplotlib rules
# ****************
$(PY_PLOT_PDF) : $(FIGURES_DIR)/paper/%.pdf : %.py
	@echo making $@
	@$(MATPLOTLIB) --font paper --build $(BUILD_DIR) $<
	@mv $(BUILD_DIR)/$*.pdf $@

$(PY_PLOT_GREY_PDF) : $(FIGURES_DIR)/grey/%.pdf : %.py
	@echo making $@
	@$(MATPLOTLIB) --grey --font paper --build $(BUILD_DIR) $<
	@gs -sOutputFile=$@ $(GS_GREY_FLAGS) $(BUILD_DIR)/$*.pdf
	@rm $(BUILD_DIR)/$*.pdf

define do-talk-matplotlib
@echo making $@
@$(MATPLOTLIB) --font presentation --build $(BUILD_DIR) $*.py
@mv $(BUILD_DIR)/$*.pdf $(FIGURES_DIR)/presentation/$*.pdf
endef

$(PY_PLOT_TALK_PDF) $(PY_MOVIE_PDF) : $(FIGURES_DIR)/presentation/%.pdf : %.py
	@$(do-talk-matplotlib)

$(PY_MOVIE_TIMELINE) : $(BUILD_DIR)/%.timeline : %.py
	@$(do-talk-matplotlib)
	@touch $@

$(PY_PLOT_POSTER_PDF) : $(FIGURES_DIR)/poster/%.pdf : %.py
	@echo making $@
	@$(MATPLOTLIB) --font poster --build $(BUILD_DIR) $<
	@mv $(BUILD_DIR)/$*.pdf $@
# ****************

# ***************
# Rule to make a figure and move it to the base directory.
$(PY_PLOT_LOCAL_PDF) : %.pdf : $(FIGURES_DIR)/$(DEFAULT_FIGURE_TYPE)/%.pdf
	@echo copying $@
	@cp $< $@
# ***************

$(OCT_PLOT_PDF) : %.pdf : %.m
	@(do-octave-command)
	@mv $(BUILD_DIR)/$*.pdf $@

$(OCT_TABLE_TEX) : $(BUILD_DIR)/%.tex : %.m
	@$(do-octave-command)

$(PY_TABLE_TEX) : $(BUILD_DIR)/%.tex : %.py
	@$(do-python-command)

$(CUSTOM_TEX) : %.tex : %.gp %.dat
	@$(do-gnuplot-command)

$(CUSTOM_EPS) : %.eps : %.gp %.dat
	@$(do-gnuplot-command)

## Copy mat files to backup directory.
$(BACKUP_MAT) : $(BACKUP_DIR)/%.mat : $(BUILD_DIR)/%.mat
	@echo Saving backups of $<
	@cp -f $< $@

$(BACKUP_PICKLE) : $(BACKUP_DIR)/%.pickle : $(BUILD_DIR)/%.pickle
	@echo Saving backups of $<
	@cp -f $< $@

## Copy dia eps files to backup directory.
$(BACKUP_DIA) : $(BACKUP_DIR)/%.eps : $(BUILD_DIR)/%.eps
	@echo Saving backups of $<
	@cp -f $< $@

## xbib files. Use bibexport.
$(PAPER_BUILD_XBIB) : %.xbib : %.pdf
	@echo Making $@
	@cd $(BUILD_DIR) && bibexport -o $(@F) $(basename $(@F))
	@mv $@.bib $@

$(PAPER_XBIB) : %.xbib : $(BUILD_DIR)/%.xbib
	@cp $< $@

## zip distribution for tex files.
$(PAPER_DIST_ZIP) : %.zip : $(BUILD_DIR)/%.pdf $(BUILD_DIR)/%.xbib
	@echo Making $@
	@./$(SCRIPT_DIR)/maketexdist.py --xbib $(BUILD_DIR)/$*.xbib --output $@ $(BUILD_DIR)/$*.tex

# Rule for Makefile JSON database.
$(MAKEFILE_JSON) : Makefile
	@echo Getting Makefile database JSON file.
	@./$(SCRIPT_DIR)/getmakedatabase.py --output $@ $<

## ************************************************************************************************
## End of standard recipes.
## ************************************************************************************************

## Start of custom recipies.