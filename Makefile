# ensure conda environment do not use user site-packages
export PYTHONNOUSERSITE=1

PYTHON_VERSION ?= 3.8
CONDA_VENV_PATH ?= $(PWD)/venv
KERNEL_NAME ?= $(shell basename $(CURDIR))

NOTEBOOKS_SRC := $(wildcard notebooks/*.py)
NOTEBOOKS := $(NOTEBOOKS_SRC:.py=.ipynb)
HTML_FILES := $(NOTEBOOKS:.ipynb=.html)

CONDA_BASE := $(shell conda info --base)
CONDA_VENV := . $(CONDA_BASE)/etc/profile.d/conda.sh && conda activate $(CONDA_VENV_PATH) &&
KERNEL_DIR := $(HOME)/.local/share/jupyter/kernels/$(KERNEL_NAME)

all: help

## Display this help message
help: Makefile
	@echo 'Available targets:'
	@echo ''
	@tac Makefile | \
	    awk '/^##/ { sub(/## /,"#"); print "    " prev $$0 } { FS=":"; prev = $$1 }' | \
	    column -t -s '#' | \
	    tac
	@echo ''
	@echo 'Configurable variables (current value):'
	@echo ''
	@echo '    PYTHON_VERSION   Python interpreter version ($(PYTHON_VERSION))'
	@echo '    CONDA_VENV_PATH  Path of the Conda virtual environment ($(CONDA_VENV_PATH))'
	@echo '    KERNEL_NAME      Jupyter kernel name ($(KERNEL_NAME))'
	@echo ''

## Convert some scripts into notebooks and .html files
notebooks: $(NOTEBOOKS) $(HTML_FILES)

# convert a notebook into .html document
%.html: %.ipynb
	$(CONDA_VENV) jupyter nbconvert --to html "$<"

# convert a script into a notebook and reformat both with black
%.ipynb: %.py requirements.txt
	$(CONDA_VENV) jupytext --sync --pipe black "$<"

# freeze the dependencies installed in the virtual environment for reproducibility
requirements.txt: venv/.canary
	$(CONDA_VENV) pip freeze > "$@"

# create a virtual environment and register it as a jupyter kernel
venv/.canary: setup.cfg setup.py
	conda create -y -p $(CONDA_VENV_PATH) python=$(PYTHON_VERSION)
	$(CONDA_VENV) conda install -y -c conda-forge cartopy
	$(CONDA_VENV) pip install -e .[dev]
	$(CONDA_VENV) python -m ipykernel install --user --name $(KERNEL_NAME)
	touch "$@"

## Create a Conda virtual environment and register it as a Jupyter kernel
venv: venv/.canary

## Create a Conda virtual environment for Jupyter on NeSI
venv_nesi: venv
	cp nesi/template_wrapper.bash $(KERNEL_DIR)/wrapper.bash
	sed -i 's|##CONDA_VENV_PATH##|$(CONDA_VENV_PATH)|' $(KERNEL_DIR)/wrapper.bash
	cp nesi/template_kernel.json $(KERNEL_DIR)/kernel.json
	sed -i 's|##KERNEL_DIR##|$(KERNEL_DIR)|; s|##KERNEL_NAME##|$(KERNEL_NAME)|' $(KERNEL_DIR)/kernel.json

## Remove the Conda virtual environment
clean_venv:
	rm -rf $(KERNEL_DIR)
	conda env remove -p $(CONDA_VENV_PATH)
	rm -rf src/*.egg-info

## Format Python scripts
format:
	$(CONDA_VENV) black src/

.PHONY: help venv clean_venv venv_nesi notebooks format
