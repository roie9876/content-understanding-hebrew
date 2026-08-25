.PHONY: setup test check run clean

PYTHON ?= python3
VENV ?= .venv
VENV_PYTHON := $(VENV)/bin/python

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PYTHON) -m pip install -r requirements.txt
	npm ci

test:
	@test -x $(VENV_PYTHON) || (echo "Run 'make setup' first." && exit 1)
	$(VENV_PYTHON) -m unittest discover -s tests -v

check: test
	PYTHONPYCACHEPREFIX=$(CURDIR)/__pycache__ $(VENV_PYTHON) -m py_compile *.py

run:
	@test -x $(VENV_PYTHON) || (echo "Run 'make setup' first." && exit 1)
	$(VENV_PYTHON) app.py

clean:
	rm -rf __pycache__ .pytest_cache
