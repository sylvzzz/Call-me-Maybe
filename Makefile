FOLDER := $(shell pwd)
MAIN = main.py
VENV = $(FOLDER)/.venv
REQ = requirements.txt
PY = $(VENV)/bin/python

install:
	uv venv $(VENV)
	uv pip install -r $(REQ) --python $(PY)
	unzip llm_sdk.zip
	uv pip install ./llm_sdk --python $(PY)

run:
	uv run $(PY) -m src

debug:
	python3 -m pdb $(MAIN)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +
	find . -type f -name "*.pyc" -exec rm -f {} +

lint:
	python3 -m flake8 . --exclude virtual_env
	python3 -m mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	python3 -m flake8 . --exclude virtual_env
	python3 -m mypy . --strict --exclude virtual_env


.PHONY: build install run debug clean lint lint-strict test-all