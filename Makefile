# Makefile for DSS Pollution Extraction

.PHONY: help test lint format docs clean

help:
	@echo "Available targets:"
	@echo "  test    Run all tests"
	@echo "  lint    Run flake8 and mypy checks"
	@echo "  format  Run black formatter"
	@echo "  docs    Build documentation"
	@echo "  clean   Remove build artifacts"

test:
	python -m pytest tests/ -v

lint:
	flake8 pollution_extraction/
	mypy pollution_extraction/

format:
	black pollution_extraction/

docs:
	cd docs && make html

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache .mypy_cache .coverage htmlcov
