#!/bin/bash
# Test script for DSS Pollution Extraction

echo "Running tests for DSS Pollution Extraction..."

# Run pytest with coverage
python -m pytest tests/ -v --cov=pollution_extraction --cov-report=html --cov-report=term

echo "Tests completed!"
