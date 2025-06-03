#!/bin/bash
# Development environment setup script

echo "Setting up development environment for DSS Pollution Extraction..."

# Virtual environment is not needed in Docker container
# as we're already in an isolated environment

# Install development dependencies
echo "Installing development dependencies..."
uv pip install -r requirements-dev.txt

# Install package in development mode
echo "Installing package in development mode..."
uv pip install -e .

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
pre-commit install

echo "Development environment setup completed!"
