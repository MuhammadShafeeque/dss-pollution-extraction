#!/bin/bash
# Development environment setup script

echo "Setting up development environment for DSS Pollution Extraction..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install development dependencies
echo "Installing development dependencies..."
pip install -r requirements-dev.txt

# Install package in development mode
echo "Installing package in development mode..."
pip install -e .

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
pre-commit install

echo "Development environment setup completed!"
echo "To activate the environment, run: source venv/bin/activate"
