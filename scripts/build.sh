#!/bin/bash
# Build script for DSS Pollution Extraction

echo "Building DSS Pollution Extraction package..."

# Clean previous builds
rm -rf build/ dist/ *.egg-info/

# Build the package
python -m build

echo "Build completed successfully!"
