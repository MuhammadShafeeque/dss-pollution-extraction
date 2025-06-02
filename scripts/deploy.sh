#!/bin/bash
# Deployment script for DSS Pollution Extraction

echo "Deploying DSS Pollution Extraction package..."

# Build the package
./build.sh

# Upload to PyPI (test repository first)
echo "Uploading to PyPI..."
python -m twine upload --repository testpypi dist/*

echo "Deployment completed successfully!"
