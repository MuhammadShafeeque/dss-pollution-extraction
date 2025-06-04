#!/bin/bash
set -e

echo "🔼 Bumping patch version..."
bumpver update --patch

echo "🧹 Cleaning old builds..."
rm -rf dist/ *.egg-info

echo "📦 Building package..."
python -m build

echo "🚀 Uploading to PyPI..."
twine upload -u $TWINE_USERNAME -p $TWINE_PASSWORD dist/*


echo "✅ Release completed successfully!"
