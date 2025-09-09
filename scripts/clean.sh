#!/bin/bash

# Clean script for AlphaScrabble

set -e

echo "🧹 Cleaning AlphaScrabble"
echo "========================"

# Remove build artifacts
echo "🗑️  Removing build artifacts..."
rm -rf build/
rm -rf dist/
rm -rf *.egg-info/
rm -rf .pytest_cache/
rm -rf .coverage
rm -rf htmlcov/
rm -rf .mypy_cache/
rm -rf .tox/

# Remove Python cache
echo "🗑️  Removing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true

# Remove temporary files
echo "🗑️  Removing temporary files..."
rm -rf .DS_Store
rm -rf Thumbs.db
rm -rf *.log
rm -rf *.tmp
rm -rf *.temp

# Remove data files (optional)
if [ "$1" = "--all" ]; then
    echo "🗑️  Removing data files..."
    rm -rf data/
    rm -rf checkpoints/
    rm -rf logs/
    rm -rf lexica_cache/
fi

# Remove third-party builds (optional)
if [ "$1" = "--all" ]; then
    echo "🗑️  Removing third-party builds..."
    rm -rf third_party/quackle/build/
fi

echo "✅ Clean complete!"
echo ""
echo "Usage:"
echo "  ./scripts/clean.sh        - Clean build artifacts only"
echo "  ./scripts/clean.sh --all  - Clean everything including data"
