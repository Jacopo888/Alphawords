#!/bin/bash

# Uninstallation script for AlphaScrabble

set -e

echo "🗑️  Uninstalling AlphaScrabble"
echo "============================="

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️  No virtual environment detected. Consider using one."
fi

# Uninstall AlphaScrabble
echo "📦 Uninstalling AlphaScrabble..."
if pip show alphascrabble &> /dev/null; then
    pip uninstall -y alphascrabble
    echo "✅ AlphaScrabble uninstalled"
else
    echo "⚠️  AlphaScrabble not found in pip"
fi

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

# Ask about data removal
echo ""
read -p "Do you want to remove data files (data/, checkpoints/, logs/, lexica_cache/)? [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🗑️  Removing data files..."
    rm -rf data/
    rm -rf checkpoints/
    rm -rf logs/
    rm -rf lexica_cache/
    echo "✅ Data files removed"
else
    echo "⚠️  Data files preserved"
fi

# Ask about third-party removal
echo ""
read -p "Do you want to remove third-party builds (third_party/quackle/build/)? [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🗑️  Removing third-party builds..."
    rm -rf third_party/quackle/build/
    echo "✅ Third-party builds removed"
else
    echo "⚠️  Third-party builds preserved"
fi

echo ""
echo "✅ Uninstallation complete!"
echo ""
echo "Note:"
echo "- Python dependencies are still installed"
echo "- Virtual environment is preserved"
echo "- Source code is preserved"
echo ""
echo "To completely remove everything:"
echo "1. Remove the virtual environment: rm -rf venv/"
echo "2. Remove the source code: rm -rf alphascrabble/"
echo "3. Uninstall Python dependencies: pip freeze | xargs pip uninstall -y"
