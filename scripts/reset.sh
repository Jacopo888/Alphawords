#!/bin/bash

# Reset script for AlphaScrabble

set -e

echo "🔄 Resetting AlphaScrabble"
echo "========================="

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️  No virtual environment detected"
fi

# Check Python version
python_version=$(python --version 2>&1 | cut -d' ' -f2)
echo "🐍 Python version: $python_version"

# Stop any running processes
echo "🛑 Stopping running processes..."
if pgrep -f "alphascrabble\|tensorboard\|jupyter\|train\|selfplay\|eval\|play" > /dev/null; then
    pkill -f "alphascrabble\|tensorboard\|jupyter\|train\|selfplay\|eval\|play"
    sleep 3
    echo "✅ Processes stopped"
else
    echo "✅ No processes to stop"
fi

# Ask for confirmation
echo ""
echo "⚠️  This will reset AlphaScrabble to a clean state!"
echo "This will remove:"
echo "  - All data files (data/, checkpoints/, logs/)"
echo "  - All build artifacts"
echo "  - All temporary files"
echo "  - All Python cache files"
echo ""
read -p "Do you want to continue? [y/N]: " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Reset cancelled"
    exit 1
fi

# Remove data files
echo "🗑️  Removing data files..."
rm -rf data/
rm -rf checkpoints/
rm -rf logs/
echo "✅ Data files removed"

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
echo "✅ Build artifacts removed"

# Remove Python cache
echo "🗑️  Removing Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
echo "✅ Python cache removed"

# Remove temporary files
echo "🗑️  Removing temporary files..."
rm -rf .DS_Store
rm -rf Thumbs.db
rm -rf *.log
rm -rf *.tmp
rm -rf *.temp
echo "✅ Temporary files removed"

# Ask about lexicon files
echo ""
read -p "Do you want to remove lexicon files? [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🗑️  Removing lexicon files..."
    rm -rf lexica_cache/
    echo "✅ Lexicon files removed"
else
    echo "⚠️  Lexicon files preserved"
fi

# Ask about third-party builds
echo ""
read -p "Do you want to remove third-party builds? [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🗑️  Removing third-party builds..."
    rm -rf third_party/quackle/build/
    echo "✅ Third-party builds removed"
else
    echo "⚠️  Third-party builds preserved"
fi

# Reinstall AlphaScrabble
echo "🔧 Reinstalling AlphaScrabble..."
pip install -e .

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/selfplay
mkdir -p data/training
mkdir -p checkpoints
mkdir -p logs
mkdir -p lexica_cache
echo "✅ Directories created"

# Run basic tests
echo "🧪 Running basic tests..."
if command -v pytest &> /dev/null; then
    pytest tests/test_rules.py -v
    echo "✅ Basic tests passed"
else
    echo "⚠️  pytest not found, skipping tests"
fi

# Check installation
echo "🔍 Checking installation..."
if python -c "import alphascrabble" 2>/dev/null; then
    echo "✅ AlphaScrabble reinstalled successfully!"
else
    echo "❌ AlphaScrabble reinstallation failed"
    exit 1
fi

echo ""
echo "🎉 Reset complete!"
echo ""
echo "Next steps:"
echo "1. Run './scripts/setup_lexicon.sh' to set up lexicon (if removed)"
echo "2. Run './scripts/run_demo.sh' to test the installation"
echo "3. Run 'alphascrabble --help' to see available commands"
echo ""
echo "Available commands:"
echo "  alphascrabble --help                    - Show help"
echo "  alphascrabble selfplay --games 10       - Generate training data"
echo "  alphascrabble train --data data/selfplay - Train model"
echo "  alphascrabble eval --net-a model.pt     - Evaluate model"
echo "  alphascrabble play --net model.pt       - Play interactively"
