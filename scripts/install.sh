#!/bin/bash

# Installation script for AlphaScrabble

set -e

echo "🚀 Installing AlphaScrabble"
echo "=========================="

# Check Python version
python_version=$(python --version 2>&1 | cut -d' ' -f2)
echo "🐍 Python version: $python_version"

# Check if Python version is 3.10+
if ! python -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null; then
    echo "❌ Python version is too old (need 3.10+)"
    exit 1
fi

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️  No virtual environment detected. Consider using one."
fi

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install system dependencies (Linux only)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "🔧 Installing system dependencies..."
    sudo apt-get update
    sudo apt-get install -y qtbase5-dev libqt5core5a build-essential cmake ninja-build
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Install AlphaScrabble
echo "🔧 Installing AlphaScrabble..."
pip install -e .

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/selfplay
mkdir -p data/training
mkdir -p checkpoints
mkdir -p logs
mkdir -p lexica_cache

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
    echo "✅ AlphaScrabble installed successfully!"
else
    echo "❌ AlphaScrabble installation failed"
    exit 1
fi

echo ""
echo "🎉 Installation complete!"
echo ""
echo "Next steps:"
echo "1. Run './scripts/setup_lexicon.sh' to set up the lexicon"
echo "2. Run './scripts/run_demo.sh' to test the installation"
echo "3. Run 'alphascrabble --help' to see available commands"
echo ""
echo "Available commands:"
echo "  alphascrabble --help                    - Show help"
echo "  alphascrabble selfplay --games 10       - Generate training data"
echo "  alphascrabble train --data data/selfplay - Train model"
echo "  alphascrabble eval --net-a model.pt     - Evaluate model"
echo "  alphascrabble play --net model.pt       - Play interactively"
