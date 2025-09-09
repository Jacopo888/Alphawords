#!/bin/bash

# Start script for AlphaScrabble

set -e

echo "🎮 Starting AlphaScrabble"
echo "========================"

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️  No virtual environment detected. Consider using one."
fi

# Check Python version
python_version=$(python --version 2>&1 | cut -d' ' -f2)
echo "🐍 Python version: $python_version"

# Check if AlphaScrabble is installed
if ! python -c "import alphascrabble" 2>/dev/null; then
    echo "❌ AlphaScrabble is not installed"
    echo "Run: ./scripts/install.sh"
    exit 1
fi

# Check lexicon files
if [ ! -f "lexica_cache/enable1.txt" ] || [ ! -f "lexica_cache/english_enable1.dawg" ] || [ ! -f "lexica_cache/english_enable1.gaddag" ]; then
    echo "⚠️  Lexicon files not found"
    echo "Run: ./scripts/setup_lexicon.sh"
    echo ""
    echo "Continuing without lexicon (some features may not work)..."
fi

# Create necessary directories
mkdir -p data/selfplay
mkdir -p data/training
mkdir -p checkpoints
mkdir -p logs
mkdir -p lexica_cache

# Show available commands
echo ""
echo "🎯 Available Commands"
echo "===================="
echo "  alphascrabble --help                    - Show help"
echo "  alphascrabble selfplay --games 10       - Generate training data"
echo "  alphascrabble train --data data/selfplay - Train model"
echo "  alphascrabble eval --net-a model.pt     - Evaluate model"
echo "  alphascrabble play --net model.pt       - Play interactively"

echo ""
echo "🎮 Quick Start Options"
echo "====================="
echo "1. Run demo: ./scripts/run_demo.sh"
echo "2. Check status: ./scripts/status.sh"
echo "3. Run tests: ./scripts/run_tests.sh"
echo "4. Run benchmarks: ./scripts/benchmark.sh"

echo ""
echo "🚀 Starting AlphaScrabble CLI..."
echo "Type 'exit' to quit"
echo ""

# Start interactive CLI
alphascrabble --help
