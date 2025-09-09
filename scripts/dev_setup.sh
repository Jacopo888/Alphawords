#!/bin/bash

# Development setup script for AlphaScrabble

set -e

echo "🔧 Setting up AlphaScrabble development environment"
echo "=================================================="

# Check Python version
python_version=$(python --version 2>&1 | cut -d' ' -f2)
echo "🐍 Python version: $python_version"

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️  No virtual environment detected. Creating one..."
    python -m venv venv
    source venv/bin/activate
    echo "✅ Virtual environment created and activated"
fi

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install development dependencies
echo "📦 Installing development dependencies..."
pip install -e ".[dev]"

# Install pre-commit hooks
echo "🔗 Installing pre-commit hooks..."
if command -v pre-commit &> /dev/null; then
    pre-commit install
    echo "✅ Pre-commit hooks installed"
else
    echo "⚠️  pre-commit not found, skipping hook installation"
fi

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

# Check code formatting
echo "🎨 Checking code formatting..."
if command -v black &> /dev/null; then
    black --check alphascrabble/ tests/
    echo "✅ Code formatting is correct"
else
    echo "⚠️  black not found, skipping format check"
fi

# Check imports
echo "📋 Checking imports..."
if command -v isort &> /dev/null; then
    isort --check-only alphascrabble/ tests/
    echo "✅ Imports are correctly sorted"
else
    echo "⚠️  isort not found, skipping import check"
fi

# Check linting
echo "🔍 Checking linting..."
if command -v flake8 &> /dev/null; then
    flake8 alphascrabble/ tests/
    echo "✅ Linting passed"
else
    echo "⚠️  flake8 not found, skipping linting"
fi

# Check type hints
echo "🔍 Checking type hints..."
if command -v mypy &> /dev/null; then
    mypy alphascrabble/ --ignore-missing-imports
    echo "✅ Type checking passed"
else
    echo "⚠️  mypy not found, skipping type checking"
fi

echo ""
echo "🎉 Development environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Run 'make test' to run all tests"
echo "2. Run 'make lint' to check code quality"
echo "3. Run 'make format' to format code"
echo "4. Run 'make benchmark' to run performance benchmarks"
echo "5. Run 'make selfplay-demo' to test self-play functionality"
echo ""
echo "Available commands:"
echo "  make help          - Show all available commands"
echo "  make test          - Run tests"
echo "  make lint          - Run linting"
echo "  make format        - Format code"
echo "  make benchmark     - Run benchmarks"
echo "  make clean         - Clean build artifacts"
