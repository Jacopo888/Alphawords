#!/bin/bash

# Test runner script for AlphaScrabble

set -e

echo "🧪 Running AlphaScrabble Tests"
echo "=============================="

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️  No virtual environment detected. Consider using one."
fi

# Check Python version
python_version=$(python --version 2>&1 | cut -d' ' -f2)
echo "🐍 Python version: $python_version"

# Check if pytest is available
if ! command -v pytest &> /dev/null; then
    echo "❌ pytest not found. Installing..."
    pip install pytest pytest-cov
fi

# Run tests with different configurations
echo ""
echo "🔍 Running basic tests..."
pytest tests/ -v --tb=short

echo ""
echo "🔍 Running tests with coverage..."
pytest tests/ --cov=alphascrabble --cov-report=term-missing --cov-report=html

echo ""
echo "🔍 Running specific test categories..."
echo "  - Rules tests..."
pytest tests/test_rules.py -v

echo "  - Move generation tests..."
pytest tests/test_movegen.py -v

echo "  - MCTS tests..."
pytest tests/test_mcts_basic.py -v

echo "  - End-to-end tests..."
pytest tests/test_end2end_smoke.py -v

echo ""
echo "✅ All tests completed!"
echo ""
echo "📊 Coverage report available in htmlcov/index.html"
echo "📊 Test results saved in .coverage"
