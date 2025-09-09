#!/bin/bash

# Update script for AlphaScrabble

set -e

echo "🔄 Updating AlphaScrabble"
echo "======================="

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️  No virtual environment detected. Consider using one."
fi

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "❌ Not in a git repository. Cannot update."
    exit 1
fi

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo "⚠️  You have uncommitted changes."
    read -p "Do you want to stash them before updating? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git stash
        echo "✅ Changes stashed"
    else
        echo "❌ Cannot update with uncommitted changes"
        exit 1
    fi
fi

# Fetch latest changes
echo "📥 Fetching latest changes..."
git fetch origin

# Check current branch
current_branch=$(git branch --show-current)
echo "📍 Current branch: $current_branch"

# Check if there are updates
if git diff --quiet HEAD origin/$current_branch; then
    echo "✅ Already up to date!"
    exit 0
fi

# Show what will be updated
echo "📋 Changes to be applied:"
git log --oneline HEAD..origin/$current_branch

# Ask for confirmation
echo ""
read -p "Do you want to update? [y/N]: " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Update cancelled"
    exit 1
fi

# Update the code
echo "🔄 Updating code..."
git pull origin $current_branch

# Reinstall AlphaScrabble
echo "🔧 Reinstalling AlphaScrabble..."
pip install -e .

# Run tests
echo "🧪 Running tests..."
if command -v pytest &> /dev/null; then
    pytest tests/test_rules.py -v
    echo "✅ Tests passed"
else
    echo "⚠️  pytest not found, skipping tests"
fi

# Check installation
echo "🔍 Checking installation..."
if python -c "import alphascrabble" 2>/dev/null; then
    echo "✅ AlphaScrabble updated successfully!"
else
    echo "❌ AlphaScrabble update failed"
    exit 1
fi

echo ""
echo "🎉 Update complete!"
echo ""
echo "Next steps:"
echo "1. Run './scripts/run_demo.sh' to test the update"
echo "2. Run 'alphascrabble --help' to see available commands"
echo ""
echo "Available commands:"
echo "  alphascrabble --help                    - Show help"
echo "  alphascrabble selfplay --games 10       - Generate training data"
echo "  alphascrabble train --data data/selfplay - Train model"
echo "  alphascrabble eval --net-a model.pt     - Evaluate model"
echo "  alphascrabble play --net model.pt       - Play interactively"
