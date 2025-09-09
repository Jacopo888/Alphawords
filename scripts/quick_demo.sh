#!/bin/bash

# Quick demo script for AlphaScrabble

set -e

echo "🚀 AlphaScrabble Quick Demo"
echo "=========================="

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️  No virtual environment detected. Consider using one."
fi

# Check Python version
python_version=$(python --version 2>&1 | cut -d' ' -f2)
echo "🐍 Python version: $python_version"

# Run the demo
echo ""
echo "🎯 Running AlphaScrabble demo..."
python colab/demo.py

# Check if demo was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Demo completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Run 'alphascrabble selfplay --games 10' to generate training data"
    echo "2. Run 'alphascrabble train --data data/selfplay --epochs 5' to train the model"
    echo "3. Run 'alphascrabble eval --net-a checkpoints/best_model.pt --opponent random' to evaluate"
    echo "4. Run 'alphascrabble play --net checkpoints/best_model.pt' to play interactively"
else
    echo ""
    echo "❌ Demo failed. Check the error messages above."
    echo ""
    echo "Troubleshooting:"
    echo "1. Make sure AlphaScrabble is installed: pip install -e ."
    echo "2. Check that all dependencies are installed"
    echo "3. Run './scripts/check_environment.sh' to check your environment"
    echo "4. Run './scripts/dev_setup.sh' to set up development environment"
fi
