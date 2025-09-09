#!/bin/bash

# Run demo script for AlphaScrabble

set -e

echo "🎮 AlphaScrabble Demo Runner"
echo "==========================="

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️  No virtual environment detected. Consider using one."
fi

# Check Python version
python_version=$(python --version 2>&1 | cut -d' ' -f2)
echo "🐍 Python version: $python_version"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/selfplay
mkdir -p data/training
mkdir -p checkpoints
mkdir -p logs
mkdir -p lexica_cache

# Run different demos based on argument
case "${1:-all}" in
    "basic")
        echo "🎯 Running basic demo..."
        python colab/demo.py
        ;;
    "quick")
        echo "⚡ Running quick start demo..."
        python colab/quick_start.py
        ;;
    "selfplay")
        echo "🎮 Running self-play demo..."
        alphascrabble selfplay --games 5 --simulations 50 --out data/demo
        ;;
    "train")
        echo "🎓 Running training demo..."
        alphascrabble train --data data/demo --epochs 2 --batch-size 64
        ;;
    "eval")
        echo "🏆 Running evaluation demo..."
        alphascrabble eval --net-a checkpoints/demo_model.pt --opponent random --games 10
        ;;
    "play")
        echo "🎯 Running interactive play demo..."
        alphascrabble play --net checkpoints/demo_model.pt --human-first
        ;;
    "all")
        echo "🎯 Running all demos..."
        
        # Basic demo
        echo ""
        echo "=== Basic Demo ==="
        python colab/demo.py
        
        # Quick start
        echo ""
        echo "=== Quick Start ==="
        python colab/quick_start.py
        
        # Self-play demo
        echo ""
        echo "=== Self-Play Demo ==="
        alphascrabble selfplay --games 5 --simulations 50 --out data/demo
        
        # Training demo
        echo ""
        echo "=== Training Demo ==="
        alphascrabble train --data data/demo --epochs 2 --batch-size 64
        
        # Evaluation demo
        echo ""
        echo "=== Evaluation Demo ==="
        alphascrabble eval --net-a checkpoints/demo_model.pt --opponent random --games 10
        
        echo ""
        echo "🎉 All demos completed!"
        ;;
    *)
        echo "Usage: $0 [basic|quick|selfplay|train|eval|play|all]"
        echo ""
        echo "Available demos:"
        echo "  basic     - Basic component demo"
        echo "  quick     - Quick start demo"
        echo "  selfplay  - Self-play demo"
        echo "  train     - Training demo"
        echo "  eval      - Evaluation demo"
        echo "  play      - Interactive play demo"
        echo "  all       - Run all demos (default)"
        exit 1
        ;;
esac

echo ""
echo "✅ Demo completed!"
echo ""
echo "Available commands:"
echo "  alphascrabble --help                    - Show help"
echo "  alphascrabble selfplay --games 10       - Generate training data"
echo "  alphascrabble train --data data/selfplay - Train model"
echo "  alphascrabble eval --net-a model.pt     - Evaluate model"
echo "  alphascrabble play --net model.pt       - Play interactively"
