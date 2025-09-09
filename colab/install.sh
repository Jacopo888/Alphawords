#!/bin/bash

# Installation script for AlphaScrabble in Google Colab

set -e

echo "🚀 Installing AlphaScrabble in Google Colab..."

# Update system packages
echo "📦 Updating system packages..."
apt-get update
apt-get install -y qtbase5-dev libqt5core5a build-essential cmake ninja-build

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install -U pip wheel cmake ninja pybind11 pytest tensorboard pandas pyarrow rich tqdm click

# Install PyTorch with CUDA support
echo "🔥 Installing PyTorch with CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Clone and install AlphaScrabble
echo "📥 Cloning AlphaScrabble..."
git clone https://github.com/alphascrabble/alphascrabble.git
cd alphascrabble

# Install in development mode
echo "🔧 Installing AlphaScrabble..."
pip install -e .

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/selfplay data/training checkpoints logs lexica_cache

# Download and compile lexicon
echo "📚 Setting up lexicon..."
if [ -f "scripts/setup_lexicon.sh" ]; then
    chmod +x scripts/setup_lexicon.sh
    ./scripts/setup_lexicon.sh
else
    echo "⚠️  Lexicon setup script not found, skipping..."
fi

echo "✅ AlphaScrabble installation complete!"
echo ""
echo "You can now run:"
echo "  alphascrabble --help"
echo "  alphascrabble selfplay --games 10"
echo "  alphascrabble train --data data/selfplay --epochs 5"
