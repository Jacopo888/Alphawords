#!/bin/bash

# Benchmark script for AlphaScrabble

set -e

echo "⚡ AlphaScrabble Benchmark"
echo "========================="

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️  No virtual environment detected. Consider using one."
fi

# Check Python version
python_version=$(python --version 2>&1 | cut -d' ' -f2)
echo "🐍 Python version: $python_version"

# Check GPU availability
echo ""
echo "🖥️  Checking GPU availability..."
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('No GPU available, using CPU')
"

# Run benchmarks
echo ""
echo "🏃 Running benchmarks..."

# Benchmark 1: Model creation
echo "  📊 Benchmarking model creation..."
python -c "
import time
import sys
sys.path.append('alphascrabble')
from alphascrabble.nn.model import AlphaScrabbleNet

start_time = time.time()
model = AlphaScrabbleNet()
creation_time = time.time() - start_time
param_count = sum(p.numel() for p in model.parameters())

print(f'    Model creation time: {creation_time:.3f}s')
print(f'    Parameter count: {param_count:,}')
print(f'    Parameters per second: {param_count/creation_time:,.0f}')
"

# Benchmark 2: Forward pass
echo "  📊 Benchmarking forward pass..."
python -c "
import time
import torch
import numpy as np
import sys
sys.path.append('alphascrabble')
from alphascrabble.nn.model import AlphaScrabbleNet

model = AlphaScrabbleNet()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Create dummy data
board_features = torch.randn(1, 32, 15, 15).to(device)
rack_features = torch.randn(1, 27).to(device)
move_features = torch.randn(1, 10, 64).to(device)

# Warmup
for _ in range(10):
    with torch.no_grad():
        _ = model(board_features, rack_features, move_features)

# Benchmark
start_time = time.time()
num_iterations = 100
for _ in range(num_iterations):
    with torch.no_grad():
        _ = model(board_features, rack_features, move_features)
end_time = time.time()

avg_time = (end_time - start_time) / num_iterations
print(f'    Average forward pass time: {avg_time*1000:.2f}ms')
print(f'    Forward passes per second: {1/avg_time:.0f}')
"

# Benchmark 3: Feature extraction
echo "  📊 Benchmarking feature extraction..."
python -c "
import time
import sys
sys.path.append('alphascrabble')
from alphascrabble.engine.features import FeatureExtractor
from alphascrabble.rules.board import Board
from alphascrabble.rules.tiles_en import Tile

extractor = FeatureExtractor()
board = Board()
rack = [Tile('H'), Tile('E'), Tile('L'), Tile('L'), Tile('O')]

# Benchmark board features
start_time = time.time()
num_iterations = 1000
for _ in range(num_iterations):
    _ = extractor.extract_board_features(board, 0)
board_time = (time.time() - start_time) / num_iterations

# Benchmark rack features
start_time = time.time()
for _ in range(num_iterations):
    _ = extractor.extract_rack_features(rack)
rack_time = (time.time() - start_time) / num_iterations

print(f'    Board feature extraction: {board_time*1000:.2f}ms')
print(f'    Rack feature extraction: {rack_time*1000:.2f}ms')
"

# Benchmark 4: Move generation (if available)
echo "  📊 Benchmarking move generation..."
python -c "
import time
import sys
sys.path.append('alphascrabble')
from alphascrabble.engine.movegen import MoveGenerator
from alphascrabble.rules.board import Board
from alphascrabble.rules.tiles_en import Tile

try:
    # This will fail if qlex is not available, which is expected
    from alphascrabble.lexicon.gaddag_loader import GaddagLoader
    loader = GaddagLoader('lexica_cache/english_enable1.dawg', 'lexica_cache/english_enable1.gaddag')
    if loader.load():
        move_generator = MoveGenerator(loader)
        board = Board()
        rack = [Tile('H'), Tile('E'), Tile('L'), Tile('L'), Tile('O')]
        
        start_time = time.time()
        num_iterations = 100
        for _ in range(num_iterations):
            _ = move_generator.generate_moves(board, rack)
        move_time = (time.time() - start_time) / num_iterations
        
        print(f'    Move generation time: {move_time*1000:.2f}ms')
        print(f'    Moves per second: {1/move_time:.0f}')
    else:
        print('    Lexicon not available, skipping move generation benchmark')
except Exception as e:
    print(f'    Move generation not available: {e}')
"

echo ""
echo "✅ Benchmark complete!"
echo ""
echo "💡 Performance tips:"
echo "  - Use GPU for faster training and inference"
echo "  - Increase batch size for better GPU utilization"
echo "  - Use larger MCTS simulations for stronger play"
echo "  - Cache feature extraction results when possible"
