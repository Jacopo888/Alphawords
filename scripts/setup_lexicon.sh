#!/bin/bash

# Setup script for AlphaScrabble lexicon
# Downloads ENABLE1 and compiles DAWG/GADDAG files

set -e

echo "🔧 Setting up AlphaScrabble lexicon..."

# Create directories
mkdir -p lexica_cache
mkdir -p third_party

# Download ENABLE1 if not exists
if [ ! -f "lexica_cache/enable1.txt" ]; then
    echo "📥 Downloading ENABLE1 word list..."
    wget -q https://norvig.com/ngrams/enable1.txt -O lexica_cache/enable1.txt
    echo "✅ ENABLE1 downloaded"
else
    echo "✅ ENABLE1 already exists"
fi

# Clone Quackle if not exists
if [ ! -d "third_party/quackle" ]; then
    echo "📥 Cloning Quackle..."
    git clone --depth 1 https://github.com/quackle/quackle.git third_party/quackle
    echo "✅ Quackle cloned"
else
    echo "✅ Quackle already exists"
fi

# Build Quackle if not built
if [ ! -f "third_party/quackle/build/src/makedawg" ]; then
    echo "🔨 Building Quackle..."
    cd third_party/quackle
    mkdir -p build
    cd build
    
    # Configure with CMake
    cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DCMAKE_CXX_FLAGS="-fPIC -O3" \
        ..
    
    # Build
    make -j$(nproc)
    cd ../../..
    echo "✅ Quackle built"
else
    echo "✅ Quackle already built"
fi

# Compile DAWG if not exists
if [ ! -f "lexica_cache/english_enable1.dawg" ]; then
    echo "🔨 Compiling DAWG..."
    third_party/quackle/build/src/makedawg \
        lexica_cache/enable1.txt \
        english.quackle_alphabet \
        > lexica_cache/english_enable1.dawg
    echo "✅ DAWG compiled"
else
    echo "✅ DAWG already exists"
fi

# Compile GADDAG if not exists
if [ ! -f "lexica_cache/english_enable1.gaddag" ]; then
    echo "🔨 Compiling GADDAG..."
    third_party/quackle/build/src/makegaddag \
        lexica_cache/english_enable1.dawg \
        > lexica_cache/english_enable1.gaddag
    echo "✅ GADDAG compiled"
else
    echo "✅ GADDAG already exists"
fi

# Verify files
echo "🔍 Verifying lexicon files..."
if [ -f "lexica_cache/english_enable1.dawg" ] && [ -f "lexica_cache/english_enable1.gaddag" ]; then
    dawg_size=$(wc -c < lexica_cache/english_enable1.dawg)
    gaddag_size=$(wc -c < lexica_cache/english_enable1.gaddag)
    echo "✅ Lexicon files created successfully"
    echo "📊 DAWG size: $dawg_size bytes"
    echo "📊 GADDAG size: $gaddag_size bytes"
else
    echo "❌ Lexicon compilation failed"
    exit 1
fi

echo "🎉 Lexicon setup complete!"
echo ""
echo "Files created:"
echo "  - lexica_cache/enable1.txt"
echo "  - lexica_cache/english_enable1.dawg"
echo "  - lexica_cache/english_enable1.gaddag"
echo ""
echo "You can now run AlphaScrabble with:"
echo "  alphascrabble selfplay --games 10"
