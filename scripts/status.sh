#!/bin/bash

# Status script for AlphaScrabble

set -e

echo "📊 AlphaScrabble Status"
echo "======================"

# Check Python version
python_version=$(python --version 2>&1 | cut -d' ' -f2)
echo "🐍 Python version: $python_version"

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment: $VIRTUAL_ENV"
else
    echo "⚠️  No virtual environment detected"
fi

# Check AlphaScrabble installation
echo ""
echo "📦 Installation Status"
echo "---------------------"
if python -c "import alphascrabble" 2>/dev/null; then
    echo "✅ AlphaScrabble is installed"
    
    # Get version if available
    if python -c "import alphascrabble; print(alphascrabble.__version__)" 2>/dev/null; then
        version=$(python -c "import alphascrabble; print(alphascrabble.__version__)")
        echo "   Version: $version"
    fi
else
    echo "❌ AlphaScrabble is not installed"
fi

# Check dependencies
echo ""
echo "🔧 Dependencies Status"
echo "---------------------"
dependencies=("torch" "numpy" "pandas" "click" "rich" "tqdm" "tensorboard")
for dep in "${dependencies[@]}"; do
    if python -c "import $dep" 2>/dev/null; then
        version=$(python -c "import $dep; print($dep.__version__)")
        echo "✅ $dep: $version"
    else
        echo "❌ $dep: not found"
    fi
done

# Check GPU availability
echo ""
echo "🖥️  Hardware Status"
echo "------------------"
if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    cuda_version=$(python -c "import torch; print(torch.version.cuda)")
    gpu_count=$(python -c "import torch; print(torch.cuda.device_count())")
    gpu_name=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    echo "✅ CUDA: $cuda_version"
    echo "✅ GPUs: $gpu_count"
    echo "✅ GPU: $gpu_name"
else
    echo "⚠️  CUDA: not available (CPU only)"
fi

# Check lexicon files
echo ""
echo "📚 Lexicon Status"
echo "----------------"
if [ -f "lexica_cache/enable1.txt" ]; then
    word_count=$(wc -l < lexica_cache/enable1.txt)
    echo "✅ ENABLE1: $word_count words"
else
    echo "❌ ENABLE1: not found"
fi

if [ -f "lexica_cache/english_enable1.dawg" ]; then
    dawg_size=$(wc -c < lexica_cache/english_enable1.dawg)
    echo "✅ DAWG: $dawg_size bytes"
else
    echo "❌ DAWG: not found"
fi

if [ -f "lexica_cache/english_enable1.gaddag" ]; then
    gaddag_size=$(wc -c < lexica_cache/english_enable1.gaddag)
    echo "✅ GADDAG: $gaddag_size bytes"
else
    echo "❌ GADDAG: not found"
fi

# Check Quackle
echo ""
echo "🔧 Quackle Status"
echo "----------------"
if [ -d "third_party/quackle" ]; then
    echo "✅ Quackle source: found"
    
    if [ -f "third_party/quackle/build/src/makedawg" ]; then
        echo "✅ Quackle tools: built"
    else
        echo "❌ Quackle tools: not built"
    fi
else
    echo "❌ Quackle source: not found"
fi

# Check directories
echo ""
echo "📁 Directory Status"
echo "------------------"
directories=("data" "checkpoints" "logs" "lexica_cache")
for dir in "${directories[@]}"; do
    if [ -d "$dir" ]; then
        file_count=$(find "$dir" -type f | wc -l)
        echo "✅ $dir: $file_count files"
    else
        echo "❌ $dir: not found"
    fi
done

# Check git status
echo ""
echo "📋 Git Status"
echo "-------------"
if [ -d ".git" ]; then
    current_branch=$(git branch --show-current 2>/dev/null || echo "unknown")
    echo "✅ Branch: $current_branch"
    
    if git diff-index --quiet HEAD -- 2>/dev/null; then
        echo "✅ Working directory: clean"
    else
        echo "⚠️  Working directory: has changes"
    fi
    
    if git diff --quiet HEAD origin/$current_branch 2>/dev/null; then
        echo "✅ Remote: up to date"
    else
        echo "⚠️  Remote: has updates"
    fi
else
    echo "❌ Git: not a repository"
fi

# Check development tools
echo ""
echo "🛠️  Development Tools Status"
echo "---------------------------"
dev_tools=("pytest" "black" "isort" "flake8" "mypy")
for tool in "${dev_tools[@]}"; do
    if command -v $tool &> /dev/null; then
        version=$($tool --version 2>&1 | head -n1 | cut -d' ' -f2)
        echo "✅ $tool: $version"
    else
        echo "❌ $tool: not found"
    fi
done

# Check system dependencies
echo ""
echo "🔧 System Dependencies Status"
echo "-----------------------------"
sys_deps=("cmake" "make" "gcc" "git")
for dep in "${sys_deps[@]}"; do
    if command -v $dep &> /dev/null; then
        version=$($dep --version 2>&1 | head -n1 | cut -d' ' -f3)
        echo "✅ $dep: $version"
    else
        echo "❌ $dep: not found"
    fi
done

# Summary
echo ""
echo "📊 Summary"
echo "=========="

# Count issues
issues=0
if ! python -c "import alphascrabble" 2>/dev/null; then
    ((issues++))
fi

if ! python -c "import torch" 2>/dev/null; then
    ((issues++))
fi

if [ ! -f "lexica_cache/enable1.txt" ]; then
    ((issues++))
fi

if [ ! -f "lexica_cache/english_enable1.dawg" ]; then
    ((issues++))
fi

if [ ! -f "lexica_cache/english_enable1.gaddag" ]; then
    ((issues++))
fi

if [ $issues -eq 0 ]; then
    echo "🎉 AlphaScrabble is ready to use!"
    echo ""
    echo "Available commands:"
    echo "  alphascrabble --help                    - Show help"
    echo "  alphascrabble selfplay --games 10       - Generate training data"
    echo "  alphascrabble train --data data/selfplay - Train model"
    echo "  alphascrabble eval --net-a model.pt     - Evaluate model"
    echo "  alphascrabble play --net model.pt       - Play interactively"
else
    echo "⚠️  Found $issues issues that need to be resolved"
    echo ""
    echo "To fix issues:"
    echo "1. Run './scripts/install.sh' to install AlphaScrabble"
    echo "2. Run './scripts/setup_lexicon.sh' to set up lexicon"
    echo "3. Run './scripts/dev_setup.sh' for development setup"
fi
