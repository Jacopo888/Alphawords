#!/bin/bash

# Environment check script for AlphaScrabble

set -e

echo "🔍 Checking AlphaScrabble Environment"
echo "===================================="

# Check Python version
echo "🐍 Checking Python version..."
python_version=$(python --version 2>&1 | cut -d' ' -f2)
echo "  Python version: $python_version"

# Check if Python version is 3.10+
if python -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null; then
    echo "  ✅ Python version is compatible (3.10+)"
else
    echo "  ❌ Python version is too old (need 3.10+)"
    exit 1
fi

# Check pip
echo ""
echo "📦 Checking pip..."
if command -v pip &> /dev/null; then
    pip_version=$(pip --version 2>&1 | cut -d' ' -f2)
    echo "  ✅ pip version: $pip_version"
else
    echo "  ❌ pip not found"
    exit 1
fi

# Check virtual environment
echo ""
echo "🌐 Checking virtual environment..."
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "  ✅ Virtual environment: $VIRTUAL_ENV"
else
    echo "  ⚠️  No virtual environment detected (recommended for development)"
fi

# Check system dependencies
echo ""
echo "🔧 Checking system dependencies..."

# Check cmake
if command -v cmake &> /dev/null; then
    cmake_version=$(cmake --version 2>&1 | head -n1 | cut -d' ' -f3)
    echo "  ✅ cmake version: $cmake_version"
else
    echo "  ❌ cmake not found (required for C++ compilation)"
fi

# Check make
if command -v make &> /dev/null; then
    make_version=$(make --version 2>&1 | head -n1 | cut -d' ' -f3)
    echo "  ✅ make version: $make_version"
else
    echo "  ❌ make not found (required for building)"
fi

# Check gcc/g++
if command -v gcc &> /dev/null; then
    gcc_version=$(gcc --version 2>&1 | head -n1 | cut -d' ' -f4)
    echo "  ✅ gcc version: $gcc_version"
else
    echo "  ❌ gcc not found (required for C++ compilation)"
fi

# Check git
if command -v git &> /dev/null; then
    git_version=$(git --version 2>&1 | cut -d' ' -f3)
    echo "  ✅ git version: $git_version"
else
    echo "  ❌ git not found (required for cloning repositories)"
fi

# Check Python dependencies
echo ""
echo "🐍 Checking Python dependencies..."

# Check if AlphaScrabble is installed
if python -c "import alphascrabble" 2>/dev/null; then
    echo "  ✅ AlphaScrabble is installed"
else
    echo "  ❌ AlphaScrabble is not installed"
    echo "     Run: pip install -e ."
fi

# Check PyTorch
if python -c "import torch" 2>/dev/null; then
    torch_version=$(python -c "import torch; print(torch.__version__)")
    echo "  ✅ PyTorch version: $torch_version"
    
    # Check CUDA
    if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
        cuda_version=$(python -c "import torch; print(torch.version.cuda)")
        gpu_count=$(python -c "import torch; print(torch.cuda.device_count())")
        echo "  ✅ CUDA available: $cuda_version (GPUs: $gpu_count)"
    else
        echo "  ⚠️  CUDA not available (CPU only)"
    fi
else
    echo "  ❌ PyTorch not found"
fi

# Check other dependencies
dependencies=("numpy" "pandas" "click" "rich" "tqdm" "tensorboard")
for dep in "${dependencies[@]}"; do
    if python -c "import $dep" 2>/dev/null; then
        version=$(python -c "import $dep; print($dep.__version__)")
        echo "  ✅ $dep version: $version"
    else
        echo "  ❌ $dep not found"
    fi
done

# Check development tools
echo ""
echo "🛠️  Checking development tools..."

dev_tools=("pytest" "black" "isort" "flake8" "mypy")
for tool in "${dev_tools[@]}"; do
    if command -v $tool &> /dev/null; then
        version=$($tool --version 2>&1 | head -n1 | cut -d' ' -f2)
        echo "  ✅ $tool version: $version"
    else
        echo "  ⚠️  $tool not found (optional for development)"
    fi
done

# Check lexicon files
echo ""
echo "📚 Checking lexicon files..."

if [ -f "lexica_cache/enable1.txt" ]; then
    word_count=$(wc -l < lexica_cache/enable1.txt)
    echo "  ✅ ENABLE1 word list: $word_count words"
else
    echo "  ❌ ENABLE1 word list not found"
    echo "     Run: ./scripts/setup_lexicon.sh"
fi

if [ -f "lexica_cache/english_enable1.dawg" ]; then
    dawg_size=$(wc -c < lexica_cache/english_enable1.dawg)
    echo "  ✅ DAWG lexicon: $dawg_size bytes"
else
    echo "  ❌ DAWG lexicon not found"
    echo "     Run: ./scripts/setup_lexicon.sh"
fi

if [ -f "lexica_cache/english_enable1.gaddag" ]; then
    gaddag_size=$(wc -c < lexica_cache/english_enable1.gaddag)
    echo "  ✅ GADDAG lexicon: $gaddag_size bytes"
else
    echo "  ❌ GADDAG lexicon not found"
    echo "     Run: ./scripts/setup_lexicon.sh"
fi

# Check Quackle
echo ""
echo "🔧 Checking Quackle..."

if [ -d "third_party/quackle" ]; then
    echo "  ✅ Quackle source code found"
    
    if [ -f "third_party/quackle/build/src/makedawg" ]; then
        echo "  ✅ Quackle tools built"
    else
        echo "  ❌ Quackle tools not built"
        echo "     Run: ./scripts/setup_lexicon.sh"
    fi
else
    echo "  ❌ Quackle source code not found"
    echo "     Run: ./scripts/setup_lexicon.sh"
fi

# Check directories
echo ""
echo "📁 Checking directories..."

directories=("data" "checkpoints" "logs" "lexica_cache")
for dir in "${directories[@]}"; do
    if [ -d "$dir" ]; then
        echo "  ✅ $dir directory exists"
    else
        echo "  ⚠️  $dir directory not found (will be created when needed)"
    fi
done

# Summary
echo ""
echo "📊 Environment Check Summary"
echo "============================"

# Count issues
issues=0
if ! python -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null; then
    ((issues++))
fi

if ! command -v cmake &> /dev/null; then
    ((issues++))
fi

if ! command -v make &> /dev/null; then
    ((issues++))
fi

if ! command -v gcc &> /dev/null; then
    ((issues++))
fi

if ! command -v git &> /dev/null; then
    ((issues++))
fi

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
    echo "🎉 Environment is ready for AlphaScrabble!"
    echo ""
    echo "Next steps:"
    echo "1. Run 'make test' to run tests"
    echo "2. Run 'make selfplay-demo' to test self-play"
    echo "3. Run 'make train-demo' to test training"
    echo "4. Run 'make eval-demo' to test evaluation"
else
    echo "⚠️  Found $issues issues that need to be resolved"
    echo ""
    echo "To fix issues:"
    echo "1. Install missing system dependencies"
    echo "2. Run 'pip install -e .' to install AlphaScrabble"
    echo "3. Run './scripts/setup_lexicon.sh' to set up lexicon"
    echo "4. Run './scripts/dev_setup.sh' for development setup"
fi
