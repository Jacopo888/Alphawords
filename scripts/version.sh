#!/bin/bash

# Version script for AlphaScrabble

set -e

echo "📋 AlphaScrabble Version Information"
echo "==================================="

# Check Python version
python_version=$(python --version 2>&1 | cut -d' ' -f2)
echo "🐍 Python version: $python_version"

# Check AlphaScrabble version
echo ""
echo "📦 AlphaScrabble Version"
echo "-----------------------"
if python -c "import alphascrabble" 2>/dev/null; then
    if python -c "import alphascrabble; print(alphascrabble.__version__)" 2>/dev/null; then
        version=$(python -c "import alphascrabble; print(alphascrabble.__version__)")
        echo "✅ AlphaScrabble version: $version"
    else
        echo "✅ AlphaScrabble: installed (version not available)"
    fi
else
    echo "❌ AlphaScrabble: not installed"
fi

# Check git version
echo ""
echo "📋 Git Version"
echo "-------------"
if [ -d ".git" ]; then
    current_branch=$(git branch --show-current 2>/dev/null || echo "unknown")
    echo "✅ Branch: $current_branch"
    
    if git describe --tags 2>/dev/null; then
        git_version=$(git describe --tags)
        echo "✅ Git version: $git_version"
    else
        echo "✅ Git: no tags found"
    fi
    
    commit_hash=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
    echo "✅ Commit: $commit_hash"
    
    commit_date=$(git log -1 --format=%ci 2>/dev/null || echo "unknown")
    echo "✅ Last commit: $commit_date"
else
    echo "❌ Git: not a repository"
fi

# Check dependencies versions
echo ""
echo "📦 Dependencies Versions"
echo "-----------------------"
dependencies=("torch" "numpy" "pandas" "click" "rich" "tqdm" "tensorboard")
for dep in "${dependencies[@]}"; do
    if python -c "import $dep" 2>/dev/null; then
        version=$(python -c "import $dep; print($dep.__version__)")
        echo "✅ $dep: $version"
    else
        echo "❌ $dep: not found"
    fi
done

# Check system versions
echo ""
echo "🔧 System Versions"
echo "-----------------"
if command -v cmake &> /dev/null; then
    cmake_version=$(cmake --version 2>&1 | head -n1 | cut -d' ' -f3)
    echo "✅ cmake: $cmake_version"
else
    echo "❌ cmake: not found"
fi

if command -v make &> /dev/null; then
    make_version=$(make --version 2>&1 | head -n1 | cut -d' ' -f3)
    echo "✅ make: $make_version"
else
    echo "❌ make: not found"
fi

if command -v gcc &> /dev/null; then
    gcc_version=$(gcc --version 2>&1 | head -n1 | cut -d' ' -f4)
    echo "✅ gcc: $gcc_version"
else
    echo "❌ gcc: not found"
fi

if command -v git &> /dev/null; then
    git_version=$(git --version 2>&1 | cut -d' ' -f3)
    echo "✅ git: $git_version"
else
    echo "❌ git: not found"
fi

# Check OS version
echo ""
echo "🖥️  System Information"
echo "--------------------"
if command -v uname &> /dev/null; then
    os_info=$(uname -a)
    echo "✅ OS: $os_info"
else
    echo "❌ OS: unknown"
fi

# Check CUDA version
echo ""
echo "🔥 CUDA Information"
echo "------------------"
if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    cuda_version=$(python -c "import torch; print(torch.version.cuda)")
    cudnn_version=$(python -c "import torch; print(torch.backends.cudnn.version())")
    gpu_count=$(python -c "import torch; print(torch.cuda.device_count())")
    gpu_name=$(python -c "import torch; print(torch.cuda.get_device_name(0))")
    echo "✅ CUDA: $cuda_version"
    echo "✅ cuDNN: $cudnn_version"
    echo "✅ GPUs: $gpu_count"
    echo "✅ GPU: $gpu_name"
else
    echo "⚠️  CUDA: not available (CPU only)"
fi

# Check lexicon versions
echo ""
echo "📚 Lexicon Information"
echo "--------------------"
if [ -f "lexica_cache/enable1.txt" ]; then
    word_count=$(wc -l < lexica_cache/enable1.txt)
    file_size=$(wc -c < lexica_cache/enable1.txt)
    echo "✅ ENABLE1: $word_count words ($file_size bytes)"
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

# Check Quackle version
echo ""
echo "🔧 Quackle Information"
echo "--------------------"
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

# Check development tools versions
echo ""
echo "🛠️  Development Tools Versions"
echo "-----------------------------"
dev_tools=("pytest" "black" "isort" "flake8" "mypy")
for tool in "${dev_tools[@]}"; do
    if command -v $tool &> /dev/null; then
        version=$($tool --version 2>&1 | head -n1 | cut -d' ' -f2)
        echo "✅ $tool: $version"
    else
        echo "❌ $tool: not found"
    fi
done

# Summary
echo ""
echo "📊 Version Summary"
echo "================="
echo "Python: $python_version"
if python -c "import alphascrabble" 2>/dev/null; then
    if python -c "import alphascrabble; print(alphascrabble.__version__)" 2>/dev/null; then
        version=$(python -c "import alphascrabble; print(alphascrabble.__version__)")
        echo "AlphaScrabble: $version"
    else
        echo "AlphaScrabble: installed"
    fi
else
    echo "AlphaScrabble: not installed"
fi

if [ -d ".git" ]; then
    if git describe --tags 2>/dev/null; then
        git_version=$(git describe --tags)
        echo "Git: $git_version"
    else
        echo "Git: no tags"
    fi
else
    echo "Git: not a repository"
fi

if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    cuda_version=$(python -c "import torch; print(torch.version.cuda)")
    echo "CUDA: $cuda_version"
else
    echo "CUDA: not available"
fi

echo ""
echo "🎉 Version information complete!"
