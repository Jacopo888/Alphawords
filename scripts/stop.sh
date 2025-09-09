#!/bin/bash

# Stop script for AlphaScrabble

set -e

echo "🛑 Stopping AlphaScrabble"
echo "========================"

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️  No virtual environment detected"
fi

# Check Python version
python_version=$(python --version 2>&1 | cut -d' ' -f2)
echo "🐍 Python version: $python_version"

# Stop any running processes
echo "🛑 Stopping running processes..."

# Kill any Python processes running AlphaScrabble
if pgrep -f "alphascrabble" > /dev/null; then
    echo "🛑 Stopping AlphaScrabble processes..."
    pkill -f "alphascrabble"
    sleep 2
    echo "✅ AlphaScrabble processes stopped"
else
    echo "✅ No AlphaScrabble processes running"
fi

# Kill any TensorBoard processes
if pgrep -f "tensorboard" > /dev/null; then
    echo "🛑 Stopping TensorBoard processes..."
    pkill -f "tensorboard"
    sleep 2
    echo "✅ TensorBoard processes stopped"
else
    echo "✅ No TensorBoard processes running"
fi

# Kill any Jupyter processes
if pgrep -f "jupyter" > /dev/null; then
    echo "🛑 Stopping Jupyter processes..."
    pkill -f "jupyter"
    sleep 2
    echo "✅ Jupyter processes stopped"
else
    echo "✅ No Jupyter processes running"
fi

# Kill any training processes
if pgrep -f "train" > /dev/null; then
    echo "🛑 Stopping training processes..."
    pkill -f "train"
    sleep 2
    echo "✅ Training processes stopped"
else
    echo "✅ No training processes running"
fi

# Kill any self-play processes
if pgrep -f "selfplay" > /dev/null; then
    echo "🛑 Stopping self-play processes..."
    pkill -f "selfplay"
    sleep 2
    echo "✅ Self-play processes stopped"
else
    echo "✅ No self-play processes running"
fi

# Kill any evaluation processes
if pgrep -f "eval" > /dev/null; then
    echo "🛑 Stopping evaluation processes..."
    pkill -f "eval"
    sleep 2
    echo "✅ Evaluation processes stopped"
else
    echo "✅ No evaluation processes running"
fi

# Kill any interactive play processes
if pgrep -f "play" > /dev/null; then
    echo "🛑 Stopping interactive play processes..."
    pkill -f "play"
    sleep 2
    echo "✅ Interactive play processes stopped"
else
    echo "✅ No interactive play processes running"
fi

# Check for any remaining processes
echo "🔍 Checking for remaining processes..."
remaining_processes=$(pgrep -f "alphascrabble\|tensorboard\|jupyter\|train\|selfplay\|eval\|play" | wc -l)
if [ $remaining_processes -gt 0 ]; then
    echo "⚠️  $remaining_processes processes still running"
    echo "You may need to stop them manually"
else
    echo "✅ All processes stopped"
fi

# Show system status
echo ""
echo "📊 System Status"
echo "==============="
echo "Python: $python_version"
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "Virtual environment: $VIRTUAL_ENV"
else
    echo "Virtual environment: none"
fi

# Check GPU status
if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    gpu_count=$(python -c "import torch; print(torch.cuda.device_count())")
    echo "GPU: $gpu_count available"
else
    echo "GPU: not available"
fi

# Check memory usage
if command -v free &> /dev/null; then
    memory_usage=$(free -h | grep "Mem:" | awk '{print $3 "/" $2}')
    echo "Memory usage: $memory_usage"
fi

# Check disk usage
if command -v df &> /dev/null; then
    disk_usage=$(df -h . | tail -1 | awk '{print $3 "/" $2 " (" $5 ")"}')
    echo "Disk usage: $disk_usage"
fi

echo ""
echo "✅ AlphaScrabble stopped successfully!"
echo ""
echo "To start again:"
echo "1. Run './scripts/start.sh' to start the CLI"
echo "2. Run './scripts/run_demo.sh' to run demos"
echo "3. Run 'alphascrabble --help' to see available commands"
