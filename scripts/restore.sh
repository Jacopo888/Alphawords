#!/bin/bash

# Restore script for AlphaScrabble

set -e

echo "🔄 Restoring AlphaScrabble from Backup"
echo "====================================="

# Check if backup file is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <backup_file.tar.gz>"
    echo ""
    echo "Available backup files:"
    ls -la backup_*.tar.gz 2>/dev/null || echo "No backup files found"
    exit 1
fi

backup_file="$1"

# Check if backup file exists
if [ ! -f "$backup_file" ]; then
    echo "❌ Backup file not found: $backup_file"
    exit 1
fi

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️  No virtual environment detected. Consider using one."
fi

# Extract backup
echo "📦 Extracting backup..."
backup_dir="${backup_file%.tar.gz}"
tar -xzf "$backup_file"
echo "✅ Backup extracted to: $backup_dir"

# Check if backup directory exists
if [ ! -d "$backup_dir" ]; then
    echo "❌ Backup directory not found: $backup_dir"
    exit 1
fi

# Show backup info
if [ -f "$backup_dir/backup_info.txt" ]; then
    echo ""
    echo "📋 Backup Information:"
    echo "====================="
    cat "$backup_dir/backup_info.txt"
    echo ""
fi

# Ask for confirmation
echo "⚠️  This will overwrite existing files!"
read -p "Do you want to continue? [y/N]: " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Restore cancelled"
    rm -rf "$backup_dir"
    exit 1
fi

# Backup current files (if they exist)
echo "💾 Creating safety backup of current files..."
safety_backup="safety_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$safety_backup"

# Backup existing files
if [ -d "alphascrabble" ]; then
    cp -r alphascrabble/ "$safety_backup/"
fi
if [ -d "tests" ]; then
    cp -r tests/ "$safety_backup/"
fi
if [ -d "colab" ]; then
    cp -r colab/ "$safety_backup/"
fi
if [ -d "scripts" ]; then
    cp -r scripts/ "$safety_backup/"
fi
if [ -d "cpp" ]; then
    cp -r cpp/ "$safety_backup/"
fi
if [ -d "third_party" ]; then
    cp -r third_party/ "$safety_backup/"
fi

# Backup configuration files
for file in pyproject.toml setup.py requirements.txt Makefile README.md LICENSE .gitignore; do
    if [ -f "$file" ]; then
        cp "$file" "$safety_backup/"
    fi
done

echo "✅ Safety backup created: $safety_backup"

# Restore files
echo "🔄 Restoring files..."

# Restore source code
if [ -d "$backup_dir/alphascrabble" ]; then
    rm -rf alphascrabble/
    cp -r "$backup_dir/alphascrabble/" ./
    echo "✅ Source code restored"
fi

if [ -d "$backup_dir/tests" ]; then
    rm -rf tests/
    cp -r "$backup_dir/tests/" ./
    echo "✅ Tests restored"
fi

if [ -d "$backup_dir/colab" ]; then
    rm -rf colab/
    cp -r "$backup_dir/colab/" ./
    echo "✅ Colab files restored"
fi

if [ -d "$backup_dir/scripts" ]; then
    rm -rf scripts/
    cp -r "$backup_dir/scripts/" ./
    echo "✅ Scripts restored"
fi

if [ -d "$backup_dir/cpp" ]; then
    rm -rf cpp/
    cp -r "$backup_dir/cpp/" ./
    echo "✅ C++ files restored"
fi

if [ -d "$backup_dir/third_party" ]; then
    rm -rf third_party/
    cp -r "$backup_dir/third_party/" ./
    echo "✅ Third-party code restored"
fi

# Restore configuration files
for file in pyproject.toml setup.py requirements.txt Makefile README.md LICENSE .gitignore; do
    if [ -f "$backup_dir/$file" ]; then
        cp "$backup_dir/$file" ./
        echo "✅ $file restored"
    fi
done

# Restore data files (optional)
echo "📊 Restoring data files..."
if [ -d "$backup_dir/data" ]; then
    rm -rf data/
    cp -r "$backup_dir/data/" ./
    echo "✅ Data files restored"
else
    echo "⚠️  No data files to restore"
fi

# Restore checkpoints (optional)
echo "💾 Restoring checkpoints..."
if [ -d "$backup_dir/checkpoints" ]; then
    rm -rf checkpoints/
    cp -r "$backup_dir/checkpoints/" ./
    echo "✅ Checkpoints restored"
else
    echo "⚠️  No checkpoints to restore"
fi

# Restore logs (optional)
echo "📝 Restoring logs..."
if [ -d "$backup_dir/logs" ]; then
    rm -rf logs/
    cp -r "$backup_dir/logs/" ./
    echo "✅ Logs restored"
else
    echo "⚠️  No logs to restore"
fi

# Restore lexicon files (optional)
echo "📚 Restoring lexicon files..."
if [ -d "$backup_dir/lexica_cache" ]; then
    rm -rf lexica_cache/
    cp -r "$backup_dir/lexica_cache/" ./
    echo "✅ Lexicon files restored"
else
    echo "⚠️  No lexicon files to restore"
fi

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
    echo "✅ AlphaScrabble restored successfully!"
else
    echo "❌ AlphaScrabble restore failed"
    exit 1
fi

# Cleanup
echo "🗑️  Cleaning up..."
rm -rf "$backup_dir"

echo ""
echo "🎉 Restore complete!"
echo ""
echo "Restored from: $backup_file"
echo "Safety backup: $safety_backup"
echo ""
echo "Next steps:"
echo "1. Run './scripts/run_demo.sh' to test the restore"
echo "2. Run 'alphascrabble --help' to see available commands"
echo ""
echo "Available commands:"
echo "  alphascrabble --help                    - Show help"
echo "  alphascrabble selfplay --games 10       - Generate training data"
echo "  alphascrabble train --data data/selfplay - Train model"
echo "  alphascrabble eval --net-a model.pt     - Evaluate model"
echo "  alphascrabble play --net model.pt       - Play interactively"
