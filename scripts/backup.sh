#!/bin/bash

# Backup script for AlphaScrabble

set -e

echo "💾 Creating AlphaScrabble Backup"
echo "==============================="

# Create backup directory
backup_dir="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup_dir"

echo "📁 Creating backup in: $backup_dir"

# Backup source code
echo "📦 Backing up source code..."
cp -r alphascrabble/ "$backup_dir/"
cp -r tests/ "$backup_dir/"
cp -r colab/ "$backup_dir/"
cp -r scripts/ "$backup_dir/"
cp -r cpp/ "$backup_dir/"
cp -r third_party/ "$backup_dir/" 2>/dev/null || echo "⚠️  third_party not found"

# Backup configuration files
echo "⚙️  Backing up configuration files..."
cp pyproject.toml "$backup_dir/" 2>/dev/null || echo "⚠️  pyproject.toml not found"
cp setup.py "$backup_dir/" 2>/dev/null || echo "⚠️  setup.py not found"
cp requirements.txt "$backup_dir/" 2>/dev/null || echo "⚠️  requirements.txt not found"
cp Makefile "$backup_dir/" 2>/dev/null || echo "⚠️  Makefile not found"
cp README.md "$backup_dir/" 2>/dev/null || echo "⚠️  README.md not found"
cp LICENSE "$backup_dir/" 2>/dev/null || echo "⚠️  LICENSE not found"
cp .gitignore "$backup_dir/" 2>/dev/null || echo "⚠️  .gitignore not found"

# Backup data files (optional)
echo "📊 Backing up data files..."
if [ -d "data" ]; then
    cp -r data/ "$backup_dir/"
    echo "✅ Data files backed up"
else
    echo "⚠️  No data files to backup"
fi

# Backup checkpoints (optional)
echo "💾 Backing up checkpoints..."
if [ -d "checkpoints" ]; then
    cp -r checkpoints/ "$backup_dir/"
    echo "✅ Checkpoints backed up"
else
    echo "⚠️  No checkpoints to backup"
fi

# Backup logs (optional)
echo "📝 Backing up logs..."
if [ -d "logs" ]; then
    cp -r logs/ "$backup_dir/"
    echo "✅ Logs backed up"
else
    echo "⚠️  No logs to backup"
fi

# Backup lexicon files (optional)
echo "📚 Backing up lexicon files..."
if [ -d "lexica_cache" ]; then
    cp -r lexica_cache/ "$backup_dir/"
    echo "✅ Lexicon files backed up"
else
    echo "⚠️  No lexicon files to backup"
fi

# Create backup info file
echo "📋 Creating backup info..."
cat > "$backup_dir/backup_info.txt" << EOF
AlphaScrabble Backup
===================
Created: $(date)
Version: $(git describe --tags 2>/dev/null || echo "unknown")
Branch: $(git branch --show-current 2>/dev/null || echo "unknown")
Commit: $(git rev-parse HEAD 2>/dev/null || echo "unknown")
Python: $(python --version 2>/dev/null || echo "unknown")
OS: $(uname -a)

Contents:
- Source code (alphascrabble/, tests/, colab/, scripts/, cpp/)
- Configuration files (pyproject.toml, setup.py, requirements.txt, etc.)
- Data files (data/, checkpoints/, logs/, lexica_cache/)
- Third-party code (third_party/)

To restore:
1. Copy the backup directory to your desired location
2. Run 'pip install -e .' to reinstall AlphaScrabble
3. Run './scripts/setup_lexicon.sh' to set up lexicon (if needed)
EOF

# Create compressed archive
echo "🗜️  Creating compressed archive..."
tar -czf "${backup_dir}.tar.gz" "$backup_dir"
echo "✅ Compressed archive created: ${backup_dir}.tar.gz"

# Show backup summary
echo ""
echo "📊 Backup Summary"
echo "================"
echo "Backup directory: $backup_dir"
echo "Compressed archive: ${backup_dir}.tar.gz"
echo "Total size: $(du -sh "$backup_dir" | cut -f1)"
echo "Compressed size: $(du -sh "${backup_dir}.tar.gz" | cut -f1)"

# Ask about cleanup
echo ""
read -p "Do you want to remove the uncompressed backup directory? [y/N]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf "$backup_dir"
    echo "✅ Uncompressed backup directory removed"
else
    echo "⚠️  Uncompressed backup directory preserved"
fi

echo ""
echo "🎉 Backup complete!"
echo ""
echo "Backup files:"
echo "  - ${backup_dir}.tar.gz (compressed archive)"
if [ -d "$backup_dir" ]; then
    echo "  - $backup_dir/ (uncompressed directory)"
fi
echo ""
echo "To restore from backup:"
echo "1. Extract: tar -xzf ${backup_dir}.tar.gz"
echo "2. Install: cd $backup_dir && pip install -e ."
echo "3. Setup: ./scripts/setup_lexicon.sh"
