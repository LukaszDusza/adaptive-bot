#!/bin/bash

# Script to remove tracked files that are now in .gitignore
# This removes them from git index but keeps them on disk

set -e

echo "=========================================="
echo "🧹 CLEANING GIT CACHE"
echo "=========================================="
echo ""
echo "This will remove from git (but keep on disk):"
echo "  - PNG files (analysis plots)"
echo "  - CSV files (analysis results)"
echo "  - DB files (Optuna databases)"
echo "  - Backup files"
echo "  - Temporary analysis files"
echo ""

read -p "Continue? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cancelled"
    exit 0
fi

echo ""
echo "📊 Removing PNG files from git..."
git ls-files | grep "\.png$" | xargs -r git rm --cached

echo ""
echo "📄 Removing CSV files from git..."
git ls-files | grep "\.csv$" | xargs -r git rm --cached

echo ""
echo "🗄️ Removing DB files from git..."
git ls-files | grep "\.db$" | xargs -r git rm --cached

echo ""
echo "💾 Removing backup files from git..."
git ls-files | grep -E "\.(backup|bak)$" | xargs -r git rm --cached

echo ""
echo "📝 Removing temporary analysis files from git..."
git rm --cached -f price_action/OPTIMIZATION_SUMMARY.md 2>/dev/null || true
git rm --cached -f price_action/RESULTS.md 2>/dev/null || true
git rm --cached -f price_action/THRESHOLD_EXPLANATION.md 2>/dev/null || true
git rm --cached -f price_action/compare_models.py 2>/dev/null || true
git rm --cached -f price_action/analyze_*.py 2>/dev/null || true
git rm --cached -f price_action/check_models.sh 2>/dev/null || true
git rm --cached -f price_action/apply_optimization.sh 2>/dev/null || true
git rm --cached -f price_action/models_analysis_full.csv 2>/dev/null || true
git rm --cached -f price_action/live_trades_analysis.csv 2>/dev/null || true

echo ""
echo "=========================================="
echo "✅ DONE!"
echo "=========================================="
echo ""
echo "Files removed from git index (but kept on disk)."
echo "Next steps:"
echo "  1. Check status: git status"
echo "  2. Commit changes: git add .gitignore && git commit -m 'Update .gitignore and remove large files'"
echo ""
echo "⚠️  NOTE: Files are still on disk, just not tracked by git anymore."
echo ""
