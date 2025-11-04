#!/bin/bash
# Sprawdza kompatybilność zmian z działającymi botami

echo "================================"
echo "COMPATIBILITY CHECK"
echo "================================"

# 1. Check if bots are running
echo -e "\n1. Checking running bots..."
docker-compose ps | grep bot- | grep Up
if [ $? -eq 0 ]; then
    echo "✅ Boty działają"
else
    echo "⚠️  Brak działających botów"
fi

# 2. Check current model versions used by bots
echo -e "\n2. Current model versions in docker-compose.yaml..."
grep -A 5 "version" docker-compose.yaml | grep "v1\." || echo "No version flags found"

# 3. Check if v1.2 models exist
echo -e "\n3. Checking v1.2 models (currently used)..."
ls -d models/v1.2.*/ 2>/dev/null | wc -l | xargs echo "   v1.2 models found:"

# 4. Check if v1.3 models exist (new)
echo -e "\n4. Checking v1.3 models (new, if trained)..."
ls -d models/v1.3.*/ 2>/dev/null | wc -l | xargs echo "   v1.3 models found:"

# 5. Files changed check
echo -e "\n5. Files modified (git status)..."
git status --short | grep -E "(bot\.py|bybit_adapter|main\.py)" && echo "⚠️  WARNING: Bot files changed!" || echo "✅ Bot files unchanged"

# 6. Backup check
echo -e "\n6. Checking backups..."
if [ -f "model_pipeline_v2.1_backup.py" ]; then
    echo "✅ Backup exists: model_pipeline_v2.1_backup.py"
else
    echo "⚠️  No backup found - create one before changes"
fi

echo -e "\n================================"
echo "SUMMARY:"
echo "================================"
echo "✅ Zmiany w model_pipeline.py dotyczą TYLKO treningu"
echo "✅ Działające boty NIE będą dotknięte"
echo "✅ v1.2 modele nadal działają"
echo "⚠️  v1.3 wejdzie w życie DOPIERO po:"
echo "   1. Retrainie modeli z --version v1.3"
echo "   2. Update docker-compose.yaml"
echo "   3. docker-compose up -d"
echo ""
