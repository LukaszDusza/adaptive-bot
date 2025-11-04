# 🔄 Migration Guide: v1.2 → v1.3 (Label Balance Fix)

## ✅ Kompatybilność: ZERO IMPACT na Działające Boty

**WAŻNE:** Zmiany w `model_pipeline.py` dotyczą **TYLKO treningu**. Działające boty w Docker **NIE są dotknięte** dopóki nie zrobisz redeploymentu.

### Dlaczego Safe?
```
model_pipeline.py (ZMIENIONE)
    ↓
    Używane TYLKO przez: python main.py --train
    ↓
    Tworzy NOWE modele w: models/v1.3.*/
    ↓
    Boty używają STARYCH modeli: models/v1.2.*/
    ↓
    Zero impact! ✅
```

## 📋 Pre-Migration Checklist

```bash
# 1. Verify running bots
docker-compose ps
# Expected: Wszystkie boty "Up"

# 2. Check current versions
grep "version v1" docker-compose.yaml
# Expected: --version v1.2.* wszędzie

# 3. Backup production models
tar -czf models_v1.2_backup_$(date +%Y%m%d).tar.gz models/v1.2.*/

# 4. Backup code
cp model_pipeline.py model_pipeline_v2.1_backup.py

# 5. Verify no pending changes to bot files
git status | grep -E "(bot.py|bybit_adapter|main.py)"
# Expected: Empty (no changes to runtime files)
```

## 🚀 Migration Strategy Options

### Option 1: Stopniowa Migracja (ZALECANA dla Production)

**Timeline: 2-3 tygodnie**

#### Tydzień 1: Training & Backtesting
```bash
# DAY 1-2: Apply fixes
cd price_action
cp model_pipeline.py model_pipeline_backup.py
# Apply 3 fixes (see LABEL_FIX_README.md)
python verify_fixes.py

# DAY 3-5: Train v1.3 models (PARALLEL z działającymi botami v1.2)
# Najpierw 2-3 główne tickery dla testu
for ticker in SOLUSDT XRPUSDT; do
  for side in long short; do
    echo "Training $ticker $side..."
    python main.py --train --side $side --ticker $ticker \
      --timeframe 15m --helper-timeframes 1h 4h \
      --version v1.3 \
      --label-trials 500 --model-trials 400
  done
done

# DAY 6: Analyze results
python analyze_label_distribution.py --version v1.3
# Expected: HOLD ~60%, ACTION ~40%

# DAY 7: Backtest comparison
python main.py --backtest --ticker SOLUSDT --timeframe 15m \
  --helper-timeframes 1h 4h --version v1.2
python main.py --backtest --ticker SOLUSDT --timeframe 15m \
  --helper-timeframes 1h 4h --version v1.3

# Compare: PnL, Win Rate, Drawdown
```

#### Tydzień 2: Pilot Deployment (1-2 boty)
```bash
# DAY 8: Deploy TYLKO 1 bot na v1.3 (smallest position size)
# Edit docker-compose.yaml:

services:
  # PILOT: v1.3
  bot-syl-sol-dynamic-tp:
    command: >
      python main.py --run-bot
      --version v1.3  # <--- CHANGE
      --ticker SOLUSDT --timeframe 15m
      --helper-timeframes 1h 4h
      --trade-size 50.0  # <--- REDUCE (było 100)
      --leverage 10
      --dynamic-tp
    # ... rest unchanged

  # CONTROL: v1.2 (pozostałe boty bez zmian)
  bot-luk-sol-dynamic-tp:
    command: >
      python main.py --run-bot
      --version v1.2.sol  # <--- NO CHANGE
      ...

# Restart TYLKO pilota
docker-compose up -d bot-syl-sol-dynamic-tp
docker-compose logs -f bot-syl-sol-dynamic-tp

# DAY 9-14: Monitor
# Compare metrics:
python analyze_trades.py  # Check v1.2 vs v1.3 live performance
```

**Monitoring Criteria (Tydzień 2):**
```bash
# Pilot bot (v1.3) powinien mieć:
- ✅ Win rate: 45-52% (was 35-40% on v1.2)
- ✅ Więcej sygnałów: +40-60%
- ✅ Mniej dead periods (>2 dni bez trade)
- ✅ Drawdown < 15%
- ❌ If win rate < 38% → ROLLBACK to v1.2
```

#### Tydzień 3: Full Migration
```bash
# If pilot successful:

# DAY 15: Train pozostałe tickery
for ticker in DOGEUSDT 1000PEPEUSDT KASUSDT ...; do
  for side in long short; do
    python main.py --train --side $side --ticker $ticker \
      --version v1.3 --label-trials 500 --model-trials 400
  done
done

# DAY 16: Update docker-compose.yaml (wszystkie boty → v1.3)
# Use sed lub manual edit:
sed -i 's/--version v1\.2\.\w\+/--version v1.3/g' docker-compose.yaml

# DAY 17: Full redeploy
docker-compose down
docker-compose up --build -d

# DAY 18-21: Monitor all bots
docker-compose logs -f
```

---

### Option 2: Shadow Mode (Production-Safe Testing)

**Timeline: 3-4 tygodnie**

```bash
# 1. Train v1.3 models

# 2. Add SHADOW bots to docker-compose.yaml (alongside v1.2)
services:
  # PRODUCTION: v1.2 (100% trade size)
  bot-syl-sol-v1.2:
    container_name: syl-sol-v1.2-prod
    command: >
      --version v1.2.sol
      --trade-size 100.0
      --leverage 10

  # SHADOW: v1.3 (10% trade size - for testing)
  bot-syl-sol-v1.3-shadow:
    container_name: syl-sol-v1.3-test
    env_file: .env_sylwia  # Same account!
    command: >
      --version v1.3
      --trade-size 10.0  # 10x smaller!
      --leverage 10

# 3. Run both side-by-side
docker-compose up -d

# 4. Monitor po 2-3 tygodnie
# Compare:
# - v1.2: 100 USD/trade, N trades
# - v1.3: 10 USD/trade, M trades (should be M > N, higher frequency)

# 5. Decision:
# If v1.3 better → full migration
# If v1.3 worse → kill shadow, stay v1.2
```

---

### Option 3: Fast Migration (Dev/Testnet Only)

**⚠️ NIE używać na Production bez testów!**

```bash
# 1. Train all v1.3 models (1-2 dni)
./run_model_workflow.sh
# Choose: Train ALL models
# Version: v1.3

# 2. Update docker-compose.yaml
sed -i 's/v1\.2\.\w\+/v1.3/g' docker-compose.yaml

# 3. Full redeploy
docker-compose down
docker-compose up --build -d

# 4. Monitor closely przez 7 dni
```

---

## 🔍 Monitoring Post-Migration

### Key Metrics to Track

```bash
# 1. Daily PnL comparison
# Compare v1.2 vs v1.3:
python -c "
import pandas as pd
import glob

# Load all trade logs
v12_trades = glob.glob('logs/trades/*v1.2*.json')
v13_trades = glob.glob('logs/trades/*v1.3*.json')

# Compare metrics
print('v1.2 Trades:', len(v12_trades))
print('v1.3 Trades:', len(v13_trades))
# Expected: v1.3 > v1.2 (more signals)
"

# 2. Win rate
python analyze_trades.py
# Filter by version, compare WR%

# 3. Signal frequency
docker-compose logs bot-syl-sol-dynamic-tp | grep "SIGNAL:" | wc -l

# 4. Dead periods (days without trades)
# Should decrease with v1.3
```

### Rollback Procedure (if needed)

```bash
# If v1.3 performance worse than expected:

# 1. Quick rollback (revert docker-compose.yaml)
git checkout docker-compose.yaml  # If committed
# Or manual edit: change v1.3 → v1.2

# 2. Restart bots
docker-compose down
docker-compose up -d

# 3. Restore models if needed
tar -xzf models_v1.2_backup_YYYYMMDD.tar.gz

# 4. Investigate
# - Check label distribution: python analyze_label_distribution.py --version v1.3
# - Check if constraint too tight: ACTION% should be 40%, not 30%
# - Loosen constraint if needed: (0.30, 0.55) instead of (0.35, 0.50)
```

---

## 📊 Success Criteria

### Migrate to v1.3 if:
- ✅ Win rate: 45-52% (improvement z 35-40%)
- ✅ Signal frequency: +40-60%
- ✅ Drawdown: < 20% (same lub better)
- ✅ PnL: Positive po 2 tygodniach
- ✅ No critical bugs

### Stay on v1.2 if:
- ❌ Win rate: < 40% (worse niż v1.2)
- ❌ Signal frequency: Nie wzrosła significantly
- ❌ Drawdown: > 25%
- ❌ Critical bugs w predictions

---

## 🎯 Timeline Summary

| Strategy | Timeline | Risk | Best For |
|----------|----------|------|----------|
| **Stopniowa** | 2-3 weeks | Low | Production |
| **Shadow Mode** | 3-4 weeks | Very Low | High-stakes |
| **Fast** | 3-7 days | High | Dev/Testnet |

**Zalecenie:** Stopniowa migracja dla production accounts.

---

## 📞 Support

**Pytania?**
- Quick guide: `LABEL_FIX_README.md`
- Analiza problemu: `LABEL_OPTIMIZATION_DIAGNOSIS.md`
- Full summary: `FINAL_SUMMARY.md`

**Verify compatibility:**
```bash
bash check_compatibility.sh
```

**Check if fixes applied:**
```bash
python verify_fixes.py
```
