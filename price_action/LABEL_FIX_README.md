# 🔧 Quick Fix: Rozkład Klas 60/40

## Problem
```
Obecnie: HOLD 75% / ACTION 25%  ❌ (za dużo HOLD)
Cel:     HOLD 60% / ACTION 40%  ✅ (realistic dla tradingu)

Uwaga: Po multiclass labeling (0=HOLD, 1=BUY, 2=SELL) następuje
konwersja na binary:
- LONG: HOLD (0) vs BUY (1) - odrzuca SELL (2)
- SHORT: HOLD (0) vs SELL (2→1) - odrzuca BUY (1)

Rozkład 60:40 jest lepszy niż 50:50, bo:
✓ Rynek nie zawsze ma opportunity → więcej HOLD jest naturalne
✓ Unika overtrading (za dużo sygnałów)
✓ Sweet spot: wystarczająco opportunities + quality signals
```

## Rozwiązanie (3 zmiany w model_pipeline.py)

### ✅ ZMIANA #1: Zwiększ time_limit range (linia ~698)
```python
'time_limit': (24, 72)  # było: (12, 48)
```

### ✅ ZMIANA #2: Zwiększ barrier_size range (linia ~697)
```python
'barrier_size': (0.020, 0.045)  # było: (0.015, 0.040)
```

### ✅ ZMIANA #3: Balance constraint dodany automatycznie (linia ~1069)
```python
# POPRAWKA #11 (v1.3): BALANCE CONSTRAINT
action_pct = actual_dist.get(1, 0) + actual_dist.get(2, 0)  # BUY + SELL
trial.set_user_attr("action_pct", float(action_pct))

# Cel: 35-50% ACTION w multiclass (po binary → 60% HOLD / 40% ACTION)
if action_pct < 0.35 or action_pct > 0.50:
    logging.debug(f"Trial {trial.number} REJECTED: ACTION%={action_pct*100:.1f}%")
    return 0.0
```

## ✅ STATUS: Zmiany Zastosowane!

Wszystkie 3 zmiany zostały automatycznie zastosowane w model_pipeline.py.

## Test
```bash
# 1. Verify changes
cd price_action
grep "'barrier_size':" model_pipeline.py
grep "'time_limit':" model_pipeline.py
grep "POPRAWKA #11" model_pipeline.py

# 2. Train test model
python main.py --train --side long \
  --ticker SOLUSDT --timeframe 15m \
  --helper-timeframes 1h 4h \
  --version v1.3 \
  --label-trials 500 --model-trials 400

# 3. Check result
python analyze_label_distribution.py --version v1.3
# Expected: ~60% HOLD / ~40% ACTION (target: 60:40 ratio)
```

## Expected Changes
- ACTION%: 25% → 40-45% (po binary: 60% HOLD / 40% ACTION)
- Barrier: 1.5% → 2.5%
- Time: 25 → 40 candles
- F1: może spaść 0.65→0.58 (OK! - bardziej realistyczny)

## Files
- `LABEL_FIX_README.md` - Ten quick guide
- `LABEL_OPTIMIZATION_DIAGNOSIS.md` - Szczegółowa analiza problemu
- `LABEL_OPTIMIZATION_FIX.py` - Pełny kod z komentarzami
- `FINAL_SUMMARY.md` - Kompletne podsumowanie
- `MIGRATION_GUIDE.md` - Migration v1.2 → v1.3
- `analyze_label_distribution.py` - Tool do analizy

## Pytania?
Zobacz `FINAL_SUMMARY.md` lub `MIGRATION_GUIDE.md`
