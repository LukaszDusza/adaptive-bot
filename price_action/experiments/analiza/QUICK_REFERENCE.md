# 🚀 Quick Reference - Bot Improvements

**Date:** November 22, 2025

---

## ✅ WHAT WAS IMPLEMENTED

### 1. RSI REVERSAL FILTER (Enabled by default)

**Your insight:**
> "bot powinien wchodzic w trade short gdy rsi z wysokiego poziomu spada ponizej 70"

**What it does:**
- ✅ Enters SHORT when RSI crosses 70 DOWN (reversal from overbought)
- ✅ Enters LONG when RSI crosses 30 UP (reversal from oversold)
- ✅ Blocks LONG if RSI >65 without 30-cross (prevents late entries)
- ✅ Blocks SHORT if RSI <35 without 70-cross (prevents late entries)

**Expected result:**
- Nov 7 would have been: **-35 USDT → -12 USDT** (+23 USDT improvement!)

---

### 2. SINGLE ATR-BASED DCA (1 level instead of 3)

**Your request:**
> "ma byc jeden poziom... opartego o ten wspolczynnik atr"

**What changed:**
- ❌ OLD: 3 levels (Fixed 0.3%, ATR 0.5x, Swing 1.5%)
- ✅ NEW: 1 level (ATR 1.5x = ~1.18% avg distance)

**Based on historical data:**
- Analyzed 16 tickers (Nov 2025)
- Average ATR: 0.785%
- Optimal multiplier: 1.5x
- Average distance: 1.177%

---

## 🎮 HOW TO USE

### Default (No Changes Needed):

```bash
# RSI filter is ENABLED by default
# DCA uses 1.5x ATR by default
docker compose up --build -d
```

### Customize RSI Filter:

```bash
# Disable RSI filter (test without)
--no-rsi-reversal-filter

# Adjust thresholds (more aggressive)
--rsi-high-threshold 60      # Block LONG if RSI >60 (was 65)
--rsi-low-threshold 40       # Block SHORT if RSI <40 (was 35)
--rsi-overbought-level 75    # SHORT signal at 75 (was 70)
--rsi-oversold-level 25      # LONG signal at 25 (was 30)
```

### Customize DCA:

```bash
# Enable DCA mode
--dca-mode

# Adjust ATR multiplier
--dca-atr-multiplier 2.0     # Wider distance (2x ATR)
--dca-atr-multiplier 1.0     # Tighter distance (1x ATR)
```

---

## 📊 WHAT TO EXPECT

### RSI Filter Logs:

**Reversal detected:**
```
🔴 RSI REVERSAL: Crossed 70.0 down (72.0 → 68.0) - OVERRIDE to SHORT
🔄 RSI Filter Applied: BUY → SELL (RSI: 68.0)
```

**Late entry blocked:**
```
⚠️ RSI HIGH FILTER: RSI=67.3 >65.0 without 30.0-cross - BLOCKING LONG
🔄 RSI Filter Applied: BUY → HOLD (RSI: 67.3)
```

### DCA Logs:

**Single ATR-based order:**
```
📊 DCA Mode: Single ATR-based limit order
   Level: $168.1234 (1.18%)
   Quantity: 5.0000
✓ DCA order placed | Wait max 300s
   🛡️  SL/TP protection EMBEDDED in order
```

---

## 🧪 TESTING CHECKLIST

### 1. Backtest (Recommended):

```bash
cd price_action

# Test Nov 1-21 data with filter
python main.py --backtest \
  --ticker DOGEUSDT --timeframe 15m \
  --helper-timeframes 1h 4h 6h 12h 1d \
  --version v1.2.doge \
  --prob-threshold 0.6039 \
  --tp-pct 0.025 --tsl-pct 0.008 \
  --trade-size 100.0 --leverage 20 \
  --dynamic-tp

# Compare: test without filter
# (add --no-rsi-reversal-filter)
```

**Check:**
- [ ] PnL improved?
- [ ] Win rate improved?
- [ ] Trade count reduced 15-20%?
- [ ] Max drawdown reduced?

### 2. Paper Trading (1-2 weeks):

```bash
# Use testnet
env_file: .env_testnet

# Monitor
docker compose logs -f bot-name
grep "RSI REVERSAL" logs/bot_*.log
grep "RSI.*FILTER" logs/bot_*.log
```

**Check:**
- [ ] Filter triggers correctly?
- [ ] No crashes/errors?
- [ ] PnL better than before?

### 3. Live (After successful testing):

```bash
docker compose down
docker compose up --build -d
```

---

## 🎯 SUCCESS INDICATORS

### RSI Filter Working:

✅ **Good:**
- Many "RSI HIGH FILTER" during uptrends
- Few trades opened with RSI >65
- Reversals generating profit
- Nov 7-like days avoided

❌ **Bad:**
- Too many blocked trades (>40%)
- Missing good opportunities
- Reversals failing

**Fix:** Adjust `--rsi-high-threshold` or `--rsi-overbought-level`

### DCA Working:

✅ **Good:**
- Orders fill at ~1.18% distance
- Better entry than market orders
- SL/TP working correctly

❌ **Bad:**
- Orders timing out frequently
- Distance too wide/tight

**Fix:** Adjust `--dca-atr-multiplier`

---

## 📁 DOCUMENTATION

Full details in:
- **`BOT_IMPROVEMENTS_SUMMARY.md`** - Complete implementation summary
- **`RSI_REVERSAL_FILTER_IMPLEMENTATION.md`** - RSI filter guide with examples
- **`RSI_REVERSAL_ANALYSIS.md`** - Theory and problem analysis
- **`ANALYSIS_SUMMARY.md`** - Original worst trades analysis

---

## ⚡ QUICK COMMANDS

### View RSI filter status:

```bash
docker compose logs bot-name | grep "RSI Filter"
# Should show: RSI Filter: ON (reversal-based entry)
```

### View DCA configuration:

```bash
docker compose logs bot-name | grep "Order Type"
# Should show: Order Type: DCA LIMIT (ATR-based, timeout: 300s)
```

### Check filter activations:

```bash
grep "RSI REVERSAL" logs/bot_*.log    # Crossing signals
grep "RSI.*FILTER" logs/bot_*.log     # Blocking signals
```

---

## 🔄 ROLLBACK (If Needed)

### Disable RSI filter:

```bash
# Add to docker-compose.yaml command:
--no-rsi-reversal-filter
```

### Disable DCA:

```bash
# Remove from docker-compose.yaml:
--dca-mode
```

---

*Quick reference guide - November 22, 2025*
*For full details, see BOT_IMPROVEMENTS_SUMMARY.md*
