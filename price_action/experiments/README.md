# Experiments Directory

This directory contains **ad-hoc analysis scripts** and experimental tools that are NOT part of the core production pipeline.

## Purpose

These scripts are for:
- One-time analysis and investigation
- Model comparison and evaluation
- Historical data analysis
- Debugging and troubleshooting
- Research and experimentation

**IMPORTANT:** These files are standalone utilities and are NOT imported or used by the main pipeline (bot.py, model_pipeline.py, etc.).

## Files

### Model Analysis
- `analyze_all_models.py` - Analyze all trained models
- `analyze_live_trades.py` - Analyze live trading performance
- `analyze_recent_trades.py` - Recent trades analysis
- `analyze_detailed_trades.py` - Detailed trade breakdown
- `analyze_rsi_signals.py` - RSI signal analysis
- `optimize_all_models.py` - Batch optimization for multiple models

### Comparisons
- `compare_models.py` - Compare different model versions
- `compare_backtest_vs_live.py` - Compare backtest vs live results
- `compare_rsi_importance.py` - Compare RSI feature importance
- `analyze_live_vs_backtest.py` - Live vs backtest performance gap analysis

### Deep Analysis
- `deep_loss_analysis.py` - Deep dive into losing trades
- `deep_feature_analysis.py` - Feature importance deep dive
- `diagnose_friday_losses.py` - Day-of-week loss patterns

### Debugging
- `debug_candle_timestamps.py` - Debug candle timing issues
- `check_trading_history.py` - Verify trading history
- `check_trend_direction.py` - Trend direction verification

### Data Tools
- `fetch_bybit_history.py` - Fetch historical data from Bybit
- `migrate_trades.py` - Trade data migration utility

### Historical Analysis
- `backtest_nov_2025.py` - November 2025 backtest
- `analyze_bybit_today.py` - Today's Bybit market analysis

### Subdirectories

#### `analiza/`
Contains 66 analysis scripts from various historical investigations.
These are organized by analysis date/topic and include trade enrichment tools,
worst trade analysis, ticker-specific studies, and market event analysis.

## Usage

Run these scripts manually as needed:

```bash
cd price_action/experiments

# Example: Analyze all models
python analyze_all_models.py

# Example: Compare backtest vs live
python compare_backtest_vs_live.py
```

## Notes

- These scripts may have hardcoded parameters (dates, tickers, etc.)
- Not all scripts are maintained or guaranteed to work with latest code
- Use as reference or starting point for new analysis
- Consider adding new analysis scripts here to keep main directory clean
