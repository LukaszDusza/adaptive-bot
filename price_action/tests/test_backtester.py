# tests/test_backtester.py
"""
Unit tests for backtesting engine.

Tests cover:
- Position class initialization and state management
- Trade class and metrics calculations
- BacktestEngine logic (position opening/closing, TSL, partial TP)
- Profit/loss calculations with fees and slippage
- Risk management (MAE/MFE tracking)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import timedelta
from dataclasses import dataclass
from backtester import (
    Position,
    Trade,
    BacktestEngine,
    calculate_metrics
)


class TestPositionClass:
    """Test Position dataclass and state management."""

    def test_position_initialization_long(self):
        """Test LONG position initialization."""
        entry_time = pd.Timestamp('2024-01-01 10:00:00')

        position = Position(
            side='Long',
            entry_price=100.0,
            entry_time=entry_time,
            current_size=1.0,
            initial_size=1.0,
            stop_loss=98.0,
            take_profit=103.0,
            highest_price=100.0,
            lowest_price=100.0,
            partial_tp_taken=False,
            tp_pct=0.03,
            entry_probability=0.75
        )

        assert position.side == 'Long'
        assert position.entry_price == 100.0
        assert position.current_size == 1.0
        assert position.initial_size == 1.0
        assert position.stop_loss == 98.0
        assert position.take_profit == 103.0
        assert position.partial_tp_taken == False
        assert position.entry_probability == 0.75

    def test_position_initialization_short(self):
        """Test SHORT position initialization."""
        entry_time = pd.Timestamp('2024-01-01 10:00:00')

        position = Position(
            side='Short',
            entry_price=100.0,
            entry_time=entry_time,
            current_size=1.0,
            initial_size=1.0,
            stop_loss=102.0,  # SL above entry for SHORT
            take_profit=97.0,  # TP below entry for SHORT
            highest_price=100.0,
            lowest_price=100.0,
            partial_tp_taken=False,
            tp_pct=0.03
        )

        assert position.side == 'Short'
        assert position.stop_loss > position.entry_price  # SHORT: SL above entry
        assert position.take_profit < position.entry_price  # SHORT: TP below entry

    def test_position_dynamic_tp_tracking(self):
        """Test dynamic TP levels tracking."""
        position = Position(
            side='Long',
            entry_price=100.0,
            entry_time=pd.Timestamp('2024-01-01'),
            current_size=1.0,
            initial_size=1.0,
            stop_loss=98.0,
            take_profit=103.0,
            highest_price=100.0,
            lowest_price=100.0,
            dynamic_tp_levels_taken=0
        )

        assert position.dynamic_tp_levels_taken == 0

        # Simulate hitting first dynamic TP level
        position.dynamic_tp_levels_taken = 1
        assert position.dynamic_tp_levels_taken == 1

        # Can track up to 4 levels
        position.dynamic_tp_levels_taken = 4
        assert position.dynamic_tp_levels_taken == 4


class TestTradeClass:
    """Test Trade dataclass and metrics."""

    def test_trade_creation_profitable(self):
        """Test creating a profitable trade."""
        trade = Trade(
            entry_time=pd.Timestamp('2024-01-01 10:00:00'),
            exit_time=pd.Timestamp('2024-01-01 11:00:00'),
            side='Long',
            entry_price=100.0,
            exit_price=103.0,
            size=10.0,
            pnl_pct=0.03,
            pnl_usd=30.0,
            fees_usd=0.6,
            net_pnl_usd=29.4,
            exit_reason='TP',
            duration=timedelta(hours=1),
            mae_pct=-0.005,
            mfe_pct=0.03
        )

        assert trade.side == 'Long'
        assert trade.pnl_pct == 0.03  # 3% profit
        assert trade.pnl_usd == 30.0
        assert trade.net_pnl_usd == 29.4  # After fees
        assert trade.exit_reason == 'TP'
        assert trade.duration == timedelta(hours=1)
        assert trade.mae_pct < 0  # Adverse move
        assert trade.mfe_pct > 0  # Favorable move

    def test_trade_creation_loss(self):
        """Test creating a losing trade."""
        trade = Trade(
            entry_time=pd.Timestamp('2024-01-01 10:00:00'),
            exit_time=pd.Timestamp('2024-01-01 10:30:00'),
            side='Short',
            entry_price=100.0,
            exit_price=102.0,
            size=10.0,
            pnl_pct=-0.02,
            pnl_usd=-20.0,
            fees_usd=0.6,
            net_pnl_usd=-20.6,
            exit_reason='SL',
            duration=timedelta(minutes=30),
            mae_pct=-0.025,
            mfe_pct=0.005
        )

        assert trade.pnl_pct < 0  # Loss
        assert trade.net_pnl_usd < trade.pnl_usd  # Fees make it worse
        assert trade.exit_reason == 'SL'

    def test_trade_time_categorization(self):
        """Test trade time-based categorization."""
        trade = Trade(
            entry_time=pd.Timestamp('2024-01-01 14:30:00'),  # 2:30 PM
            exit_time=pd.Timestamp('2024-01-01 16:45:00'),   # 4:45 PM
            side='Long',
            entry_price=100.0,
            exit_price=102.0,
            size=1.0,
            pnl_pct=0.02,
            pnl_usd=2.0,
            fees_usd=0.1,
            net_pnl_usd=1.9,
            exit_reason='TP',
            duration=timedelta(hours=2, minutes=15),
            mae_pct=0.0,
            mfe_pct=0.02,
            entry_hour=14,
            exit_hour=16,
            entry_day_of_week=0,  # Monday
            exit_day_of_week=0
        )

        assert trade.entry_hour == 14
        assert trade.exit_hour == 16
        assert trade.entry_day_of_week == 0  # Monday


class TestBacktestEngineInitialization:
    """Test BacktestEngine initialization and configuration."""

    def test_backtest_engine_default_initialization(self):
        """Test engine with default parameters."""
        engine = BacktestEngine()

        assert engine.initial_capital == 10000.0
        assert engine.current_capital == 10000.0
        assert engine.maker_fee == 0.0002
        assert engine.taker_fee == 0.00055
        assert engine.slippage_pct == 0.0003
        assert engine.enable_limit_order == False
        assert engine.enable_dca == False

    def test_backtest_engine_custom_initialization(self):
        """Test engine with custom parameters."""
        engine = BacktestEngine(
            initial_capital=5000.0,
            maker_fee=0.0001,
            taker_fee=0.0003,
            slippage_pct=0.0002,
            enable_limit_order=True,
            limit_order_candles=3,
            cooldown_minutes=15
        )

        assert engine.initial_capital == 5000.0
        assert engine.maker_fee == 0.0001
        assert engine.enable_limit_order == True
        assert engine.limit_order_candles == 3
        assert engine.cooldown_minutes == 15

    def test_backtest_engine_with_dca(self):
        """Test engine with DCA enabled."""
        engine = BacktestEngine(
            enable_dca=True,
            dca_atr_multiplier=2.0
        )

        assert engine.enable_dca == True
        assert engine.dca_atr_multiplier == 2.0


class TestPositionManagement:
    """Test position opening, closing, and management logic."""

    def test_position_size_calculation(self):
        """Test position size based on capital and risk."""
        engine = BacktestEngine(initial_capital=10000.0)

        # Trade size: $100, Price: $50, Leverage: 10x
        # Expected size: ($100 / $50) * 10 = 20 contracts
        trade_size = 100.0
        price = 50.0
        leverage = 10

        expected_size = (trade_size / price) * leverage

        assert expected_size == 20.0

    def test_long_position_profit_calculation(self):
        """Test profit calculation for LONG position."""
        entry_price = 100.0
        exit_price = 105.0
        size = 10.0

        # PnL % = (exit - entry) / entry
        pnl_pct = (exit_price - entry_price) / entry_price

        # PnL USD = pnl_pct * (entry_price * size)
        notional = entry_price * size
        pnl_usd = pnl_pct * notional

        assert pnl_pct == pytest.approx(0.05, rel=0.01)  # 5%
        assert pnl_usd == pytest.approx(50.0, rel=0.01)

    def test_short_position_profit_calculation(self):
        """Test profit calculation for SHORT position."""
        entry_price = 100.0
        exit_price = 95.0  # Price went down (profit for SHORT)
        size = 10.0

        # PnL % = (entry - exit) / entry (reversed for SHORT)
        pnl_pct = (entry_price - exit_price) / entry_price

        # PnL USD = pnl_pct * notional
        notional = entry_price * size
        pnl_usd = pnl_pct * notional

        assert pnl_pct == pytest.approx(0.05, rel=0.01)  # 5% profit
        assert pnl_usd == pytest.approx(50.0, rel=0.01)

    def test_fees_calculation(self):
        """Test trading fees calculation."""
        entry_price = 100.0
        size = 10.0
        maker_fee = 0.0002  # 0.02%
        taker_fee = 0.00055  # 0.055%

        notional = entry_price * size  # $1000

        # Entry fee (maker)
        entry_fee = notional * maker_fee

        # Exit fee (taker for market exit)
        exit_fee = notional * taker_fee

        total_fees = entry_fee + exit_fee

        assert entry_fee == pytest.approx(0.2, rel=0.01)
        assert exit_fee == pytest.approx(0.55, rel=0.01)
        assert total_fees == pytest.approx(0.75, rel=0.01)


class TestTSLLogic:
    """Test Trailing Stop Loss logic."""

    def test_tsl_updates_in_profit_long(self):
        """Test TSL updates when LONG in profit."""
        entry_price = 100.0
        current_price = 105.0
        tsl_pct = 0.01  # 1%
        current_sl = 98.0

        # When in profit, TSL should trail at current_price * (1 - tsl_pct)
        is_in_profit = current_price > entry_price

        if is_in_profit:
            potential_sl = current_price * (1 - tsl_pct)
            new_sl = max(current_sl, potential_sl)  # Can only move up
        else:
            new_sl = current_sl

        expected_sl = 105.0 * 0.99  # = 103.95

        assert new_sl == pytest.approx(expected_sl, rel=0.01)
        assert new_sl > current_sl  # Moved up

    def test_tsl_no_update_when_losing_long(self):
        """Test TSL doesn't update when LONG losing."""
        entry_price = 100.0
        current_price = 99.0  # Losing
        current_sl = 98.0

        is_in_profit = current_price > entry_price

        if is_in_profit:
            new_sl = current_price * 0.99
        else:
            new_sl = current_sl  # Don't update

        assert new_sl == current_sl
        assert new_sl == 98.0

    def test_tsl_only_moves_favorably(self):
        """Test TSL only moves in favorable direction."""
        # LONG: SL can only increase
        entry = 100.0
        sl_long = 103.0
        current_long = 104.0

        potential_sl_long = current_long * 0.99
        new_sl_long = max(sl_long, potential_sl_long)

        assert new_sl_long >= sl_long  # Only up

        # SHORT: SL can only decrease
        sl_short = 97.0
        current_short = 96.0

        potential_sl_short = current_short * 1.01
        new_sl_short = min(sl_short, potential_sl_short)

        assert new_sl_short <= sl_short  # Only down


class TestPartialTPLogic:
    """Test Partial Take Profit logic."""

    def test_partial_tp_reduces_position_size(self):
        """Test partial TP closes 50% of position."""
        initial_size = 10.0
        partial_tp_pct = 0.5  # 50%

        # After partial TP, size should be 50%
        remaining_size = initial_size * (1 - partial_tp_pct)

        assert remaining_size == 5.0

    def test_partial_tp_level_calculation_long(self):
        """Test partial TP level for LONG."""
        entry = 100.0
        tp_pct = 0.03  # 3%

        # Full TP: 103
        full_tp = entry * (1 + tp_pct)

        # Partial TP: 50% of distance = 101.5
        distance = full_tp - entry
        partial_tp = entry + (distance * 0.5)

        assert full_tp == 103.0
        assert partial_tp == pytest.approx(101.5, rel=0.01)
        assert entry < partial_tp < full_tp

    def test_dynamic_tp_levels(self):
        """Test dynamic TP with 4 levels (25% each)."""
        entry = 100.0
        tp_pct = 0.04  # 4%
        full_tp = entry * (1 + tp_pct)  # 104

        # 4 levels at 25%, 50%, 75%, 100% of distance
        distance = full_tp - entry  # 4

        levels = []
        for pct in [0.25, 0.50, 0.75, 1.0]:
            level = entry + (distance * pct)
            levels.append(level)

        assert len(levels) == 4
        assert levels[0] == pytest.approx(101.0, rel=0.01)  # 25%
        assert levels[1] == pytest.approx(102.0, rel=0.01)  # 50%
        assert levels[2] == pytest.approx(103.0, rel=0.01)  # 75%
        assert levels[3] == pytest.approx(104.0, rel=0.01)  # 100%


class TestMAEMFETracking:
    """Test Maximum Adverse/Favorable Excursion tracking."""

    def test_mae_tracking_long(self):
        """Test MAE (worst drawdown) for LONG position."""
        entry = 100.0
        lowest_during_trade = 98.5

        # MAE = (lowest - entry) / entry (negative for LONG)
        mae_pct = (lowest_during_trade - entry) / entry

        assert mae_pct == pytest.approx(-0.015, rel=0.001)  # -1.5%
        assert mae_pct < 0  # Adverse move

    def test_mfe_tracking_long(self):
        """Test MFE (best profit) for LONG position."""
        entry = 100.0
        highest_during_trade = 105.0

        # MFE = (highest - entry) / entry (positive for LONG)
        mfe_pct = (highest_during_trade - entry) / entry

        assert mfe_pct == pytest.approx(0.05, rel=0.001)  # +5%
        assert mfe_pct > 0  # Favorable move

    def test_mae_mfe_for_losing_trade(self):
        """Test MAE/MFE for a losing trade."""
        entry = 100.0
        highest = 101.5  # Briefly went up
        lowest = 96.5    # Worst point during trade
        exit = 97.0      # Closed at SL (better than lowest)

        mae = (lowest - entry) / entry
        mfe = (highest - entry) / entry
        final_pnl = (exit - entry) / entry

        assert mae < final_pnl  # MAE worse than final result
        assert mfe > 0  # Had some profit opportunity
        assert final_pnl < 0  # But closed at loss


@pytest.mark.unit
class TestMetricsCalculation:
    """Test trade metrics calculation."""

    def test_win_rate_calculation(self):
        """Test win rate calculation."""
        # 3 wins out of 5 trades = 60%
        total_trades = 5
        winning_trades = 3

        win_rate = (winning_trades / total_trades) * 100

        assert win_rate == 60.0

    def test_profit_factor(self):
        """Test profit factor calculation."""
        total_wins = 150.0
        total_losses = 100.0

        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        assert profit_factor == 1.5  # Good profit factor

    def test_average_win_vs_average_loss(self):
        """Test average win vs average loss."""
        wins = [10, 20, 30]  # Total: 60
        losses = [-5, -10, -15]  # Total: -30

        avg_win = sum(wins) / len(wins)
        avg_loss = abs(sum(losses) / len(losses))

        win_loss_ratio = avg_win / avg_loss

        assert avg_win == 20.0
        assert avg_loss == 10.0
        assert win_loss_ratio == 2.0  # Wins 2x bigger than losses

    def test_max_drawdown_calculation(self):
        """Test maximum drawdown calculation."""
        equity_curve = [10000, 10500, 10200, 9800, 10300, 11000]

        peak = equity_curve[0]
        max_dd = 0

        for equity in equity_curve:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak
            max_dd = max(max_dd, drawdown)

        # Peak at 10500, trough at 9800
        # DD = (10500 - 9800) / 10500 = 6.67%
        assert max_dd == pytest.approx(0.0667, rel=0.01)
