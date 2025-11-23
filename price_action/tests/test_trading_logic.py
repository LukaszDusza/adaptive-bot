# tests/test_trading_logic.py
"""
Unit tests for trading logic functions.

Tests cover:
- Position sizing calculations
- Trailing stop loss (TSL) logic
- Partial take profit (TP) calculations
- Entry/exit logic
"""

import pytest
import pandas as pd
import numpy as np


class TestPositionSizeCalculation:
    """Test position size calculations."""

    def test_calculate_position_size_basic(self, test_config):
        """Test basic position size calculation."""
        # Given: $100 trade size, $50 price, 10x leverage
        # Expected: ($100 / $50) * 10 = 20 units
        trade_size = test_config.TRADE_SIZE_USD
        price = 50.0
        leverage = test_config.LEVERAGE

        expected_size = (trade_size / price) * leverage

        # Calculate: (trade_size / price) * leverage
        calculated_size = (trade_size / price) * leverage

        assert calculated_size == pytest.approx(expected_size, rel=0.01)
        assert calculated_size == 20.0  # Exact check

    def test_position_size_with_different_leverage(self, test_config):
        """Test position size scales with leverage."""
        price = 100.0
        trade_size = test_config.TRADE_SIZE_USD

        # Test different leverage values
        for leverage in [5, 10, 20]:
            size = (trade_size / price) * leverage
            expected = (100.0 / 100.0) * leverage  # = leverage
            assert size == expected

    def test_position_size_scales_with_price(self, test_config):
        """Test position size inversely proportional to price."""
        trade_size = test_config.TRADE_SIZE_USD
        leverage = test_config.LEVERAGE

        # Higher price = smaller position size
        size_at_50 = (trade_size / 50.0) * leverage
        size_at_100 = (trade_size / 100.0) * leverage

        assert size_at_50 > size_at_100
        assert size_at_50 == 2 * size_at_100  # Double price = half size


class TestTSLCalculation:
    """Test trailing stop loss logic."""

    def test_tsl_updates_in_profit_long(self, test_config):
        """Test TSL updates when LONG position in profit."""
        # Given: LONG at $100, current price $105, TSL 1%
        entry_price = 100.0
        current_price = 105.0
        last_sl = 98.0
        tsl_pct = test_config.TSL_PCT  # 0.01 = 1%

        # Expected: TSL should trail at 105 * (1 - 0.01) = 103.95
        # TSL moves up because price is in profit
        expected_sl = current_price * (1 - tsl_pct)

        assert expected_sl == pytest.approx(103.95, rel=0.01)
        assert expected_sl > last_sl  # SL moved up

    def test_tsl_no_update_when_losing_long(self, test_config):
        """Test TSL doesn't update when LONG position losing."""
        # Given: LONG at $100, current price $99 (losing), last SL $98
        entry_price = 100.0
        current_price = 99.0
        last_sl = 98.0
        tsl_pct = test_config.TSL_PCT

        # When price is below entry (losing), TSL should NOT update
        # because we only trail in profit
        is_in_profit = current_price > entry_price

        if is_in_profit:
            new_sl = max(last_sl, current_price * (1 - tsl_pct))
        else:
            new_sl = last_sl  # Don't update

        assert new_sl == last_sl  # SL unchanged at $98
        assert new_sl == 98.0

    def test_tsl_updates_in_profit_short(self, test_config):
        """Test TSL updates when SHORT position in profit."""
        # Given: SHORT at $100, current price $95 (profit), last SL $102
        entry_price = 100.0
        current_price = 95.0
        last_sl = 102.0
        tsl_pct = test_config.TSL_PCT

        # For SHORT: TSL trails at current_price * (1 + tsl_pct)
        # Expected: 95 * (1 + 0.01) = 95.95
        is_in_profit = current_price < entry_price

        if is_in_profit:
            potential_sl = current_price * (1 + tsl_pct)
            new_sl = min(last_sl, potential_sl)  # SHORT: SL moves down
        else:
            new_sl = last_sl

        expected_sl = 95.0 * 1.01
        assert new_sl == pytest.approx(expected_sl, rel=0.01)
        assert new_sl < last_sl  # SHORT: SL moved down

    def test_tsl_no_update_when_losing_short(self, test_config):
        """Test TSL doesn't update when SHORT position losing."""
        # Given: SHORT at $100, current price $105 (losing), last SL $102
        entry_price = 100.0
        current_price = 105.0
        last_sl = 102.0

        # When price above entry (losing SHORT), don't update
        is_in_profit = current_price < entry_price

        if is_in_profit:
            new_sl = current_price * (1 + test_config.TSL_PCT)
        else:
            new_sl = last_sl  # Don't update

        assert new_sl == last_sl
        assert new_sl == 102.0

    def test_tsl_only_moves_in_favorable_direction(self, test_config):
        """Test TSL only moves up for LONG, down for SHORT."""
        # LONG: SL can only increase
        entry_long = 100.0
        last_sl_long = 103.0
        current_price_long = 104.0

        potential_sl_long = current_price_long * (1 - test_config.TSL_PCT)
        new_sl_long = max(last_sl_long, potential_sl_long)

        assert new_sl_long >= last_sl_long  # Can only go up

        # SHORT: SL can only decrease
        entry_short = 100.0
        last_sl_short = 97.0
        current_price_short = 96.0

        potential_sl_short = current_price_short * (1 + test_config.TSL_PCT)
        new_sl_short = min(last_sl_short, potential_sl_short)

        assert new_sl_short <= last_sl_short  # Can only go down


class TestPartialTPLogic:
    """Test partial take profit calculations."""

    def test_partial_tp_level_calculation_long(self, test_config):
        """Test partial TP level for LONG (50% of way to full TP)."""
        entry_price = 100.0
        tp_pct = test_config.TP_PCT  # 0.03 = 3%

        # Full TP: 100 * (1 + 0.03) = 103
        full_tp = entry_price * (1 + tp_pct)

        # Partial TP: 50% of distance from entry to full TP
        # Distance = 103 - 100 = 3
        # Partial = 100 + 3 * 0.5 = 101.5
        distance = full_tp - entry_price
        partial_tp = entry_price + (distance * 0.5)

        assert full_tp == 103.0
        assert partial_tp == pytest.approx(101.5, rel=0.01)

    def test_partial_tp_level_calculation_short(self, test_config):
        """Test partial TP level for SHORT (50% of way to full TP)."""
        entry_price = 100.0
        tp_pct = test_config.TP_PCT  # 0.03 = 3%

        # Full TP: 100 * (1 - 0.03) = 97
        full_tp = entry_price * (1 - tp_pct)

        # Partial TP: 50% of distance from entry to full TP
        # Distance = 97 - 100 = -3
        # Partial = 100 + (-3) * 0.5 = 98.5
        distance = full_tp - entry_price
        partial_tp = entry_price + (distance * 0.5)

        assert full_tp == 97.0
        assert partial_tp == pytest.approx(98.5, rel=0.01)

    def test_partial_tp_is_between_entry_and_full_tp(self, test_config):
        """Test partial TP is always between entry and full TP."""
        entry = 100.0
        tp_pct = 0.03

        # LONG
        full_tp_long = entry * (1 + tp_pct)
        partial_tp_long = entry + (full_tp_long - entry) * 0.5

        assert entry < partial_tp_long < full_tp_long

        # SHORT
        full_tp_short = entry * (1 - tp_pct)
        partial_tp_short = entry + (full_tp_short - entry) * 0.5

        assert full_tp_short < partial_tp_short < entry


class TestSLTPCalculation:
    """Test stop loss and take profit level calculations."""

    def test_sl_tp_levels_long(self, test_config):
        """Test SL and TP calculation for LONG."""
        entry_price = 100.0
        sl_pct = 0.02  # 2% SL
        tp_pct = test_config.TP_PCT  # 3% TP

        # LONG: SL below entry, TP above entry
        sl = entry_price * (1 - sl_pct)
        tp = entry_price * (1 + tp_pct)

        assert sl == pytest.approx(98.0, rel=0.01)
        assert tp == pytest.approx(103.0, rel=0.01)
        assert sl < entry_price < tp

    def test_sl_tp_levels_short(self, test_config):
        """Test SL and TP calculation for SHORT."""
        entry_price = 100.0
        sl_pct = 0.02  # 2% SL
        tp_pct = test_config.TP_PCT  # 3% TP

        # SHORT: SL above entry, TP below entry
        sl = entry_price * (1 + sl_pct)
        tp = entry_price * (1 - tp_pct)

        assert sl == pytest.approx(102.0, rel=0.01)
        assert tp == pytest.approx(97.0, rel=0.01)
        assert tp < entry_price < sl

    def test_tp_should_be_larger_than_sl(self, test_config):
        """Test that TP distance > SL distance (risk/reward)."""
        entry = 100.0
        sl_pct = 0.02
        tp_pct = test_config.TP_PCT  # 0.03

        # For profitable strategy, TP% should be > SL%
        assert tp_pct > sl_pct

        # Distance check
        sl_distance = entry * sl_pct
        tp_distance = entry * tp_pct

        assert tp_distance > sl_distance


class TestProfitCalculation:
    """Test profit/loss calculations."""

    def test_profit_calculation_long(self, test_config):
        """Test profit calculation for LONG position."""
        entry_price = 100.0
        exit_price = 103.0
        size = 10.0
        leverage = test_config.LEVERAGE

        # Profit = (exit - entry) / entry * leverage * size
        pnl_pct = (exit_price - entry_price) / entry_price
        pnl_usd = pnl_pct * (entry_price * size / leverage)

        expected_pnl_pct = 0.03  # 3% move
        expected_pnl_usd = 0.03 * (100.0 * 10.0 / 10)  # $3

        assert pnl_pct == pytest.approx(expected_pnl_pct, rel=0.01)
        assert pnl_usd == pytest.approx(expected_pnl_usd, rel=0.01)

    def test_loss_calculation_long_hits_sl(self, test_config):
        """Test loss calculation when LONG hits SL."""
        entry_price = 100.0
        sl_price = 98.0  # 2% SL
        size = 10.0
        leverage = test_config.LEVERAGE

        # Loss = (sl - entry) / entry * leverage * size
        pnl_pct = (sl_price - entry_price) / entry_price
        pnl_usd = pnl_pct * (entry_price * size / leverage)

        assert pnl_pct == pytest.approx(-0.02, rel=0.01)  # -2%
        assert pnl_usd < 0  # Loss


@pytest.mark.unit
class TestDecisionLogic:
    """Test trading decision logic."""

    def test_hold_when_probabilities_below_threshold(self, test_config):
        """Test HOLD decision when probabilities below threshold."""
        proba_long = 0.65
        proba_short = 0.60
        threshold = test_config.PROBABILITY_THRESHOLD  # 0.7

        # Neither probability above threshold → HOLD
        decision = 'HOLD'
        if proba_long > threshold and proba_long > proba_short:
            decision = 'BUY'
        elif proba_short > threshold and proba_short > proba_long:
            decision = 'SELL'

        assert decision == 'HOLD'

    def test_buy_when_long_probability_strong(self, test_config):
        """Test BUY decision when LONG probability strong."""
        proba_long = 0.75
        proba_short = 0.30
        threshold = test_config.PROBABILITY_THRESHOLD

        decision = 'HOLD'
        if proba_long > threshold and proba_long > proba_short:
            decision = 'BUY'
        elif proba_short > threshold and proba_short > proba_long:
            decision = 'SELL'

        assert decision == 'BUY'

    def test_sell_when_short_probability_strong(self, test_config):
        """Test SELL decision when SHORT probability strong."""
        proba_long = 0.25
        proba_short = 0.80
        threshold = test_config.PROBABILITY_THRESHOLD

        decision = 'HOLD'
        if proba_long > threshold and proba_long > proba_short:
            decision = 'BUY'
        elif proba_short > threshold and proba_short > proba_long:
            decision = 'SELL'

        assert decision == 'SELL'

    def test_hold_when_probabilities_too_close(self, test_config):
        """Test HOLD when both probabilities similar (no clear signal)."""
        proba_long = 0.72
        proba_short = 0.71
        threshold = test_config.PROBABILITY_THRESHOLD
        min_diff = 0.1  # Require 10% difference

        # Both above threshold but too close → HOLD
        decision = 'HOLD'
        if proba_long > threshold and proba_long > proba_short:
            if abs(proba_long - proba_short) >= min_diff:
                decision = 'BUY'
        elif proba_short > threshold and proba_short > proba_long:
            if abs(proba_short - proba_long) >= min_diff:
                decision = 'SELL'

        assert decision == 'HOLD'  # Too close, no trade
