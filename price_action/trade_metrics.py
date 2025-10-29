"""
Trade Metrics Calculator

Calculates performance metrics for completed trades:
- MFE (Maximum Favorable Excursion) - best profit reached during trade
- MAE (Maximum Adverse Excursion) - worst drawdown during trade
- Trading fees (entry + exit + partial TP)
- Exit reason determination

SOLID: Single Responsibility Principle
- This module is ONLY responsible for trade metric calculations
- Separated from bot logic and logging
"""

from typing import Dict, Optional
import logging

log = logging.getLogger(__name__)


class TradeMetricsCalculator:
    """
    Calculates performance metrics for trades.

    Immutable and stateless - all methods are pure functions.
    """

    # Bybit fee structure (as of 2024)
    MAKER_FEE_PCT = 0.02  # 0.02% for limit orders
    TAKER_FEE_PCT = 0.055  # 0.055% for market orders

    @staticmethod
    def calculate_mfe_mae(
        side: str,
        entry_price: float,
        highest_price: float,
        lowest_price: float
    ) -> Dict[str, float]:
        """
        Calculate Maximum Favorable/Adverse Excursion.

        MFE = Best profit reached during trade (% from entry)
        MAE = Worst drawdown during trade (% from entry)

        Args:
            side: "Long" or "Short"
            entry_price: Entry price
            highest_price: Highest price reached during trade
            lowest_price: Lowest price reached during trade

        Returns:
            Dictionary with 'mfe' and 'mae' in percent
        """
        if entry_price <= 0:
            log.warning("Invalid entry_price for MFE/MAE calculation")
            return {"mfe": 0.0, "mae": 0.0}

        if side == "Long":
            # MFE: highest price vs entry (upside potential)
            mfe = ((highest_price - entry_price) / entry_price) * 100 if highest_price > 0 else 0.0

            # MAE: entry vs lowest price (downside risk)
            mae = ((entry_price - lowest_price) / entry_price) * 100 if lowest_price > 0 else 0.0

        elif side == "Short":
            # MFE: entry vs lowest price (downside potential for short)
            mfe = ((entry_price - lowest_price) / entry_price) * 100 if lowest_price > 0 else 0.0

            # MAE: highest price vs entry (upside risk for short)
            mae = ((highest_price - entry_price) / entry_price) * 100 if highest_price > 0 else 0.0

        else:
            log.warning(f"Invalid side for MFE/MAE: {side}")
            return {"mfe": 0.0, "mae": 0.0}

        return {
            "mfe": round(mfe, 4),
            "mae": round(mae, 4)
        }

    @classmethod
    def calculate_fees(
        cls,
        entry_value_usd: float,
        exit_value_usd: float,
        partial_tp_value_usd: float = 0.0,
        entry_is_limit: bool = False,
        exit_is_limit: bool = False
    ) -> float:
        """
        Calculate total trading fees paid.

        Bybit fee structure:
        - Market orders: 0.055% (taker fee)
        - Limit orders: 0.02% (maker fee)

        Args:
            entry_value_usd: Entry position value in USD
            exit_value_usd: Exit position value in USD
            partial_tp_value_usd: Partial TP value in USD (if taken)
            entry_is_limit: True if entry was limit order
            exit_is_limit: True if exit was limit order

        Returns:
            Total fees paid in USD
        """
        # Entry fee
        entry_fee_pct = cls.MAKER_FEE_PCT if entry_is_limit else cls.TAKER_FEE_PCT
        entry_fee = (entry_value_usd * entry_fee_pct) / 100

        # Exit fee
        exit_fee_pct = cls.MAKER_FEE_PCT if exit_is_limit else cls.TAKER_FEE_PCT
        exit_fee = (exit_value_usd * exit_fee_pct) / 100

        # Partial TP fee (always market order)
        partial_tp_fee = (partial_tp_value_usd * cls.TAKER_FEE_PCT) / 100 if partial_tp_value_usd > 0 else 0.0

        total_fees = entry_fee + exit_fee + partial_tp_fee

        log.debug(f"Fees: entry={entry_fee:.4f}, exit={exit_fee:.4f}, partial_tp={partial_tp_fee:.4f}, total={total_fees:.4f}")

        return round(total_fees, 4)

    @staticmethod
    def determine_exit_reason(
        position_closed: bool,
        sl_hit: bool = False,
        tp_hit: bool = False,
        partial_tp_taken: bool = False,
        dynamic_tp_levels_taken: int = 0,
        timeout: bool = False,
        manual_close: bool = False,
        error_close: bool = False
    ) -> str:
        """
        Determine exit reason based on trade conditions.

        Priority order:
        1. Error/Manual close (highest priority)
        2. Timeout
        3. Dynamic TP
        4. Partial TP
        5. Full TP
        6. SL/TSL
        7. Unknown

        Args:
            position_closed: Position was closed
            sl_hit: Stop loss was hit
            tp_hit: Take profit was hit
            partial_tp_taken: Partial TP was taken
            dynamic_tp_levels_taken: Number of dynamic TP levels hit
            timeout: Timeout occurred
            manual_close: Manual close by user
            error_close: Closed due to error

        Returns:
            Exit reason string
        """
        if not position_closed:
            return "ACTIVE"

        if error_close:
            return "ERROR"

        if manual_close:
            return "MANUAL"

        if timeout:
            return "TIMEOUT"

        # Dynamic TP has higher priority than partial TP
        if dynamic_tp_levels_taken >= 4:
            return "DYNAMIC_TP_LEVEL_4"
        elif dynamic_tp_levels_taken >= 1:
            return f"DYNAMIC_TP_LEVEL_{dynamic_tp_levels_taken}"

        if partial_tp_taken:
            # If partial TP taken, exit could be:
            # - TSL (trailing stop after partial TP)
            # - Remaining TP hit
            if tp_hit:
                return "PARTIAL_TP_THEN_FULL_TP"
            elif sl_hit:
                return "PARTIAL_TP_THEN_TSL"
            else:
                return "PARTIAL_TP"

        if tp_hit:
            return "TP"

        if sl_hit:
            return "SL_TSL"

        return "UNKNOWN"

    @staticmethod
    def calculate_trade_summary(
        side: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        highest_price: float,
        lowest_price: float,
        leverage: int = 1,
        entry_is_limit: bool = False,
        exit_is_limit: bool = False,
        partial_tp_value_usd: float = 0.0,
        duration_seconds: float = 0.0,
        exit_reason: str = "UNKNOWN",
        initial_sl: float = None,
        initial_tp: float = None,
        final_sl: float = None,
        final_tp: float = None
    ) -> Dict[str, any]:
        """
        Calculate complete trade summary with all metrics.

        Args:
            side: "Long" or "Short"
            entry_price: Entry price
            exit_price: Exit price
            quantity: Position quantity
            highest_price: Highest price during trade
            lowest_price: Lowest price during trade
            leverage: Leverage used
            entry_is_limit: Entry was limit order
            exit_is_limit: Exit was limit order
            partial_tp_value_usd: Partial TP value
            duration_seconds: Trade duration
            exit_reason: Exit reason
            initial_sl: Initial stop loss
            initial_tp: Initial take profit
            final_sl: Final stop loss (may differ from initial)
            final_tp: Final take profit

        Returns:
            Complete trade summary dictionary
        """
        # Calculate PnL
        if side == "Long":
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100 if entry_price > 0 else 0
        elif side == "Short":
            pnl_pct = ((entry_price - exit_price) / entry_price) * 100 if entry_price > 0 else 0
        else:
            pnl_pct = 0

        position_value_usd = entry_price * quantity
        pnl_usd = (position_value_usd * pnl_pct) / 100

        # Calculate MFE/MAE
        mfe_mae = TradeMetricsCalculator.calculate_mfe_mae(
            side, entry_price, highest_price, lowest_price
        )

        # Calculate fees
        exit_value_usd = exit_price * quantity
        fees_paid = TradeMetricsCalculator.calculate_fees(
            entry_value_usd=position_value_usd,
            exit_value_usd=exit_value_usd,
            partial_tp_value_usd=partial_tp_value_usd,
            entry_is_limit=entry_is_limit,
            exit_is_limit=exit_is_limit
        )

        # Net PnL (after fees)
        net_pnl_usd = pnl_usd - fees_paid

        return {
            # PnL
            "pnl_usd": round(pnl_usd, 2),
            "pnl_percent": round(pnl_pct, 4),
            "net_pnl_usd": round(net_pnl_usd, 2),

            # Performance metrics
            "max_favorable_excursion": mfe_mae["mfe"],
            "max_adverse_excursion": mfe_mae["mae"],
            "fees_paid": fees_paid,

            # Trade info
            "duration_seconds": int(duration_seconds),
            "exit_reason": exit_reason,

            # Risk management
            "initial_sl": initial_sl,
            "initial_tp": initial_tp,
            "final_sl": final_sl,
            "final_tp": final_tp,

            # Position details
            "entry_price": entry_price,
            "exit_price": exit_price,
            "quantity": quantity,
            "leverage": leverage,
        }
