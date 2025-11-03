"""
Bybit service for fetching pending orders
"""
import sys
import os
import logging
from typing import List, Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Add price_action directory to Python path
# In Docker: /app/price_action
# Local dev: ../../../../price_action
if os.path.exists("/app/price_action"):
    PRICE_ACTION_DIR = "/app/price_action"
else:
    PRICE_ACTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../price_action"))

if PRICE_ACTION_DIR not in sys.path:
    sys.path.insert(0, PRICE_ACTION_DIR)
    logger.info(f"Added price_action to sys.path: {PRICE_ACTION_DIR}")

try:
    from bybit_adapter import BybitAdapter
    BYBIT_ADAPTER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"BybitAdapter not available: {e}")
    BYBIT_ADAPTER_AVAILABLE = False


class BybitService:
    """Service for interacting with Bybit API"""

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize Bybit service.

        Args:
            api_key: Bybit API key (if None, will try to load from env)
            api_secret: Bybit API secret (if None, will try to load from env)
            base_url: Bybit base URL (if None, uses production)
        """
        if not BYBIT_ADAPTER_AVAILABLE:
            raise RuntimeError("BybitAdapter not available - check import path")

        # Try to load from environment if not provided
        self.api_key = api_key or os.getenv("BYBIT_API_KEY")
        self.api_secret = api_secret or os.getenv("BYBIT_API_SECRET")
        self.base_url = base_url or os.getenv("BYBIT_BASE_URL", "https://api.bybit.com")

        if not self.api_key or not self.api_secret:
            logger.warning("Bybit API credentials not configured - pending orders feature will be limited")
            self.adapter = None
        else:
            try:
                self.adapter = BybitAdapter(
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                    base_url=self.base_url if self.base_url != "https://api.bybit.com" else None
                )
                logger.info("✓ BybitAdapter initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize BybitAdapter: {e}")
                self.adapter = None

    def get_open_orders(self, ticker: str) -> List[Dict]:
        """
        Get open orders for a ticker from Bybit.

        Args:
            ticker: Trading pair (e.g., SOLUSDT)

        Returns:
            List of open order dictionaries
        """
        if not self.adapter:
            logger.warning("BybitAdapter not initialized - cannot fetch orders")
            return []

        try:
            orders = self.adapter.get_open_orders(ticker)
            logger.info(f"✓ Retrieved {len(orders)} open orders for {ticker}")
            return orders
        except Exception as e:
            logger.error(f"Failed to get open orders for {ticker}: {e}")
            return []

    def get_multiple_open_orders(self, tickers: List[str]) -> Dict[str, List[Dict]]:
        """
        Get open orders for multiple tickers.

        Args:
            tickers: List of trading pairs

        Returns:
            Dictionary mapping ticker to list of orders
        """
        results = {}
        for ticker in tickers:
            results[ticker] = self.get_open_orders(ticker)
        return results

    def is_available(self) -> bool:
        """Check if Bybit service is available and configured."""
        return self.adapter is not None

    def cancel_order(self, ticker: str, order_id: str) -> bool:
        """
        Cancel a specific order.

        Args:
            ticker: Trading pair (e.g., SOLUSDT)
            order_id: Order ID to cancel

        Returns:
            True if successful, False otherwise
        """
        if not self.adapter:
            logger.warning("BybitAdapter not initialized - cannot cancel order")
            return False

        try:
            self.adapter.cancel_order(ticker, order_id)
            logger.info(f"✓ Cancelled order {order_id} for {ticker}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id} for {ticker}: {e}")
            return False

    def cancel_all_orders(self, ticker: str) -> bool:
        """
        Cancel all open orders for a ticker.

        Args:
            ticker: Trading pair (e.g., SOLUSDT)

        Returns:
            True if successful, False otherwise
        """
        if not self.adapter:
            logger.warning("BybitAdapter not initialized - cannot cancel orders")
            return False

        try:
            self.adapter.cancel_all_orders(ticker)
            logger.info(f"✓ Cancelled all orders for {ticker}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel all orders for {ticker}: {e}")
            return False

    def get_current_price(self, ticker: str) -> float:
        """
        Get current price for a ticker.

        Args:
            ticker: Trading pair (e.g., SOLUSDT)

        Returns:
            Current price or 0.0 on error
        """
        if not self.adapter:
            logger.warning("BybitAdapter not initialized - cannot get price")
            return 0.0

        try:
            price = self.adapter.latest_price(ticker)
            return price
        except Exception as e:
            logger.error(f"Failed to get price for {ticker}: {e}")
            return 0.0

    def get_wallet_balance(self) -> float:
        """
        Get current wallet equity (USDT) from Bybit.

        Returns:
            Current equity in USDT or 0.0 on error
        """
        if not self.adapter:
            logger.warning("BybitAdapter not initialized - cannot get balance")
            return 0.0

        try:
            balance = self.adapter.get_balance(use_available=False)
            logger.info(f"✓ Current wallet equity: ${balance:.2f}")
            return balance
        except Exception as e:
            logger.error(f"Failed to get wallet balance: {e}")
            return 0.0

    def get_closed_pnl_history(self, symbol: str = None, limit: int = 100) -> list:
        """
        Get closed P&L history from Bybit.

        Args:
            symbol: Trading pair (e.g., SOLUSDT). If None, gets all symbols.
            limit: Max records to return (default 100, max 100)

        Returns:
            List of closed P&L records with timestamp and cumulative PnL
        """
        if not self.adapter:
            logger.warning("BybitAdapter not initialized - cannot get P&L history")
            return []

        try:
            # Get closed P&L from Bybit
            pnl_records = self.adapter.get_closed_pnl_history(
                symbol=symbol,
                limit=limit
            )

            logger.info(f"✓ Retrieved {len(pnl_records)} closed P&L records from Bybit")
            return pnl_records
        except Exception as e:
            logger.error(f"Failed to get closed P&L history: {e}")
            return []

    def build_equity_curve_from_bybit(self, limit: int = 100) -> list:
        """
        Build equity curve from real Bybit data (closed P&L history).

        This is the SOURCE OF TRUTH - gets actual trading results from the exchange.

        Args:
            limit: Number of historical records to fetch (max 100 per API call)

        Returns:
            List of dicts with 'timestamp' and 'equity' keys, sorted chronologically
        """
        if not self.adapter:
            logger.warning("BybitAdapter not initialized - cannot build equity curve")
            return []

        try:
            # Get current balance
            current_equity = self.get_wallet_balance()

            # Get closed P&L history
            pnl_records = self.get_closed_pnl_history(limit=limit)

            if not pnl_records:
                logger.warning("No P&L history available from Bybit")
                return []

            # Sort by timestamp ascending (oldest first)
            pnl_records.sort(key=lambda x: int(x.get('createdTime', 0)))

            # Build equity curve by working backwards from current equity
            equity_curve = []
            running_equity = current_equity

            # Add current point
            from datetime import datetime
            equity_curve.append({
                'timestamp': datetime.now().isoformat(),
                'equity': running_equity,
                'cumulative_pnl': 0.0  # Current is baseline
            })

            # Work backwards through closed trades
            cumulative_pnl = 0.0
            for record in reversed(pnl_records):
                closed_pnl = float(record.get('closedPnl', 0))
                cumulative_pnl -= closed_pnl  # Subtract because we're going backwards

                # Equity at this point in time = current equity - cumulative PnL since then
                historical_equity = current_equity - cumulative_pnl

                timestamp_ms = int(record.get('createdTime', 0))
                timestamp = datetime.fromtimestamp(timestamp_ms / 1000).isoformat()

                equity_curve.append({
                    'timestamp': timestamp,
                    'equity': historical_equity,
                    'cumulative_pnl': cumulative_pnl,
                    'symbol': record.get('symbol', ''),
                    'side': record.get('side', ''),
                    'pnl': closed_pnl
                })

            # Reverse to get chronological order (oldest to newest)
            equity_curve.reverse()

            logger.info(f"✓ Built equity curve with {len(equity_curve)} points from Bybit API")
            logger.info(f"  Starting equity: ${equity_curve[0]['equity']:.2f}")
            logger.info(f"  Current equity: ${equity_curve[-1]['equity']:.2f}")
            logger.info(f"  Total PnL: ${equity_curve[-1]['equity'] - equity_curve[0]['equity']:.2f}")

            return equity_curve

        except Exception as e:
            logger.error(f"Failed to build equity curve from Bybit: {e}")
            return []
