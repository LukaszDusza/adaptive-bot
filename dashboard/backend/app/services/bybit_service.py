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
