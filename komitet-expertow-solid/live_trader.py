# services/bybit_service.py
"""
Bybit API service for live trading integration.
Handles connection, position management, and order execution.
"""

import pandas as pd
import time
import logging
import os
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Production Bybit API imports
try:
    from pybit.unified_trading import HTTP

    PYBIT_AVAILABLE = True
except ImportError:
    print("Warning: pybit not installed. Install with: pip install pybit")
    PYBIT_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BybitService:
    """
    Service for Bybit API integration.
    Handles live data fetching and position management.
    """

    def __init__(self, mode='paper', api_key: Optional[str] = None, api_secret: Optional[str] = None,
                 testnet: bool = True):
        """
        Initialize Bybit service.

        Args:
            mode: 'paper' for paper trading, 'live' for real trading
            api_key: Bybit API key (optional, will use .env if not provided)
            api_secret: Bybit API secret (optional, will use .env if not provided)
            testnet: Use testnet environment (default True for safety)
        """
        self.mode = mode
        # Load API credentials from .env file if not provided
        self.api_key = api_key or os.getenv('BYBIT_API_KEY')
        self.api_secret = api_secret or os.getenv('BYBIT_API_SECRET')
        self.testnet = testnet
        self.client: Optional[HTTP] = None

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 10 requests per second limit

        # Initialize API client
        self._init_client()

    def _init_client(self):
        """Initialize Bybit API client."""
        if self.mode == 'paper':
            logger.info("Bybit service initialized in PAPER TRADING mode")
            self.client = None
        else:
            if not PYBIT_AVAILABLE:
                raise ImportError("pybit library is required for live trading. Install with: pip install pybit")

            if not self.api_key or not self.api_secret:
                raise ValueError("API key and secret are required for live trading")

            try:
                self.client = HTTP(
                    testnet=self.testnet,
                    api_key=self.api_key,
                    api_secret=self.api_secret
                )
                logger.info(f"Bybit service initialized in LIVE TRADING mode (testnet: {self.testnet})")

                # Test connection
                server_time = self.get_server_time()
                if server_time:
                    logger.info(
                        f"Bybit API connection successful. Server time: {datetime.fromtimestamp(server_time / 1000)}")
                else:
                    logger.warning("Bybit API connection test failed")

            except Exception as e:
                logger.error(f"Failed to initialize Bybit API client: {e}")
                raise

    def _rate_limit(self):
        """Apply rate limiting between API requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()

    def fetch_recent_candles(self, symbol: str, interval_minutes: int = 5, limit: int = 500) -> pd.DataFrame:
        """
        Fetch recent candlestick data from Bybit.

        Args:
            symbol: Trading pair symbol (e.g., 'ETHUSDT')
            interval_minutes: Candle interval in minutes (default 5)
            limit: Number of candles to fetch (default 500)

        Returns:
            DataFrame with OHLCV data, indexed by timestamp.
        """
        if self.mode == 'paper' or not self.client:
            return self._fetch_mock_data(symbol, interval_minutes, limit)

        try:
            self._rate_limit()

            interval_map = {
                1: '1', 3: '3', 5: '5', 15: '15', 30: '30', 60: '60',
                120: '120', 240: '240', 360: '360', 720: '720', 1440: 'D'
            }
            interval_str = interval_map.get(interval_minutes, '5')

            response = self.client.get_kline(
                category="linear", symbol=symbol, interval=interval_str, limit=limit
            )

            if response['retCode'] == 0 and response['result']['list']:
                data = response['result']['list']
                df = pd.DataFrame(data, columns=['start_time', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
                df['timestamp'] = pd.to_datetime(df['start_time'].astype(int), unit='ms')
                df.set_index('timestamp', inplace=True)
                df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
                df = df.sort_index()
                logger.info(f"Fetched {len(df)} candles for {symbol} ({interval_minutes}m)")
                return df
            else:
                logger.error(f"Failed to fetch candles: {response.get('retMsg', 'Unknown error')}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Error fetching candles for {symbol}: {e}")
            return pd.DataFrame()

    def _fetch_mock_data(self, symbol: str, interval_minutes: int, limit: int) -> pd.DataFrame:
        """Fetch mock data for paper trading (placeholder)."""
        logger.info(f"Fetching {limit} mock candles of {symbol} {interval_minutes}m for paper trading")
        # TODO: Implement mock data or fetch from alternative source
        return pd.DataFrame()

    def get_current_positions(self) -> List[Dict[str, Any]]:
        """
        Get current open positions.

        Returns:
            List of active position dictionaries.
        """
        if self.mode == 'paper' or not self.client:
            return []

        try:
            self._rate_limit()
            response = self.client.get_positions(category="linear", settleCoin="USDT")

            if response['retCode'] == 0:
                positions = response['result']['list']
                active_positions = [
                    {
                        'symbol': pos['symbol'],
                        'side': pos['side'],
                        'size': float(pos.get('size', 0)),
                        'entry_price': float(pos.get('avgPrice', 0)),
                        'unrealised_pnl': float(pos.get('unrealisedPnl', 0)),
                        'leverage': float(pos.get('leverage', 1)),
                        'position_value': float(pos.get('positionValue', 0)),
                        'raw_data': pos
                    }
                    for pos in positions if float(pos.get('size', 0)) > 0
                ]
                logger.info(f"Found {len(active_positions)} active positions")
                return active_positions
            else:
                logger.error(f"Failed to get positions: {response.get('retMsg', 'Unknown error')}")
                return []

        except Exception as e:
            logger.error(f"Error getting current positions: {e}")
            return []

    def place_order(self, symbol: str, side: str, order_type: str, qty: float,
                    price: Optional[float] = None, stop_loss: Optional[float] = None,
                    take_profit: Optional[float] = None, reduce_only: bool = False) -> Dict[str, Any]:
        """
        Place a new order.

        Args:
            symbol: Trading pair symbol
            side: 'Buy' or 'Sell'
            order_type: 'Market' or 'Limit'
            qty: Order quantity
            price: Order price (for limit orders)
            stop_loss: Stop loss price
            take_profit: Take profit price
            reduce_only: True if the order should only reduce a position

        Returns:
            Order response dictionary.
        """
        if self.mode == 'paper' or not self.client:
            return self._place_paper_order(symbol, side, order_type, qty, price, stop_loss, take_profit)

        try:
            self._rate_limit()

            order_params = {
                "category": "linear", "symbol": symbol, "side": side,
                "orderType": order_type, "qty": str(qty),
            }

            if order_type == "Limit":
                if price is None:
                    raise ValueError("Price is required for Limit orders")
                order_params["price"] = str(price)
                order_params["timeInForce"] = "GTC"
            else:  # Market order
                order_params["timeInForce"] = "IOC"

            if stop_loss: order_params["stopLoss"] = str(stop_loss)
            if take_profit: order_params["takeProfit"] = str(take_profit)
            if reduce_only: order_params["reduceOnly"] = True

            response = self.client.place_order(**order_params)

            if response['retCode'] == 0:
                order_id = response['result']['orderId']
                logger.info(f"Order placed successfully: {side} {qty} {symbol} (ID: {order_id})")
                return {'ret_code': 0, 'order_id': order_id, 'response': response}
            else:
                logger.error(f"Failed to place order: {response.get('retMsg', 'Unknown error')}")
                return {'ret_code': response['retCode'], 'error': response.get('retMsg', 'Unknown error'),
                        'response': response}

        except Exception as e:
            logger.error(f"Error placing order for {symbol}: {e}")
            return {'ret_code': -1, 'error': str(e)}

    def _place_paper_order(self, symbol, side, order_type, qty, price, stop_loss, take_profit):
        """Place paper trading order."""
        order_id = f"paper_{int(time.time() * 1000)}"
        logger.info(f"PAPER ORDER: {side} {qty} {symbol} at {price or 'market'} (SL: {stop_loss}, TP: {take_profit})")
        return {'ret_code': 0, 'order_id': order_id, 'status': 'Filled'}

    def modify_position(self, symbol: str, stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> \
    Dict[str, Any]:
        """
        Modify existing position's stop loss or take profit.

        Args:
            symbol: Trading pair symbol
            stop_loss: New stop loss price
            take_profit: New take profit price

        Returns:
            Response dictionary.
        """
        if self.mode == 'paper' or not self.client:
            logger.info(f"PAPER MODIFY: {symbol} SL: {stop_loss}, TP: {take_profit}")
            return {'ret_code': 0, 'message': 'Paper trading position modified'}

        try:
            self._rate_limit()

            if not stop_loss and not take_profit:
                logger.warning("No modifications specified for position")
                return {'ret_code': -1, 'error': 'No stop loss or take profit specified'}

            modify_params = {"category": "linear", "symbol": symbol}
            if stop_loss: modify_params["stopLoss"] = str(stop_loss)
            if take_profit: modify_params["takeProfit"] = str(take_profit)

            response = self.client.set_trading_stop(**modify_params)

            if response['retCode'] == 0:
                logger.info(f"Position modified successfully for {symbol}: SL={stop_loss}, TP={take_profit}")
                return {'ret_code': 0, 'message': 'Position modified successfully', 'response': response}
            else:
                logger.error(f"Failed to modify position: {response.get('retMsg', 'Unknown error')}")
                return {'ret_code': response['retCode'], 'error': response.get('retMsg', 'Unknown error'),
                        'response': response}

        except Exception as e:
            logger.error(f"Error modifying position for {symbol}: {e}")
            return {'ret_code': -1, 'error': str(e)}

    def close_position(self, symbol: str) -> Dict[str, Any]:
        """
        Close an entire position for a given symbol using a market order.

        Args:
            symbol: Trading pair symbol to close.

        Returns:
            Response dictionary from the closing order.
        """
        if self.mode == 'paper' or not self.client:
            logger.info(f"PAPER CLOSE: Closing position for {symbol}")
            return {'ret_code': 0, 'message': 'Paper trading position closed'}

        try:
            self._rate_limit()

            # Use get_current_positions to find the position to close
            positions = self.get_current_positions()
            active_position = next((p for p in positions if p['symbol'] == symbol), None)

            if not active_position:
                logger.warning(f"No active position found for {symbol} to close.")
                return {'ret_code': -1, 'error': f'No active position found for {symbol}'}

            current_side = active_position['side']
            position_size = active_position['size']
            close_side = 'Sell' if current_side == 'Buy' else 'Buy'

            logger.info(f"Attempting to close {current_side} position of size {position_size} for {symbol}...")

            # Place market order to close position with reduce_only=True
            return self.place_order(
                symbol=symbol,
                side=close_side,
                order_type='Market',
                qty=position_size,
                reduce_only=True
            )

        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
            return {'ret_code': -1, 'error': str(e)}

    def get_account_balance(self) -> Dict[str, Any]:
        """
        Get Unified Trading Account balance information.

        Returns:
            Balance dictionary for the unified account.
        """
        if self.mode == 'paper' or not self.client:
            return {'accountType': 'PAPER', 'totalEquity': '10000.0',
                    'coin': [{'coin': 'USDT', 'availableToWithdraw': '10000.0'}]}

        try:
            self._rate_limit()
            response = self.client.get_wallet_balance(accountType="UNIFIED")
            if response['retCode'] == 0 and response['result']['list']:
                logger.info("Successfully fetched account balance.")
                # The main unified account info is the first item in the list
                return response['result']['list'][0]
            else:
                logger.error(f"Failed to fetch account balance: {response.get('retMsg', 'Unknown error')}")
                return {}
        except Exception as e:
            logger.error(f"Error fetching account balance: {e}")
            return {}

    def get_server_time(self) -> Optional[int]:
        """
        Get Bybit server time in milliseconds.

        Returns:
            Server timestamp in milliseconds or None if failed.
        """
        if self.mode == 'paper' or not self.client:
            return int(time.time() * 1000)

        try:
            self._rate_limit()
            response = self.client.get_server_time()
            if response.get('retCode') == 0:
                # timeNano is in nanoseconds, convert to milliseconds
                return int(response['result']['timeNano']) // 1_000_000
            else:
                logger.error(f"Failed to get server time: {response.get('retMsg')}")
                return None
        except Exception as e:
            logger.error(f"Error fetching server time: {e}")
            return None