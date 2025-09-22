# services/bybit_service.py
"""
Bybit API service for live trading integration.
Handles connection, position management, and order execution.
"""

import pandas as pd
import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

# Production Bybit API imports
try:
    from pybit.unified_trading import HTTP
    PYBIT_AVAILABLE = True
except ImportError:
    print("Warning: pybit not installed. Install with: pip install pybit")
    PYBIT_AVAILABLE = False

# Setup logging
logger = logging.getLogger(__name__)

class BybitService:
    """
    Service for Bybit API integration.
    Handles live data fetching and position management.
    """
    
    def __init__(self, mode='paper', api_key=None, api_secret=None, testnet=True):
        """
        Initialize Bybit service.
        
        Args:
            mode: 'paper' for paper trading, 'live' for real trading
            api_key: Bybit API key (required for live trading)
            api_secret: Bybit API secret (required for live trading)
            testnet: Use testnet environment (default True for safety)
        """
        self.mode = mode
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.client = None
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
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
                # Initialize real Bybit API client
                endpoint = 'https://api-testnet.bybit.com' if self.testnet else 'https://api.bybit.com'
                self.client = HTTP(
                    testnet=self.testnet,
                    api_key=self.api_key,
                    api_secret=self.api_secret
                )
                logger.info(f"Bybit service initialized in LIVE TRADING mode (testnet: {self.testnet})")
                
                # Test connection
                server_time = self.get_server_time()
                if server_time:
                    logger.info("Bybit API connection successful")
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
            DataFrame with OHLCV data
        """
        if self.mode == 'paper':
            # For paper trading, return mock data or fetch from another source
            return self._fetch_mock_data(symbol, interval_minutes, limit)
        
        try:
            self._rate_limit()
            
            # Map interval minutes to Bybit interval format
            interval_map = {
                1: '1',
                3: '3', 
                5: '5',
                15: '15',
                30: '30',
                60: '60',
                120: '120',
                240: '240',
                360: '360',
                720: '720',
                1440: 'D'
            }
            
            interval_str = interval_map.get(interval_minutes, '5')
            
            # Fetch kline data from Bybit
            response = self.client.get_kline(
                category="spot",
                symbol=symbol,
                interval=interval_str,
                limit=min(limit, 1000)  # Bybit limit is 1000
            )
            
            if response['retCode'] == 0 and response['result']['list']:
                data = response['result']['list']
                
                # Convert to DataFrame
                df = pd.DataFrame(data, columns=[
                    'start_time', 'open', 'high', 'low', 'close', 'volume', 'turnover'
                ])
                
                # Convert timestamp and set as index
                df['timestamp'] = pd.to_datetime(df['start_time'].astype(int), unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Select and convert OHLCV columns
                df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
                
                # Sort by timestamp (Bybit returns newest first)
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
        print(f"Fetching {limit} candles of {symbol} {interval_minutes}m for paper trading")
        # TODO: Implement mock data or fetch from alternative source
        return pd.DataFrame()
    
    def get_current_positions(self) -> List[Dict[str, Any]]:
        """
        Get current open positions.
        
        Returns:
            List of position dictionaries
        """
        if self.mode == 'paper':
            # Return paper trading positions
            return []
        
        # TODO: Implement real Bybit position query
        # response = self.client.my_position(symbol=symbol)
        # if response['ret_code'] == 0:
        #     return response['result']
        
        return []
    
    def place_order(self, symbol: str, side: str, order_type: str, qty: float, 
                   price: Optional[float] = None, stop_loss: Optional[float] = None,
                   take_profit: Optional[float] = None) -> Dict[str, Any]:
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
            
        Returns:
            Order response dictionary
        """
        if self.mode == 'paper':
            return self._place_paper_order(symbol, side, order_type, qty, price, stop_loss, take_profit)
        
        try:
            self._rate_limit()
            
            # Prepare order parameters
            order_params = {
                "category": "spot",
                "symbol": symbol,
                "side": side,
                "orderType": order_type,
                "qty": str(qty),
                "timeInForce": "GTC" if order_type == "Limit" else "IOC"
            }
            
            # Add price for limit orders
            if order_type == "Limit" and price is not None:
                order_params["price"] = str(price)
            
            # Add stop loss and take profit if provided
            if stop_loss is not None:
                order_params["stopLoss"] = str(stop_loss)
            
            if take_profit is not None:
                order_params["takeProfit"] = str(take_profit)
            
            # Place order
            response = self.client.place_order(**order_params)
            
            if response['retCode'] == 0:
                order_id = response['result']['orderId']
                logger.info(f"Order placed successfully: {side} {qty} {symbol} (ID: {order_id})")
                
                return {
                    'ret_code': 0,
                    'order_id': order_id,
                    'symbol': symbol,
                    'side': side,
                    'order_type': order_type,
                    'qty': qty,
                    'price': price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'status': 'New',
                    'response': response
                }
            else:
                logger.error(f"Failed to place order: {response.get('retMsg', 'Unknown error')}")
                return {
                    'ret_code': response['retCode'],
                    'error': response.get('retMsg', 'Unknown error'),
                    'response': response
                }
                
        except Exception as e:
            logger.error(f"Error placing order for {symbol}: {e}")
            return {
                'ret_code': -1,
                'error': str(e)
            }
    
    def _place_paper_order(self, symbol: str, side: str, order_type: str, qty: float,
                          price: Optional[float], stop_loss: Optional[float], 
                          take_profit: Optional[float]) -> Dict[str, Any]:
        """Place paper trading order."""
        order_id = f"paper_{int(time.time())}"
        print(f"PAPER ORDER: {side} {qty} {symbol} at {price} (SL: {stop_loss}, TP: {take_profit})")
        
        return {
            'ret_code': 0,
            'order_id': order_id,
            'symbol': symbol,
            'side': side,
            'order_type': order_type,
            'qty': qty,
            'price': price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'status': 'Filled' if order_type == 'Market' else 'New'
        }
    
    def modify_position(self, symbol: str, stop_loss: Optional[float] = None,
                       take_profit: Optional[float] = None) -> Dict[str, Any]:
        """
        Modify existing position's stop loss or take profit.
        
        Args:
            symbol: Trading pair symbol
            stop_loss: New stop loss price
            take_profit: New take profit price
            
        Returns:
            Response dictionary
        """
        if self.mode == 'paper':
            logger.info(f"PAPER MODIFY: {symbol} SL: {stop_loss}, TP: {take_profit}")
            return {'ret_code': 0, 'message': 'Paper trading position modified'}
        
        try:
            self._rate_limit()
            
            # Prepare modification parameters
            modify_params = {
                "category": "spot",
                "symbol": symbol
            }
            
            # Add stop loss if provided
            if stop_loss is not None:
                modify_params["stopLoss"] = str(stop_loss)
            
            # Add take profit if provided
            if take_profit is not None:
                modify_params["takeProfit"] = str(take_profit)
            
            if not stop_loss and not take_profit:
                logger.warning("No modifications specified for position")
                return {'ret_code': -1, 'error': 'No stop loss or take profit specified'}
            
            # Modify position using trading stop
            response = self.client.set_trading_stop(**modify_params)
            
            if response['retCode'] == 0:
                logger.info(f"Position modified successfully for {symbol}: SL={stop_loss}, TP={take_profit}")
                return {
                    'ret_code': 0,
                    'symbol': symbol,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'message': 'Position modified successfully',
                    'response': response
                }
            else:
                logger.error(f"Failed to modify position: {response.get('retMsg', 'Unknown error')}")
                return {
                    'ret_code': response['retCode'],
                    'error': response.get('retMsg', 'Unknown error'),
                    'response': response
                }
                
        except Exception as e:
            logger.error(f"Error modifying position for {symbol}: {e}")
            return {
                'ret_code': -1,
                'error': str(e)
            }
    
    def close_position(self, symbol: str) -> Dict[str, Any]:
        """
        Close position by market order.
        
        Args:
            symbol: Trading pair symbol
            
        Returns:
            Response dictionary
        """
        if self.mode == 'paper':
            print(f"PAPER CLOSE: Closing position for {symbol}")
            return {'ret_code': 0, 'message': 'Paper trading position closed'}
        
        # TODO: Implement real position closing
        # First get current position
        # positions = self.get_current_positions()
        # for pos in positions:
        #     if pos['symbol'] == symbol and float(pos['size']) > 0:
        #         side = 'Sell' if pos['side'] == 'Buy' else 'Buy'
        #         response = self.place_order(
        #             symbol=symbol,
        #             side=side,
        #             order_type='Market',
        #             qty=float(pos['size'])
        #         )
        #         return response
        
        return {}
    
    def get_account_balance(self) -> Dict[str, Any]:
        """
        Get account balance information.
        
        Returns:
            Balance dictionary
        """
        if self.mode == 'paper':
            return {
                'ret_code': 0,
                'result': {
                    'USDT': {
                        'available_balance': 1000.0,
                        'wallet_balance': 1000.0
                    }
                }
            }
        
        # TODO: Implement real balance query
        # response = self.client.get_wallet_balance()
        # return response
        
        return {}
    
    def get_server_time(self) -> int:
        """Get Bybit server timestamp."""
        if self.mode == 'paper':
            return int(time.time() * 1000)
        
        # TODO: Implement real server time query
        # response = self.client.server_time()
        # return response['time_now']
        
        return int(time.time() * 1000)