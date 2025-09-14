#!/usr/bin/env python3
"""
Bybit Data Provider for Adaptive Trading Bot

This module implements the Bybit API integration for fetching historical data,
real-time data, and managing WebSocket connections as recommended in the analysis.
"""

import ccxt
import pandas as pd
import numpy as np
import asyncio
import websockets
import json
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from ccxt.base.errors import NetworkError, ExchangeError, RateLimitExceeded
from dataclasses import dataclass
import time

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class BybitConfig:
    """Configuration for Bybit connection"""
    api_key: Optional[str] = None
    secret: Optional[str] = None
    testnet: bool = True
    enable_rate_limit: bool = True
    timeout: int = 10000
    
class BybitDataProvider:
    """
    Bybit data provider implementing the recommendations from ANALIZA_BOTA_ADAPTACYJNEGO.md
    
    Features:
    - Historical data fetching for backtesting
    - WebSocket connection for live trading
    - Rate limiting and error handling
    - Support for multiple timeframes
    """
    
    def __init__(self, config: BybitConfig):
        """Initialize Bybit data provider"""
        self.config = config
        self.exchange = None
        self.ws_connection = None
        self.ws_callbacks = {}
        self.is_connected = False
        self._init_exchange()
        
    def _init_exchange(self):
        """Initialize CCXT exchange instance"""
        try:
            self.exchange = ccxt.bybit({
                'apiKey': self.config.api_key,
                'secret': self.config.secret,
                'testnet': self.config.testnet,
                'enableRateLimit': self.config.enable_rate_limit,
                'timeout': self.config.timeout,
                'options': {
                    'defaultType': 'linear',  # Use linear derivatives
                },
            })
            logger.info(f"Bybit exchange initialized (testnet: {self.config.testnet})")
        except Exception as e:
            logger.error(f"Failed to initialize Bybit exchange: {e}")
            raise
    
    def get_historical_data(
        self, 
        symbol: str, 
        timeframe: str = '15m',
        limit: int = 1000,
        since: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from Bybit
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
            limit: Number of candles to fetch (max 1000)
            since: Start date for historical data
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Convert datetime to timestamp if provided
            since_ts = None
            if since:
                since_ts = int(since.timestamp() * 1000)
            
            # Fetch OHLCV data with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    ohlcv = self.exchange.fetch_ohlcv(
                        symbol=symbol,
                        timeframe=timeframe,
                        limit=limit,
                        since=since_ts
                    )
                    break
                except RateLimitExceeded:
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff
                        logger.warning(f"Rate limit exceeded, waiting {wait_time}s")
                        time.sleep(wait_time)
                    else:
                        raise
                except (NetworkError, ExchangeError) as e:
                    if attempt < max_retries - 1:
                        wait_time = 1 + attempt
                        logger.warning(f"Network error: {e}, retrying in {wait_time}s")
                        time.sleep(wait_time)
                    else:
                        raise
            
            # Convert to DataFrame
            if not ohlcv:
                raise ValueError("No data received from Bybit")
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Ensure numeric data types
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].astype(float)
            
            logger.info(f"Fetched {len(df)} candles for {symbol} ({timeframe})")
            return df
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error fetching historical data: {error_msg}")
            
            # Provide specific guidance for authentication errors
            if "10003" in error_msg or "API key is invalid" in error_msg:
                logger.error("Authentication Error: API key is invalid or expired")
                logger.error("Troubleshooting steps:")
                logger.error("1. Verify API key and secret are correct")
                logger.error(f"2. Check testnet setting matches your API key (current: testnet={self.config.testnet})")
                logger.error("3. Ensure API key has required permissions")
            elif "10004" in error_msg:
                logger.error("Authentication Error: API secret is invalid")
            elif "10005" in error_msg:
                logger.error("Permission Error: API key lacks required permissions")
            
            raise
    
    def get_klines(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        """
        Fetch klines data from Bybit (alias for get_historical_data)
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT' or 'BTCUSDT')
            timeframe: Candle timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
            limit: Number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data and timestamp column
        """
        try:
            # Ensure symbol is in correct format for CCXT
            if '/' not in symbol:
                # Convert BTCUSDT to BTC/USDT format
                if symbol.endswith('USDT'):
                    base = symbol[:-4]
                    quote = 'USDT'
                    symbol = f"{base}/{quote}"
                elif symbol.endswith('USD'):
                    base = symbol[:-3]
                    quote = 'USD'
                    symbol = f"{base}/{quote}"
            
            # Use get_historical_data to fetch the data
            df = self.get_historical_data(symbol, timeframe, limit)
            
            # Reset index to make timestamp a column (as expected by calling code)
            df_reset = df.reset_index()
            
            return df_reset
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error fetching klines data: {error_msg}")
            
            # Provide specific guidance for authentication errors
            if "10003" in error_msg or "API key is invalid" in error_msg:
                logger.error("Authentication Error: API key is invalid or expired")
                logger.error("Troubleshooting steps:")
                logger.error("1. Verify API key and secret are correct")
                logger.error(f"2. Check testnet setting matches your API key (current: testnet={self.config.testnet})")
                logger.error("3. Ensure API key has required permissions")
            elif "10004" in error_msg:
                logger.error("Authentication Error: API secret is invalid")
            elif "10005" in error_msg:
                logger.error("Permission Error: API key lacks required permissions")
            
            raise
    
    def get_multiple_timeframe_data(
        self,
        symbol: str,
        timeframes: List[str] = ['15m', '1h', '4h'],
        limit: int = 1000,
        since: Optional[datetime] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple timeframes
        
        Args:
            symbol: Trading pair
            timeframes: List of timeframes to fetch
            limit: Number of candles per timeframe
            since: Start date
            
        Returns:
            Dictionary with timeframe as key and DataFrame as value
        """
        data = {}
        for timeframe in timeframes:
            try:
                df = self.get_historical_data(symbol, timeframe, limit, since)
                data[timeframe] = df
                
                # Add delay between requests to avoid rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Failed to fetch data for {timeframe}: {e}")
                data[timeframe] = pd.DataFrame()
        
        return data
    
    async def connect_websocket(self, symbols: List[str], timeframe: str = '15m'):
        """
        Connect to Bybit WebSocket for real-time data
        
        Args:
            symbols: List of symbols to subscribe to
            timeframe: Kline timeframe
        """
        try:
            # Bybit WebSocket URL
            if self.config.testnet:
                ws_url = "wss://stream-testnet.bybit.com/v5/public/linear"
            else:
                ws_url = "wss://stream.bybit.com/v5/public/linear"
            
            # Connect to WebSocket
            self.ws_connection = await websockets.connect(ws_url)
            
            # Subscribe to kline data
            subscription_topics = []
            for symbol in symbols:
                # Convert symbol format (BTC/USDT -> BTCUSDT)
                bybit_symbol = symbol.replace('/', '')
                topic = f"kline.{timeframe}.{bybit_symbol}"
                subscription_topics.append(topic)
            
            subscribe_msg = {
                "op": "subscribe",
                "args": subscription_topics
            }
            
            await self.ws_connection.send(json.dumps(subscribe_msg))
            logger.info(f"Subscribed to WebSocket topics: {subscription_topics}")
            
            self.is_connected = True
            
            # Start listening for messages
            asyncio.create_task(self._listen_websocket())
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            self.is_connected = False
            raise
    
    async def _listen_websocket(self):
        """Listen for WebSocket messages"""
        try:
            async for message in self.ws_connection:
                data = json.loads(message)
                
                # Handle different message types
                if 'topic' in data and data['topic'].startswith('kline'):
                    await self._handle_kline_data(data)
                elif 'success' in data:
                    logger.info(f"WebSocket subscription successful: {data}")
                elif 'ret_msg' in data:
                    logger.warning(f"WebSocket message: {data['ret_msg']}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
            self.is_connected = False
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            self.is_connected = False
    
    async def _handle_kline_data(self, data: Dict[str, Any]):
        """Process incoming kline data"""
        try:
            topic = data['topic']
            kline_data = data['data'][0]  # First kline in the array
            
            # Extract symbol and timeframe from topic
            parts = topic.split('.')
            timeframe = parts[1]
            symbol = parts[2]
            
            # Convert to standard format
            ohlcv = {
                'timestamp': pd.to_datetime(int(kline_data['start']), unit='ms'),
                'open': float(kline_data['open']),
                'high': float(kline_data['high']),
                'low': float(kline_data['low']),
                'close': float(kline_data['close']),
                'volume': float(kline_data['volume']),
                'symbol': symbol,
                'timeframe': timeframe
            }
            
            # Call registered callbacks
            callback_key = f"{symbol}_{timeframe}"
            if callback_key in self.ws_callbacks:
                for callback in self.ws_callbacks[callback_key]:
                    try:
                        await callback(ohlcv)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")
            
        except Exception as e:
            logger.error(f"Error processing kline data: {e}")
    
    def register_callback(self, symbol: str, timeframe: str, callback: Callable):
        """
        Register callback function for real-time data
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            callback: Async callback function to handle data
        """
        key = f"{symbol}_{timeframe}"
        if key not in self.ws_callbacks:
            self.ws_callbacks[key] = []
        self.ws_callbacks[key].append(callback)
        logger.info(f"Registered callback for {key}")
    
    async def disconnect_websocket(self):
        """Disconnect from WebSocket"""
        if self.ws_connection:
            await self.ws_connection.close()
            self.is_connected = False
            logger.info("WebSocket disconnected")
    
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get current ticker information"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            raise
    
    def get_orderbook(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """Get current orderbook"""
        try:
            orderbook = self.exchange.fetch_order_book(symbol, limit)
            return orderbook
        except Exception as e:
            logger.error(f"Error fetching orderbook for {symbol}: {e}")
            raise
    
    def test_connection(self) -> bool:
        """Test connection to Bybit with proper API key validation"""
        try:
            # First try to load markets (doesn't require authentication)
            self.exchange.load_markets()
            logger.info("Bybit markets loaded successfully")
            
            # If API key is provided, test authenticated endpoints
            if self.config.api_key and self.config.secret:
                try:
                    # Test API key by fetching account balance (requires authentication)
                    balance = self.exchange.fetch_balance()
                    logger.info("Bybit API authentication successful")
                    logger.info(f"Account balance fetched: {len(balance)} currencies")
                    return True
                except Exception as auth_error:
                    # Enhanced error logging for authentication failures
                    error_msg = str(auth_error)
                    logger.error(f"Bybit API authentication failed: {error_msg}")
                    
                    # Check for common API key issues
                    if "10003" in error_msg or "API key is invalid" in error_msg:
                        logger.error("API Key Error: The provided API key is invalid or expired")
                        logger.error("Please check:")
                        logger.error("1. API key is correctly copied without extra spaces")
                        logger.error("2. API secret is correctly copied without extra spaces") 
                        logger.error("3. API key has the required permissions (Account Read, Position Read)")
                        logger.error(f"4. Testnet setting matches your API key type (current: testnet={self.config.testnet})")
                    elif "10004" in error_msg:
                        logger.error("API Secret Error: The API secret is invalid")
                    elif "10005" in error_msg:
                        logger.error("Permission Error: API key doesn't have required permissions")
                    elif "network" in error_msg.lower():
                        logger.error("Network Error: Check your internet connection")
                    
                    return False
            else:
                logger.info("No API credentials provided - public endpoints only")
                return True
                
        except Exception as e:
            logger.error(f"Bybit connection test failed: {e}")
            return False
    
    def get_available_symbols(self, base_currency: str = 'USDT') -> List[str]:
        """Get list of available trading symbols"""
        try:
            markets = self.exchange.load_markets()
            symbols = [symbol for symbol, market in markets.items() 
                      if market['quote'] == base_currency and market['active']]
            return sorted(symbols)
        except Exception as e:
            logger.error(f"Error fetching available symbols: {e}")
            return []


class BybitDataManager:
    """High-level data manager for coordinating multiple data providers"""
    
    def __init__(self, config: BybitConfig):
        self.provider = BybitDataProvider(config)
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes
    
    def get_backtest_data(
        self,
        symbol: str,
        timeframe: str = '15m',
        days_back: int = 365
    ) -> pd.DataFrame:
        """Get data suitable for backtesting"""
        since = datetime.now() - timedelta(days=days_back)
        
        # For backtesting, we may need more data than the API limit
        # Implement pagination if needed
        all_data = []
        current_since = since
        
        while current_since < datetime.now():
            df = self.provider.get_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                limit=1000,
                since=current_since
            )
            
            if df.empty:
                break
                
            all_data.append(df)
            
            # Move to the next batch
            current_since = df.index[-1] + timedelta(minutes=15)  # Adjust based on timeframe
            
            # Avoid rate limiting
            time.sleep(0.2)
        
        if all_data:
            result = pd.concat(all_data)
            result = result[~result.index.duplicated(keep='first')]
            result.sort_index(inplace=True)
            return result
        else:
            return pd.DataFrame()