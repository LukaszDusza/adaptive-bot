#!/usr/bin/env python3
"""
Live Trading Infrastructure for Adaptive Trading Bot

This module implements the live trading system including order management,
position synchronization, and real-time risk monitoring as recommended
in ANALIZA_BOTA_ADAPTACYJNEGO.md
"""

import asyncio
import logging
import ccxt
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import time
from decimal import Decimal, ROUND_DOWN

# Import our modules
from core.regime_detector import RegimeDetector, MarketRegime
from strategies.consolidation_strategy import ConsolidationStrategy, SignalType as ConsSignalType
from strategies.trend_strategy import TrendStrategy, TrendSignalType
from core.risk_manager import RiskManager, PositionSide, Position
from indicators.technical import TechnicalIndicators
from data.bybit_provider import BybitDataProvider, BybitConfig

# Setup logging
logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = "Market"
    LIMIT = "Limit"
    STOP = "Stop"
    STOP_LIMIT = "StopLimit"

class OrderStatus(Enum):
    NEW = "New"
    PARTIALLY_FILLED = "PartiallyFilled"
    FILLED = "Filled"
    CANCELED = "Canceled"
    REJECTED = "Rejected"

@dataclass
class Order:
    """Trading order representation"""
    id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    amount: float
    price: Optional[float]
    order_type: OrderType
    status: OrderStatus = OrderStatus.NEW
    filled_amount: float = 0.0
    average_price: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    strategy: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LivePosition:
    """Live position representation"""
    symbol: str
    side: str
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    strategy: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def pnl_percentage(self) -> float:
        """Calculate PnL percentage"""
        if self.entry_price == 0:
            return 0.0
        
        if self.side == 'buy':
            return ((self.current_price - self.entry_price) / self.entry_price) * 100
        else:  # sell/short
            return ((self.entry_price - self.current_price) / self.entry_price) * 100

@dataclass
class LiveTradingConfig:
    """Configuration for live trading"""
    # Exchange settings
    api_key: str
    api_secret: str
    testnet: bool = True
    
    # Trading settings
    symbols: List[str] = field(default_factory=lambda: ['BTC/USDT'])
    timeframe: str = '15m'
    max_positions: int = 1
    risk_per_trade: float = 0.02
    initial_capital: float = 10000.0
    leverage: int = 2  # default leverage for futures
    
    # Risk management
    max_daily_loss: float = 0.05  # 5% max daily loss
    max_drawdown: float = 0.15    # 15% max drawdown
    max_symbol_exposure_pct: float = 0.5  # cap per symbol exposure as % of equity
    var_threshold: float = 0.05  # 5% VaR(95%) threshold
    use_kelly: bool = False
    kelly_cap: float = 0.03  # cap Kelly fraction (3%)
    
    # Order settings
    default_order_type: OrderType = OrderType.MARKET
    slippage_tolerance: float = 0.001
    order_timeout_seconds: int = 60
    
    # Monitoring
    heartbeat_interval: int = 30  # seconds
    log_level: str = 'INFO'

class OrderManager:
    """Order management system for live trading"""
    
    def __init__(self, exchange: ccxt.Exchange):
        self.exchange = exchange
        self.active_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.callbacks: Dict[str, List[Callable]] = {
            'order_filled': [],
            'order_canceled': [],
            'order_rejected': []
        }
        
    def register_callback(self, event_type: str, callback: Callable):
        """Register callback for order events"""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
            logger.info(f"Registered callback for {event_type}")
    
    async def create_order(self, order: Order) -> Optional[str]:
        """
        Create a new order on the exchange
        
        Args:
            order: Order object to create
            
        Returns:
            Order ID if successful, None if failed
        """
        try:
            # Prepare order parameters
            params = {}
            if order.order_type == OrderType.LIMIT:
                params['price'] = order.price
            elif order.order_type == OrderType.STOP:
                params['stopPrice'] = order.price
            
            # Create order on exchange
            result = await self.exchange.create_order(
                symbol=order.symbol,
                type=order.order_type.value.lower(),
                side=order.side,
                amount=order.amount,
                price=order.price,
                params=params
            )
            
            # Update order with exchange response
            order.id = result['id']
            order.status = OrderStatus.NEW
            order.updated_at = datetime.now()
            
            # Store in active orders
            self.active_orders[order.id] = order
            
            logger.info(f"Created order {order.id}: {order.side} {order.amount} {order.symbol}")
            return order.id
            
        except Exception as e:
            logger.error(f"Failed to create order: {e}")
            order.status = OrderStatus.REJECTED
            await self._trigger_callbacks('order_rejected', order)
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an active order"""
        try:
            if order_id not in self.active_orders:
                logger.warning(f"Order {order_id} not found in active orders")
                return False
            
            order = self.active_orders[order_id]
            
            # Cancel on exchange
            await self.exchange.cancel_order(order_id, order.symbol)
            
            # Update status
            order.status = OrderStatus.CANCELED
            order.updated_at = datetime.now()
            
            # Move to history
            self.order_history.append(order)
            del self.active_orders[order_id]
            
            logger.info(f"Canceled order {order_id}")
            await self._trigger_callbacks('order_canceled', order)
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def update_orders(self):
        """Update status of all active orders"""
        for order_id in list(self.active_orders.keys()):
            await self._update_order_status(order_id)
    
    async def _update_order_status(self, order_id: str):
        """Update individual order status"""
        try:
            order = self.active_orders[order_id]
            
            # Fetch order status from exchange
            exchange_order = await self.exchange.fetch_order(order_id, order.symbol)
            
            # Update order details
            old_status = order.status
            order.status = OrderStatus(exchange_order['status'])
            order.filled_amount = exchange_order['filled']
            order.average_price = exchange_order['average']
            order.updated_at = datetime.now()
            
            # Handle status changes
            if old_status != order.status:
                if order.status == OrderStatus.FILLED:
                    self.order_history.append(order)
                    del self.active_orders[order_id]
                    await self._trigger_callbacks('order_filled', order)
                    logger.info(f"Order {order_id} filled at {order.average_price}")
                    
                elif order.status == OrderStatus.CANCELED:
                    self.order_history.append(order)
                    del self.active_orders[order_id]
                    await self._trigger_callbacks('order_canceled', order)
                    
        except Exception as e:
            logger.error(f"Error updating order {order_id}: {e}")
    
    async def _trigger_callbacks(self, event_type: str, order: Order):
        """Trigger callbacks for order events"""
        for callback in self.callbacks.get(event_type, []):
            try:
                await callback(order)
            except Exception as e:
                logger.error(f"Callback error for {event_type}: {e}")

class PositionManager:
    """Position management and synchronization"""
    
    def __init__(self, exchange: ccxt.Exchange):
        self.exchange = exchange
        self.positions: Dict[str, LivePosition] = {}
        self.position_history: List[LivePosition] = []
        
    async def sync_positions(self):
        """Synchronize positions with exchange, but only track positions opened by the bot (in DB)."""
        try:
            # Try to get managed symbols from DB (positions opened by the bot)
            managed_symbols: Optional[set] = None
            try:
                from database.models import get_position_repository, PositionStatus
                pos_repo = get_position_repository()
                open_positions = pos_repo.get_positions(status=PositionStatus.OPEN, limit=1000)
                managed_symbols = set(p.symbol for p in open_positions)
            except Exception:
                # If DB not available, default to empty set to avoid touching any positions
                managed_symbols = set()
            
            # Fetch positions from exchange
            exchange_positions = await self.exchange.fetch_positions()
            
            current_symbols = set()
            
            for pos in exchange_positions:
                # Normalize and validate fields safely
                if not isinstance(pos, dict):
                    continue
                symbol = pos.get('symbol')
                # Determine size using multiple possible keys and consider non-zero only
                size_value = None
                for size_field in ['size', 'contracts', 'amount', 'baseSize', 'quoteSize']:
                    if size_field in pos and pos.get(size_field) not in (None, 0):
                        size_value = pos.get(size_field)
                        break
                if size_value in (None, 0) or not symbol:
                    continue  # Skip inactive or malformed positions
                
                # Filter: only positions that the bot manages (present in DB)
                if managed_symbols and symbol not in managed_symbols:
                    # Ignore human/opened elsewhere positions
                    continue
                elif not managed_symbols:
                    # When DB is unavailable or has no open positions, do not track any positions
                    continue
                
                current_symbols.add(symbol)
                
                # Map common fields safely
                side = pos.get('side') or ('buy' if (pos.get('positionSide') == 'Long') else 'sell')
                entry_price = pos.get('entryPrice') or pos.get('avgEntryPrice') or pos.get('entry_price') or 0.0
                mark_price = pos.get('markPrice') or pos.get('lastPrice') or pos.get('price') or entry_price
                unrealized_pnl = pos.get('unrealizedPnl') or pos.get('unrealizedPnlUsd') or 0.0
                
                if symbol in self.positions:
                    # Update existing position
                    self.positions[symbol].size = float(size_value)
                    self.positions[symbol].current_price = float(mark_price)
                    self.positions[symbol].unrealized_pnl = float(unrealized_pnl)
                    self.positions[symbol].updated_at = datetime.now()
                else:
                    # Create new position (managed by bot)
                    self.positions[symbol] = LivePosition(
                        symbol=symbol,
                        side=side,
                        size=float(size_value),
                        entry_price=float(entry_price),
                        current_price=float(mark_price),
                        unrealized_pnl=float(unrealized_pnl)
                    )
                    logger.info(f"Managed position detected (DB): {symbol} {side} {size_value}")
            
            # Remove closed positions
            closed_symbols = set(self.positions.keys()) - current_symbols
            for symbol in closed_symbols:
                closed_pos = self.positions.pop(symbol)
                self.position_history.append(closed_pos)
                logger.info(f"Managed position closed: {symbol}")
                
        except Exception as e:
            logger.error(f"Error syncing positions: {e}")
    
    def get_position(self, symbol: str) -> Optional[LivePosition]:
        """Get current position for symbol"""
        return self.positions.get(symbol)
    
    def get_total_exposure(self) -> float:
        """Get total position exposure"""
        return sum(abs(pos.size * pos.current_price) for pos in self.positions.values())
    
    def get_unrealized_pnl(self) -> float:
        """Get total unrealized PnL"""
        return sum(pos.unrealized_pnl for pos in self.positions.values())

class LiveTradingEngine:
    """Main live trading engine orchestrating all components"""
    
    def __init__(self, config: LiveTradingConfig):
        self.config = config
        self.running = False
        
        # Initialize exchange
        self.exchange = ccxt.bybit({
            'apiKey': config.api_key,
            'secret': config.api_secret,
            'testnet': config.testnet,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'linear',
            }
        })
        
        # Initialize components
        self.bybit_provider = BybitDataProvider(BybitConfig(
            api_key=config.api_key,
            secret=config.api_secret,
            testnet=config.testnet
        ))
        
        self.order_manager = OrderManager(self.exchange)
        self.position_manager = PositionManager(self.exchange)
        
        # Trading components
        self.regime_detector = RegimeDetector()
        self.trend_strategy = TrendStrategy()
        self.consolidation_strategy = ConsolidationStrategy()
        self.risk_manager = RiskManager(
            initial_capital=config.initial_capital,
            risk_per_trade=config.risk_per_trade
        )
        self.technical = TechnicalIndicators()
        
        # Data storage
        self.data_buffer: Dict[str, pd.DataFrame] = {}
        self.last_signals: Dict[str, Any] = {}
        
        # Performance tracking
        self.start_capital = config.initial_capital
        self.daily_pnl = 0.0
        self.max_drawdown_reached = 0.0
        
        # Register callbacks
        self.order_manager.register_callback('order_filled', self._on_order_filled)
        
    async def start(self):
        """Start the live trading engine"""
        logger.info("Starting live trading engine...")
        
        try:
            # Test exchange connection
            await self.exchange.load_markets()
            logger.info("Exchange connection established")
            
            # Try to set leverage for each symbol (futures mode)
            for sym in self.config.symbols:
                try:
                    if hasattr(self.exchange, 'setLeverage'):
                        await self.exchange.setLeverage(self.config.leverage, sym, params={'buyLeverage': self.config.leverage, 'sellLeverage': self.config.leverage})
                    elif hasattr(self.exchange, 'set_leverage'):
                        await self.exchange.set_leverage(self.config.leverage, sym)
                    logger.info(f"Leverage set for {sym}: x{self.config.leverage}")
                except Exception as le:
                    logger.warning(f"Could not set leverage for {sym}: {le}")
            
            # Initialize data buffers
            await self._initialize_data_buffers()
            
            # Start WebSocket connections
            await self.bybit_provider.connect_websocket(
                self.config.symbols, 
                self.config.timeframe
            )
            
            # Register WebSocket callbacks
            for symbol in self.config.symbols:
                self.bybit_provider.register_callback(
                    symbol.replace('/', ''), 
                    self.config.timeframe,
                    self._on_new_data
                )
            
            self.running = True
            
            # Start main trading loop
            await asyncio.gather(
                self._trading_loop(),
                self._monitoring_loop(),
                self._risk_monitoring_loop()
            )
            
        except Exception as e:
            logger.error(f"Error starting live trading: {e}")
            await self.stop()
    
    async def stop(self):
        """Stop the live trading engine"""
        logger.info("Stopping live trading engine...")
        self.running = False
        
        # Close all positions (optional, depends on strategy)
        # await self._close_all_positions()
        
        # Disconnect WebSocket
        await self.bybit_provider.disconnect_websocket()
        
        logger.info("Live trading engine stopped")
    
    async def _initialize_data_buffers(self):
        """Initialize historical data buffers"""
        for symbol in self.config.symbols:
            try:
                # Get initial historical data
                df = self.bybit_provider.get_historical_data(
                    symbol=symbol,
                    timeframe=self.config.timeframe,
                    limit=200  # Enough for indicators
                )
                
                # Add technical indicators
                df = self.technical.add_all_indicators(df)
                
                self.data_buffer[symbol] = df
                logger.info(f"Initialized data buffer for {symbol}: {len(df)} candles")
                
            except Exception as e:
                logger.error(f"Failed to initialize data for {symbol}: {e}")
    
    async def _on_new_data(self, ohlcv_data: Dict[str, Any]):
        """Handle new OHLCV data from WebSocket"""
        try:
            symbol = ohlcv_data['symbol']
            
            # Convert to DataFrame row
            new_row = pd.DataFrame([{
                'open': ohlcv_data['open'],
                'high': ohlcv_data['high'],
                'low': ohlcv_data['low'],
                'close': ohlcv_data['close'],
                'volume': ohlcv_data['volume']
            }], index=[ohlcv_data['timestamp']])
            
            # Update data buffer
            if symbol in self.data_buffer:
                # Remove oldest data and add new
                self.data_buffer[symbol] = pd.concat([
                    self.data_buffer[symbol].iloc[1:],  # Remove first row
                    new_row
                ])
                
                # Recalculate indicators
                self.data_buffer[symbol] = self.technical.add_all_indicators(
                    self.data_buffer[symbol]
                )
                
                # Trigger trading decision
                await self._process_trading_signal(symbol)
                
        except Exception as e:
            logger.error(f"Error processing new data: {e}")
    
    async def _process_trading_signal(self, symbol: str):
        """Process trading signals for a symbol"""
        try:
            data = self.data_buffer[symbol]
            if len(data) < 50:  # Not enough data
                return
            
            # Detect current regime
            regime, confidence = self.regime_detector.detect_regime(data)
            
            # Skip trading in STAGNANT or PANIC regimes
            if regime in [MarketRegime.STAGNANT, MarketRegime.PANIC]:
                return
            
            # Get current position
            current_position = self.position_manager.get_position(symbol)
            
            # Generate signals based on regime
            signal = None
            signal_confidence = 0.0
            
            if regime == MarketRegime.TRENDING:
                signal, signal_confidence = self.trend_strategy.generate_signal(data)
            elif regime == MarketRegime.CONSOLIDATION:
                signal, signal_confidence = self.consolidation_strategy.generate_signal(data)
            
            # Process signal
            if signal and signal_confidence > 0.6:  # Minimum confidence threshold
                await self._execute_signal(symbol, signal, signal_confidence, current_position)
                
        except Exception as e:
            logger.error(f"Error processing signal for {symbol}: {e}")
    
    async def _execute_signal(
        self,
        symbol: str,
        signal: Any,
        confidence: float,
        current_position: Optional[LivePosition]
    ):
        """Execute trading signal"""
        try:
            current_price = self.data_buffer[symbol]['close'].iloc[-1]
            
            # Check risk limits
            if not await self._check_risk_limits():
                logger.warning("Risk limits exceeded, skipping signal")
                return
            
            # Determine action based on signal and current position
            if isinstance(signal, TrendSignalType):
                if signal == TrendSignalType.LONG_ENTRY and not current_position:
                    await self._open_long_position(symbol, current_price, confidence)
                elif signal == TrendSignalType.SHORT_ENTRY and not current_position:
                    await self._open_short_position(symbol, current_price, confidence)
                elif signal == TrendSignalType.EXIT and current_position:
                    await self._close_position(symbol, current_position)
            
            elif isinstance(signal, ConsSignalType):
                if signal == ConsSignalType.BUY and not current_position:
                    await self._open_long_position(symbol, current_price, confidence)
                elif signal == ConsSignalType.SELL and not current_position:
                    await self._open_short_position(symbol, current_price, confidence)
                    
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
    
    async def _open_long_position(self, symbol: str, price: float, confidence: float):
        """Open a long position"""
        try:
            # Compute equity
            equity = await self._get_account_balance()
            # Stop distance (simple 2% of price; TODO: use ATR)
            stop_distance = max(price * 0.02, 1e-8)
            
            # Optional Kelly-based sizing from DB history
            base_risk = self.config.risk_per_trade
            if self.config.use_kelly:
                try:
                    from database.models import get_position_repository, PositionStatus
                    repo = get_position_repository()
                    closed = repo.get_positions(status=PositionStatus.CLOSED, limit=200)
                    wins = [float(p.pnl) for p in closed if p.pnl and float(p.pnl) > 0]
                    losses = [float(p.pnl) for p in closed if p.pnl and float(p.pnl) < 0]
                    if len(wins) >= 5 and len(losses) >= 5:
                        win_rate = len(wins) / (len(wins) + len(losses))
                        avg_win = float(np.mean(wins))
                        avg_loss = float(-np.mean(losses))
                        if avg_loss > 0 and 0 < win_rate < 1:
                            R = avg_win / avg_loss
                            kelly = win_rate - (1 - win_rate) / R
                            if kelly is not None and kelly > 0:
                                base_risk = min(base_risk, min(kelly, self.config.kelly_cap))
                except Exception:
                    pass
            
            # Exposure cap per symbol
            max_exposure_value = equity * self.config.max_symbol_exposure_pct
            desired_risk_amount = equity * base_risk
            raw_size = desired_risk_amount / stop_distance
            max_contracts = max_exposure_value / price
            position_size = float(min(raw_size, max_contracts))
            if position_size <= 0:
                logger.info("Calculated position size is zero; skipping")
                return
            
            # Round position size to exchange precision
            position_size = self._round_to_precision(position_size, symbol)
            
            # Create market buy order
            order = Order(
                id="",  # Will be set by order manager
                symbol=symbol,
                side="buy",
                amount=position_size,
                price=None,  # Market order
                order_type=OrderType.MARKET,
                strategy="adaptive_bot",
                metadata={"confidence": confidence, "signal_type": "long_entry"}
            )
            
            order_id = await self.order_manager.create_order(order)
            if order_id:
                logger.info(f"Long position order created: {symbol} size={position_size}")
            
        except Exception as e:
            logger.error(f"Error opening long position: {e}")
    
    async def _open_short_position(self, symbol: str, price: float, confidence: float):
        """Open a short position"""
        try:
            # Compute equity
            equity = await self._get_account_balance()
            stop_distance = max(price * 0.02, 1e-8)
            
            base_risk = self.config.risk_per_trade
            if self.config.use_kelly:
                try:
                    from database.models import get_position_repository, PositionStatus
                    repo = get_position_repository()
                    closed = repo.get_positions(status=PositionStatus.CLOSED, limit=200)
                    wins = [float(p.pnl) for p in closed if p.pnl and float(p.pnl) > 0]
                    losses = [float(p.pnl) for p in closed if p.pnl and float(p.pnl) < 0]
                    if len(wins) >= 5 and len(losses) >= 5:
                        win_rate = len(wins) / (len(wins) + len(losses))
                        avg_win = float(np.mean(wins))
                        avg_loss = float(-np.mean(losses))
                        if avg_loss > 0 and 0 < win_rate < 1:
                            R = avg_win / avg_loss
                            kelly = win_rate - (1 - win_rate) / R
                            if kelly is not None and kelly > 0:
                                base_risk = min(base_risk, min(kelly, self.config.kelly_cap))
                except Exception:
                    pass
            
            max_exposure_value = equity * self.config.max_symbol_exposure_pct
            desired_risk_amount = equity * base_risk
            raw_size = desired_risk_amount / stop_distance
            max_contracts = max_exposure_value / price
            position_size = float(min(raw_size, max_contracts))
            if position_size <= 0:
                logger.info("Calculated position size is zero; skipping")
                return
            
            # Round position size to exchange precision
            position_size = self._round_to_precision(position_size, symbol)
            
            # Create market sell order
            order = Order(
                id="",  # Will be set by order manager
                symbol=symbol,
                side="sell",
                amount=position_size,
                price=None,  # Market order
                order_type=OrderType.MARKET,
                strategy="adaptive_bot",
                metadata={"confidence": confidence, "signal_type": "short_entry"}
            )
            
            order_id = await self.order_manager.create_order(order)
            if order_id:
                logger.info(f"Short position order created: {symbol} size={position_size}")
            
        except Exception as e:
            logger.error(f"Error opening short position: {e}")
    
    async def _close_position(self, symbol: str, position: LivePosition):
        """Close existing position that is managed by the bot (exists in DB)."""
        try:
            # Verify that this symbol has an OPEN position in DB managed by the bot
            try:
                from database.models import get_position_repository, PositionStatus
                pos_repo = get_position_repository()
                open_positions = pos_repo.get_positions(symbol=symbol, status=PositionStatus.OPEN, limit=1)
                if not open_positions:
                    logger.info(f"Skip closing non-managed position for {symbol}")
                    return
            except Exception:
                # If DB not available, do not close to avoid touching human positions
                logger.warning("DB unavailable while attempting to close position; skipping to avoid touching human positions")
                return
            
            # Create opposite side order
            opposite_side = "sell" if position.side == "buy" else "buy"
            
            order = Order(
                id="",
                symbol=symbol,
                side=opposite_side,
                amount=abs(position.size),
                price=None,  # Market order
                order_type=OrderType.MARKET,
                strategy="adaptive_bot",
                metadata={"signal_type": "exit", "reason": "strategy_exit"}
            )
            
            order_id = await self.order_manager.create_order(order)
            if order_id:
                logger.info(f"Position close order created: {symbol}")
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
    
    def _round_to_precision(self, amount: float, symbol: str) -> float:
        """Round amount to exchange precision"""
        # This should be implemented based on exchange market info
        # For now, use a simple rounding to 8 decimal places
        return float(Decimal(str(amount)).quantize(Decimal('0.00000001'), rounding=ROUND_DOWN))
    
    async def _check_risk_limits(self) -> bool:
        """Check if risk limits allow new trades"""
        try:
            # Check daily loss limit
            current_balance = await self._get_account_balance()
            daily_pnl_pct = (current_balance - self.start_capital) / self.start_capital
            
            if daily_pnl_pct < -self.config.max_daily_loss:
                logger.warning(f"Daily loss limit reached: {daily_pnl_pct:.2%}")
                return False
            
            # Check maximum drawdown
            if self.max_drawdown_reached > self.config.max_drawdown:
                logger.warning(f"Max drawdown limit reached: {self.max_drawdown_reached:.2%}")
                return False
            
            # Check maximum positions
            if len(self.position_manager.positions) >= self.config.max_positions:
                logger.warning("Maximum positions reached")
                return False
            
            # Per-symbol exposure cap and VaR check
            total_equity = current_balance
            # Exposure cap across tracked positions
            for sym, pos in self.position_manager.positions.items():
                exposure = abs(pos.size * pos.current_price)
                if exposure > total_equity * self.config.max_symbol_exposure_pct:
                    logger.warning(f"Exposure cap exceeded for {sym}: {exposure/total_equity:.2%}")
                    return False
            
            # Simple portfolio VaR(95%) based on recent returns
            try:
                returns = []
                for sym, df in self.data_buffer.items():
                    if isinstance(df, pd.DataFrame) and 'close' in df.columns and len(df) > 50:
                        ret = df['close'].pct_change().dropna().tail(200)
                        if not ret.empty:
                            returns.append(ret.values)
                if returns:
                    all_rets = np.concatenate(returns)
                    var95 = -np.percentile(all_rets, 5)
                    if var95 > self.config.var_threshold:
                        logger.warning(f"VaR(95) {var95:.2%} exceeds threshold {self.config.var_threshold:.2%}")
                        return False
            except Exception:
                pass
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {e}")
            return False
    
    async def _get_account_balance(self) -> float:
        """Get current account balance"""
        try:
            balance = await self.exchange.fetch_balance()
            return balance['total']['USDT']  # Assuming USDT as base currency
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
            return self.start_capital
    
    async def _on_order_filled(self, order: Order):
        """Handle order filled callback"""
        logger.info(f"Order filled: {order.id} {order.side} {order.amount} {order.symbol} @ {order.average_price}")
        
        # Sync positions after order fill
        await self.position_manager.sync_positions()
    
    async def _trading_loop(self):
        """Main trading loop"""
        while self.running:
            try:
                # Update order statuses
                await self.order_manager.update_orders()
                
                # Sync positions
                await self.position_manager.sync_positions()
                
                # Wait before next iteration
                await asyncio.sleep(5)  # 5 second interval
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(10)
    
    async def _monitoring_loop(self):
        """Monitoring and logging loop"""
        while self.running:
            try:
                # Log status
                balance = await self._get_account_balance()
                unrealized_pnl = self.position_manager.get_unrealized_pnl()
                
                logger.info(f"Balance: ${balance:.2f}, Unrealized PnL: ${unrealized_pnl:.2f}, Active Positions: {len(self.position_manager.positions)}")
                
                # Wait for next heartbeat
                await asyncio.sleep(self.config.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(30)
    
    async def _risk_monitoring_loop(self):
        """Risk monitoring loop"""
        while self.running:
            try:
                # Update drawdown tracking
                current_balance = await self._get_account_balance()
                current_drawdown = (self.start_capital - current_balance) / self.start_capital
                self.max_drawdown_reached = max(self.max_drawdown_reached, current_drawdown)
                
                # Emergency shutdown if extreme drawdown
                if current_drawdown > self.config.max_drawdown * 1.5:  # 1.5x the limit
                    logger.error("EMERGENCY SHUTDOWN: Extreme drawdown detected")
                    await self.stop()
                    break
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in risk monitoring: {e}")
                await asyncio.sleep(60)