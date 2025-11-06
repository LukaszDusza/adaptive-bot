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

        # Check if hedge mode is enabled (True if env var is "true", "1", "yes")
        hedge_mode_env = os.getenv("BYBIT_HEDGE_MODE", "true").lower()
        self.hedge_mode = hedge_mode_env in ("true", "1", "yes")

        if not self.api_key or not self.api_secret:
            logger.warning("Bybit API credentials not configured - pending orders feature will be limited")
            self.adapter = None
        else:
            try:
                self.adapter = BybitAdapter(
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                    base_url=self.base_url if self.base_url != "https://api.bybit.com" else None,
                    hedge_mode=self.hedge_mode
                )
                logger.info(f"✓ BybitAdapter initialized successfully (hedge_mode={self.hedge_mode})")
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

    def get_all_open_orders(self) -> Dict[str, List[Dict]]:
        """
        Fetch ALL open orders from Bybit (all symbols).

        Since Bybit API requires at least one filter parameter (symbol/settleCoin/baseCoin),
        we fetch orders for all active positions + use settleCoin=USDT to catch any other orders.

        Returns:
            Dictionary mapping ticker to list of orders
        """
        if not self.adapter:
            logger.warning("BybitAdapter not initialized")
            return {}

        try:
            # Strategy: Get all positions first, then fetch orders for each ticker
            # Also fetch by settleCoin=USDT to catch any orders without positions

            # Step 1: Get all active positions to know which tickers have orders
            active_positions = self.get_active_positions_from_bybit()
            active_tickers = {pos.get('ticker') for pos in active_positions}
            logger.info(f"Found {len(active_tickers)} active positions: {active_tickers}")

            # Step 2: Fetch orders by settleCoin=USDT (gets all USDT perpetual orders)
            all_orders = self.adapter.get_open_orders_by_settle_coin('USDT')

            # Group by symbol
            orders_by_ticker = {}
            for order in all_orders:
                symbol = order.get('symbol', 'UNKNOWN')
                if symbol not in orders_by_ticker:
                    orders_by_ticker[symbol] = []
                orders_by_ticker[symbol].append(order)

            logger.info(f"✓ Retrieved {len(all_orders)} open orders across {len(orders_by_ticker)} symbols")
            return orders_by_ticker

        except Exception as e:
            logger.error(f"Failed to get all open orders: {e}")
            return {}

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

    def get_closed_pnl_history_paginated(self, symbol: str = None, start_time_ms: int = None, max_records: int = 1000) -> list:
        """
        Get closed P&L history from Bybit with pagination to fetch more than 100 records.

        Args:
            symbol: Trading pair (e.g., SOLUSDT). If None, gets all symbols.
            start_time_ms: Start timestamp in milliseconds (e.g., Nov 1, 2025)
            max_records: Maximum total records to fetch (will make multiple API calls)

        Returns:
            List of closed P&L records sorted by createdTime descending
        """
        if not self.adapter:
            logger.warning("BybitAdapter not initialized - cannot get P&L history")
            return []

        try:
            all_records = []
            current_end_time = None  # Start from most recent

            # Fetch in batches of 100 (Bybit's limit per request)
            while len(all_records) < max_records:
                batch_size = min(100, max_records - len(all_records))

                pnl_records = self.adapter.get_closed_pnl_history(
                    symbol=symbol,
                    start_time_ms=start_time_ms,
                    end_time_ms=current_end_time,
                    limit=batch_size
                )

                if not pnl_records:
                    break  # No more records

                all_records.extend(pnl_records)

                # Use oldest record's timestamp as next end_time for pagination
                oldest_record_time = min(int(r.get('createdTime', 0)) for r in pnl_records)

                # If we got less than requested or hit the start time, we're done
                if len(pnl_records) < batch_size or (start_time_ms and oldest_record_time <= start_time_ms):
                    break

                # Set next end_time to 1ms before oldest record (to avoid duplicates)
                current_end_time = oldest_record_time - 1

                logger.info(f"📄 Fetched {len(pnl_records)} records (total: {len(all_records)}), continuing pagination...")

            logger.info(f"✓ Retrieved {len(all_records)} closed P&L records from Bybit (paginated)")
            return all_records

        except Exception as e:
            logger.error(f"Failed to get closed P&L history (paginated): {e}")
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
                cumulative_pnl += closed_pnl  # Add PnL going backwards

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

    def get_active_positions_from_bybit(self) -> list:
        """
        Get all active positions from Bybit (SOURCE OF TRUTH).

        Returns:
            List of active position dictionaries with full trade details
        """
        if not self.adapter:
            logger.warning("BybitAdapter not initialized - cannot get positions")
            return []

        try:
            # Get all open positions across all symbols
            positions = self.adapter.get_all_positions()

            if not positions:
                logger.info("No active positions found on Bybit")
                return []

            active_positions = []
            for pos in positions:
                # Filter only positions with size > 0
                size = abs(float(pos.get('size', 0)))
                if size > 0:
                    active_positions.append({
                        'ticker': pos.get('symbol'),
                        'side': pos.get('side'),  # 'Buy' or 'Sell'
                        'entry_price': float(pos.get('avgPrice', 0)),
                        'quantity': size,
                        'leverage': int(float(pos.get('leverage', 1))),
                        'unrealized_pnl': float(pos.get('unrealisedPnl', 0)),
                        'stop_loss': float(pos.get('stopLoss', 0)) if pos.get('stopLoss') else None,
                        'take_profit': float(pos.get('takeProfit', 0)) if pos.get('takeProfit') else None,
                        'created_time': pos.get('createdTime', ''),
                        'updated_time': pos.get('updatedTime', ''),
                    })

            logger.info(f"✓ Retrieved {len(active_positions)} active positions from Bybit")
            return active_positions

        except Exception as e:
            logger.error(f"Failed to get active positions from Bybit: {e}")
            return []

    def get_trade_history_stats_from_bybit(self, limit: int = 500) -> dict:
        """
        Calculate trading statistics from Bybit closed PnL history (SOURCE OF TRUTH).

        Args:
            limit: Number of recent trades to analyze (default 500)

        Returns:
            Dictionary with comprehensive trading stats
        """
        if not self.adapter:
            logger.warning("BybitAdapter not initialized")
            return {}

        try:
            pnl_records = self.get_closed_pnl_history(limit=limit)

            if not pnl_records:
                return {
                    'total_trades': 0,
                    'total_pnl': 0.0,
                    'win_rate': 0.0,
                    'avg_win': 0.0,
                    'avg_loss': 0.0,
                    'profit_factor': None,
                    'max_win': 0.0,
                    'max_loss': 0.0,
                }

            # Calculate stats
            total_trades = len(pnl_records)
            pnls = [float(r.get('closedPnl', 0)) for r in pnl_records]

            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]

            total_pnl = sum(pnls)
            win_rate = len(wins) / total_trades if total_trades > 0 else 0.0

            avg_win = sum(wins) / len(wins) if wins else 0.0
            avg_loss = sum(losses) / len(losses) if losses else 0.0

            total_wins = sum(wins) if wins else 0.0
            total_losses = abs(sum(losses)) if losses else 0.0
            profit_factor = total_wins / total_losses if total_losses > 0 else None

            max_win = max(pnls) if pnls else 0.0
            max_loss = min(pnls) if pnls else 0.0

            # Calculate fees (if available in records)
            total_fees = sum([abs(float(r.get('execFee', 0))) for r in pnl_records])

            # Calculate trades per day (last 7 days average)
            from datetime import datetime, timedelta
            now_ms = int(datetime.now().timestamp() * 1000)
            seven_days_ago_ms = now_ms - (7 * 24 * 60 * 60 * 1000)

            trades_last_7_days = [
                r for r in pnl_records
                if int(r.get('createdTime', 0)) >= seven_days_ago_ms
            ]
            trades_per_day = len(trades_last_7_days) / 7.0

            stats = {
                'total_trades': total_trades,
                'total_pnl': total_pnl,
                'win_rate': win_rate,
                'winning_trades': len(wins),
                'losing_trades': len(losses),
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'max_win': max_win,
                'max_loss': max_loss,
                'total_fees': total_fees,
                'avg_fee_per_trade': total_fees / total_trades if total_trades > 0 else 0.0,
                'trades_per_day': trades_per_day,
                'trades_last_7_days': len(trades_last_7_days),
            }

            logger.info(f"✓ Calculated stats from {total_trades} Bybit trades: PnL=${total_pnl:.2f}, WR={win_rate*100:.1f}%")
            return stats

        except Exception as e:
            logger.error(f"Failed to calculate stats from Bybit: {e}")
            return {}

    def analyze_execution_quality(self, limit: int = 200) -> dict:
        """
        Analyze execution quality: slippage, maker/taker ratio, fill times.

        Args:
            limit: Number of recent executions to analyze

        Returns:
            Dictionary with execution quality metrics
        """
        if not self.adapter:
            return {
                'total_executions': 0,
                'avg_slippage_pct': 0.0,
                'maker_count': 0,
                'taker_count': 0,
                'maker_ratio': 0.0,
                'taker_ratio': 0.0,
                'best_slippage_pct': 0.0,
                'worst_slippage_pct': 0.0,
                'total_fees': 0.0,
                'avg_fee_per_execution': 0.0
            }

        try:
            executions = self.adapter.get_execution_list(limit=limit)

            if not executions:
                return {
                    'total_executions': 0,
                    'avg_slippage_pct': 0.0,
                    'maker_count': 0,
                    'taker_count': 0,
                    'maker_ratio': 0.0,
                    'taker_ratio': 0.0,
                    'best_slippage_pct': 0.0,
                    'worst_slippage_pct': 0.0,
                    'total_fees': 0.0,
                    'avg_fee_per_execution': 0.0
                }

            # Calculate slippage for each execution
            slippages = []
            maker_count = 0
            taker_count = 0
            total_fees = 0.0

            for exec in executions:
                order_price = float(exec.get('orderPrice', 0))
                exec_price = float(exec.get('execPrice', 0))

                # Calculate slippage (positive = worse than expected)
                if order_price > 0:
                    if exec.get('side') == 'Buy':
                        # Buy: positive slippage = paid more
                        slippage = ((exec_price - order_price) / order_price) * 100
                    else:
                        # Sell: positive slippage = received less
                        slippage = ((order_price - exec_price) / order_price) * 100

                    slippages.append(slippage)

                # Count maker vs taker
                if exec.get('isMaker'):
                    maker_count += 1
                else:
                    taker_count += 1

                # Sum fees
                total_fees += abs(float(exec.get('execFee', 0)))

            total_count = len(executions)
            avg_slippage = sum(slippages) / len(slippages) if slippages else 0.0
            best_slippage = min(slippages) if slippages else 0.0
            worst_slippage = max(slippages) if slippages else 0.0

            maker_ratio = maker_count / total_count if total_count > 0 else 0.0
            taker_ratio = taker_count / total_count if total_count > 0 else 0.0

            avg_fee_rate = total_fees / total_count if total_count > 0 else 0.0

            result = {
                'total_executions': total_count,
                'avg_slippage_pct': avg_slippage,
                'maker_count': maker_count,
                'taker_count': taker_count,
                'maker_ratio': maker_ratio,
                'taker_ratio': taker_ratio,
                'best_slippage_pct': best_slippage,
                'worst_slippage_pct': worst_slippage,
                'total_fees': total_fees,
                'avg_fee_per_execution': avg_fee_rate
            }

            logger.info(f"✓ Execution quality: Slippage={avg_slippage:.3f}%, Maker={maker_ratio*100:.1f}%, Fees=${total_fees:.2f}")
            return result

        except Exception as e:
            logger.error(f"Failed to analyze execution quality: {e}")
            return {
                'total_executions': 0,
                'avg_slippage_pct': 0.0,
                'maker_count': 0,
                'taker_count': 0,
                'maker_ratio': 0.0,
                'taker_ratio': 0.0,
                'best_slippage_pct': 0.0,
                'worst_slippage_pct': 0.0,
                'total_fees': 0.0,
                'avg_fee_per_execution': 0.0
            }

    def analyze_funding_costs(self, days: int = 30) -> dict:
        """
        Analyze funding costs (fees for holding positions).

        Args:
            days: Number of days to analyze

        Returns:
            Dictionary with funding cost analysis
        """
        if not self.adapter:
            return {
                'total_funding_fees': 0.0,
                'funding_count': 0,
                'avg_funding_per_event': 0.0,
                'daily_avg_funding': 0.0,
                'monthly_projected_funding': 0.0,
                'funding_by_symbol': {}
            }

        try:
            # Calculate time range
            from datetime import datetime, timedelta
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)

            start_ms = int(start_time.timestamp() * 1000)
            end_ms = int(end_time.timestamp() * 1000)

            # Get transaction log - fetch in batches (max 50 per request)
            all_transactions = []
            current_start = start_ms

            while current_start < end_ms:
                transactions = self.adapter.get_transaction_log(
                    type=None,  # Get all types
                    start_time_ms=current_start,
                    end_time_ms=end_ms,
                    limit=50
                )

                if not transactions:
                    break

                all_transactions.extend(transactions)

                # Get oldest transaction time for next batch
                oldest_time = min([int(t.get('transactionTime', end_ms)) for t in transactions])
                if oldest_time <= current_start:
                    break  # Prevent infinite loop
                current_start = oldest_time + 1

                if len(all_transactions) >= 500:  # Limit total fetched
                    break

            # Filter funding fees
            funding_transactions = [
                t for t in all_transactions
                if t.get('type') in ['FUNDING_FEE', 'SETTLEMENT']
            ]

            if not funding_transactions:
                return {
                    'total_funding_fees': 0.0,
                    'funding_count': 0,
                    'avg_funding_per_event': 0.0,
                    'daily_avg_funding': 0.0,
                    'monthly_projected_funding': 0.0,
                    'funding_by_symbol': {}
                }

            # Calculate metrics
            total_funding = sum([abs(float(t.get('funding', 0))) for t in funding_transactions])
            funding_count = len(funding_transactions)
            avg_per_event = total_funding / funding_count if funding_count > 0 else 0.0
            daily_avg = total_funding / days if days > 0 else 0.0

            # Group by symbol
            from collections import defaultdict
            funding_by_symbol = defaultdict(float)
            for t in funding_transactions:
                symbol = t.get('symbol', 'UNKNOWN')
                funding_by_symbol[symbol] += abs(float(t.get('funding', 0)))

            result = {
                'total_funding_fees': total_funding,
                'funding_count': funding_count,
                'avg_funding_per_event': avg_per_event,
                'daily_avg_funding': daily_avg,
                'monthly_projected_funding': daily_avg * 30,
                'funding_by_symbol': dict(funding_by_symbol)
            }

            logger.info(f"✓ Funding analysis ({days}d): Total=${total_funding:.2f}, Daily=${daily_avg:.2f}")
            return result

        except Exception as e:
            logger.error(f"Failed to analyze funding costs: {e}")
            return {
                'total_funding_fees': 0.0,
                'funding_count': 0,
                'avg_funding_per_event': 0.0,
                'daily_avg_funding': 0.0,
                'monthly_projected_funding': 0.0,
                'funding_by_symbol': {}
            }

    def analyze_sl_tp_effectiveness(self, limit: int = 100) -> dict:
        """
        Analyze SL/TP effectiveness from order history.

        Args:
            limit: Number of recent orders to analyze

        Returns:
            Dictionary with SL/TP stats
        """
        if not self.adapter:
            return {
                'total_orders': 0,
                'tp_hit_count': 0,
                'sl_hit_count': 0,
                'tp_hit_rate': 0.0,
                'sl_hit_rate': 0.0,
                'avg_tp_distance_pct': 0.0,
                'avg_sl_distance_pct': 0.0,
                'risk_reward_ratio': 0.0
            }

        try:
            orders = self.adapter.get_order_history(order_status='Filled', limit=limit)

            if not orders:
                return {
                    'total_orders': 0,
                    'tp_hit_count': 0,
                    'sl_hit_count': 0,
                    'tp_hit_rate': 0.0,
                    'sl_hit_rate': 0.0,
                    'avg_tp_distance_pct': 0.0,
                    'avg_sl_distance_pct': 0.0,
                    'risk_reward_ratio': 0.0
                }

            # Analyze TP/SL
            tp_orders = []
            sl_orders = []
            tp_distances = []
            sl_distances = []

            for order in orders:
                stop_type = order.get('stopOrderType', 'UNKNOWN')
                entry_price = float(order.get('lastPriceOnCreated', 0))

                if stop_type in ['TakeProfit', 'PartialTakeProfit']:
                    tp_orders.append(order)

                    # Calculate TP distance
                    tp_price = float(order.get('takeProfit', 0))
                    if tp_price > 0 and entry_price > 0:
                        side = order.get('side')
                        if side == 'Buy':
                            tp_dist = ((tp_price - entry_price) / entry_price) * 100
                        else:
                            tp_dist = ((entry_price - tp_price) / entry_price) * 100
                        tp_distances.append(abs(tp_dist))

                elif stop_type in ['StopLoss', 'TrailingStop']:
                    sl_orders.append(order)

                    # Calculate SL distance
                    sl_price = float(order.get('stopLoss', 0))
                    if sl_price > 0 and entry_price > 0:
                        side = order.get('side')
                        if side == 'Buy':
                            sl_dist = ((entry_price - sl_price) / entry_price) * 100
                        else:
                            sl_dist = ((sl_price - entry_price) / entry_price) * 100
                        sl_distances.append(abs(sl_dist))

            total_orders = len(orders)
            tp_count = len(tp_orders)
            sl_count = len(sl_orders)

            tp_rate = tp_count / total_orders if total_orders > 0 else 0.0
            sl_rate = sl_count / total_orders if total_orders > 0 else 0.0

            avg_tp_dist = sum(tp_distances) / len(tp_distances) if tp_distances else 0.0
            avg_sl_dist = sum(sl_distances) / len(sl_distances) if sl_distances else 0.0

            # Risk/Reward ratio
            rr_ratio = avg_tp_dist / avg_sl_dist if avg_sl_dist > 0 else 0.0

            result = {
                'total_orders': total_orders,
                'tp_hit_count': tp_count,
                'sl_hit_count': sl_count,
                'tp_hit_rate': tp_rate,
                'sl_hit_rate': sl_rate,
                'avg_tp_distance_pct': avg_tp_dist,
                'avg_sl_distance_pct': avg_sl_dist,
                'risk_reward_ratio': rr_ratio
            }

            logger.info(f"✓ SL/TP analysis: TP={tp_rate*100:.1f}%, SL={sl_rate*100:.1f}%, R:R={rr_ratio:.2f}")
            return result

        except Exception as e:
            logger.error(f"Failed to analyze SL/TP effectiveness: {e}")
            return {
                'total_orders': 0,
                'tp_hit_count': 0,
                'sl_hit_count': 0,
                'tp_hit_rate': 0.0,
                'sl_hit_rate': 0.0,
                'avg_tp_distance_pct': 0.0,
                'avg_sl_distance_pct': 0.0,
                'risk_reward_ratio': 0.0
            }

    def close_position_by_symbol(self, symbol: str) -> dict:
        """
        Close position for a specific symbol via Bybit API.

        Args:
            symbol: Trading pair (e.g., SOLUSDT)

        Returns:
            Dict with success status and message

        Raises:
            Exception if position close fails
        """
        if not self.adapter:
            raise RuntimeError("BybitAdapter not initialized - cannot close position")

        try:
            logger.info(f"🚨 Closing position for {symbol}")
            self.adapter.close_position(symbol)
            logger.info(f"✓ Position closed successfully for {symbol}")
            return {
                'success': True,
                'symbol': symbol,
                'message': f'Position closed for {symbol}'
            }
        except Exception as e:
            logger.error(f"Failed to close position for {symbol}: {e}")
            raise

    def set_stop_loss_to_breakeven(self, symbol: str, side: str, entry_price: float) -> dict:
        """
        Set stop loss to breakeven + fee buffer (entry price + trading fees).

        Args:
            symbol: Trading pair (e.g., SOLUSDT)
            side: Position side ("Long" or "Short")
            entry_price: Entry price to set as stop loss

        Returns:
            Dict with success status and message

        Raises:
            Exception if SL update fails
        """
        if not self.adapter:
            raise RuntimeError("BybitAdapter not initialized - cannot update SL")

        try:
            # Calculate breakeven with fee buffer
            # Bybit fees: ~0.055% taker (entry) + ~0.055% taker (exit) = ~0.11%
            # Use 0.15% buffer to be safe (0.0015)
            fee_buffer_pct = 0.0015

            side_upper = side.upper()
            if side_upper in ("BUY", "LONG"):
                # LONG: need higher price to breakeven (entry + fees)
                breakeven_price = entry_price * (1 + fee_buffer_pct)
            else:
                # SHORT: need lower price to breakeven (entry - fees)
                breakeven_price = entry_price * (1 - fee_buffer_pct)

            logger.info(f"Setting SL to BE+fee for {symbol} {side}: entry={entry_price}, BE+fee={breakeven_price:.6f}")
            self.adapter.set_stop_loss(symbol, breakeven_price, side)
            logger.info(f"✓ SL set to breakeven+fee for {symbol}")
            return {
                'success': True,
                'symbol': symbol,
                'side': side,
                'stop_loss': breakeven_price,
                'message': f'SL set to breakeven+fee (${breakeven_price:.6f}) for {symbol}'
            }
        except Exception as e:
            logger.error(f"Failed to set SL for {symbol}: {e}")
            raise
