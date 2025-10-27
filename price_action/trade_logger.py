"""
Advanced Logging System for Trading Bot
Enables full trade reconstruction and post-factum analysis
"""

import json
import os
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path


class TradeLogger:
    """
    Advanced logging system for trade reconstruction and analysis.
    
    Features:
    - JSON logs for each trade with full event history
    - Candle data storage (Parquet format)
    - Indicator snapshots
    - Performance analytics
    """
    
    def __init__(self, base_dir: str = "logs"):
        self.base_dir = Path(base_dir)
        self._setup_directories()
        
        # Current trade being tracked
        self.current_trade: Optional[Dict] = None
        self.trade_id: Optional[str] = None
        
        # Candle buffer
        self.candle_buffer: List[Dict] = []
        self.candle_buffer_size = 100  # Save every 100 candles
        
        logging.info("✓ TradeLogger initialized")
    
    def _setup_directories(self):
        """Create directory structure"""
        dirs = ['trades', 'candles', 'indicators', 'analytics']
        for dir_name in dirs:
            (self.base_dir / dir_name).mkdir(parents=True, exist_ok=True)
    
    def _generate_trade_id(self, ticker: str, side: str) -> str:
        """Generate unique trade ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{timestamp}_{ticker}_{side}"
    
    def start_trade(self, ticker: str, side: str, decision_data: Dict[str, Any]):
        """
        Start tracking a new trade.
        
        Args:
            ticker: Trading pair
            side: LONG or SHORT
            decision_data: Model decision info (probabilities, etc.)
        """
        self.trade_id = self._generate_trade_id(ticker, side)
        
        self.current_trade = {
            "trade_id": self.trade_id,
            "ticker": ticker,
            "side": side,
            "start_time": datetime.now().isoformat(),
            "events": [],
            "indicators": {},
            "candles": [],
            "summary": {}
        }
        
        # Log initial signal
        self.log_event("SIGNAL", decision_data)
        
        logging.info(f"📝 Started tracking trade: {self.trade_id}")
    
    def log_event(self, event_type: str, data: Dict[str, Any]):
        """
        Log a trade event.
        
        Event types:
        - SIGNAL: Model decision
        - ENTRY: Position opened
        - TSL_UPDATE: Trailing stop updated
        - PARTIAL_TP: Partial take profit
        - EXIT: Position closed
        - ERROR: Error occurred
        """
        if not self.current_trade:
            logging.warning(f"Attempted to log {event_type} but no active trade")
            return
        
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "data": data
        }
        
        self.current_trade["events"].append(event)
        
        # Log to console with emoji
        emoji_map = {
            "SIGNAL": "🎯",
            "ENTRY": "🚀",
            "TSL_UPDATE": "📈",
            "PARTIAL_TP": "💰",
            "EXIT": "🏁",
            "ERROR": "❌"
        }
        emoji = emoji_map.get(event_type, "📝")
        
        logging.info(f"{emoji} {event_type}: {json.dumps(data, default=str)}")
    
    def log_candle(self, candle_data: Dict[str, Any], position_data: Optional[Dict] = None):
        """
        Log candle data with optional position state.
        
        Args:
            candle_data: OHLCV data
            position_data: Current position state (optional)
        """
        candle = {
            "timestamp": candle_data.get('timestamp', datetime.now().isoformat()),
            "open": candle_data.get('open', 0.0),
            "high": candle_data.get('high', 0.0),
            "low": candle_data.get('low', 0.0),
            "close": candle_data.get('close', 0.0),
            "volume": candle_data.get('volume', 0.0),
            "turnover": candle_data.get('turnover', 0.0)
        }
        
        # Add position data if in trade
        if position_data:
            candle.update({
                "in_position": True,
                "position_side": position_data.get('side'),
                "entry_price": position_data.get('entry_price'),
                "current_sl": position_data.get('sl'),
                "current_tp": position_data.get('tp'),
                "current_pnl_pct": position_data.get('pnl_pct'),
                "highest_price": position_data.get('highest_price'),
                "lowest_price": position_data.get('lowest_price')
            })
        else:
            candle["in_position"] = False
        
        self.candle_buffer.append(candle)
        
        # Save to file when buffer is full
        if len(self.candle_buffer) >= self.candle_buffer_size:
            self._flush_candle_buffer()
    
    def _flush_candle_buffer(self):
        """Save buffered candles to Parquet file"""
        if not self.candle_buffer:
            return
        
        try:
            df = pd.DataFrame(self.candle_buffer)
            
            # Generate filename with date
            date_str = datetime.now().strftime('%Y-%m-%d')
            ticker = self.current_trade.get('ticker', 'UNKNOWN') if self.current_trade else 'UNKNOWN'
            filename = f"{ticker}_1h_{date_str}.parquet"
            filepath = self.base_dir / 'candles' / filename
            
            # Append to existing file or create new
            if filepath.exists():
                existing_df = pd.read_parquet(filepath)
                df = pd.concat([existing_df, df], ignore_index=True)
            
            df.to_parquet(filepath, index=False)
            
            logging.debug(f"💾 Saved {len(self.candle_buffer)} candles to {filename}")
            self.candle_buffer = []
            
        except Exception as e:
            logging.error(f"Failed to save candles: {e}")
    
    def log_indicators(self, indicators: Dict[str, Any], model_probas: Dict[str, float]):
        """
        Log indicator values at entry.
        
        Args:
            indicators: Dictionary of indicator values
            model_probas: Model probability outputs
        """
        if not self.current_trade:
            return
        
        indicator_snapshot = {
            "timestamp": datetime.now().isoformat(),
            "indicators": indicators,
            "model_probas": model_probas
        }
        
        # Store in trade
        self.current_trade["indicators"]["entry"] = indicator_snapshot
        
        # Also save to separate file for analysis
        self._save_indicator_snapshot(indicator_snapshot)
    
    def _save_indicator_snapshot(self, snapshot: Dict):
        """Save indicator snapshot to file"""
        try:
            date_str = datetime.now().strftime('%Y-%m-%d')
            ticker = self.current_trade.get('ticker', 'UNKNOWN')
            filename = f"{ticker}_indicators_{date_str}.json"
            filepath = self.base_dir / 'indicators' / filename
            
            # Append to file
            snapshots = []
            if filepath.exists():
                with open(filepath, 'r') as f:
                    snapshots = json.load(f)
            
            snapshots.append(snapshot)
            
            with open(filepath, 'w') as f:
                json.dump(snapshots, f, indent=2)
                
        except Exception as e:
            logging.error(f"Failed to save indicator snapshot: {e}")
    
    def end_trade(self, summary: Dict[str, Any]):
        """
        End trade tracking and save to file.
        
        Args:
            summary: Trade summary (P&L, duration, etc.)
        """
        if not self.current_trade:
            logging.warning("Attempted to end trade but no active trade")
            return
        
        self.current_trade["summary"] = summary
        self.current_trade["end_time"] = datetime.now().isoformat()
        
        # Save trade to file
        self._save_trade()
        
        # Update analytics
        self._update_analytics(self.current_trade)
        
        # Flush any remaining candles
        self._flush_candle_buffer()
        
        logging.info(f"✓ Trade completed and saved: {self.trade_id}")
        
        # Reset
        self.current_trade = None
        self.trade_id = None
    
    def _save_trade(self):
        """Save complete trade to JSON file"""
        try:
            filename = f"{self.trade_id}.json"
            filepath = self.base_dir / 'trades' / filename
            
            with open(filepath, 'w') as f:
                json.dump(self.current_trade, f, indent=2, default=str)
            
            logging.info(f"💾 Trade saved: {filename}")
            
        except Exception as e:
            logging.error(f"Failed to save trade: {e}", exc_info=True)
    
    def _update_analytics(self, trade: Dict):
        """Update aggregate analytics"""
        try:
            # Daily summary
            date_str = datetime.now().strftime('%Y-%m-%d')
            summary_file = self.base_dir / 'analytics' / f"daily_summary_{date_str}.json"
            
            summary = {}
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
            
            # Initialize if new
            if not summary:
                summary = {
                    "date": date_str,
                    "total_trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "total_pnl": 0.0,
                    "trades": []
                }
            
            # Update
            pnl = trade["summary"].get("total_pnl_usd", 0)
            summary["total_trades"] += 1
            summary["total_pnl"] += pnl
            
            if pnl > 0:
                summary["wins"] += 1
            elif pnl < 0:
                summary["losses"] += 1
            
            summary["trades"].append({
                "trade_id": trade["trade_id"],
                "side": trade["side"],
                "pnl": pnl,
                "duration": trade["summary"].get("duration_seconds", 0)
            })
            
            # Calculate win rate
            if summary["total_trades"] > 0:
                summary["win_rate"] = summary["wins"] / summary["total_trades"] * 100
            
            # Save
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
                
        except Exception as e:
            logging.error(f"Failed to update analytics: {e}")
    
    def get_trade_history(self, days: int = 7) -> List[Dict]:
        """
        Load trade history for last N days.
        
        Args:
            days: Number of days to look back
        
        Returns:
            List of trade dictionaries
        """
        trades = []
        trades_dir = self.base_dir / 'trades'
        
        try:
            for trade_file in sorted(trades_dir.glob('*.json'), reverse=True):
                # Check if within date range
                # Filename format: YYYYMMDD_HHMMSS_TICKER_SIDE.json
                date_str = trade_file.stem[:8]
                trade_date = datetime.strptime(date_str, '%Y%m%d')
                
                if (datetime.now() - trade_date).days <= days:
                    with open(trade_file, 'r') as f:
                        trades.append(json.load(f))
            
            return trades
            
        except Exception as e:
            logging.error(f"Failed to load trade history: {e}")
            return []


class TradeVisualizer:
    """
    Visualize trades with full reconstruction from logs.
    """
    
    def __init__(self, trade_logger: TradeLogger):
        self.logger = trade_logger
    
    def reconstruct_trade(self, trade_id: str) -> Optional[Dict]:
        """
        Load complete trade data for visualization.
        
        Args:
            trade_id: Trade ID to reconstruct
        
        Returns:
            Complete trade data with candles
        """
        try:
            # Load trade JSON
            trade_file = self.logger.base_dir / 'trades' / f"{trade_id}.json"
            
            if not trade_file.exists():
                logging.error(f"Trade file not found: {trade_id}")
                return None
            
            with open(trade_file, 'r') as f:
                trade = json.load(f)
            
            # Load associated candles
            ticker = trade['ticker']
            start_time = datetime.fromisoformat(trade['start_time'])
            end_time = datetime.fromisoformat(trade['end_time'])
            
            candles = self._load_candles_for_period(
                ticker, 
                start_time.date(), 
                end_time.date()
            )
            
            # Filter candles to trade period
            if candles is not None:
                candles = candles[
                    (candles['timestamp'] >= start_time.isoformat()) &
                    (candles['timestamp'] <= end_time.isoformat())
                ]
                trade['candles_df'] = candles
            
            return trade
            
        except Exception as e:
            logging.error(f"Failed to reconstruct trade: {e}", exc_info=True)
            return None
    
    def _load_candles_for_period(self, ticker: str, start_date, end_date) -> Optional[pd.DataFrame]:
        """Load candles from Parquet files for date range"""
        dfs = []
        
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            candle_file = self.logger.base_dir / 'candles' / f"{ticker}_1h_{date_str}.parquet"
            
            if candle_file.exists():
                df = pd.read_parquet(candle_file)
                dfs.append(df)
            
            current_date = pd.to_datetime(current_date) + pd.Timedelta(days=1)
            current_date = current_date.date()
        
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return None
    
    def create_chart(self, trade_id: str, output_file: str = None):
        """
        Create interactive chart with all trade events.
        
        Requires: plotly
        
        Args:
            trade_id: Trade to visualize
            output_file: Output HTML file (optional)
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            logging.error("plotly not installed. Run: pip install plotly")
            return
        
        trade = self.reconstruct_trade(trade_id)
        
        if not trade:
            return
        
        candles_df = trade.get('candles_df')
        
        if candles_df is None or candles_df.empty:
            logging.error("No candle data available for visualization")
            return
        
        # Create figure
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('Price Action', 'Volume'),
            row_heights=[0.7, 0.3]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=candles_df['timestamp'],
                open=candles_df['open'],
                high=candles_df['high'],
                low=candles_df['low'],
                close=candles_df['close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Add trade events as annotations
        for event in trade['events']:
            event_type = event['type']
            event_data = event['data']
            timestamp = event['timestamp']
            
            # Get price for annotation
            candle = candles_df[candles_df['timestamp'] == timestamp]
            
            if not candle.empty:
                price = candle.iloc[0]['high']
                
                color_map = {
                    'ENTRY': 'green',
                    'TSL_UPDATE': 'blue',
                    'PARTIAL_TP': 'orange',
                    'EXIT': 'red'
                }
                
                fig.add_annotation(
                    x=timestamp,
                    y=price,
                    text=event_type,
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor=color_map.get(event_type, 'gray'),
                    ax=0,
                    ay=-40,
                    row=1, col=1
                )
        
        # Add SL/TP lines if in position
        if 'entry_price' in candles_df.columns:
            in_position = candles_df[candles_df['in_position'] == True]
            
            if not in_position.empty:
                # Entry price
                fig.add_hline(
                    y=in_position.iloc[0]['entry_price'],
                    line_dash="dash",
                    line_color="yellow",
                    annotation_text="Entry",
                    row=1, col=1
                )
                
                # Current SL (changes over time)
                for idx, row in in_position.iterrows():
                    if pd.notna(row.get('current_sl')):
                        fig.add_scatter(
                            x=[row['timestamp']],
                            y=[row['current_sl']],
                            mode='markers',
                            marker=dict(color='red', size=3),
                            showlegend=False,
                            row=1, col=1
                        )
        
        # Volume bars
        colors = ['red' if row['close'] < row['open'] else 'green' 
                 for _, row in candles_df.iterrows()]
        
        fig.add_trace(
            go.Bar(
                x=candles_df['timestamp'],
                y=candles_df['volume'],
                marker_color=colors,
                name='Volume'
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f"Trade Reconstruction: {trade_id}",
            xaxis_rangeslider_visible=False,
            height=800
        )
        
        # Save or show
        if output_file:
            fig.write_html(output_file)
            logging.info(f"✓ Chart saved to {output_file}")
        else:
            fig.show()


# Example usage
if __name__ == "__main__":
    # Initialize logger
    trade_logger = TradeLogger()
    
    # Start tracking a trade
    trade_logger.start_trade(
        ticker="SOLUSDT",
        side="LONG",
        decision_data={
            "decision": "BUY",
            "proba_buy": 0.847,
            "proba_sell": 0.342,
            "threshold": 0.80
        }
    )
    
    # Log entry
    trade_logger.log_event("ENTRY", {
        "entry_price": 102.45,
        "quantity": 0.976,
        "sl_price": 100.20,
        "tp_price": 108.60
    })
    
    # Log TSL update
    trade_logger.log_event("TSL_UPDATE", {
        "old_sl": 100.20,
        "new_sl": 101.50,
        "current_price": 105.25,
        "reason": "in_profit"
    })
    
    # End trade
    trade_logger.end_trade({
        "total_pnl_usd": 4.42,
        "duration_seconds": 10800
    })
    
    print("✓ Example logging completed")
