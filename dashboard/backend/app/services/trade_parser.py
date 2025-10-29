"""
Trade log parser - reads and processes trade JSON files + SQLite database

V2: Hybrid approach - reads from SQLite if available, falls back to JSON
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from app.config import settings
from app.models import Trade, TradeSummary, TradeEvent, TradeIndicators, TradeSide
from app.services.state_reader import StateReader

# Database imports with graceful fallback
try:
    from app.services.database import get_database, get_repository
    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Database module not available - using JSON-only mode")

logger = logging.getLogger(__name__)


class TradeParser:
    """
    Parses trade logs from JSON files or SQLite database.

    V2: Hybrid approach
    - Tries to read from SQLite database first (faster, structured queries)
    - Falls back to JSON files if database unavailable
    - Maintains backward compatibility
    """

    def __init__(self, use_database: bool = True):
        self.logs_dir = Path(settings.LOGS_DIR)
        self.trades_dir = self.logs_dir / "trades"
        self.state_reader = StateReader()

        # Initialize database repository if available
        self.use_database = use_database and DATABASE_AVAILABLE
        self.repository = None

        if self.use_database:
            try:
                db_path = self.logs_dir / "trades.db"
                self.repository = get_repository()
                logger.info(f"✓ Database repository initialized: {db_path}")
            except Exception as e:
                logger.warning(f"Failed to initialize database: {e}. Falling back to JSON mode.")
                self.use_database = False

    def get_all_trades(self, limit: Optional[int] = None) -> List[Trade]:
        """
        Load all trade logs (from database or JSON files).

        Args:
            limit: Maximum number of trades to return (most recent first)

        Returns:
            List of Trade objects sorted by start_time descending
        """
        # Try database first
        if self.use_database and self.repository:
            try:
                return self._get_trades_from_database(limit=limit)
            except Exception as e:
                logger.error(f"Database query failed: {e}. Falling back to JSON.")
                self.use_database = False

        # Fallback to JSON files
        if not self.trades_dir.exists():
            logger.warning(f"Trades directory not found: {self.trades_dir}")
            return []

        trades = []
        trade_files = sorted(
            self.trades_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        for trade_file in trade_files[:limit] if limit else trade_files:
            try:
                trade = self._parse_trade_file(trade_file)
                if trade:
                    trades.append(trade)
            except Exception as e:
                logger.error(f"Failed to parse {trade_file}: {e}")

        return trades

    def get_trade_by_id(self, trade_id: str) -> Optional[Trade]:
        """Get single trade by ID"""
        trade_file = self.trades_dir / f"{trade_id}.json"

        if not trade_file.exists():
            return None

        return self._parse_trade_file(trade_file)

    def get_active_trades(self) -> List[Trade]:
        """
        Get all currently active trades.

        Checks two sources:
        1. Trade log files (logs/trades/*.json) with no end_time
        2. Bot state files (bot_state/*_state.json) for positions without trade logs

        IMPORTANT: State files are ignored if there's a NEWER closed trade log for the same ticker
        (prevents showing already-closed positions as active due to stale state files)
        """
        # Get active trades from trade logs
        all_trades = self.get_all_trades()
        active_from_logs = [t for t in all_trades if t.is_active and t.end_time is None]

        # Build a map of ticker -> most recent closed trade end_time
        latest_closed_by_ticker: Dict[str, datetime] = {}
        for trade in all_trades:
            if trade.end_time is not None:
                ticker = trade.ticker
                if ticker not in latest_closed_by_ticker or trade.end_time > latest_closed_by_ticker[ticker]:
                    latest_closed_by_ticker[ticker] = trade.end_time

        # Get active positions from bot state files
        active_from_state = self.state_reader.get_active_positions_from_state()

        # Filter out state files if they're older than the most recent closed trade
        filtered_state_positions = []
        filtered_count = 0

        for pos in active_from_state:
            latest_closed_time = latest_closed_by_ticker.get(pos.ticker)

            if latest_closed_time is None:
                # No closed trades for this ticker, keep the state position
                filtered_state_positions.append(pos)
            elif pos.start_time > latest_closed_time:
                # State file is NEWER than the latest closed trade, keep it
                filtered_state_positions.append(pos)
            else:
                # State file is OLDER than the latest closed trade, it's stale
                filtered_count += 1
                logger.info(f"Filtered stale state file for {pos.ticker}: state from {pos.start_time}, last closed at {latest_closed_time}")

        if filtered_count > 0:
            logger.info(f"Filtered out {filtered_count} stale state files (older than closed trades)")

        # Combine both sources
        all_active = active_from_logs + filtered_state_positions

        logger.info(f"Active trades: {len(active_from_logs)} from logs, {len(filtered_state_positions)} from state files")

        return all_active

    def get_trades_by_ticker(self, ticker: str) -> List[Trade]:
        """Get all trades for specific ticker"""
        all_trades = self.get_all_trades()
        return [t for t in all_trades if t.ticker == ticker]

    def _parse_trade_file(self, file_path: Path) -> Optional[Trade]:
        """
        Parse single trade JSON file.

        Expected structure:
        {
            "trade_id": "20231028_120000_SOLUSDT_Long",
            "ticker": "SOLUSDT",
            "side": "Long",
            "start_time": "2023-10-28T12:00:00",
            "end_time": "2023-10-28T14:30:00",
            "events": [...],
            "indicators": {...},
            "summary": {...}
        }
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Parse events
            events = []
            for event in data.get("events", []):
                events.append(TradeEvent(
                    timestamp=datetime.fromisoformat(event["timestamp"]),
                    type=event["type"],
                    data=event.get("data", {})
                ))

            # Parse summary
            summary = None
            if "summary" in data and data["summary"]:
                summary_data = data["summary"]
                # Handle both formats: new (total_pnl_usd) and legacy (pnl)
                pnl = summary_data.get("total_pnl_usd", summary_data.get("pnl", 0.0))
                pnl_percent = summary_data.get("total_pnl_pct", summary_data.get("pnl_percent", 0.0))

                logger.debug(f"Parsing summary for {file_path.name}: pnl={pnl}, pnl_pct={pnl_percent}")

                summary = TradeSummary(
                    pnl=pnl,
                    pnl_percent=pnl_percent,
                    exit_reason=summary_data.get("exit_reason", "Unknown"),
                    duration_seconds=summary_data.get("duration_seconds", 0.0),
                    max_favorable_excursion=summary_data.get("max_favorable_excursion"),
                    max_adverse_excursion=summary_data.get("max_adverse_excursion"),
                    fees_paid=summary_data.get("fees_paid", 0.0)
                )

            # Parse indicators
            indicators = None
            if "indicators" in data and data["indicators"]:
                indicators = TradeIndicators(
                    entry=data["indicators"].get("entry"),
                    exit=data["indicators"].get("exit")
                )

            # Extract entry/exit info from events
            entry_price = None
            exit_price = None
            quantity = None
            leverage = None
            initial_sl = None
            initial_tp = None
            current_sl = None
            current_tp = None

            for event in events:
                if event.type == "ENTRY":
                    entry_price = event.data.get("entry_price")
                    quantity = event.data.get("quantity")
                    leverage = event.data.get("leverage")
                    initial_sl = event.data.get("initial_sl")
                    initial_tp = event.data.get("initial_tp")
                    current_sl = initial_sl
                    current_tp = initial_tp
                elif event.type == "EXIT":
                    exit_price = event.data.get("exit_price")
                elif event.type == "TSL_UPDATE":
                    current_sl = event.data.get("new_sl")

            # Determine if trade is active
            is_active = data.get("end_time") is None

            return Trade(
                trade_id=data["trade_id"],
                ticker=data["ticker"],
                side=TradeSide(data["side"]),
                start_time=datetime.fromisoformat(data["start_time"]),
                end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=quantity,
                leverage=leverage,
                initial_sl=initial_sl,
                initial_tp=initial_tp,
                current_sl=current_sl,
                current_tp=current_tp,
                summary=summary,
                events=events,
                indicators=indicators,
                is_active=is_active,
                notes=data.get("notes")
            )

        except Exception as e:
            logger.error(f"Error parsing trade file {file_path}: {e}")
            return None

    def _get_trades_from_database(self, limit: Optional[int] = None, ticker: str = None) -> List[Trade]:
        """
        Load trades from SQLite database.

        Args:
            limit: Maximum number of trades to return
            ticker: Filter by ticker (optional)

        Returns:
            List of Trade objects from database
        """
        db_trades = self.repository.get_all_trades(limit=limit, ticker=ticker)

        # Convert SQLAlchemy models to Pydantic models
        return [self._convert_db_trade_to_model(db_trade) for db_trade in db_trades]

    def _convert_db_trade_to_model(self, db_trade) -> Trade:
        """
        Convert SQLAlchemy TradeModel to Pydantic Trade model.

        Args:
            db_trade: TradeModel from database

        Returns:
            Trade Pydantic model for API
        """
        # Convert events
        events = []
        for db_event in db_trade.events:
            events.append(TradeEvent(
                timestamp=db_event.timestamp,
                type=db_event.event_type,
                data=db_event.data
            ))

        # Convert summary
        summary = None
        if db_trade.end_time:
            summary = TradeSummary(
                pnl=db_trade.pnl_usd or 0.0,
                pnl_percent=db_trade.pnl_percent or 0.0,
                exit_reason=db_trade.exit_reason or "Unknown",
                duration_seconds=db_trade.duration_seconds or 0.0,
                max_favorable_excursion=db_trade.max_favorable_excursion,
                max_adverse_excursion=db_trade.max_adverse_excursion,
                fees_paid=db_trade.fees_paid or 0.0
            )

        # Convert indicators
        indicators = None
        if db_trade.indicators:
            entry_indicators = None
            exit_indicators = None

            for db_indicator in db_trade.indicators:
                snapshot = {
                    "timestamp": db_indicator.timestamp.isoformat(),
                    "indicators": db_indicator.indicators,
                    "model_probas": {
                        "proba_buy": db_indicator.proba_buy,
                        "proba_sell": db_indicator.proba_sell
                    }
                }

                if db_indicator.snapshot_type == "entry":
                    entry_indicators = snapshot
                elif db_indicator.snapshot_type == "exit":
                    exit_indicators = snapshot

            indicators = TradeIndicators(
                entry=entry_indicators,
                exit=exit_indicators
            )

        return Trade(
            trade_id=db_trade.trade_id,
            ticker=db_trade.ticker,
            side=TradeSide(db_trade.side),
            start_time=db_trade.start_time,
            end_time=db_trade.end_time,
            entry_price=db_trade.entry_price,
            exit_price=db_trade.exit_price,
            quantity=db_trade.quantity,
            leverage=db_trade.leverage,
            initial_sl=db_trade.initial_sl,
            initial_tp=db_trade.initial_tp,
            current_sl=db_trade.final_sl or db_trade.initial_sl,  # Use final if available
            current_tp=db_trade.final_tp or db_trade.initial_tp,
            summary=summary,
            events=events,
            indicators=indicators,
            is_active=db_trade.is_active,
            notes=db_trade.notes
        )

    def update_trade_note(self, trade_id: str, note: str) -> bool:
        """Add/update note for a trade"""
        trade_file = self.trades_dir / f"{trade_id}.json"

        if not trade_file.exists():
            return False

        try:
            with open(trade_file, 'r') as f:
                data = json.load(f)

            data["notes"] = note

            with open(trade_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Updated note for trade {trade_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to update note for {trade_id}: {e}")
            return False
