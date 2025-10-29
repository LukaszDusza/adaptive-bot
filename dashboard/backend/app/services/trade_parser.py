"""
Trade log parser - reads and processes trade JSON files
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

logger = logging.getLogger(__name__)


class TradeParser:
    """Parses trade logs from JSON files"""

    def __init__(self):
        self.logs_dir = Path(settings.LOGS_DIR)
        self.trades_dir = self.logs_dir / "trades"
        self.state_reader = StateReader()

    def get_all_trades(self, limit: Optional[int] = None) -> List[Trade]:
        """
        Load all trade logs from JSON files.

        Args:
            limit: Maximum number of trades to return (most recent first)

        Returns:
            List of Trade objects sorted by start_time descending
        """
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
        """
        # Get active trades from trade logs
        all_trades = self.get_all_trades()
        active_from_logs = [t for t in all_trades if t.is_active and t.end_time is None]

        # Get active positions from bot state files
        active_from_state = self.state_reader.get_active_positions_from_state()

        # Combine both sources (state files should only have positions not in logs)
        # Deduplicate by trade_id (though state files use synthetic IDs, so unlikely to clash)
        all_active = active_from_logs + active_from_state

        logger.info(f"Active trades: {len(active_from_logs)} from logs, {len(active_from_state)} from state files")

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
                summary = TradeSummary(
                    pnl=summary_data.get("pnl", 0.0),
                    pnl_percent=summary_data.get("pnl_percent", 0.0),
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
