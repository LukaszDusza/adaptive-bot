"""
Bot state file reader - reads active positions from bot_state/*.json files
"""
import json
import os
import re
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime
import logging

from app.config import settings
from app.models import Trade, TradeSide, TradeEvent

logger = logging.getLogger(__name__)


class StateReader:
    """Reads bot state files to detect active positions"""

    def __init__(self, bybit_service=None):
        self.state_dir = Path(settings.BOT_STATE_DIR)
        self.bybit_service = bybit_service

    def get_active_positions_from_state(self, verify_on_bybit: bool = True) -> List[Trade]:
        """
        Read bot_state/*.json files to find active positions.

        These files are created when a bot has an active position but may not
        have a corresponding trade log file (e.g., after restart with order recovery).

        Returns:
            List of Trade objects representing active positions
        """
        if not self.state_dir.exists():
            logger.warning(f"Bot state directory not found: {self.state_dir}")
            return []

        active_positions = []
        state_files = list(self.state_dir.glob("*_state.json"))

        logger.info(f"Found {len(state_files)} bot state files")

        for state_file in state_files:
            try:
                position = self._parse_state_file(state_file)
                if position:
                    # Verify position actually exists on Bybit (prevents showing stale bot_state data)
                    if verify_on_bybit and self.bybit_service and self.bybit_service.is_available():
                        try:
                            # Check if position exists on Bybit by fetching open orders
                            # If there are NO open orders AND position has no trade log, it's likely stale
                            bybit_orders = self.bybit_service.adapter.get_open_orders(position.ticker)
                            bybit_position = self.bybit_service.adapter.get_position(position.ticker)

                            # Position is valid ONLY if it actually exists on Bybit with size > 0
                            # Pending limit orders alone do NOT count as active position
                            has_position = bybit_position and bybit_position.get('size', 0) > 0

                            if not has_position:
                                logger.info(f"Filtering stale state file {state_file.name}: no actual position on Bybit for {position.ticker} (only pending orders)")
                                continue

                        except Exception as e:
                            logger.warning(f"Failed to verify position on Bybit for {position.ticker}: {e}")
                            # On verification error, include position anyway (fail-safe)

                    active_positions.append(position)
                    logger.info(f"Active position found in {state_file.name}: {position.ticker} {position.side}")
            except Exception as e:
                logger.error(f"Failed to parse state file {state_file}: {e}")

        return active_positions

    def _parse_state_file(self, file_path: Path) -> Optional[Trade]:
        """
        Parse bot state JSON file.

        Expected structure:
        {
            "side": "Long" | "Short",
            "entry_price": 123.45,
            "initial_tp": 126.78,
            "original_qty": 10.0,
            "initial_size": 1234.5,
            "partial_tp_taken": false,
            "dynamic_tp_levels_taken": 0,
            "last_sl": 120.0,
            "highest_price": 125.0,
            "lowest_price": 123.0,
            "dca_fills": 1,
            "last_updated": "2025-10-29T09:51:38.125291"
        }
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Extract ticker and strategy info from filename
            # Format: {TICKER}_{TIMEFRAME}_plus_{HELPERS}_state.json
            # Example: DOGEUSDT_15m_plus_1h_4h_6h_12h_1d_state.json
            filename = file_path.stem  # Remove .json
            match = re.match(r'(.+?)_(\d+[mhd])_plus_(.+)_state$', filename)

            if not match:
                logger.warning(f"Could not parse ticker from filename: {filename}")
                return None

            ticker = match.group(1)
            timeframe = match.group(2)
            helpers = match.group(3).replace('_', ' ')

            # Parse side
            side_str = data.get("side", "").capitalize()
            if side_str not in ["Long", "Short"]:
                # State file is empty or doesn't have active position
                logger.debug(f"No active position in {file_path} (side: '{side_str}')")
                return None

            side = TradeSide.LONG if side_str == "Long" else TradeSide.SHORT

            # Parse entry info
            entry_price = data.get("entry_price")
            quantity = data.get("original_qty")
            initial_tp = data.get("initial_tp")
            current_sl = data.get("last_sl")

            if not all([entry_price, quantity]):
                logger.debug(f"State file {file_path} missing position data (probably just closed)")
                return None

            # Parse timestamp
            last_updated_str = data.get("last_updated")
            if last_updated_str:
                try:
                    start_time = datetime.fromisoformat(last_updated_str)
                except:
                    start_time = datetime.fromtimestamp(file_path.stat().st_mtime)
            else:
                start_time = datetime.fromtimestamp(file_path.stat().st_mtime)

            # Create synthetic trade_id
            trade_id = f"{start_time.strftime('%Y%m%d_%H%M%S')}_{ticker}_{side_str.upper()}_RECOVERED"

            # Create events list from state data
            events = []

            # Add synthetic ENTRY event
            events.append(TradeEvent(
                timestamp=start_time,
                type="STATE_RECOVERED",
                data={
                    "source": "bot_state_file",
                    "entry_price": entry_price,
                    "quantity": quantity,
                    "initial_tp": initial_tp,
                    "initial_sl": current_sl,
                    "dca_fills": data.get("dca_fills", 0),
                    "partial_tp_taken": data.get("partial_tp_taken", False),
                    "dynamic_tp_levels_taken": data.get("dynamic_tp_levels_taken", 0)
                }
            ))

            # Build Trade object
            return Trade(
                trade_id=trade_id,
                ticker=ticker,
                side=side,
                start_time=start_time,
                end_time=None,  # Active position
                entry_price=entry_price,
                exit_price=None,
                quantity=quantity,
                leverage=None,  # Not stored in state file
                initial_sl=current_sl,  # Best we have
                initial_tp=initial_tp,
                current_sl=current_sl,
                current_tp=initial_tp,  # State file doesn't track TP updates
                summary=None,
                events=events,
                indicators=None,
                is_active=True,
                notes=f"Position recovered from bot_state file (no trade log). DCA fills: {data.get('dca_fills', 0)}. Partial TP: {'taken' if data.get('partial_tp_taken', False) else 'not taken'}. partial_tp_taken={data.get('partial_tp_taken', False)}"
            )

        except Exception as e:
            logger.error(f"Error parsing state file {file_path}: {e}")
            return None
