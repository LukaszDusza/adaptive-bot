"""
Bot state management - pause/resume, config updates
"""
import json
import os
from pathlib import Path
from typing import Dict, Optional, List
import logging

from app.config import settings

logger = logging.getLogger(__name__)


class BotStateManager:
    """
    Manages bot state files and configuration.

    State files are stored in bot_state/ directory:
    - {strategy_id}_state.json - Trading position state
    - dashboard_control.json - Dashboard control flags (pause, etc.)
    """

    def __init__(self):
        self.bot_state_dir = Path(settings.BOT_STATE_DIR)
        self.control_file = self.bot_state_dir / "dashboard_control.json"
        self._ensure_directories()

    def _ensure_directories(self):
        """Ensure state directories exist"""
        self.bot_state_dir.mkdir(parents=True, exist_ok=True)

    def get_pause_state(self) -> Dict[str, bool]:
        """
        Get current pause state for all tickers.

        Returns:
            Dict mapping ticker -> is_paused (bool)
        """
        if not self.control_file.exists():
            return {}

        try:
            with open(self.control_file, 'r') as f:
                data = json.load(f)
                return data.get("paused_tickers", {})
        except Exception as e:
            logger.error(f"Failed to read control file: {e}")
            return {}

    def set_pause_state(self, tickers: Optional[List[str]], paused: bool, reason: str = "") -> bool:
        """
        Set pause state for specific tickers or all.

        Args:
            tickers: List of tickers to pause/resume, or None for all
            paused: True to pause, False to resume
            reason: Optional reason for pause

        Returns:
            True if successful
        """
        try:
            # Load existing data
            data = {}
            if self.control_file.exists():
                with open(self.control_file, 'r') as f:
                    data = json.load(f)

            if "paused_tickers" not in data:
                data["paused_tickers"] = {}

            # Update pause state
            if tickers is None:
                # Pause/resume all
                data["global_pause"] = paused
                data["global_pause_reason"] = reason
                logger.info(f"Global trading {'paused' if paused else 'resumed'}: {reason}")
            else:
                # Pause/resume specific tickers
                for ticker in tickers:
                    data["paused_tickers"][ticker] = paused
                    logger.info(f"Ticker {ticker} {'paused' if paused else 'resumed'}: {reason}")

            # Save
            with open(self.control_file, 'w') as f:
                json.dump(data, f, indent=2)

            return True

        except Exception as e:
            logger.error(f"Failed to set pause state: {e}")
            return False

    def is_ticker_paused(self, ticker: str) -> bool:
        """Check if specific ticker is paused"""
        pause_state = self.get_pause_state()

        # Check global pause
        try:
            with open(self.control_file, 'r') as f:
                data = json.load(f)
                if data.get("global_pause", False):
                    return True
        except:
            pass

        return pause_state.get(ticker, False)

    def get_bot_config(self, strategy_id: str) -> Optional[Dict]:
        """
        Get current bot configuration from state file.

        Args:
            strategy_id: e.g., "SOLUSDT_15m_plus_1h_4h_long"

        Returns:
            Config dict or None if not found
        """
        state_file = self.bot_state_dir / f"{strategy_id}_state.json"

        if not state_file.exists():
            return None

        try:
            with open(state_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read bot config for {strategy_id}: {e}")
            return None

    def update_bot_config(self, strategy_id: str, updates: Dict) -> bool:
        """
        Update bot configuration in state file.

        NOTE: Bot must be restarted to pick up changes!

        Args:
            strategy_id: e.g., "SOLUSDT_15m_plus_1h_4h_long"
            updates: Dict with config updates

        Returns:
            True if successful
        """
        state_file = self.bot_state_dir / f"{strategy_id}_state.json"

        try:
            # Load existing state
            state = {}
            if state_file.exists():
                with open(state_file, 'r') as f:
                    state = json.load(f)

            # Update config section
            if "config" not in state:
                state["config"] = {}

            state["config"].update(updates)

            # Save
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)

            logger.info(f"Updated config for {strategy_id}: {updates}")
            return True

        except Exception as e:
            logger.error(f"Failed to update bot config: {e}")
            return False

    def get_all_bot_states(self) -> List[Dict]:
        """Get all bot state files"""
        states = []

        for state_file in self.bot_state_dir.glob("*_state.json"):
            # Skip control file
            if state_file == self.control_file:
                continue

            try:
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    state["strategy_id"] = state_file.stem.replace("_state", "")
                    states.append(state)
            except Exception as e:
                logger.error(f"Failed to read {state_file}: {e}")

        return states
