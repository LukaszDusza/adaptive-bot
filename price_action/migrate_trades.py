"""
Trade Migration Script

Imports existing JSON trade logs into SQLite database for historical analysis.

Features:
- Batch processing with progress tracking
- Data validation and error handling
- Duplicate detection (skip existing trades)
- Statistics reporting

Usage:
    python migrate_trades.py [--logs-dir logs/trades] [--limit 100]
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from tqdm import tqdm

from database import get_database, get_repository, TradeModel
from trade_metrics import TradeMetricsCalculator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)


class TradeMigrator:
    """
    Migrates JSON trade logs to SQLite database.

    Features:
    - Parses JSON files
    - Extracts complete metrics
    - Validates data
    - Handles errors gracefully
    """

    def __init__(self, logs_dir: str = "logs/trades"):
        self.logs_dir = Path(logs_dir)
        self.repository = get_repository()

        self.stats = {
            "total_files": 0,
            "migrated": 0,
            "skipped": 0,
            "errors": 0
        }

    def migrate_all(self, limit: int = None):
        """
        Migrate all JSON trade files to database.

        Args:
            limit: Maximum number of trades to migrate (for testing)
        """
        log.info(f"Starting migration from: {self.logs_dir}")

        # Get all JSON files
        json_files = sorted(self.logs_dir.glob("*.json"))
        self.stats["total_files"] = len(json_files)

        if limit:
            json_files = json_files[:limit]
            log.info(f"Limiting to {limit} files")

        log.info(f"Found {len(json_files)} trade JSON files")

        # Process each file with progress bar
        for trade_file in tqdm(json_files, desc="Migrating trades"):
            try:
                self._migrate_trade_file(trade_file)
            except Exception as e:
                log.error(f"Failed to migrate {trade_file.name}: {e}")
                self.stats["errors"] += 1

        # Print summary
        self._print_summary()

    def _migrate_trade_file(self, trade_file: Path):
        """Migrate single trade JSON file"""
        # Load JSON
        with open(trade_file, 'r') as f:
            trade_data = json.load(f)

        trade_id = trade_data.get("trade_id")

        if not trade_id:
            log.warning(f"Skipping {trade_file.name}: No trade_id")
            self.stats["skipped"] += 1
            return

        # Check if already exists
        existing = self.repository.get_trade(trade_id)
        if existing:
            log.debug(f"Skipping {trade_id}: Already in database")
            self.stats["skipped"] += 1
            return

        # Parse trade data
        trade_dict = self._parse_trade_json(trade_data)

        if not trade_dict:
            log.warning(f"Skipping {trade_id}: Invalid data")
            self.stats["skipped"] += 1
            return

        # Create trade record
        self.repository.create_trade(trade_dict)
        log.debug(f"✓ Migrated trade: {trade_id}")

        # Migrate events
        self._migrate_events(trade_id, trade_data.get("events", []))

        # Migrate indicators
        self._migrate_indicators(trade_id, trade_data.get("indicators", {}))

        self.stats["migrated"] += 1

    def _parse_trade_json(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse JSON trade data into database format.

        Returns:
            Dictionary with trade attributes for database
        """
        try:
            summary = trade_data.get("summary", {})
            trade_id = trade_data.get("trade_id")
            ticker = trade_data.get("ticker")
            side = trade_data.get("side")

            # Parse timestamps
            start_time = self._parse_timestamp(trade_data.get("start_time"))
            end_time = self._parse_timestamp(trade_data.get("end_time"))

            if not all([trade_id, ticker, side, start_time]):
                return None

            # Extract metrics from summary
            # Handle both old format (total_pnl_usd) and new format (pnl_usd)
            pnl_usd = summary.get("pnl_usd") or summary.get("total_pnl_usd", 0.0)
            pnl_percent = summary.get("pnl_percent") or summary.get("total_pnl_pct", 0.0)

            # Build trade dict
            trade_dict = {
                "trade_id": trade_id,
                "ticker": ticker,
                "side": side,
                "start_time": start_time,
                "end_time": end_time,
                "duration_seconds": summary.get("duration_seconds"),

                # Entry/Exit
                "entry_price": summary.get("entry_price"),
                "exit_price": summary.get("exit_price"),
                "quantity": summary.get("quantity"),
                "leverage": summary.get("leverage"),

                # Risk Management
                "initial_sl": summary.get("initial_sl"),
                "initial_tp": summary.get("initial_tp"),
                "final_sl": summary.get("final_sl"),
                "final_tp": summary.get("final_tp"),

                # Performance
                "pnl_usd": pnl_usd,
                "pnl_percent": pnl_percent,
                "max_favorable_excursion": summary.get("max_favorable_excursion", 0.0),
                "max_adverse_excursion": summary.get("max_adverse_excursion", 0.0),
                "fees_paid": summary.get("fees_paid", 0.0),

                # Exit Info
                "exit_reason": summary.get("exit_reason") or self._extract_exit_reason(trade_data),

                # Status
                "is_active": end_time is None,
            }

            return trade_dict

        except Exception as e:
            log.error(f"Error parsing trade data: {e}")
            return None

    def _migrate_events(self, trade_id: str, events: List[Dict]):
        """Migrate trade events"""
        for event in events:
            try:
                event_type = event.get("type")
                event_data = event.get("data", {})

                if not event_type:
                    continue

                self.repository.add_event(trade_id, event_type, event_data)

            except Exception as e:
                log.error(f"Failed to migrate event: {e}")

    def _migrate_indicators(self, trade_id: str, indicators: Dict):
        """Migrate indicator snapshots"""
        for snapshot_type, snapshot_data in indicators.items():
            try:
                if not isinstance(snapshot_data, dict):
                    continue

                indicator_values = snapshot_data.get("indicators", {})
                model_probas = snapshot_data.get("model_probas", {})

                if not indicator_values:
                    continue

                self.repository.add_indicators(
                    trade_id,
                    snapshot_type,
                    indicator_values,
                    model_probas
                )

            except Exception as e:
                log.error(f"Failed to migrate indicators: {e}")

    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse ISO timestamp string"""
        if not timestamp_str:
            return None

        try:
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except Exception as e:
            log.warning(f"Failed to parse timestamp: {timestamp_str}: {e}")
            return None

    def _extract_exit_reason(self, trade_data: Dict) -> str:
        """
        Extract exit reason from events if not in summary.

        Fallback for old trade format.
        """
        events = trade_data.get("events", [])

        for event in reversed(events):  # Check from end
            event_type = event.get("type")
            if event_type == "EXIT":
                trigger = event.get("data", {}).get("trigger")
                if trigger:
                    return trigger.upper()

        # Default
        return "UNKNOWN"

    def _print_summary(self):
        """Print migration statistics"""
        log.info("\n" + "=" * 60)
        log.info("MIGRATION SUMMARY")
        log.info("=" * 60)
        log.info(f"Total files found:     {self.stats['total_files']}")
        log.info(f"Successfully migrated: {self.stats['migrated']}")
        log.info(f"Skipped (duplicates):  {self.stats['skipped']}")
        log.info(f"Errors:                {self.stats['errors']}")
        log.info("=" * 60)

        if self.stats["errors"] > 0:
            log.warning(f"⚠️  {self.stats['errors']} trades failed to migrate")

        if self.stats["migrated"] > 0:
            log.info(f"✅ {self.stats['migrated']} trades migrated successfully!")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Migrate JSON trade logs to SQLite database"
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default="logs/trades",
        help="Directory containing JSON trade logs (default: logs/trades)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of trades to migrate (for testing)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create migrator
    migrator = TradeMigrator(logs_dir=args.logs_dir)

    # Run migration
    try:
        migrator.migrate_all(limit=args.limit)
    except KeyboardInterrupt:
        log.info("\nMigration interrupted by user")
    except Exception as e:
        log.error(f"Migration failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
