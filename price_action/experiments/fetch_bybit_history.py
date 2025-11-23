#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skrypt do pobierania pełnej historii pozycji z Bybit od 1 listopada 2025.

Pobiera:
- Closed P&L history (pozycje + PnL)
- Execution list (szczegółowe egzekucje)
- Order history (ordery + SL/TP)

Zapisuje CSV per ticker w katalogu /analiza
"""

import os
import sys
import pandas as pd
from datetime import datetime
from typing import List, Dict
import time
import logging

# Import bybit adapter
from bybit_adapter import BybitAdapter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)

# Date range
START_DATE = "2025-11-01"  # 1 listopada 2025
END_DATE = datetime.now().strftime("%Y-%m-%d")  # Dzisiaj

# Tickery (z docker-compose.yaml)
TICKERS = [
    "SOLUSDT",
    "DOGEUSDT",
    "ETHUSDT",
    "LINKUSDT",
    "HBARUSDT",
    "SUIUSDT",
    "NEARUSDT",
    "TRXUSDT",
    "KASUSDT",
    "1000PEPEUSDT",
    "XRPUSDT",
    "BRETTUSDT",
    "ICPUSDT",
    "ONDOUSDT",
    "PONKEUSDT",
    "WIFUSDT",
]

# Output directory
OUTPUT_DIR = "/Users/lukasz/projects/adaptive-bot/price_action/analiza"


def parse_date_to_ms(date_str: str) -> int:
    """Convert YYYY-MM-DD to milliseconds timestamp (start of day)."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return int(dt.timestamp() * 1000)


def fetch_all_closed_pnl(adapter: BybitAdapter, symbol: str, start_ms: int, end_ms: int) -> List[Dict]:
    """
    Fetch ALL closed P&L records for a symbol with pagination and 7-day chunking.

    Bybit API limits:
    - Max 100 records per request
    - Max 7 days time range per request
    """
    all_records = []

    log.info(f"  📊 Fetching closed P&L for {symbol}...")

    # Split into 7-day chunks (Bybit API limit)
    SEVEN_DAYS_MS = 7 * 24 * 60 * 60 * 1000
    current_start = start_ms
    chunk_num = 1

    while current_start < end_ms:
        current_end = min(current_start + SEVEN_DAYS_MS, end_ms)

        log.info(f"    Chunk {chunk_num}: {datetime.fromtimestamp(current_start/1000).strftime('%Y-%m-%d')} → {datetime.fromtimestamp(current_end/1000).strftime('%Y-%m-%d')}")

        # Paginate within this 7-day chunk
        cursor = None
        page = 1

        while True:
            try:
                params = {
                    "category": "linear",
                    "symbol": symbol,
                    "startTime": current_start,
                    "endTime": current_end,
                    "limit": 100
                }

                if cursor:
                    params["cursor"] = cursor

                response = adapter.client.get_closed_pnl(**params)

                if str(response.get("retCode")) != "0":
                    log.error(f"      ❌ API error: {response.get('retMsg')}")
                    break

                result = response.get("result", {})
                records = result.get("list", [])
                next_cursor = result.get("nextPageCursor")

                if not records:
                    break

                all_records.extend(records)
                log.info(f"      Page {page}: +{len(records)} records | Chunk total: {len([r for r in all_records if current_start <= int(r.get('createdTime', 0)) <= current_end])}")

                if not next_cursor or next_cursor == "":
                    break

                cursor = next_cursor
                page += 1
                time.sleep(0.2)

            except Exception as e:
                log.error(f"      ❌ Error: {e}")
                break

        current_start = current_end
        chunk_num += 1

    log.info(f"  ✓ Total closed P&L records: {len(all_records)}")
    return all_records


def fetch_all_executions(adapter: BybitAdapter, symbol: str, start_ms: int, end_ms: int) -> List[Dict]:
    """
    Fetch ALL execution records for a symbol with pagination and 7-day chunking.

    Bybit API limits:
    - Max 100 records per request
    - Max 7 days time range per request
    """
    all_records = []

    log.info(f"  📋 Fetching executions for {symbol}...")

    # Split into 7-day chunks
    SEVEN_DAYS_MS = 7 * 24 * 60 * 60 * 1000
    current_start = start_ms
    chunk_num = 1

    while current_start < end_ms:
        current_end = min(current_start + SEVEN_DAYS_MS, end_ms)

        log.info(f"    Chunk {chunk_num}: {datetime.fromtimestamp(current_start/1000).strftime('%Y-%m-%d')} → {datetime.fromtimestamp(current_end/1000).strftime('%Y-%m-%d')}")

        cursor = None
        page = 1

        while True:
            try:
                params = {
                    "category": "linear",
                    "symbol": symbol,
                    "startTime": current_start,
                    "endTime": current_end,
                    "limit": 100
                }

                if cursor:
                    params["cursor"] = cursor

                response = adapter.client.get_executions(**params)

                if str(response.get("retCode")) != "0":
                    log.error(f"      ❌ API error: {response.get('retMsg')}")
                    break

                result = response.get("result", {})
                records = result.get("list", [])
                next_cursor = result.get("nextPageCursor")

                if not records:
                    break

                all_records.extend(records)
                log.info(f"      Page {page}: +{len(records)} records")

                if not next_cursor or next_cursor == "":
                    break

                cursor = next_cursor
                page += 1
                time.sleep(0.2)

            except Exception as e:
                log.error(f"      ❌ Error: {e}")
                break

        current_start = current_end
        chunk_num += 1

    log.info(f"  ✓ Total execution records: {len(all_records)}")
    return all_records


def fetch_all_orders(adapter: BybitAdapter, symbol: str, start_ms: int, end_ms: int) -> List[Dict]:
    """
    Fetch ALL order history for a symbol with pagination and 7-day chunking.

    Bybit API limits:
    - Max 50 records per request
    - Max 7 days time range per request
    """
    all_records = []

    log.info(f"  📝 Fetching order history for {symbol}...")

    # Split into 7-day chunks
    SEVEN_DAYS_MS = 7 * 24 * 60 * 60 * 1000
    current_start = start_ms
    chunk_num = 1

    while current_start < end_ms:
        current_end = min(current_start + SEVEN_DAYS_MS, end_ms)

        log.info(f"    Chunk {chunk_num}: {datetime.fromtimestamp(current_start/1000).strftime('%Y-%m-%d')} → {datetime.fromtimestamp(current_end/1000).strftime('%Y-%m-%d')}")

        cursor = None
        page = 1

        while True:
            try:
                params = {
                    "category": "linear",
                    "symbol": symbol,
                    "startTime": current_start,
                    "endTime": current_end,
                    "limit": 50
                }

                if cursor:
                    params["cursor"] = cursor

                response = adapter.client.get_order_history(**params)

                if str(response.get("retCode")) != "0":
                    log.error(f"      ❌ API error: {response.get('retMsg')}")
                    break

                result = response.get("result", {})
                records = result.get("list", [])
                next_cursor = result.get("nextPageCursor")

                if not records:
                    break

                all_records.extend(records)
                log.info(f"      Page {page}: +{len(records)} records")

                if not next_cursor or next_cursor == "":
                    break

                cursor = next_cursor
                page += 1
                time.sleep(0.2)

            except Exception as e:
                log.error(f"      ❌ Error: {e}")
                break

        current_start = current_end
        chunk_num += 1

    log.info(f"  ✓ Total order records: {len(all_records)}")
    return all_records


def save_to_csv(data: List[Dict], filepath: str, data_type: str):
    """Save data to CSV with proper formatting."""
    if not data:
        log.warning(f"  ⚠️  No data to save for {data_type}")
        return

    df = pd.DataFrame(data)

    # Convert timestamp fields to readable dates
    timestamp_fields = ['createdTime', 'updatedTime', 'execTime', 'fundingRateTimestamp']
    for field in timestamp_fields:
        if field in df.columns:
            df[f'{field}_readable'] = pd.to_datetime(df[field], unit='ms').dt.strftime('%Y-%m-%d %H:%M:%S')

    # Sort by timestamp (most recent first)
    if 'createdTime' in df.columns:
        df = df.sort_values('createdTime', ascending=False)
    elif 'execTime' in df.columns:
        df = df.sort_values('execTime', ascending=False)

    # Save
    df.to_csv(filepath, index=False)
    log.info(f"  ✓ Saved {len(df)} records to: {filepath}")
    log.info(f"    Columns: {', '.join(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}")


def process_ticker(adapter: BybitAdapter, ticker: str, start_ms: int, end_ms: int):
    """Process all data for a single ticker."""
    log.info(f"\n{'='*60}")
    log.info(f"Processing: {ticker}")
    log.info(f"{'='*60}")

    # 1. Fetch closed P&L (main data)
    closed_pnl = fetch_all_closed_pnl(adapter, ticker, start_ms, end_ms)
    if closed_pnl:
        filepath = os.path.join(OUTPUT_DIR, f"{ticker}_closed_pnl.csv")
        save_to_csv(closed_pnl, filepath, "Closed P&L")

    # 2. Fetch executions (detailed fills)
    executions = fetch_all_executions(adapter, ticker, start_ms, end_ms)
    if executions:
        filepath = os.path.join(OUTPUT_DIR, f"{ticker}_executions.csv")
        save_to_csv(executions, filepath, "Executions")

    # 3. Fetch orders (with SL/TP info)
    orders = fetch_all_orders(adapter, ticker, start_ms, end_ms)
    if orders:
        filepath = os.path.join(OUTPUT_DIR, f"{ticker}_orders.csv")
        save_to_csv(orders, filepath, "Orders")

    log.info(f"\n✓ Completed {ticker}")


def main():
    """Main execution."""
    log.info("=" * 70)
    log.info("BYBIT HISTORY FETCHER - Pobieranie pełnej historii pozycji")
    log.info("=" * 70)
    log.info(f"Data range: {START_DATE} → {END_DATE}")
    log.info(f"Tickers: {len(TICKERS)}")
    log.info(f"Output dir: {OUTPUT_DIR}")
    log.info("")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log.info(f"✓ Created output directory: {OUTPUT_DIR}\n")

    # Load API credentials from .env_luk
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(__file__), '.env_luk')
    load_dotenv(env_path)

    api_key = os.getenv('BYBIT_API_KEY')
    api_secret = os.getenv('BYBIT_API_SECRET')
    base_url = os.getenv('BYBIT_BASE_URL', 'https://api.bybit.com')

    if not api_key or not api_secret:
        log.error("❌ Missing BYBIT_API_KEY or BYBIT_API_SECRET in .env_luk")
        sys.exit(1)

    log.info(f"✓ Loaded credentials from .env_luk")
    log.info(f"  Base URL: {base_url}\n")

    # Initialize adapter
    adapter = BybitAdapter(
        api_key=api_key,
        api_secret=api_secret,
        base_url=base_url,
        category="linear"
    )

    # Convert dates to milliseconds
    start_ms = parse_date_to_ms(START_DATE)
    end_ms = parse_date_to_ms(END_DATE) + (24 * 60 * 60 * 1000)  # End of day

    log.info(f"📅 Time range: {start_ms} → {end_ms}")
    log.info(f"   ({START_DATE} 00:00:00 → {END_DATE} 23:59:59)\n")

    # Process each ticker
    start_time = time.time()

    for i, ticker in enumerate(TICKERS, 1):
        log.info(f"\n[{i}/{len(TICKERS)}] Processing {ticker}...")
        try:
            process_ticker(adapter, ticker, start_ms, end_ms)
        except Exception as e:
            log.error(f"❌ Failed to process {ticker}: {e}", exc_info=True)
            continue

    # Summary
    elapsed = time.time() - start_time
    log.info("\n" + "=" * 70)
    log.info("✓ COMPLETED - All tickers processed")
    log.info("=" * 70)
    log.info(f"Total time: {elapsed:.1f}s")
    log.info(f"Output directory: {OUTPUT_DIR}")
    log.info(f"\nPliki CSV:")
    for ticker in TICKERS:
        pnl_file = os.path.join(OUTPUT_DIR, f"{ticker}_closed_pnl.csv")
        exec_file = os.path.join(OUTPUT_DIR, f"{ticker}_executions.csv")
        order_file = os.path.join(OUTPUT_DIR, f"{ticker}_orders.csv")

        if os.path.exists(pnl_file):
            pnl_size = len(pd.read_csv(pnl_file))
            log.info(f"  ✓ {ticker}_closed_pnl.csv ({pnl_size} records)")
        if os.path.exists(exec_file):
            exec_size = len(pd.read_csv(exec_file))
            log.info(f"  ✓ {ticker}_executions.csv ({exec_size} records)")
        if os.path.exists(order_file):
            order_size = len(pd.read_csv(order_file))
            log.info(f"  ✓ {ticker}_orders.csv ({order_size} records)")


if __name__ == "__main__":
    main()