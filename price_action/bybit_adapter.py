# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import time
import math
from typing import Any, Dict, List, Optional
import logging
import aiohttp
from urllib.parse import urlencode
import ssl
import certifi
import requests
from pybit.unified_trading import HTTP
from pybit.exceptions import InvalidRequestError
import pandas as pd

log = logging.getLogger("bybit")


class _DemoHTTP(HTTP):
    def __init__(self, api_key: str, api_secret: str, base_url: str, recv_window: int = 20000):
        super().__init__(api_key=api_key, api_secret=api_secret, testnet=False, recv_window=recv_window)
        base_url = base_url.rstrip("/")
        if hasattr(self, "endpoint"):
            self.endpoint = base_url
        try:
            setattr(self, "_domain", base_url)
        except Exception:
            pass
        log.info(f"Bybit HTTP endpoint ustawiony na: {base_url}")


def _norm_symbol(symbol: str) -> str:
    if ":" in symbol:
        symbol = symbol.split(":")[0]
    symbol = symbol.replace("/", "")
    return symbol


def _tf_to_interval(tf: str) -> str:
    VALID_INTERVALS = {
        '1m': '1', '3m': '3', '5m': '5', '15m': '15', '30m': '30',
        '1h': '60', '2h': '120', '4h': '240', '6h': '360', '12h': '720',
        '1D': 'D', '1d': 'D', 'd': 'D',
        '1W': 'W', '1w': 'W', 'w': 'W',
        '1M': 'M', '1mth': 'M', 'm': 'M', 'month': 'M'
    }

    tf_lower = tf.strip().lower().replace(' ', '')

    if tf_lower in VALID_INTERVALS:
        return VALID_INTERVALS[tf_lower]

    if tf_lower.isnumeric() and tf_lower in VALID_INTERVALS.values():
        return tf_lower

    raise ValueError(
        f"Nieprawidłowy lub nieobsługiwany przez Bybit timeframe: '{tf}'. "
        f"Dozwolone wartości to np.: {list(VALID_INTERVALS.keys())}"
    )


def _bar_ms(interval: str) -> int:
    if interval == "D":
        return 24 * 60 * 60 * 1000
    if interval == "W":
        return 7 * 24 * 60 * 60 * 1000
    if interval == "M": return 30 * 24 * 60 * 60 * 1000
    return int(interval) * 60 * 1000


class BybitAPIError(RuntimeError):
    pass


class BybitAdapter:

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: Optional[str] = None,
        category: str = "linear",
        recv_window: int = 20000,
        hedge_mode: bool = False,
    ) -> None:
        self.category = category
        self.recv_window = recv_window
        self.hedge_mode = hedge_mode

        if not api_key or not api_secret:
            raise ValueError("Brak kluczy API (BYBIT_API_KEY / BYBIT_API_SECRET).")

        if base_url:
            self.client = _DemoHTTP(api_key, api_secret, base_url, recv_window=recv_window)
            self.base_url = base_url.rstrip("/")
        else:
            self.client = HTTP(api_key=api_key, api_secret=api_secret, testnet=False, recv_window=recv_window)
            self.base_url = "https://api.bybit.com"

        self._lot_step: Dict[str, float] = {}
        self._tick_size: Dict[str, float] = {}

        try:
            self._prime_instrument_cache()
        except Exception as e:
            log.warning(f"Nie udało się wczytać specyfikacji instrumentów: {e}")

    def _get_position_idx(self, side: str) -> int:
        """
        Calculate positionIdx based on hedge_mode and side.
        
        One-Way Mode (hedge_mode=False): positionIdx = 0
        Hedge Mode (hedge_mode=True):
            - "Buy" or "Long" -> positionIdx = 1
            - "Sell" or "Short" -> positionIdx = 2
        """
        if not self.hedge_mode:
            return 0
        
        # Hedge mode: determine based on side
        if side in ("Buy", "Long"):
            return 1
        elif side in ("Sell", "Short"):
            return 2
        else:
            log.warning(f"Unknown side '{side}', defaulting to positionIdx=0")
            return 0

    def _prime_instrument_cache(self):
        log.info("Pobieranie specyfikacji instrumentów dla 'linear'...")
        try:
            resp = self.client.get_instruments_info(category=self.category)
            if str(resp.get("retCode")) == "0":
                instruments = (resp.get("result") or {}).get("list", [])
                for instrument in instruments:
                    symbol = instrument.get("symbol")
                    lot_filter = instrument.get("lotSizeFilter")
                    if symbol and lot_filter:
                        qty_step_str = lot_filter.get("qtyStep")
                        if qty_step_str:
                            self._lot_step[symbol] = float(qty_step_str)
                log.info(f"Wczytano specyfikację dla {len(self._lot_step)} instrumentów.")
            else:
                log.error(f"Błąd podczas pobierania specyfikacji instrumentów: {resp.get('retMsg')}")
        except Exception as e:
            log.error(f"Nie udało się pobrać specyfikacji instrumentów: {e}", exc_info=True)

    def round_qty(self, symbol: str, qty: float) -> float:
        """
        Round quantity to ticker-specific precision based on qtyStep.

        Examples:
        - qtyStep=0.01 → 2 decimals (SOLUSDT, ETHUSDT)
        - qtyStep=0.001 → 3 decimals (BTCUSDT)
        - qtyStep=0.1 → 1 decimal
        - qtyStep=1 → 0 decimals
        """
        step = self._lot_step.get(symbol, 0.01)  # Default 0.01 if not found

        if step >= 1:
            decimals = 0
        else:
            # Calculate decimals from step size
            # 0.01 → 2, 0.001 → 3, 0.0001 → 4
            decimals = abs(int(math.log10(step)))

        return round(qty, decimals)

    def get_min_order_qty(self, symbol: str) -> float:
        """
        Get minimum order quantity for ticker.
        Returns qtyStep as a proxy for minOrderQty.

        Typical values:
        - SOLUSDT: 0.01
        - BTCUSDT: 0.001
        - ETHUSDT: 0.01
        """
        return self._lot_step.get(symbol, 0.01)

    def get_position_size(self, symbol: str) -> float:
        try:
            pos_data = self.get_position(symbol)
            return float(pos_data.get('size', 0.0)) if pos_data else 0.0
        except (BybitAPIError, InvalidRequestError):
            return -1.0

    def _parse_end_date(self, date_str: str) -> int:
        """
        Parse date string (YYYY-MM-DD) to milliseconds timestamp.
        Sets time to end of day (23:59:59) to include full day's data.
        """
        from datetime import datetime
        try:
            # Parse date and set to end of day (23:59:59)
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            dt = dt.replace(hour=23, minute=59, second=59, microsecond=999000)
            return int(dt.timestamp() * 1000)
        except ValueError as e:
            raise ValueError(f"Invalid date format: {date_str}. Expected YYYY-MM-DD. Error: {e}")

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int, end_date: str = None, start_date: str = None, fetch_max: bool = False) -> List[List[Any]]:
        """
        Fetch OHLCV candle data from Bybit API.

        Args:
            symbol: Trading pair (e.g., SOLUSDT)
            timeframe: Candle interval (e.g., 15m, 1h, 4h)
            limit: Maximum number of candles to fetch (ignored if start_date and end_date both provided)
            end_date: End date for data (YYYY-MM-DD). If provided, fetches backwards from this date.
            start_date: Start date for data (YYYY-MM-DD). If provided with end_date, filters to exact date range.
            fetch_max: If True, fetches ALL available history from Bybit (ignores limit)

        Returns:
            List of OHLCV candles (sorted chronologically, oldest first)
        """
        sym = _norm_symbol(symbol)
        interval = _tf_to_interval(timeframe)
        step = _bar_ms(interval)
        out: List[List[Any]] = []

        # Use provided end_date or default to current time
        end = self._parse_end_date(end_date) if end_date else int(time.time() * 1000)

        # Track initial end for logging
        initial_end = end

        # If both start_date and end_date provided, calculate required candles for exact range
        # Otherwise use limit parameter
        if start_date and end_date:
            start_ms = self._parse_end_date(start_date)
            end_ms = self._parse_end_date(end_date)
            time_diff_ms = end_ms - start_ms
            required_candles = int(time_diff_ms / step) + 100  # +100 buffer for safety
            effective_limit = required_candles
            log.info(f"📅 Date range mode: {start_date} to {end_date}")
            log.info(f"   Calculated required candles: {required_candles}")
        elif fetch_max:
            effective_limit = 999999999
        else:
            effective_limit = limit

        # For fetch_max, set limit to a very high number and increase max_empty_tries
        max_empty_tries = 50 if fetch_max else 20

        tries = 0
        consecutive_empty = 0
        retries_with_backoff = 0
        max_retries = 5

        log.info(f"fetch_ohlcv: symbol={sym}, timeframe={timeframe}, limit={limit}, "
                 f"start_date={start_date}, end_date={end_date}, fetch_max={fetch_max}, effective_limit={effective_limit}")

        while len(out) < effective_limit and consecutive_empty < max_empty_tries:
            need = min(1000, effective_limit - len(out))  # Bybit API limit: 1000 candles per request
            start = end - need * step

            try:
                resp = self.client.get_kline(
                    category=self.category,
                    symbol=sym,
                    interval=interval,
                    start=start,
                    end=end,
                    limit=need,
                )
            except InvalidRequestError as e:
                error_msg = str(e)

                # Check for rate limit errors (10006 = rate limit exceeded)
                if "10006" in error_msg or "rate limit" in error_msg.lower():
                    if retries_with_backoff < max_retries:
                        backoff_time = 2 ** retries_with_backoff  # Exponential backoff: 1s, 2s, 4s, 8s, 16s
                        log.warning(f"Rate limit hit! Backing off for {backoff_time}s (retry {retries_with_backoff+1}/{max_retries})")
                        time.sleep(backoff_time)
                        retries_with_backoff += 1
                        continue
                    else:
                        raise BybitAPIError(f"Rate limit exceeded after {max_retries} retries: {e}")
                else:
                    raise BybitAPIError(f"fetch_ohlcv get_kline request error: {e}")

            if not isinstance(resp, dict) or str(resp.get("retCode")) not in ("0",):
                ret_code = resp.get("retCode") if isinstance(resp, dict) else "unknown"
                ret_msg = resp.get("retMsg") if isinstance(resp, dict) else str(resp)

                # Check for specific error codes
                if ret_code == "10001":  # Invalid parameter
                    log.error(f"Invalid parameter error: {ret_msg}")
                    raise BybitAPIError(f"Invalid parameter: {ret_msg}")
                elif ret_code == "10003":  # Invalid symbol
                    log.error(f"Invalid symbol: {sym}")
                    raise BybitAPIError(f"Invalid symbol: {sym}")
                else:
                    raise BybitAPIError(f"fetch_ohlcv -> retCode={ret_code}, retMsg={ret_msg}")

            data = (((resp.get("result") or {}).get("list")) or [])

            if not data:
                consecutive_empty += 1
                tries += 1

                if consecutive_empty >= 3:
                    # If we've hit 3 consecutive empty responses, we've likely reached the limit of available data
                    log.info(f"Reached limit of available data after {consecutive_empty} empty responses. "
                            f"Total candles fetched: {len(out)}")
                    break

                # Move back in time and try again
                end = start
                time.sleep(0.5)  # Longer sleep on empty response
                continue

            # Reset consecutive empty counter on successful fetch
            consecutive_empty = 0
            retries_with_backoff = 0

            try:
                data_sorted = sorted(data, key=lambda r: int(r[0]))
            except Exception:
                data_sorted = sorted(data, key=lambda r: int(r.get("start", 0)))

            # Prepend older data to the beginning (we're fetching backwards in time)
            out = data_sorted + out

            # Move end pointer to before the oldest candle we just fetched
            end = int(data_sorted[0][0]) - step
            tries = 0

            # Log progress every 10k candles
            if len(out) % 10000 == 0:
                log.info(f"Progress: {len(out)} candles fetched...")

            time.sleep(0.2)  # Rate limiting

        # No need to reverse - we prepend older batches to the beginning, so list is already sorted oldest → newest

        # Filter by date range if start_date provided
        if start_date:
            start_ms = self._parse_end_date(start_date)
            out_filtered = [candle for candle in out if int(candle[0]) >= start_ms]
            log.info(f"📅 Filtered by start_date {start_date}: {len(out)} → {len(out_filtered)} candles")
            out = out_filtered

        # If not fetch_max and not using date range, trim to requested limit (keep most recent candles)
        if not fetch_max and not (start_date and end_date) and len(out) > limit:
            out = out[-limit:]

        if not out:
            log.warning(
                f"fetch_ohlcv: pusta odpowiedź. symbol={sym}, interval={interval}, limit={limit}, base={self.base_url}"
            )
        else:
            # Calculate date range
            from datetime import datetime
            oldest_ts = int(out[0][0])
            newest_ts = int(out[-1][0])
            oldest_date = datetime.fromtimestamp(oldest_ts / 1000).strftime('%Y-%m-%d %H:%M:%S')
            newest_date = datetime.fromtimestamp(newest_ts / 1000).strftime('%Y-%m-%d %H:%M:%S')

            log.info(f"✓ Successfully fetched {len(out)} candles for {sym} {timeframe}")
            log.info(f"  Date range: {oldest_date} → {newest_date}")
            log.info(f"  Requested limit: {limit}, Fetched: {len(out)}")

        return out

    def get_balance(self, use_available: bool = False) -> float:
        try:
            resp = self.client.get_wallet_balance(accountType="UNIFIED", coin="USDT")
            if str(resp.get("retCode")) not in ("0",):
                raise BybitAPIError(f"get_wallet_balance -> {resp}")
            rows = (resp.get("result") or {}).get("list") or []
            if not rows:
                return 0.0
            coins = rows[0].get("coin") or []
            for c in coins:
                if c.get("coin") == "USDT":
                    return float(c.get("availableToWithdraw" if use_available else "equity", 0.0))
            return 0.0
        except Exception as e:
            log.warning(f"get_balance fail: {e}")
            return 0.0

    def set_leverage(self, symbol_u: str, leverage: float) -> None:
        try:
            r = self.client.set_leverage(
                category=self.category,
                symbol=symbol_u,
                buyLeverage=str(leverage),
                sellLeverage=str(leverage),
            )
            rc = str(r.get("retCode"))
            if rc == "110043":
                log.info("Leverage already set (110043).")
            elif rc != "0":
                raise BybitAPIError(f"set_leverage -> {r}")
        except InvalidRequestError as e:
            msg = str(e)
            if "110043" in msg and "not modified" in msg:
                log.info("Leverage already set.")
            else:
                raise

    def _safe_parse_price(self, value, field_name: str = "price") -> float:
        """
        Safely parse price from Bybit API response.

        Handles:
        - None, empty strings, "0", "0.0" → returns 0.0
        - Valid numeric strings → returns float
        - Invalid formats → logs warning and returns 0.0

        This fixes the bug where empty strings "" from API were treated as 0
        due to `p.get("stopLoss", 0) or 0` returning 0 for empty strings.
        """
        if value is None or value == "":
            return 0.0

        try:
            parsed = float(value)
            # Treat 0 as "not set"
            return parsed if parsed != 0.0 else 0.0
        except (ValueError, TypeError) as e:
            log.warning(f"Failed to parse {field_name}='{value}': {e}")
            return 0.0

    def get_position(self, symbol_u: str) -> Dict[str, Any]:
        resp = self.client.get_positions(category=self.category, symbol=symbol_u)
        if str(resp.get("retCode")) not in ("0",):
            raise BybitAPIError(f"get_positions -> {resp}")
        arr = (resp.get("result") or {}).get("list") or []
        for p in arr:
            sz = float(p.get("size", 0) or 0)
            if abs(sz) > 0:
                side = p.get("side")
                # FIX: Handle None/empty side (Bybit can return None for classic accounts)
                if not side or side not in ("Buy", "Sell"):
                    log.warning(f"Invalid position side: {side}, skipping position")
                    continue
                entry = float(p.get("avgPrice", 0) or 0)

                # FIX: Safely parse SL/TP (handles empty strings from API)
                stop_loss = self._safe_parse_price(p.get("stopLoss"), "stopLoss")
                take_profit = self._safe_parse_price(p.get("takeProfit"), "takeProfit")

                # DEBUG: Log raw API values for SL/TP detection
                log.debug(f"Position {symbol_u}: raw stopLoss={repr(p.get('stopLoss'))}, "
                         f"raw takeProfit={repr(p.get('takeProfit'))}, "
                         f"parsed SL={stop_loss}, parsed TP={take_profit}")

                return {
                    "side": "Long" if side == "Buy" else "Short",
                    "size": sz,
                    "entryPrice": entry,
                    "stopLoss": stop_loss,
                    "takeProfit": take_profit
                }
        return {}

    def latest_price(self, symbol_u: str) -> float:
        resp = self.client.get_tickers(category=self.category, symbol=symbol_u)
        if str(resp.get("retCode")) not in ("0",):
            raise BybitAPIError(f"get_tickers -> {resp}")
        arr = (resp.get("result") or {}).get("list") or []
        if not arr:
            return 0.0
        return float(arr[0].get("lastPrice", 0) or 0)

    def _format_qty(self, qty: float) -> str:
        """Format quantity as integer string if whole number, otherwise with decimals."""
        if qty == int(qty):
            return str(int(qty))
        return str(qty)

    def market_open(self, symbol_u: str, side: str, qty: float):
        # Minimum order quantities (fallback when rounding produces 0)
        MIN_ORDER_QTY = {
            'SOLUSDT': 0.1,
            'BTCUSDT': 0.001,
            'ETHUSDT': 0.01,
            'DOGEUSDT': 1.0,
            '1000PEPEUSDT': 1000.0,
        }

        # Try 2 decimal places first
        safe_qty = round(qty, 2)
        if safe_qty <= 0:
            safe_qty = MIN_ORDER_QTY.get(symbol_u, 0.1)
            log.warning(f"⚠️ Qty rounded to 0, using minimum: {safe_qty} for {symbol_u}")

        position_idx = self._get_position_idx(side)
        log.info(f"Próba otwarcia pozycji: {qty} zaokrąglona do {safe_qty} (2 miejsca po przecinku) dla {symbol_u} (positionIdx={position_idx})")

        try:
            resp = self.client.place_order(
                category=self.category,
                symbol=symbol_u,
                side=side,
                orderType="Market",
                qty=self._format_qty(safe_qty), timeInForce="IOC", reduceOnly=False, positionIdx=position_idx,
            )
            if str(resp.get("retCode")) not in ("0",):
                raise BybitAPIError(f"place_order -> {resp}")
            return resp
        except (InvalidRequestError, BybitAPIError) as e:
            error_msg = str(e)
            # If "Qty invalid" error, retry with 1 decimal place
            if "Qty invalid" in error_msg or "10001" in error_msg:
                safe_qty = round(qty, 1)
                if safe_qty <= 0:
                    safe_qty = MIN_ORDER_QTY.get(symbol_u, 0.1)
                    log.warning(f"⚠️ Qty rounded to 0 (1 decimal), using minimum: {safe_qty} for {symbol_u}")
                else:
                    log.warning(f"Qty invalid z 2 miejscami, ponawiam z 1 miejscem: {safe_qty} dla {symbol_u}")
                try:
                    resp = self.client.place_order(
                        category=self.category,
                        symbol=symbol_u,
                        side=side,
                        orderType="Market",
                        qty=self._format_qty(safe_qty), timeInForce="IOC", reduceOnly=False, positionIdx=position_idx,
                    )
                    if str(resp.get("retCode")) not in ("0",):
                        raise BybitAPIError(f"place_order -> {resp}")
                    return resp
                except (InvalidRequestError, BybitAPIError) as e2:
                    error_msg2 = str(e2)
                    # If still "Qty invalid", retry with integer (0 decimal places) - needed for DOGEUSDT, etc.
                    if "Qty invalid" in error_msg2 or "10001" in error_msg2:
                        safe_qty = int(qty)
                        if safe_qty <= 0:
                            safe_qty = MIN_ORDER_QTY.get(symbol_u, 1.0)
                            log.warning(f"⚠️ Qty rounded to 0 (integer), using minimum: {safe_qty} for {symbol_u}")
                        else:
                            log.warning(f"Qty invalid z 1 miejscem, ponawiam z liczbą całkowitą: {safe_qty} dla {symbol_u}")
                        resp = self.client.place_order(
                            category=self.category,
                            symbol=symbol_u,
                            side=side,
                            orderType="Market",
                            qty=self._format_qty(safe_qty), timeInForce="IOC", reduceOnly=False, positionIdx=position_idx,
                        )
                        if str(resp.get("retCode")) not in ("0",):
                            raise BybitAPIError(f"place_order -> {resp}")
                        return resp
                    else:
                        raise
            else:
                raise

    def market_close(self, symbol_u: str, side: str, qty: float, position_side: str = None):
        """
        Close position with market order (reduceOnly).

        Args:
            symbol_u: Trading pair
            side: Order side ("Buy" to close SHORT, "Sell" to close LONG)
            qty: Quantity to close
            position_side: Position side ("Long" or "Short") - REQUIRED for hedge mode to get correct positionIdx
        """
        # Try 2 decimal places first
        safe_qty = round(qty, 2)

        # CRITICAL FIX: For reduceOnly orders in hedge mode, use position side (not order side) to get positionIdx
        # Example: Closing SHORT position requires positionIdx=2, even though order side is "Buy"
        if position_side:
            position_idx = self._get_position_idx(position_side)
        else:
            # Fallback for backward compatibility (one-way mode)
            position_idx = self._get_position_idx(side)

        log.info(f"Próba zamknięcia pozycji: {qty} zaokrąglona do {safe_qty} (2 miejsca po przecinku) dla {symbol_u} (positionIdx={position_idx}, position_side={position_side})")

        try:
            resp = self.client.place_order(
                category=self.category,
                symbol=symbol_u,
                side=side,
                orderType="Market",
                qty=self._format_qty(safe_qty), reduceOnly=True, timeInForce="IOC", positionIdx=position_idx,
            )
            if str(resp.get("retCode")) not in ("0",):
                raise BybitAPIError(f"close_position (partial) -> {resp}")
            return resp
        except (InvalidRequestError, BybitAPIError) as e:
            error_msg = str(e)
            # If "Qty invalid" error, retry with 1 decimal place
            if "Qty invalid" in error_msg or "10001" in error_msg:
                safe_qty = round(qty, 1)
                log.warning(f"Qty invalid z 2 miejscami, ponawiam z 1 miejscem: {safe_qty} dla {symbol_u}")
                try:
                    resp = self.client.place_order(
                        category=self.category,
                        symbol=symbol_u,
                        side=side,
                        orderType="Market",
                        qty=self._format_qty(safe_qty), reduceOnly=True, timeInForce="IOC", positionIdx=position_idx,
                    )
                    if str(resp.get("retCode")) not in ("0",):
                        raise BybitAPIError(f"close_position (partial) -> {resp}")
                    return resp
                except (InvalidRequestError, BybitAPIError) as e2:
                    error_msg2 = str(e2)
                    # If still "Qty invalid", retry with integer (0 decimal places) - needed for DOGEUSDT, etc.
                    if "Qty invalid" in error_msg2 or "10001" in error_msg2:
                        safe_qty = int(qty)
                        log.warning(f"Qty invalid z 1 miejscem, ponawiam z liczbą całkowitą: {safe_qty} dla {symbol_u}")
                        resp = self.client.place_order(
                            category=self.category,
                            symbol=symbol_u,
                            side=side,
                            orderType="Market",
                            qty=self._format_qty(safe_qty), reduceOnly=True, timeInForce="IOC", positionIdx=position_idx,
                        )
                        if str(resp.get("retCode")) not in ("0",):
                            raise BybitAPIError(f"close_position (partial) -> {resp}")
                        return resp
                    else:
                        raise
            else:
                raise

    def limit_open(self, symbol_u: str, side: str, qty: float, limit_price: float, stop_loss: float = None, take_profit: float = None):
        """
        Place a limit order to open a position.

        Args:
            symbol_u: Trading pair (e.g., SOLUSDT)
            side: "Buy" for LONG, "Sell" for SHORT
            qty: Order quantity
            limit_price: Limit price for the order
            stop_loss: Optional stop loss price (activated when order fills)
            take_profit: Optional take profit price (activated when order fills)

        Returns:
            API response with orderId
        """
        # Minimum order quantities (fallback when rounding produces 0)
        MIN_ORDER_QTY = {
            'SOLUSDT': 0.1,
            'BTCUSDT': 0.001,
            'ETHUSDT': 0.01,
            'DOGEUSDT': 1.0,
            '1000PEPEUSDT': 1000.0,
        }

        safe_qty = round(qty, 2)
        if safe_qty <= 0:
            safe_qty = MIN_ORDER_QTY.get(symbol_u, 0.1)
            log.warning(f"⚠️ Qty rounded to 0, using minimum: {safe_qty} for {symbol_u}")

        position_idx = self._get_position_idx(side)

        sl_tp_info = ""
        if stop_loss:
            sl_tp_info += f" | SL: {stop_loss:.4f}"
        if take_profit:
            sl_tp_info += f" | TP: {take_profit:.4f}"

        log.info(f"Placing LIMIT order: {side} {safe_qty} @ {limit_price:.4f}{sl_tp_info} for {symbol_u} (positionIdx={position_idx})")

        # Build order parameters
        order_params = {
            "category": self.category,
            "symbol": symbol_u,
            "side": side,
            "orderType": "Limit",
            "qty": self._format_qty(safe_qty),
            "price": str(limit_price),
            "timeInForce": "GTC",  # Good Till Cancel
            "reduceOnly": False,
            "positionIdx": position_idx,
        }

        # Add SL/TP if provided (Bybit will activate them when order fills)
        if stop_loss:
            order_params["stopLoss"] = str(stop_loss)
            order_params["slTriggerBy"] = "MarkPrice"  # Use MarkPrice to prevent manipulation
        if take_profit:
            order_params["takeProfit"] = str(take_profit)
            order_params["tpTriggerBy"] = "MarkPrice"

        try:
            resp = self.client.place_order(**order_params)
            if str(resp.get("retCode")) not in ("0",):
                raise BybitAPIError(f"place_order (LIMIT) -> {resp}")

            order_id = (resp.get("result") or {}).get("orderId")
            log.info(f"✓ Limit order placed: ID={order_id}")
            return resp

        except (InvalidRequestError, BybitAPIError) as e:
            error_msg = str(e)
            # If "Qty invalid" error, retry with 1 decimal place
            if "Qty invalid" in error_msg or "10001" in error_msg:
                safe_qty = round(qty, 1)
                if safe_qty <= 0:
                    safe_qty = MIN_ORDER_QTY.get(symbol_u, 0.1)
                    log.warning(f"⚠️ Qty rounded to 0 (1 decimal), using minimum: {safe_qty} for {symbol_u}")
                else:
                    log.warning(f"Qty invalid z 2 miejscami, ponawiam z 1 miejscem: {safe_qty} dla {symbol_u}")
                try:
                    order_params["qty"] = self._format_qty(safe_qty)
                    resp = self.client.place_order(**order_params)
                    if str(resp.get("retCode")) not in ("0",):
                        raise BybitAPIError(f"place_order (LIMIT) -> {resp}")

                    order_id = (resp.get("result") or {}).get("orderId")
                    log.info(f"✓ Limit order placed: ID={order_id}")
                    return resp
                except (InvalidRequestError, BybitAPIError) as e2:
                    error_msg2 = str(e2)
                    # If still "Qty invalid", retry with integer (0 decimal places) - needed for DOGEUSDT, etc.
                    if "Qty invalid" in error_msg2 or "10001" in error_msg2:
                        safe_qty = int(qty)
                        if safe_qty <= 0:
                            safe_qty = MIN_ORDER_QTY.get(symbol_u, 1.0)
                            log.warning(f"⚠️ Qty rounded to 0 (integer), using minimum: {safe_qty} for {symbol_u}")
                        else:
                            log.warning(f"Qty invalid z 1 miejscem, ponawiam z liczbą całkowitą: {safe_qty} dla {symbol_u}")
                        order_params["qty"] = self._format_qty(safe_qty)
                        resp = self.client.place_order(**order_params)
                        if str(resp.get("retCode")) not in ("0",):
                            raise BybitAPIError(f"place_order (LIMIT) -> {resp}")

                        order_id = (resp.get("result") or {}).get("orderId")
                        log.info(f"✓ Limit order placed: ID={order_id}")
                        return resp
                    else:
                        raise
            else:
                raise

    def get_open_orders(self, symbol_u: str = None) -> list:
        """
        Get all open (unfilled) orders for a symbol, or all symbols if None.

        Args:
            symbol_u: Trading pair (optional). If None, returns ALL open orders.

        Returns:
            List of open orders with orderId, side, price, qty, etc.
        """
        try:
            params = {"category": self.category}
            if symbol_u:
                params["symbol"] = symbol_u

            resp = self.client.get_open_orders(**params)
            if str(resp.get("retCode")) not in ("0",):
                raise BybitAPIError(f"get_open_orders -> {resp}")

            orders = (resp.get("result") or {}).get("list", [])
            return orders

        except Exception as e:
            log.error(f"Error getting open orders: {e}", exc_info=True)
            return []

    def get_open_orders_by_settle_coin(self, settle_coin: str = "USDT") -> list:
        """
        Get all open orders by settlement coin (e.g., USDT for all USDT perpetuals).

        Args:
            settle_coin: Settlement coin (default: USDT)

        Returns:
            List of open orders
        """
        try:
            params = {
                "category": self.category,
                "settleCoin": settle_coin
            }

            resp = self.client.get_open_orders(**params)
            if str(resp.get("retCode")) not in ("0",):
                raise BybitAPIError(f"get_open_orders_by_settle_coin -> {resp}")

            orders = (resp.get("result") or {}).get("list", [])
            log.info(f"✓ Retrieved {len(orders)} open orders for settleCoin={settle_coin}")
            return orders

        except Exception as e:
            log.error(f"Error getting open orders by settleCoin: {e}", exc_info=True)
            return []

    def cancel_order(self, symbol_u: str, order_id: str):
        """
        Cancel a specific order by orderId.

        Args:
            symbol_u: Trading pair
            order_id: Order ID to cancel
        """
        try:
            resp = self.client.cancel_order(
                category=self.category,
                symbol=symbol_u,
                orderId=order_id
            )
            if str(resp.get("retCode")) not in ("0",):
                raise BybitAPIError(f"cancel_order -> {resp}")

            log.info(f"✓ Order cancelled: {order_id}")
            return resp

        except Exception as e:
            log.error(f"Error cancelling order {order_id}: {e}", exc_info=True)
            raise

    def cancel_all_orders(self, symbol_u: str):
        """
        Cancel all open orders for a symbol (safety mechanism).
        """
        try:
            resp = self.client.cancel_all_orders(
                category=self.category,
                symbol=symbol_u
            )
            if str(resp.get("retCode")) not in ("0",):
                raise BybitAPIError(f"cancel_all_orders -> {resp}")

            log.info(f"✓ All orders cancelled for {symbol_u}")
            return resp

        except Exception as e:
            log.error(f"Error cancelling all orders: {e}", exc_info=True)
            raise

    def get_available_balance_usd(self) -> float:
        """
        Get available balance in USD for Unified Trading Account.

        Returns:
            Available balance in USD (totalAvailableBalance for cross margin)
            Returns 0.0 on error.
        """
        try:
            resp = self.client.get_wallet_balance(accountType="UNIFIED")

            if str(resp.get("retCode")) not in ("0",):
                log.error(f"get_wallet_balance error: {resp}")
                return 0.0

            # Extract accounts from result
            result = resp.get("result") or {}
            accounts = result.get("list", [])

            if not accounts:
                log.warning("No unified account found in wallet balance response")
                return 0.0

            first_account = accounts[0]

            # Try multiple possible balance fields
            balance_usd = 0.0

            # Strategy 1: Try totalAvailableBalance (but Bybit may return empty string "")
            total_avail = first_account.get("totalAvailableBalance", "")
            if total_avail and str(total_avail).strip():
                try:
                    balance_usd = float(total_avail)
                    log.info(f"✓ Using totalAvailableBalance: ${balance_usd:.2f}")
                except (ValueError, TypeError):
                    pass

            # Strategy 2: If still zero, try USDT coin equity (most reliable!)
            if balance_usd == 0.0:
                coins = first_account.get("coin", [])
                log.info(f"🔍 Searching USDT in {len(coins)} coins...")

                for coin in coins:
                    if coin.get("coin") == "USDT":
                        # Get all possible balance fields
                        equity = coin.get("equity", "")
                        wallet_balance = coin.get("walletBalance", "")
                        available_withdraw = coin.get("availableToWithdraw", "")

                        # FIX: Try availableToWithdraw FIRST (free funds, excludes positions)
                        if available_withdraw and str(available_withdraw).strip():
                            try:
                                balance_usd = float(available_withdraw)
                                log.info(f"✓ Using USDT availableToWithdraw: ${balance_usd:.2f} (free funds)")
                                break
                            except (ValueError, TypeError):
                                pass

                        # Fallback to equity (total balance including positions)
                        if balance_usd == 0.0 and equity and str(equity).strip():
                            try:
                                balance_usd = float(equity)
                                log.warning(f"⚠️  Using USDT equity: ${balance_usd:.2f} (may include locked funds in positions)")
                                break
                            except (ValueError, TypeError):
                                pass

                        # Last resort: walletBalance
                        if balance_usd == 0.0 and wallet_balance and str(wallet_balance).strip():
                            try:
                                balance_usd = float(wallet_balance)
                                log.warning(f"⚠️  Using USDT walletBalance: ${balance_usd:.2f}")
                                break
                            except (ValueError, TypeError):
                                pass

            # If still zero, log warning
            if balance_usd == 0.0:
                log.warning(f"⚠️ Could not find USDT balance in account")
                log.warning(f"⚠️ Available coins: {[c.get('coin') for c in first_account.get('coin', [])]}")

            log.info(f"💰 Unified Wallet Available Balance: ${balance_usd:.2f}")
            return balance_usd

        except Exception as e:
            log.error(f"Error getting wallet balance: {e}", exc_info=True)
            return 0.0

    def get_active_positions_count(self) -> int:
        """
        Get count of currently active positions (size > 0) for ALL symbols.

        Returns:
            Number of active positions across all symbols
            Returns 0 on error.
        """
        try:
            # Get all positions for linear (USDT perpetuals)
            resp = self.client.get_positions(
                category="linear",
                settleCoin="USDT"
            )

            if str(resp.get("retCode")) not in ("0",):
                log.error(f"get_positions error: {resp}")
                return 0

            result = resp.get("result") or {}
            positions = result.get("list", [])

            # Count positions with size > 0
            active_count = 0
            active_symbols = []

            for pos in positions:
                size_str = pos.get("size", "0")
                symbol = pos.get("symbol", "UNKNOWN")
                side = pos.get("side", "")

                try:
                    size = float(size_str)
                    if size > 0:
                        active_count += 1
                        active_symbols.append(f"{symbol}({side})")
                except (ValueError, TypeError):
                    continue

            if active_count > 0:
                log.info(f"📊 Active positions: {active_count} - {', '.join(active_symbols)}")
            else:
                log.info(f"📊 Active positions: 0 (no open positions)")

            return active_count

        except Exception as e:
            log.error(f"Error getting active positions count: {e}", exc_info=True)
            return 0

    def set_stop_loss(self, symbol_u: str, price: float, side: str):
        position_idx = self._get_position_idx(side)
        try:
            resp = self.client.set_trading_stop(
                category=self.category,
                symbol=symbol_u,
                stopLoss=str(price),
                tpslMode="Full",  # Required by API - applies to entire position
                slTriggerBy="MarkPrice",  # Use MarkPrice to prevent manipulation
                positionIdx=position_idx,
            )
            if str(resp.get("retCode")) not in ("0",):
                raise BybitAPIError(f"set_trading_stop (SL) -> {resp}")
            return resp
        except (InvalidRequestError, BybitAPIError) as e:
            error_msg = str(e)
            # Error 34040 = "not modified" - SL already at this price, treat as warning not error
            if "34040" in error_msg or "not modified" in error_msg.lower():
                log.warning(f"SL already set to {price:.4f} for {symbol_u} (34040 - not modified)")
                return {"retCode": "0", "retMsg": "not_modified"}
            else:
                raise

    def set_take_profit(self, symbol_u: str, price: float, side: str):
        position_idx = self._get_position_idx(side)
        try:
            resp = self.client.set_trading_stop(
                category=self.category,
                symbol=symbol_u,
                takeProfit=str(price),
                tpslMode="Full",  # Required by API - applies to entire position
                tpTriggerBy="MarkPrice",  # Use MarkPrice to prevent manipulation
                positionIdx=position_idx,
            )
            if str(resp.get("retCode")) not in ("0",):
                raise BybitAPIError(f"set_trading_stop (TP) -> {resp}")
            return resp
        except (InvalidRequestError, BybitAPIError) as e:
            error_msg = str(e)
            # Error 34040 = "not modified" - TP already at this price, treat as warning not error
            if "34040" in error_msg or "not modified" in error_msg.lower():
                log.warning(f"TP already set to {price:.4f} for {symbol_u} (34040 - not modified)")
                return {"retCode": "0", "retMsg": "not_modified"}
            else:
                raise

    def cancel_tpsl(self, symbol_u: str):
        # Get position to determine correct positionIdx
        pos = self.get_position(symbol_u)
        if pos:
            side = pos["side"]  # "Long" or "Short"
            position_idx = self._get_position_idx(side)
        else:
            # No position, use default
            position_idx = 0
        
        resp = self.client.set_trading_stop(
            category=self.category,
            symbol=symbol_u,
            takeProfit="0",
            stopLoss="0",
            tpslMode="Full",  # Required by API
            positionIdx=position_idx,
        )
        if str(resp.get("retCode")) not in ("0",):
            raise BybitAPIError(f"cancel_tpsl -> {resp}")
        return resp

    def close_position(self, symbol_u: str):
        pos = self.get_position(symbol_u)
        if not pos:
            return
        side = "Sell" if pos["side"] == "Long" else "Buy"
        # FIX: Use position side (not order side) to get correct positionIdx in hedge mode
        position_idx = self._get_position_idx(pos["side"])
        qty = abs(float(pos.get("size", 0)))
        if qty <= 0: return
        
        # Try 2 decimal places first
        safe_qty = round(qty, 2)
        log.info(f"Próba zamknięcia całej pozycji: {qty} zaokrąglona do {safe_qty} (2 miejsca po przecinku) dla {symbol_u} (positionIdx={position_idx})")
        
        try:
            resp = self.client.place_order(
                category=self.category,
                symbol=symbol_u,
                side=side,
                orderType="Market",
                qty=self._format_qty(safe_qty), reduceOnly=True, timeInForce="IOC", positionIdx=position_idx,
            )
            if str(resp.get("retCode")) not in ("0",):
                raise BybitAPIError(f"close_position -> {resp}")
            return resp
        except (InvalidRequestError, BybitAPIError) as e:
            error_msg = str(e)
            # If "Qty invalid" error, retry with 1 decimal place
            if "Qty invalid" in error_msg or "10001" in error_msg:
                safe_qty = round(qty, 1)
                log.warning(f"Qty invalid z 2 miejscami, ponawiam z 1 miejscem: {safe_qty} dla {symbol_u}")
                try:
                    resp = self.client.place_order(
                        category=self.category,
                        symbol=symbol_u,
                        side=side,
                        orderType="Market",
                        qty=self._format_qty(safe_qty), reduceOnly=True, timeInForce="IOC", positionIdx=position_idx,
                    )
                    if str(resp.get("retCode")) not in ("0",):
                        raise BybitAPIError(f"close_position -> {resp}")
                    return resp
                except (InvalidRequestError, BybitAPIError) as e2:
                    error_msg2 = str(e2)
                    # If still "Qty invalid", retry with integer (0 decimal places) - needed for DOGEUSDT, etc.
                    if "Qty invalid" in error_msg2 or "10001" in error_msg2:
                        safe_qty = int(qty)
                        log.warning(f"Qty invalid z 1 miejscem, ponawiam z liczbą całkowitą: {safe_qty} dla {symbol_u}")
                        resp = self.client.place_order(
                            category=self.category,
                            symbol=symbol_u,
                            side=side,
                            orderType="Market",
                            qty=self._format_qty(safe_qty), reduceOnly=True, timeInForce="IOC", positionIdx=position_idx,
                        )
                        if str(resp.get("retCode")) not in ("0",):
                            raise BybitAPIError(f"close_position -> {resp}")
                        return resp
                    else:
                        raise
            else:
                raise

    async def fetch_historical_liquidations_async(self, symbol: str, start_ms: int, end_ms: int) -> List[Dict]:
        log.info(f"Pobieranie historii likwidacji dla {symbol}...")
        params = {
            "category": self.category,
            "symbol": _norm_symbol(symbol),
            "startTime": start_ms, "endTime": end_ms, "limit": 1000
        }
        return await self._fetch_paginated_data_async("/v5/market/liquidations", params)

    async def _fetch_paginated_data_async(self, endpoint: str, params: Dict) -> List[Dict]:
        all_data = []
        full_url_template = f"{self.base_url}{endpoint}?"
        ssl_context = ssl.create_default_context(cafile=certifi.where())

        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
            while True:
                query_string = urlencode(params)
                try:
                    async with session.get(full_url_template + query_string) as response:
                        if response.status != 200:
                            log.error(f"Błąd HTTP {response.status} dla {endpoint}")
                            await asyncio.sleep(5)
                            continue

                        resp_json = await response.json()

                        if str(resp_json.get("retCode")) != "0":
                            log.error(f"Błąd API Bybit dla {endpoint}: {resp_json.get('retMsg')}")
                            break

                        result = resp_json.get("result", {})
                        data_list = result.get("list", [])

                        if not data_list:
                            break

                        all_data.extend(data_list)

                        next_page_cursor = result.get("nextPageCursor")
                        if not next_page_cursor:
                            break

                        params["cursor"] = next_page_cursor
                        await asyncio.sleep(0.2)

                except aiohttp.ClientError as e:
                    log.error(f"Błąd połączenia aiohttp: {e}")
                    await asyncio.sleep(5)
                    continue

        return all_data

    async def fetch_historical_funding_rate_async(self, symbol: str, start_ms: int, end_ms: int) -> List[Dict]:
        log.info(f"Pobieranie historii Funding Rate dla {symbol}...")
        params = {
            "category": self.category,
            "symbol": _norm_symbol(symbol),
            "startTime": start_ms, "endTime": end_ms, "limit": 200
        }
        all_data = []
        current_start = start_ms
        ssl_context = ssl.create_default_context(cafile=certifi.where())

        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
            while current_start < end_ms:
                params["startTime"] = current_start
                query_string = urlencode(params)

                try:
                    async with session.get(f"{self.base_url}/v5/market/funding/history?" + query_string) as response:
                        if response.status != 200:
                            log.error(f"Błąd HTTP {response.status} dla funding/history")
                            await asyncio.sleep(5)
                            continue

                        resp_json = await response.json()
                        if str(resp_json.get("retCode")) != "0":
                            log.error(f"Błąd API Bybit dla funding/history: {resp_json.get('retMsg')}")
                            break

                        data_list = resp_json.get("result", {}).get("list", [])
                        if not data_list:
                            break

                        all_data.extend(data_list)
                        last_timestamp = int(data_list[-1]["fundingRateTimestamp"])
                        current_start = last_timestamp + 1
                        await asyncio.sleep(0.2)

                except aiohttp.ClientError as e:
                    log.error(f"Błąd połączenia aiohttp: {e}")
                    await asyncio.sleep(5)
                    continue

        return all_data

    async def fetch_historical_open_interest_async(self, symbol: str, timeframe: str, start_ms: int, end_ms: int) -> List[Dict]:
        log.info(f"Pobieranie historii Open Interest dla {symbol} (interwał: {timeframe})...")

        api_timeframe = timeframe.replace('m', 'min')

        params = {
            "category": self.category,
            "symbol": _norm_symbol(symbol),
            "intervalTime": api_timeframe,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 200
        }
        all_data, current_start = [], start_ms
        ssl_context = ssl.create_default_context(cafile=certifi.where())

        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
            while current_start < end_ms:
                params["startTime"] = current_start
                query_string = urlencode(params)

                try:
                    async with session.get(f"{self.base_url}/v5/market/open-interest?" + query_string) as response:
                        if response.status != 200:
                            log.error(f"Błąd HTTP {response.status} dla open-interest")
                            await asyncio.sleep(5)
                            continue

                        resp_json = await response.json()
                        if str(resp_json.get("retCode")) != "0":
                            log.error(f"Błąd API Bybit dla open-interest: {resp_json.get('retMsg')}")
                            break

                        data_list = resp_json.get("result", {}).get("list", [])
                        if not data_list: break

                        all_data.extend(data_list)
                        last_timestamp = int(data_list[-1]["timestamp"])
                        current_start = last_timestamp + 1
                        await asyncio.sleep(0.2)

                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    log.error(f"Błąd połączenia lub timeout dla open-interest: {e}")
                    await asyncio.sleep(5)
                    continue
        return all_data

    def fetch_open_interest_history(self, symbol: str, timeframe: str, limit: int = 200) -> List[Dict]:
        """
        Pobiera historię Open Interest dla danego symbolu i interwału.
        Interwały obsługiwane przez API: 5min, 15min, 30min, 1h, 2h, 4h, 6h, 12h, 1d
        """
        log.info(f"Pobieranie historii Open Interest dla {symbol} (interwał: {timeframe})...")
        all_data = []

        # Mapowanie naszych interwałów na te akceptowane przez API
        # Bybit oczekuje np. '5min' zamiast '5m'
        api_timeframe = timeframe.replace('m', 'min').replace('D', 'd')

        params = {
            "category": self.category,
            "symbol": _norm_symbol(symbol),
            "intervalTime": api_timeframe,
            "limit": min(limit, 200)  # API pozwala na max 200 na żądanie
        }

        # Pętla do pobierania danych partiami, jeśli limit jest większy niż 200
        while len(all_data) < limit:
            remaining = limit - len(all_data)
            params['limit'] = min(remaining, 200)

            # Jeśli już pobraliśmy jakieś dane, ustawiamy endTime na timestamp ostatniego rekordu
            if all_data:
                last_timestamp = int(all_data[-1]['timestamp'])
                params['endTime'] = last_timestamp - 1

            try:
                query_string = urlencode(params)
                url = f"{self.base_url}/v5/market/open-interest?{query_string}"
                response = requests.get(url)
                response.raise_for_status()
                resp_json = response.json()

                if str(resp_json.get("retCode")) != "0":
                    log.error(f"Błąd API Bybit dla open-interest: {resp_json.get('retMsg')}")
                    break

                data_list = resp_json.get("result", {}).get("list", [])
                if not data_list:
                    break  # Brak więcej danych

                # Dane przychodzą posortowane od najnowszych, odwracamy kolejność
                all_data.extend(reversed(data_list))

            except requests.RequestException as e:
                log.error(f"Błąd połączenia podczas pobierania Open Interest: {e}")
                break

        # Sortujemy finalnie i upewniamy się, że nie mamy duplikatów
        if all_data:
            df = pd.DataFrame(all_data).drop_duplicates(subset=['timestamp']).sort_values('timestamp')
            return df.to_dict('records')

        return []

    def fetch_funding_rate_history(self, symbol: str, limit: int = 200) -> List[Dict]:
        """
        Pobiera historię stóp finansowania (Funding Rate).
        """
        log.info(f"Pobieranie historii Funding Rate dla {symbol}...")
        params = {
            "category": self.category,
            "symbol": _norm_symbol(symbol),
            "limit": limit
        }
        try:
            query_string = urlencode(params)
            url = f"{self.base_url}/v5/market/funding/history?{query_string}"
            response = requests.get(url)
            response.raise_for_status()
            resp_json = response.json()

            if str(resp_json.get("retCode")) == "0":
                return resp_json.get("result", {}).get("list", [])
            else:
                log.error(f"Błąd API Bybit dla funding/history: {resp_json.get('retMsg')}")
                return []
        except requests.RequestException as e:
            log.error(f"Błąd połączenia podczas pobierania Funding Rate: {e}")
            return []

    def fetch_recent_trades(self, symbol: str, limit: int = 1000) -> List[Dict]:
        """
        Pobiera listę ostatnich surowych transakcji na rynku.
        """
        log.info(f"Pobieranie ostatnich transakcji dla {symbol}...")
        # API pozwala na max 1000 na żądanie
        params = {
            "category": self.category,
            "symbol": _norm_symbol(symbol),
            "limit": min(limit, 1000)
        }
        try:
            query_string = urlencode(params)
            url = f"{self.base_url}/v5/market/recent-trade?{query_string}"
            response = requests.get(url)
            response.raise_for_status()  # Zgłosi błąd dla statusów 4xx/5xx
            resp_json = response.json()

            if str(resp_json.get("retCode")) == "0":
                return resp_json.get("result", {}).get("list", [])
            else:
                log.error(f"Błąd API Bybit dla recent-trade: {resp_json.get('retMsg')}")
                return []
        except requests.RequestException as e:
            log.error(f"Błąd połączenia podczas pobierania ostatnich transakcji: {e}")
            return []

    def get_trade_history(self, symbol: str = None, start_time_ms: int = None, end_time_ms: int = None, limit: int = 100) -> List[Dict]:
        """
        Get trade execution history from Bybit.

        Args:
            symbol: Trading pair (e.g., SOLUSDT). If None, gets all symbols.
            start_time_ms: Start timestamp in milliseconds
            end_time_ms: End timestamp in milliseconds
            limit: Max records to return (default 100, max 100 per request)

        Returns:
            List of execution records sorted by execTime descending

        Example response for each execution:
            {
                'symbol': 'SOLUSDT',
                'orderId': '...',
                'execId': '...',
                'execPrice': '170.50',
                'execQty': '0.1',
                'execValue': '17.05',
                'execTime': '1698765432000',
                'execFee': '0.00935',
                'side': 'Buy',
                'orderType': 'Market',
                'closedSize': '0.1'
            }
        """
        try:
            params = {
                "category": self.category,
                "limit": min(limit, 100)  # Bybit API limit: 100 per request
            }

            if symbol:
                params["symbol"] = _norm_symbol(symbol)
            if start_time_ms:
                params["startTime"] = int(start_time_ms)
            if end_time_ms:
                params["endTime"] = int(end_time_ms)

            log.info(f"📊 Fetching trade history: symbol={symbol}, start={start_time_ms}, end={end_time_ms}, limit={limit}")

            resp = self.client.get_executions(**params)

            if str(resp.get("retCode")) not in ("0",):
                log.error(f"get_executions error: {resp}")
                return []

            result = resp.get("result") or {}
            executions = result.get("list", [])

            log.info(f"✓ Retrieved {len(executions)} execution records")
            return executions

        except Exception as e:
            log.error(f"Error fetching trade history: {e}", exc_info=True)
            return []

    def get_closed_pnl_history(self, symbol: str = None, start_time_ms: int = None, end_time_ms: int = None, limit: int = 100) -> List[Dict]:
        """
        Get closed P&L history from Bybit (completed trades with profit/loss).

        Args:
            symbol: Trading pair (e.g., SOLUSDT). If None, gets all symbols.
            start_time_ms: Start timestamp in milliseconds
            end_time_ms: End timestamp in milliseconds
            limit: Max records to return (default 100, max 100 per request)

        Returns:
            List of closed P&L records sorted by createdTime descending

        Example response for each closed trade:
            {
                'symbol': 'SOLUSDT',
                'orderId': '...',
                'side': 'Buy',
                'qty': '0.1',
                'orderPrice': '170.50',
                'orderType': 'Market',
                'execType': 'Trade',
                'closedSize': '0.1',
                'cumEntryValue': '17.05',
                'avgEntryPrice': '170.50',
                'cumExitValue': '17.20',
                'avgExitPrice': '172.00',
                'closedPnl': '0.15',      # Net P&L after fees
                'fillCount': '1',
                'leverage': '10',
                'createdTime': '1698765432000'
            }
        """
        try:
            params = {
                "category": self.category,
                "limit": min(limit, 100)  # Bybit API limit: 100 per request
            }

            if symbol:
                params["symbol"] = _norm_symbol(symbol)
            if start_time_ms:
                params["startTime"] = int(start_time_ms)
            if end_time_ms:
                params["endTime"] = int(end_time_ms)

            log.info(f"📊 Fetching closed P&L history: symbol={symbol}, start={start_time_ms}, end={end_time_ms}, limit={limit}")

            resp = self.client.get_closed_pnl(**params)

            if str(resp.get("retCode")) not in ("0",):
                log.error(f"get_closed_pnl error: {resp}")
                return []

            result = resp.get("result") or {}
            trades = result.get("list", [])

            log.info(f"✓ Retrieved {len(trades)} closed P&L records")
            return trades

        except Exception as e:
            log.error(f"Error fetching closed P&L history: {e}", exc_info=True)
            return []

    def get_execution_list(self, symbol: str = None, start_time_ms: int = None, end_time_ms: int = None, limit: int = 100) -> List[Dict]:
        """
        Get execution list (trade fills) from Bybit.

        This shows individual fills/executions for orders, including:
        - Maker vs Taker classification (affects fees)
        - Actual execution price vs order price (slippage)
        - Execution fees paid
        - Fill timestamps

        Args:
            symbol: Trading pair (e.g., SOLUSDT). If None, gets all symbols.
            start_time_ms: Start timestamp in milliseconds
            end_time_ms: End timestamp in milliseconds
            limit: Max records to return (default 100, max 100)

        Returns:
            List of execution records

        Example response:
            {
                'symbol': 'SOLUSDT',
                'orderId': '...',
                'orderLinkId': '...',
                'side': 'Buy',
                'orderPrice': '170.50',
                'orderQty': '1.0',
                'leavesQty': '0.0',
                'orderType': 'Market',
                'stopOrderType': 'UNKNOWN',
                'execFee': '0.009405',  # Fee paid
                'execId': '...',
                'execPrice': '170.55',  # Actual fill price
                'execQty': '1.0',
                'execType': 'Trade',
                'execValue': '170.55',
                'execTime': '1698765432000',
                'isMaker': False,  # False = Taker (0.055% fee), True = Maker (0.02% fee)
                'feeRate': '0.00055',
                'tradeIv': '',
                'markIv': '',
                'markPrice': '170.50',
                'indexPrice': '170.48',
                'underlyingPrice': '',
                'blockTradeId': '',
                'closedSize': '1.0',
                'seq': 123456789
            }
        """
        try:
            params = {
                "category": self.category,
                "limit": min(limit, 100)
            }

            if symbol:
                params["symbol"] = _norm_symbol(symbol)
            if start_time_ms:
                params["startTime"] = int(start_time_ms)
            if end_time_ms:
                params["endTime"] = int(end_time_ms)

            log.info(f"📋 Fetching execution list: symbol={symbol}, limit={limit}")

            response = self.client.get_executions(**params)

            if str(response.get("retCode")) != "0":
                log.error(f"Error response from Bybit: {response.get('retMsg')}")
                return []

            result = response.get("result", {})
            executions = result.get("list", [])

            log.info(f"✓ Retrieved {len(executions)} execution records")
            return executions

        except Exception as e:
            log.error(f"Error fetching execution list: {e}", exc_info=True)
            return []

    def get_transaction_log(self, category: str = None, base_coin: str = "USDT", type: str = None,
                           start_time_ms: int = None, end_time_ms: int = None, limit: int = 50) -> List[Dict]:
        """
        Get transaction log (account transactions including funding fees).

        This shows ALL financial transactions on the account:
        - Trades (buy/sell executions)
        - Funding fees (periodic fees for holding positions)
        - Transfers, deposits, withdrawals
        - Position settlements

        Args:
            category: Product category (default: None = unified account)
            base_coin: Base coin (default: USDT)
            type: Transaction type filter (TRANSFER_IN, TRANSFER_OUT, TRADE, SETTLEMENT, DELIVERY, LIQUIDATION, BONUS, FEE_REFUND, INTEREST, CURRENCY_BUY, CURRENCY_SELL)
            start_time_ms: Start timestamp in milliseconds
            end_time_ms: End timestamp in milliseconds
            limit: Max records (default 50, max 50 per request)

        Returns:
            List of transaction records

        Example response:
            {
                'symbol': 'SOLUSDT',
                'category': 'linear',
                'side': 'Buy',
                'transactionTime': '1698765432000',
                'type': 'TRADE',  # or 'FUNDING_FEE', 'SETTLEMENT', etc.
                'qty': '1.0',
                'size': '1.0',
                'currency': 'USDT',
                'tradePrice': '170.50',
                'funding': '-0.0051',  # Funding fee paid (negative = paid, positive = received)
                'fee': '-0.009405',  # Trading fee
                'cashFlow': '-170.509405',  # Net cash change
                'change': '-170.509405',
                'cashBalance': '829.49',  # Balance after transaction
                'feeRate': '0.00055',
                'bonusChange': '0',
                'tradeId': '...',
                'orderId': '...',
                'orderLinkId': ''
            }
        """
        try:
            params = {
                "accountType": "UNIFIED",
                "limit": min(limit, 50)
            }

            if category:
                params["category"] = category
            if base_coin:
                params["baseCoin"] = base_coin
            if type:
                params["type"] = type
            if start_time_ms:
                params["startTime"] = int(start_time_ms)
            if end_time_ms:
                params["endTime"] = int(end_time_ms)

            log.info(f"💰 Fetching transaction log: baseCoin={base_coin}, type={type}, limit={limit}")

            response = self.client.get_transaction_log(**params)

            if str(response.get("retCode")) != "0":
                log.error(f"Error response from Bybit: {response.get('retMsg')}")
                return []

            result = response.get("result", {})
            transactions = result.get("list", [])

            log.info(f"✓ Retrieved {len(transactions)} transaction records")
            return transactions

        except Exception as e:
            log.error(f"Error fetching transaction log: {e}", exc_info=True)
            return []

    def get_order_history(self, symbol: str = None, order_status: str = None,
                          start_time_ms: int = None, end_time_ms: int = None, limit: int = 50) -> List[Dict]:
        """
        Get order history (all orders including cancelled, rejected, filled).

        Args:
            symbol: Trading pair (e.g., SOLUSDT). If None, gets all symbols.
            order_status: Order status filter (New, PartiallyFilled, Filled, Cancelled, Rejected, PartiallyFilledCanceled)
            start_time_ms: Start timestamp in milliseconds
            end_time_ms: End timestamp in milliseconds
            limit: Max records (default 50, max 50)

        Returns:
            List of order records

        Example response:
            {
                'orderId': '...',
                'orderLinkId': '...',
                'blockTradeId': '',
                'symbol': 'SOLUSDT',
                'price': '170.50',
                'qty': '1.0',
                'side': 'Buy',
                'isLeverage': '0',
                'positionIdx': 0,
                'orderStatus': 'Filled',
                'cancelType': 'UNKNOWN',
                'rejectReason': 'EC_NoError',
                'avgPrice': '170.55',
                'leavesQty': '0.0',
                'leavesValue': '0',
                'cumExecQty': '1.0',
                'cumExecValue': '170.55',
                'cumExecFee': '0.009405',
                'timeInForce': 'GTC',
                'orderType': 'Market',
                'stopOrderType': 'UNKNOWN',
                'orderIv': '',
                'triggerPrice': '0.00',
                'takeProfit': '175.00',  # TP price
                'stopLoss': '165.00',  # SL price
                'tpTriggerBy': 'LastPrice',
                'slTriggerBy': 'LastPrice',
                'triggerDirection': 0,
                'triggerBy': 'UNKNOWN',
                'lastPriceOnCreated': '170.50',
                'reduceOnly': False,
                'closeOnTrigger': False,
                'smpType': 'None',
                'smpGroup': 0,
                'smpOrderId': '',
                'tpslMode': 'Full',
                'tpLimitPrice': '0.00',
                'slLimitPrice': '0.00',
                'placeType': '',
                'createdTime': '1698765430000',
                'updatedTime': '1698765432000'
            }
        """
        try:
            params = {
                "category": self.category,
                "limit": min(limit, 50)
            }

            if symbol:
                params["symbol"] = _norm_symbol(symbol)
            if order_status:
                params["orderStatus"] = order_status
            if start_time_ms:
                params["startTime"] = int(start_time_ms)
            if end_time_ms:
                params["endTime"] = int(end_time_ms)

            log.info(f"📝 Fetching order history: symbol={symbol}, status={order_status}, limit={limit}")

            response = self.client.get_order_history(**params)

            if str(response.get("retCode")) != "0":
                log.error(f"Error response from Bybit: {response.get('retMsg')}")
                return []

            result = response.get("result", {})
            orders = result.get("list", [])

            log.info(f"✓ Retrieved {len(orders)} order records")
            return orders

        except Exception as e:
            log.error(f"Error fetching order history: {e}", exc_info=True)
            return []

    def get_all_positions(self) -> List[Dict]:
        """
        Get all active positions across all symbols.

        Returns:
            List of position dictionaries
        """
        try:
            params = {
                "category": self.category,
                "settleCoin": "USDT"
            }

            log.info("📊 Fetching all positions...")

            response = self.client.get_positions(**params)

            if str(response.get("retCode")) != "0":
                log.error(f"Error response from Bybit: {response.get('retMsg')}")
                return []

            result = response.get("result", {})
            positions = result.get("list", [])

            # Filter only positions with size > 0
            active_positions = [p for p in positions if float(p.get('size', 0)) > 0]

            log.info(f"✓ Retrieved {len(active_positions)} active positions (out of {len(positions)} total)")
            return active_positions

        except Exception as e:
            log.error(f"Error fetching positions: {e}", exc_info=True)
            return []