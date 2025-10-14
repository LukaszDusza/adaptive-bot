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
    ) -> None:
        self.category = category
        self.recv_window = recv_window

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
        s = _norm_symbol(symbol)
        step = self._lot_step.get(s)

        if step is not None and step > 0:
            precision = abs(int(math.log10(step))) if step < 1 else 0
            factor = 10 ** precision
            return math.floor(qty * factor) / factor
        return round(qty, 6)

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

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int, end_date: str = None) -> List[List[Any]]:
        sym = _norm_symbol(symbol)
        interval = _tf_to_interval(timeframe)
        step = _bar_ms(interval)
        out: List[List[Any]] = []
        # Use provided end_date or default to current time
        end = self._parse_end_date(end_date) if end_date else int(time.time() * 1000)
        tries = 0

        while len(out) < limit and tries < 20:
            need = min(1000, limit - len(out))
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
                raise BybitAPIError(f"fetch_ohlcv get_kline request error: {e}")

            if not isinstance(resp, dict) or str(resp.get("retCode")) not in ("0",):
                raise BybitAPIError(f"fetch_ohlcv -> ret={resp}")

            data = (((resp.get("result") or {}).get("list")) or [])
            if not data:
                tries += 1
                end = start
                continue

            try:
                data_sorted = sorted(data, key=lambda r: int(r[0]))
            except Exception:
                data_sorted = sorted(data, key=lambda r: int(r.get("start", 0)))

            out.extend(data_sorted)
            end = int(data_sorted[0][0]) - step
            tries = 0
            time.sleep(0.2)

        if len(out) > limit:
            out = out[-limit:]

        if not out:
            log.warning(
                f"fetch_ohlcv: pusta odpowiedź. symbol={sym}, interval={interval}, limit={limit}, base={self.base_url}"
            )
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

    def get_position(self, symbol_u: str) -> Dict[str, Any]:
        resp = self.client.get_positions(category=self.category, symbol=symbol_u)
        if str(resp.get("retCode")) not in ("0",):
            raise BybitAPIError(f"get_positions -> {resp}")
        arr = (resp.get("result") or {}).get("list") or []
        for p in arr:
            sz = float(p.get("size", 0) or 0)
            if abs(sz) > 0:
                side = p.get("side")
                entry = float(p.get("avgPrice", 0) or 0)
                return {"side": "Long" if side == "Buy" else "Short", "size": sz, "entryPrice": entry}
        return {}

    def latest_price(self, symbol_u: str) -> float:
        resp = self.client.get_tickers(category=self.category, symbol=symbol_u)
        if str(resp.get("retCode")) not in ("0",):
            raise BybitAPIError(f"get_tickers -> {resp}")
        arr = (resp.get("result") or {}).get("list") or []
        if not arr:
            return 0.0
        return float(arr[0].get("lastPrice", 0) or 0)

    def market_open(self, symbol_u: str, side: str, qty: float):
        safe_qty = self.round_qty(symbol_u, qty)
        log.info(f"Oryginalna ilość {qty} zaokrąglona do {safe_qty} dla {symbol_u}")
        resp = self.client.place_order(
            category=self.category,
            symbol=symbol_u,
            side=side,
            orderType="Market",
            qty=str(safe_qty), timeInForce="IOC", reduceOnly=False,
        )
        if str(resp.get("retCode")) not in ("0",):
            raise BybitAPIError(f"place_order -> {resp}")
        return resp

    def market_close(self, symbol_u: str, side: str, qty: float):
        safe_qty = self.round_qty(symbol_u, qty)
        log.info(f"Oryginalna ilość {qty} zaokrąglona do {safe_qty} dla {symbol_u}")
        resp = self.client.place_order(
            category=self.category,
            symbol=symbol_u,
            side=side,
            orderType="Market",
            qty=str(safe_qty), reduceOnly=True, timeInForce="IOC",
        )
        if str(resp.get("retCode")) not in ("0",):
            raise BybitAPIError(f"close_position (partial) -> {resp}")
        return resp

    def set_stop_loss(self, symbol_u: str, price: float, side: str):
        resp = self.client.set_trading_stop(
            category=self.category, symbol=symbol_u, stopLoss=str(price), positionIdx=0,
        )
        if str(resp.get("retCode")) not in ("0",):
            raise BybitAPIError(f"set_trading_stop (SL) -> {resp}")
        return resp

    def set_take_profit(self, symbol_u: str, price: float, side: str):
        resp = self.client.set_trading_stop(
            category=self.category,
            symbol=symbol_u,
            takeProfit=str(price),
            positionIdx=0,
        )
        if str(resp.get("retCode")) not in ("0",):
            raise BybitAPIError(f"set_trading_stop (TP) -> {resp}")
        return resp

    def cancel_tpsl(self, symbol_u: str):
        resp = self.client.set_trading_stop(
            category=self.category,
            symbol=symbol_u,
            takeProfit="0",
            stopLoss="0",
            positionIdx=0,
        )
        if str(resp.get("retCode")) not in ("0",):
            raise BybitAPIError(f"cancel_tpsl -> {resp}")
        return resp

    def close_position(self, symbol_u: str):
        pos = self.get_position(symbol_u)
        if not pos:
            return
        side = "Sell" if pos["side"] == "Long" else "Buy"
        qty = abs(float(pos.get("size", 0)))
        if qty <= 0: return
        safe_qty = self.round_qty(symbol_u, qty)
        log.info(f"Oryginalna ilość {qty} zaokrąglona do {safe_qty} dla {symbol_u}")
        resp = self.client.place_order(
            category=self.category,
            symbol=symbol_u,
            side=side,
            orderType="Market",
            qty=str(safe_qty), reduceOnly=True, timeInForce="IOC",
        )
        if str(resp.get("retCode")) not in ("0",):
            raise BybitAPIError(f"close_position -> {resp}")
        return resp

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