# bybit_adapter.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import time
import math
from typing import Any, Dict, List, Optional

import logging

from pybit.unified_trading import HTTP
from pybit.exceptions import InvalidRequestError

log = logging.getLogger("bybit")


# ---- HTTP z możliwością wstrzyknięcia niestandardowego endpointu (np. DEMO) ----
class _DemoHTTP(HTTP):
    def __init__(self, api_key: str, api_secret: str, base_url: str, recv_window: int = 20000):
        super().__init__(api_key=api_key, api_secret=api_secret, testnet=False, recv_window=recv_window)
        base_url = base_url.rstrip("/")
        # W pybit 5.11.0 URL jest w atrybucie 'endpoint'
        if hasattr(self, "endpoint"):
            self.endpoint = base_url
        # Niektóre wewnętrzne metody korzystają też z '_domain' – ustawiamy obie na wszelki wypadek
        try:
            setattr(self, "_domain", base_url)
        except Exception:
            pass
        log.info(f"Bybit HTTP endpoint ustawiony na: {base_url}")


def _norm_symbol(symbol: str) -> str:
    """
    'DOGE/USDT:USDT' -> 'DOGEUSDT'
    'DOGEUSDT'       -> 'DOGEUSDT'
    """
    if ":" in symbol:
        symbol = symbol.split(":")[0]
    symbol = symbol.replace("/", "")
    return symbol


def _tf_to_interval(tf: str) -> str:
    """
    Mapowanie timeframe na format v5:
      5m -> '5', 15m -> '15', 1h -> '60', 2h -> '120', 4h->'240', 1d->'D', 1w->'W', 1M->'M'
    """
    tf = tf.strip().lower()
    if tf.endswith("m"):
        return str(int(tf[:-1]))
    if tf.endswith("h"):
        return str(int(tf[:-1]) * 60)
    if tf in ("1d", "d", "1day", "day"):
        return "D"
    if tf in ("1w", "w", "1week", "week"):
        return "W"
    if tf in ("1mo", "1mth", "month", "mth"):
        return "M"
    # fallback: spróbuj minutes
    try:
        _ = int(tf)
        return tf
    except Exception:
        raise ValueError(f"Nieznany timeframe: {tf}")


def _bar_ms(interval: str) -> int:
    if interval == "D":
        return 24 * 60 * 60 * 1000
    if interval == "W":
        return 7 * 24 * 60 * 60 * 1000
    if interval == "M":
        return 30 * 24 * 60 * 60 * 1000  # przybliżenie
    return int(interval) * 60 * 1000  # minutes


class BybitAPIError(RuntimeError):
    pass


class BybitAdapter:
    """
    Cienka warstwa nad pybit v5.

    Używane w live_trader.py:
      - fetch_ohlcv(symbol, timeframe, limit)
      - round_qty(symbol, qty)
      - get_balance(use_available: bool)
      - set_leverage(symbol_u, leverage)
      - get_position(symbol_u)
      - latest_price(symbol_u)
      - market_open(symbol_u, side, qty)
      - set_stop_loss(symbol_u, price, side)
      - set_take_profit(symbol_u, price, side)
      - cancel_tpsl(symbol_u)
      - close_position(symbol_u)
    """

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
            # DEMO / custom endpoint
            self.client = _DemoHTTP(api_key, api_secret, base_url, recv_window=recv_window)
            self.base_url = base_url.rstrip("/")
        else:
            # standardowy endpoint pybit (prod/testnet wg flagi testnet – tutaj False)
            self.client = HTTP(api_key=api_key, api_secret=api_secret, testnet=False, recv_window=recv_window)
            self.base_url = "https://api.bybit.com"

        self._lot_step: Dict[str, float] = {}
        self._tick_size: Dict[str, float] = {}

        try:
            self._prime_instrument_cache()
        except Exception as e:
            log.warning(f"Nie udało się wczytać specyfikacji instrumentów: {e}")

    # ---------- Narzędzia ----------
    def _prime_instrument_cache(self):
        """Opcjonalne: wypełnij cache kroków ilości/ceny. Na razie pomijamy (on-demand)."""
        pass

    def round_qty(self, symbol: str, qty: float) -> float:
        s = _norm_symbol(symbol)
        step = self._lot_step.get(s, 0.0)
        if step and step > 0:
            return max(step, math.floor(qty / step) * step)
        return float(f"{qty:.6f}")  # bezpieczny fallback

    # ---------- Market data ----------
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> List[List[Any]]:
        """
        Zwraca listę list: [start, open, high, low, close, volume, turnover] (stringi – jak w v5).
        Paginacja chunkami do 1000, od najnowszej świecy wstecz.
        """
        sym = _norm_symbol(symbol)
        interval = _tf_to_interval(timeframe)
        step = _bar_ms(interval)
        out: List[List[Any]] = []

        # Zbieramy wstecz od teraz. Niektóre rynki DEMO mają mniejsze okna dostępności – pętla to uwzględnia.
        end = int(time.time() * 1000)
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
                # W DEMO często trzeba „kawałkować” okno. Cofamy end i próbujemy dalej.
                tries += 1
                end = start
                continue

            try:
                data_sorted = sorted(data, key=lambda r: int(r[0]))
            except Exception:
                data_sorted = sorted(data, key=lambda r: int(r.get("start", 0)))

            out.extend(data_sorted)
            end = start
            tries = 0  # reset, skoro dostaliśmy dane

        if len(out) > limit:
            out = out[-limit:]

        if not out:
            log.warning(
                f"fetch_ohlcv: pusta odpowiedź. symbol={sym}, interval={interval}, limit={limit}, base={self.base_url}"
            )
        return out

    # ---------- Account / risk ----------
    def get_balance(self, use_available: bool = False) -> float:
        """
        Zwraca totalEquity lub availableToWithdraw dla USDT na Unified Account.
        """
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

    # ---------- Pozycje / zlecenia ----------
    def get_position(self, symbol_u: str) -> Dict[str, Any]:
        resp = self.client.get_positions(category=self.category, symbol=symbol_u)
        if str(resp.get("retCode")) not in ("0",):
            raise BybitAPIError(f"get_positions -> {resp}")
        arr = (resp.get("result") or {}).get("list") or []
        for p in arr:
            sz = float(p.get("size", 0) or 0)
            if abs(sz) > 0:
                side = p.get("side")  # Buy/Sell
                entry = float(p.get("avgPrice", 0) or 0)
                return {
                    "side": "Long" if side == "Buy" else "Short",
                    "size": sz,
                    "entryPrice": entry,
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

    def market_open(self, symbol_u: str, side: str, qty: float):
        """
        side: 'Buy' dla long, 'Sell' dla short
        """
        resp = self.client.place_order(
            category=self.category,
            symbol=symbol_u,
            side=side,
            orderType="Market",
            qty=str(qty),
            timeInForce="IOC",
            reduceOnly=False,
        )
        if str(resp.get("retCode")) not in ("0",):
            raise BybitAPIError(f"place_order -> {resp}")
        return resp

    def set_stop_loss(self, symbol_u: str, price: float, side: str):
        """
        side: 'Sell' gdy mamy long (SL sprzedaje), 'Buy' gdy short
        """
        resp = self.client.set_trading_stop(
            category=self.category,
            symbol=symbol_u,
            stopLoss=str(price),
            positionIdx=0,  # net
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
        # === POPRAWKA: Aby anulować SL/TP w API v5, należy wysłać "0" ===
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
        # zlecenie odwrotne na cały wolumen (reduceOnly)
        pos = self.get_position(symbol_u)
        if not pos:
            return
        side = "Sell" if pos["side"] == "Long" else "Buy"
        qty = abs(float(pos.get("size", 0)))
        if qty <= 0:
            return
        resp = self.client.place_order(
            category=self.category,
            symbol=symbol_u,
            side=side,
            orderType="Market",
            qty=str(qty),
            reduceOnly=True,
            timeInForce="IOC",
        )
        if str(resp.get("retCode")) not in ("0",):
            raise BybitAPIError(f"close_position -> {resp}")
        return resp