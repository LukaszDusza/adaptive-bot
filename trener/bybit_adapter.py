# bybit_adapter.py
# Adapter Bybit V5 (Unified Trading) z obsługą:
# - TRYB "demo": natywne REST na https://api-demo.bybit.com (własny HTTP + podpisy)
# - TRYB "pybit": fallback na pybit (mainnet/testnet)
# Funkcje: get_kline, get_instruments_info, set_leverage, place_order (market), wallet_balance (Unified)

from __future__ import annotations
import time
import hmac
import hashlib
import json
import math
from typing import Any, Dict, List, Optional, Iterable

import requests

# Pybit jest opcjonalny (używany tylko w trybie "pybit")
try:
    from pybit.unified_trading import HTTP as PYBIT_HTTP
except Exception:
    PYBIT_HTTP = None


class BybitAPIError(RuntimeError):
    pass


def _ensure_ok(resp: Dict[str, Any], ctx: str = "", extra_ok: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """
    Sprawdza retCode. Domyślnie akceptuje '0'.
    Można przekazać extra_ok, np. {'110043'} dla 'leverage not modified'.
    """
    if not isinstance(resp, dict):
        raise BybitAPIError(f"{ctx} -> invalid response type: {type(resp)}")
    code = str(resp.get("retCode"))
    if code != "0":
        if extra_ok and code in set(extra_ok):
            return resp
        raise BybitAPIError(f"{ctx} -> retCode={resp.get('retCode')} retMsg={resp.get('retMsg')} resp={resp}")
    return resp


def _round_step(x: float, step: float) -> float:
    if step <= 0:
        return float(x)
    # floor do siatki kroku
    return math.floor(float(x) / step + 1e-12) * step


# ----------------------------- DEMO HTTP (api-demo.bybit.com) -----------------------------

class _DemoHTTP:
    """
    Minimalny klient Bybit V5 dla api-demo.bybit.com z podpisem HMAC (SIGN-TYPE=2).
    Dokumentacja: https://bybit-exchange.github.io/docs/v5/intro
    """
    def __init__(self, api_key: str, api_secret: str, base_url: str, recv_window: int = 5000, timeout: int = 15):
        self.api_key = api_key
        self.api_secret = api_secret.encode()
        self.base_url = base_url.rstrip("/")
        self.recv_window = recv_window
        self.timeout = timeout
        self.sess = requests.Session()

    def _ts(self) -> str:
        return str(int(time.time() * 1000))

    def _sign(self, ts: str, query_or_body: str) -> str:
        # SIGN-TYPE=2: sign = HMAC_SHA256(secret, ts + api_key + recv_window + query/body)
        payload = ts + self.api_key + str(self.recv_window) + query_or_body
        return hmac.new(self.api_secret, payload.encode(), hashlib.sha256).hexdigest()

    def _headers(self, ts: str, sign: str) -> Dict[str, str]:
        return {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-TIMESTAMP": ts,
            "X-BAPI-RECV-WINDOW": str(self.recv_window),
            "X-BAPI-SIGN": sign,
            "X-BAPI-SIGN-TYPE": "2",
            "Content-Type": "application/json",
        }

    def get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        # query string według Bybit: klucze sortowane alfabetycznie
        items = []
        for k in sorted(params.keys()):
            v = params[k]
            if v is None:
                continue
            items.append(f"{k}={v}")
        query = "&".join(items)
        ts = self._ts()
        sign = self._sign(ts, query)
        headers = self._headers(ts, sign)
        resp = self.sess.get(url, params=params, headers=headers, timeout=self.timeout)
        return resp.json()

    def post(self, path: str, data: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        body_str = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
        ts = self._ts()
        sign = self._sign(ts, body_str)
        headers = self._headers(ts, sign)
        resp = self.sess.post(url, data=body_str.encode("utf-8"), headers=headers, timeout=self.timeout)
        return resp.json()


# ----------------------------- ADAPTER -----------------------------

class BybitAdapter:
    """
    Jeden wspólny adapter. Sam wybierze tryb:
      - jeśli base_url zawiera 'api-demo' => 'demo' (własny HTTP)
      - inaczej => 'pybit' (wymaga pybit)
    Publiczne metody:
      fetch_ohlcv(symbol, interval, limit)
      set_leverage(symbol, leverage)
      market_entry(symbol, side, qty, reduce_only=False)
      market_close(symbol, side, qty)
      fetch_balance_usdt()
    """
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str = "https://api.bybit.com",
        category: str = "linear",
        recv_window: int = 5000,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip("/")
        self.category = category
        self.recv_window = recv_window

        if "api-demo" in self.base_url.lower():
            # DEMO: natywny klient
            self.mode = "demo"
            self.client = _DemoHTTP(api_key, api_secret, self.base_url, recv_window=self.recv_window)
        else:
            # PYBIT: testnet/mainnet rozpoznawany przez flagę testnet
            if PYBIT_HTTP is None:
                raise BybitAPIError("pybit nie jest zainstalowany, a base_url nie wygląda na demo.")
            self.mode = "pybit"
            testnet = ("testnet" in self.base_url.lower())
            self.client = PYBIT_HTTP(api_key=api_key, api_secret=api_secret, testnet=testnet, recv_window=recv_window)

        # cache specyfikacji
        self._markets: Dict[str, Dict[str, Any]] = {}

    # ---------- Market meta ----------

    def _load_symbol_specs(self, symbol: str) -> Dict[str, Any]:
        if symbol in self._markets:
            return self._markets[symbol]

        if self.mode == "demo":
            r = self.client.get("/v5/market/instruments-info", {"category": self.category, "symbol": symbol})
        else:  # pybit
            r = self.client.get_instruments_info(category=self.category, symbol=symbol)

        data = _ensure_ok(r, "get_instruments_info")["result"]
        lst = data.get("list") or []
        if not lst:
            raise BybitAPIError(f"Instrument {symbol} nie znaleziony (category={self.category})")
        info = lst[0]

        lot = info.get("lotSizeFilter", {}) or {}
        price = info.get("priceFilter", {}) or {}
        specs = {
            "qtyStep": float(lot.get("qtyStep", 0) or 0),
            "minOrderQty": float(lot.get("minOrderQty", 0) or 0),
            "maxOrderQty": float(lot.get("maxOrderQty", 0) or 0),
            "tickSize": float(price.get("tickSize", 0) or 0),
            "minPrice": float(price.get("minPrice", 0) or 0),
            "maxPrice": float(price.get("maxPrice", 0) or 0),
        }
        self._markets[symbol] = specs
        return specs

    def _round_qty_for(self, symbol: str, qty: float) -> float:
        specs = self._load_symbol_specs(symbol)
        step = specs.get("qtyStep", 0.0) or 0.0
        q = _round_step(float(qty), step) if step > 0 else float(qty)
        min_q = specs.get("minOrderQty") or 0.0
        if min_q and q < min_q:
            q = 0.0
        return q

    def _round_price_for(self, symbol: str, price: float) -> float:
        specs = self._load_symbol_specs(symbol)
        tick = specs.get("tickSize", 0.0) or 0.0
        p = _round_step(float(price), tick) if tick > 0 else float(price)
        return p

    # ---------- OHLCV ----------

    def fetch_ohlcv(self, symbol: str, interval: str, limit: int = 1000) -> List[List[Any]]:
        """
        Zwraca listę świec: [ts_ms, open, high, low, close, volume, turnover]
        interval: '1','3','5','15','60','240','D','W', ...
        """
        if self.mode == "demo":
            r = self.client.get("/v5/market/kline", {"category": self.category, "symbol": symbol, "interval": interval, "limit": limit})
        else:
            r = self.client.get_kline(category=self.category, symbol=symbol, interval=interval, limit=limit)

        data = _ensure_ok(r, "get_kline")["result"]
        kl = data.get("list") or []
        out: List[List[Any]] = []
        for k in reversed(kl):  # Bybit daje od najnowszej
            ts = int(k[0])
            o = float(k[1]); h = float(k[2]); l = float(k[3]); c = float(k[4])
            v = float(k[5]); to = float(k[6]) if len(k) > 6 else 0.0
            out.append([ts, o, h, l, c, v, to])
        return out

    # ---------- Dźwignia ----------

    def set_leverage(self, symbol: str, leverage: float) -> None:
        if self.mode == "demo":
            r = self.client.post("/v5/position/set-leverage", {
                "category": self.category,
                "symbol": symbol,
                "buyLeverage": str(leverage),
                "sellLeverage": str(leverage),
            })
        else:
            r = self.client.set_leverage(category=self.category, symbol=symbol, buyLeverage=str(leverage), sellLeverage=str(leverage))
        # Akceptuj także 110043: "leverage not modified"
        _ensure_ok(r, "set_leverage", extra_ok={"110043"})

    # ---------- Zlecenia ----------

    def market_entry(self, symbol: str, side: str, qty: float, reduce_only: bool = False) -> str:
        """
        side: 'Buy' (long) albo 'Sell' (short)
        """
        q = self._round_qty_for(symbol, float(qty))
        if q <= 0:
            raise BybitAPIError(f"Qty po zaokrągleniu = 0 (qtyStep/minOrderQty). Symbol={symbol}, qty_in={qty}")

        payload = {
            "category": self.category,
            "symbol": symbol,
            "side": side,
            "orderType": "Market",
            "qty": str(q),
            "reduceOnly": reduce_only,
            "timeInForce": "IOC",
        }

        if self.mode == "demo":
            r = self.client.post("/v5/order/create", payload)
        else:
            r = self.client.place_order(**payload)

        data = _ensure_ok(r, "place_order")["result"]
        return data.get("orderId", "")

    def market_close(self, symbol: str, side: str, qty: float) -> str:
        """
        Zamknięcie pozycji: przeciwna strona z reduceOnly=True.
        side: 'Buy' zamyka shorta, 'Sell' zamyka longa.
        """
        q = self._round_qty_for(symbol, float(qty))
        if q <= 0:
            raise BybitAPIError(f"Qty po zaokrągleniu = 0 (qtyStep/minOrderQty). Symbol={symbol}, qty_in={qty}")

        payload = {
            "category": self.category,
            "symbol": symbol,
            "side": side,
            "orderType": "Market",
            "qty": str(q),
            "reduceOnly": True,
            "timeInForce": "IOC",
        }

        if self.mode == "demo":
            r = self.client.post("/v5/order/create", payload)
        else:
            r = self.client.place_order(**payload)

        data = _ensure_ok(r, "place_order(close)")["result"]
        return data.get("orderId", "")

    # ---------- Balance (Unified) ----------

    def fetch_balance_usdt(self) -> Dict[str, float]:
        if self.mode == "demo":
            r = self.client.get("/v5/account/wallet-balance", {"accountType": "UNIFIED", "coin": "USDT"})
        else:
            r = self.client.get_wallet_balance(accountType="UNIFIED", coin="USDT")

        data = _ensure_ok(r, "get_wallet_balance")["result"]
        lst = data.get("list") or []
        if not lst:
            return {
                "total_equity": 0.0,
                "total_available": 0.0,
                "coin_equity": 0.0,
                "coin_wallet": 0.0,
                "coin_available": 0.0,
            }
        acct = lst[0]
        total_equity = float(acct.get("totalEquity", 0) or 0)
        total_avail = float(acct.get("totalAvailableBalance", 0) or 0)

        usdt_info: Optional[Dict[str, Any]] = None
        for c in acct.get("coin", []) or []:
            if str(c.get("coin")).upper() == "USDT":
                usdt_info = c
                break
        if not usdt_info:
            usdt_info = {}

        coin_equity = float(usdt_info.get("equity", 0) or 0)
        coin_wallet = float(usdt_info.get("walletBalance", 0) or 0)
        coin_available = float(usdt_info.get("availableToWithdraw", 0) or 0)

        return {
            "total_equity": total_equity,
            "total_available": total_avail,
            "coin_equity": coin_equity,
            "coin_wallet": coin_wallet,
            "coin_available": coin_available,
        }

    # ---------- Utility ----------

    def sleep(self, sec: float):
        time.sleep(sec)
