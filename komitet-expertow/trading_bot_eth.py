import os
import time
import schedule
import requests
import pandas as pd
import sys
import json
import logging
from pybit.unified_trading import HTTP
from pybit.exceptions import InvalidRequestError

# --- Konfiguracja Bota ---
TICKER = "ETHUSDT"
RISK_PER_TRADE = 0.02
LEVERAGE = "10"  # Ustawienie dźwigni
ATR_MULTIPLIER = 2
API_URL = "http://api_service_eth:8082/predict"
MIN_CONFIDENCE = 0.65
RRR = 2  # Risk-Reward Ratio
QTY_PRECISION = 2  # Precyzja dla ETH

# --- Konfiguracja Testów i Logowania ---
TEST_TICKER = "USDCUSDT"
MIN_TEST_QTY = "5.1"

# --- Zaawansowana Konfiguracja Logowania ---
logging.basicConfig(level=logging.INFO,
                    format='{\"timestamp\": \"%(asctime)s\", \"level\": \"%(levelname)s\", \"service\": \"trading_bot_eth\", \"message\": %(message)s}',
                    datefmt='%Y-%m-%dT%H:%M:%S%z')


def log(event, details):
    logging.info(json.dumps({"event": event, "details": details}))


# --- Solidna Inicjalizacja Sesji Bybit ---
api_key = os.getenv("BYBIT_API_KEY")
api_secret = os.getenv("BYBIT_API_SECRET")
session = None
if api_key and api_secret:
    try:
        session = HTTP(testnet=False, api_key=api_key, api_secret=api_secret)
        log("bybit_session_init", {"status": "success"})
    except Exception as e:
        log("bybit_session_init", {"status": "failure", "error": str(e)})
        sys.exit(1)
else:
    log("bybit_session_init", {"status": "failure", "error": "API keys not found"})
    sys.exit(1)


# --- GŁÓWNA LOGIKA BOTA ---
def job():
    log("job_start", {})

    # 1. Pobierz analizę od "Komitetu Ekspertów"
    try:
        response = requests.get(API_URL, timeout=10)
        response.raise_for_status()
        analysis = response.json()
        log("api_analysis_received", analysis)
    except requests.exceptions.RequestException as e:
        log("api_error", {"error": str(e)});
        return

    opinions = analysis['expert_opinions']

    # 2. Zlicz głosy ekspertów
    votes_long, votes_short = 0, 0
    for expert, opinion in opinions.items():
        if opinion['confidence'] >= MIN_CONFIDENCE:
            if opinion['prediction'] == 1:
                votes_long += 1
            else:
                votes_short += 1

    # 3. Sprawdź, czy należy zamknąć istniejącą pozycję
    current_positions = get_open_positions()
    if 'long' in current_positions and votes_short >= 2:
        log("exit_signal_triggered", {"strategy": "long", "reason": "Opposing vote from committee"})
        close_position('long', current_positions['long']['size'])
    if 'short' in current_positions and votes_long >= 2:
        log("exit_signal_triggered", {"strategy": "short", "reason": "Opposing vote from committee"})
        close_position('short', current_positions['short']['size'])

    # 4. Sprawdź, czy należy otworzyć nową pozycję
    current_positions = get_open_positions()  # Sprawdzamy ponownie
    strategy_to_open = None
    if votes_long >= 2 and 'long' not in current_positions:
        strategy_to_open = 'long'
    elif votes_short >= 2 and 'short' not in current_positions:
        strategy_to_open = 'short'

    if strategy_to_open:
        log("entry_signal_triggered",
            {"strategy": strategy_to_open, "votes_long": votes_long, "votes_short": votes_short})
        balance = get_wallet_balance()
        if balance is None or balance <= 0:
            log("insufficient_funds",
                {"message": "Balance is None or zero, cannot open new trade.", "balance": balance})
            return

        # <<< START: DODANA LOGIKA USTAWIANIA DŹWIGNI >>>
        try:
            log("set_leverage_attempt", {"symbol": TICKER, "leverage": LEVERAGE})
            session.set_leverage(
                category="linear",
                symbol=TICKER,
                buyLeverage=LEVERAGE,
                sellLeverage=LEVERAGE
            )
            log("set_leverage_success", {"symbol": TICKER, "leverage": LEVERAGE})
        except InvalidRequestError as e:
            # Kod 110043 oznacza, że dźwignia jest już ustawiona na tę wartość. To nie jest błąd.
            if "110043" in str(e):
                log("set_leverage_info", {"message": "Leverage already set to the desired value.", "leverage": LEVERAGE})
            else:
                log("set_leverage_error", {"error": str(e)})
                return  # Zakończ, jeśli wystąpił inny błąd przy ustawianiu dźwigni
        # <<< KONIEC: DODANA LOGIKA USTAWIANIA DŹWIGNI >>>

        entry_price = analysis['current_price']
        atr_value_price = analysis['atr_value_5m']

        stop_loss_distance = atr_value_price * ATR_MULTIPLIER
        sl_price = entry_price - stop_loss_distance if strategy_to_open == 'long' else entry_price + stop_loss_distance
        tp_price = entry_price + (abs(entry_price - sl_price) * RRR) if strategy_to_open == 'long' else entry_price - (
                    abs(entry_price - sl_price) * RRR)

        position_value = balance * RISK_PER_TRADE * float(LEVERAGE)
        position_size_units = position_value / entry_price if entry_price > 0 else 0

        place_order_with_sl_tp(
            strategy=strategy_to_open,
            qty=round(position_size_units, QTY_PRECISION),
            sl_price=round(sl_price, 2),
            tp_price=round(tp_price, 2)
        )

    log("job_end", {})


# --- FUNKCJE POMOCNICZE ---

def get_open_positions() -> dict:
    """Pobiera i zwraca otwarte pozycje dla danego tickera w trybie hedge."""
    positions = {}
    try:
        response = session.get_positions(category="linear", symbol=TICKER)
        if response.get('retCode') == 0 and response.get('result', {}).get('list'):
            for pos in response['result']['list']:
                if float(pos.get('size', 0)) > 0:
                    if pos['side'] == 'Buy':
                        positions['long'] = {'size': float(pos['size']), 'avgPrice': float(pos['avgPrice'])}
                    elif pos['side'] == 'Sell':
                        positions['short'] = {'size': float(pos['size']), 'avgPrice': float(pos['avgPrice'])}
        else:
            log("get_positions_warning", {"message": "Empty or invalid response from Bybit", "response": response})
    except Exception as e:
        log("get_positions_unexpected_error", {"error": str(e)})
    return positions


def get_wallet_balance(coin: str = "USDT") -> float | None:
    """Pobiera dostępne saldo dla danej monety z konta Unified Trading."""
    try:
        response = session.get_wallet_balance(accountType="UNIFIED", coin=coin)
        if response.get('retCode') == 0 and response.get('result', {}).get('list'):
            return float(response['result']['list'][0].get('totalWalletBalance', 0))
        log("get_balance_warning", {"message": "Could not retrieve wallet balance.", "response": response})
        return None
    except Exception as e:
        log("get_balance_unexpected_error", {"error": str(e)})
        return None


def place_order_with_sl_tp(strategy: str, qty: float, sl_price: float, tp_price: float):
    """Składa zlecenie rynkowe z jednoczesnym ustawieniem SL i TP."""
    side = "Buy" if strategy == 'long' else "Sell"
    position_idx = 1 if strategy == 'long' else 2

    log("placing_order", {"side": side, "qty": qty, "sl": sl_price, "tp": tp_price, "positionIdx": position_idx})
    try:
        session.place_order(
            category="linear",
            symbol=TICKER,
            side=side,
            orderType="Market",
            qty=str(qty),
            takeProfit=str(tp_price),
            stopLoss=str(sl_price),
            positionIdx=position_idx
        )
        log("order_placed_successfully", {"side": side, "qty": qty})
    except InvalidRequestError as e:
        log("order_placement_error", {"error_code": e.status_code, "error_message": e.message})
    except Exception as e:
        log("order_placement_unexpected_error", {"error": str(e)})


def close_position(strategy: str, qty: float):
    """Zamyka istniejącą pozycję zleceniem rynkowym."""
    side = "Sell" if strategy == 'long' else "Buy"
    position_idx = 1 if strategy == 'long' else 2

    log("closing_position", {"side": side, "qty": qty, "positionIdx": position_idx})
    try:
        session.place_order(
            category="linear",
            symbol=TICKER,
            side=side,
            orderType="Market",
            qty=str(qty),
            reduceOnly=True,
            positionIdx=position_idx
        )
        log("close_order_placed_successfully", {"side": side, "qty": qty})
    except InvalidRequestError as e:
        log("close_order_error", {"error_code": e.status_code, "error_message": e.message})
    except Exception as e:
        log("close_order_unexpected_error", {"error": str(e)})


# --- FUNKCJE POMOCNICZE DLA TESTÓW ---

def get_open_position(symbol):
    """Zwraca pojedynczą otwartą pozycję (dla testów)."""
    try:
        response = session.get_positions(category="linear", symbol=symbol)
        if response['retCode'] == 0:
            for position in response['result']['list']:
                if float(position.get('size', 0)) > 0:
                    return position
    except Exception:
        pass
    return None


def place_market_order(symbol, side, qty, reduce_only=False):
    """Uproszczona funkcja składania zleceń rynkowych (dla testów)."""
    params = {"category": "linear", "symbol": symbol, "side": side, "orderType": "Market", "qty": str(qty)}

    if reduce_only:
        params["reduceOnly"] = True
        params["positionIdx"] = 1 if side == "Sell" else 2
    else:
        params["positionIdx"] = 1 if side == "Buy" else 2

    try:
        response = session.place_order(**params)
        log("place_market_order_test", {"request": params, "response": response})
        return response
    except Exception as e:
        log("place_market_order_test_exception", {"request": params, "error": str(e)})
        return {}


# --- KOMPLETNY MODUŁ TESTÓW STARTOWYCH ---

def run_startup_smoke_tests():
    log("startup_tests_start", {})

    # Krok 1: Sprawdź połączenie z Bybit i saldo
    balance = get_wallet_balance()
    if balance is not None:
        log("startup_test_step1_ok", {"message": "Bybit connection OK", "balance": f"{balance:.2f} USDT"})
    else:
        log("startup_test_step1_fail", {"message": "Bybit connection or API keys failed"});
        sys.exit(1)

    time.sleep(2)

    # Krok 2: Sprawdź połączenie z API serwisu ML
    try:
        response = requests.get(API_URL, timeout=10)
        if response.status_code == 200:
            log("startup_test_step2_ok", {"message": "ML API connection OK", "response": response.json()})
        else:
            log("startup_test_step2_fail",
                {"message": "ML API returned status code", "status_code": response.status_code});
            sys.exit(1)
    except requests.exceptions.RequestException as e:
        log("startup_test_step2_fail", {"message": "Could not connect to ML API", "error": str(e)});
        sys.exit(1)

    time.sleep(2)

    # Krok 3: Pełny test złożenia i zamknięcia zlecenia
    try:
        log("startup_test_step3_start", {"message": "Starting end-to-end order test", "symbol": TEST_TICKER})
        if get_open_position(symbol=TEST_TICKER):
            log("startup_test_step3_fail", {"message": "A test position already exists. Aborting test."});
            sys.exit(1)

        open_resp = place_market_order(symbol=TEST_TICKER, side="Buy", qty=MIN_TEST_QTY)
        if not (open_resp and open_resp.get('retCode') == 0):
            log("startup_test_step3_fail", {"message": "Failed to place test order.", "response": open_resp});
            sys.exit(1)

        log("startup_test_step3_placed", {"message": "Test order placed successfully."});
        time.sleep(5)

        if not get_open_position(symbol=TEST_TICKER):
            log("startup_test_step3_fail", {"message": "Test position did not appear after placing order."});
            sys.exit(1)

        log("startup_test_step3_verified", {"message": "Test position verified."})
        close_resp = place_market_order(symbol=TEST_TICKER, side="Sell", qty=MIN_TEST_QTY, reduce_only=True)

        if not (close_resp and close_resp.get('retCode') == 0):
            log("startup_test_step3_fail", {"message": "Failed to close test position.", "response": close_resp});
            sys.exit(1)

        log("startup_test_step3_closed", {"message": "Test position closed successfully."});
        time.sleep(5)

        if get_open_position(symbol=TEST_TICKER):
            log("startup_test_step3_fail", {"message": "Test position still exists after closing."});
            sys.exit(1)

        log("startup_test_step3_ok", {"message": "End-to-end order test completed successfully."})
    except Exception as e:
        log("startup_test_step3_exception", {"error": str(e)});
        sys.exit(1)
    finally:
        if get_open_position(symbol=TEST_TICKER):
            log("startup_test_cleanup", {"message": "Attempting to close lingering test position."})
            place_market_order(symbol=TEST_TICKER, side="Sell", qty=MIN_TEST_QTY, reduce_only=True)

    log("startup_tests_complete", {})


# --- URUCHOMIENIE BOTA ---

if __name__ == "__main__":
    run_startup_smoke_tests()

    log("bot_startup_successful", {"ticker": TICKER, "risk": RISK_PER_TRADE, "confidence": MIN_CONFIDENCE, "rrr": RRR})

    job()

    schedule.every(5).minutes.at(":01").do(job)
    while True:
        schedule.run_pending()
        time.sleep(1)