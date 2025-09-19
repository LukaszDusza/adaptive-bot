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
TICKER = "ICPUSDT"
RISK_PER_TRADE = 0.02
LEVERAGE = "10"
API_URL = "http://api_service:8080/predict"
POSITION_TIME_EXIT_HOURS = 12
MIN_CONFIDENCE = 0.40
QTY_PRECISION = 1

# --- Konfiguracja Testów i Logowania ---
TEST_TICKER = "USDCUSDT"
MIN_TEST_QTY = "5.1"

# Ustawienie ustrukturyzowanego loggera
logging.basicConfig(level=logging.INFO,
                    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "service": "trading_bot", "message": %(message)s}',
                    datefmt='%Y-%m-%dT%H:%M:%S%z')


def log(event, details):
    logging.info(json.dumps({"event": event, "details": details}))


# --- Konfiguracja API Bybit ---
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


# --- Funkcje Pomocnicze Bybit ---
def get_wallet_balance():
    try:
        response = session.get_wallet_balance(accountType="UNIFIED", coin="USDT")
        if response['retCode'] == 0 and response['result']['list']:
            return float(response['result']['list'][0]['totalWalletBalance'])
        return None
    except Exception:
        return None


def get_open_position(symbol=TICKER):
    try:
        response = session.get_positions(category="linear", symbol=symbol)
        if response['retCode'] == 0:
            for position in response['result']['list']:
                if float(position.get('size', 0)) > 0:
                    return position
    except Exception:
        pass
    return None


def get_current_price(symbol=TICKER):
    try:
        response = session.get_tickers(category="linear", symbol=symbol)
        if response['retCode'] == 0:
            return float(response['result']['list'][0]['markPrice'])
    except Exception:
        return None


def place_market_order(symbol, side, qty, tp_price=None, sl_price=None, reduce_only=False):
    params = {}
    try:
        params = {"category": "linear", "symbol": symbol, "side": side, "orderType": "Market", "qty": str(qty)}
        if reduce_only:
            if side == "Sell":
                params["positionIdx"] = 1
            else:
                params["positionIdx"] = 2
        else:
            if side == "Buy":
                params["positionIdx"] = 1
            else:
                params["positionIdx"] = 2

        if tp_price: params["takeProfit"] = str(tp_price)
        if sl_price: params["stopLoss"] = str(sl_price)
        if reduce_only: params["reduceOnly"] = True

        response = session.place_order(**params)
        log("place_order", {"request": params, "response": response})
        return response
    except Exception as e:
        log("place_order_exception", {"request": params, "error": str(e)})
        return {}


# --- Główna Logika Bota ---
def manage_open_position(position):
    position_side = position['side']
    strategy_to_check = 'long' if position_side == 'Buy' else 'short'
    exit_action_needed = 'EXIT_LONG' if position_side == 'Buy' else 'EXIT_SHORT'
    log("manage_position_start", {"position": position})
    try:
        response = requests.post(API_URL, json={"symbol": TICKER, "strategy": strategy_to_check})
        signal = response.json()
        log("api_response_for_exit", {"raw_signal": signal})
        if signal.get("action") == exit_action_needed and signal.get("confidence", 0) >= MIN_CONFIDENCE:
            log("exit_signal_received", {"signal": signal})
            place_market_order(symbol=TICKER, side="Sell" if position_side == 'Buy' else "Buy", qty=position['size'],
                               reduce_only=True)
            return
    except requests.exceptions.RequestException as e:
        log("api_connection_error", {"error": str(e)})
    entry_time = pd.to_datetime(int(position['createdTime']), unit='ms', utc=True)
    position_age_hours = (pd.Timestamp.now(tz='UTC') - entry_time).total_seconds() / 3600
    if position_age_hours > POSITION_TIME_EXIT_HOURS:
        log("time_exit_triggered", {"position_age_hours": position_age_hours})
        place_market_order(symbol=TICKER, side="Sell" if position_side == 'Buy' else "Buy", qty=position['size'],
                           reduce_only=True)
    else:
        log("no_exit_signal", {"position_age_hours": position_age_hours})


def look_for_new_trade():
    log("look_for_trade_start", {})
    try:
        response_long = requests.post(API_URL, json={"symbol": TICKER, "strategy": "long"})
        pred_long = response_long.json()
        response_short = requests.post(API_URL, json={"symbol": TICKER, "strategy": "short"})
        pred_short = response_short.json()
        log("api_responses_for_entry", {"long_signal": pred_long, "short_signal": pred_short})
    except requests.exceptions.RequestException as e:
        log("api_connection_error", {"error": str(e)})
        return
    long_signal = pred_long if pred_long.get("action") == "ENTER_LONG" and pred_long.get("confidence",
                                                                                         0) >= MIN_CONFIDENCE else None
    short_signal = pred_short if pred_short.get("action") == "ENTER_SHORT" and pred_short.get("confidence",
                                                                                              0) >= MIN_CONFIDENCE else None
    best_signal, trade_side = None, None
    if long_signal and short_signal:
        if long_signal['confidence'] > short_signal['confidence']:
            best_signal, trade_side = long_signal, "Buy"
        else:
            best_signal, trade_side = short_signal, "Sell"
    elif long_signal:
        best_signal, trade_side = long_signal, "Buy"
    elif short_signal:
        best_signal, trade_side = short_signal, "Sell"

    if best_signal:
        log("entry_signal_selected", {"signal": best_signal})
        balance, price = get_wallet_balance(), get_current_price()
        if balance and price:
            try:
                session.set_leverage(category="linear", symbol=TICKER, buyLeverage=str(LEVERAGE),
                                     sellLeverage=str(LEVERAGE))
                log("set_leverage_success", {"symbol": TICKER, "leverage": LEVERAGE})
            except InvalidRequestError as e:
                if "110043" in str(e):
                    log("set_leverage_info",
                        {"message": "Leverage already set to the correct value.", "leverage": LEVERAGE})
                else:
                    log("set_leverage_error", {"error": str(e)})
                    return

            position_value = balance * RISK_PER_TRADE * float(LEVERAGE)
            # <<< ZMIANA: Użycie nowej stałej do poprawnego zaokrąglania ilości >>>
            qty = round(position_value / price, QTY_PRECISION)

            place_market_order(symbol=TICKER, side=trade_side, qty=qty,
                               tp_price=best_signal.get('take_profit_price'),
                               sl_price=best_signal.get('stop_loss_price'))
    else:
        log("no_entry_signal", {"min_confidence": MIN_CONFIDENCE})


def check_for_signal_and_trade():
    log("main_cycle_start", {})
    position = get_open_position(symbol=TICKER)
    if position:
        manage_open_position(position)
    else:
        look_for_new_trade()


def run_startup_smoke_tests():
    log("startup_tests_start", {})
    balance = get_wallet_balance()
    if balance is not None:
        log("startup_test_step1_ok", {"message": "Bybit connection OK", "balance": f"{balance:.2f} USDT"})
    else:
        log("startup_test_step1_fail", {"message": "Bybit connection or API keys failed"});
        sys.exit(1)
    time.sleep(5)
    try:
        response = requests.post(API_URL, json={"symbol": TICKER, "strategy": "long"})
        if response.status_code == 200:
            log("startup_test_step2_ok", {"message": "ML API connection OK"})
        else:
            log("startup_test_step2_fail",
                {"message": "ML API returned status code", "status_code": response.status_code});
            sys.exit(1)
    except requests.exceptions.ConnectionError as e:
        log("startup_test_step2_fail", {"message": "Could not connect to ML API", "error": str(e)});
        sys.exit(1)
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


if __name__ == "__main__":
    run_startup_smoke_tests()
    log("bot_startup_successful", {})
    check_for_signal_and_trade()
    schedule.every().hour.at(":02").do(check_for_signal_and_trade)
    while True:
        schedule.run_pending()
        time.sleep(1)