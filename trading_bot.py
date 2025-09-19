import os
import time
import schedule
import requests
import pandas as pd
import sys
import json
from pybit.unified_trading import HTTP

# --- Konfiguracja Bota ---
TICKER = "ICPUSDT"
RISK_PER_TRADE = 0.02
LEVERAGE = "10"
API_URL = "http://api_service:8080/predict"
POSITION_TIME_EXIT_HOURS = 12
MIN_CONFIDENCE = 0.60  # Minimalna pewność (60%), aby rozważyć sygnał

# --- Konfiguracja API Bybit (bez zmian) ---
api_key = os.getenv("BYBIT_API_KEY")
api_secret = os.getenv("BYBIT_API_SECRET")
if not api_key or not api_secret:
    print("OSTRZEŻENIE: Używam kluczy demonstracyjnych.")
    api_key = "pvXUTTIFpBuHDndmpf"
    api_secret = "uZnjNGSE4OZ3uHNyNU53XFMr2q9X2dlEJk46"
try:
    # Zmieniono na testnet=False zgodnie z api_service.py
    session = HTTP(testnet=False, api_key=api_key, api_secret=api_secret)
    print("Pomyślnie zainicjowano sesję Bybit dla bota.")
except Exception as e:
    print(f"Błąd inicjalizacji sesji Bybit dla bota: {e}")
    session = None


# --- Funkcje Pomocnicze Bybit (bez zmian) ---
def get_wallet_balance():
    try:
        response = session.get_wallet_balance(accountType="UNIFIED", coin="USDT")
        if response['retCode'] == 0 and response['result']['list']:
            balance = float(response['result']['list'][0]['totalWalletBalance'])
            return balance
        else:
            print(f"Błąd API Bybit przy pobieraniu salda: {response['retMsg']}")
            return None
    except Exception as e:
        print(f"Wyjątek podczas pobierania salda: {e}")
        return None


def get_open_position():
    try:
        response = session.get_positions(category="linear", symbol=TICKER)
        if response['retCode'] == 0:
            positions = response['result']['list']
            if positions and float(positions[0]['size']) > 0:
                print(f"Znaleziono otwartą pozycję: {positions[0]['side']} {positions[0]['size']} {TICKER}")
                return positions[0]
    except Exception as e:
        print(f"Wyjątek podczas sprawdzania pozycji: {e}")
    return None


def get_current_price():
    try:
        response = session.get_tickers(category="linear", symbol=TICKER)
        if response['retCode'] == 0:
            return float(response['result']['list'][0]['markPrice'])
    except Exception as e:
        print(f"Wyjątek podczas pobierania ceny: {e}")
    return None


def place_market_order(side, qty, tp_price, sl_price):
    try:
        # Upewniamy się, że parametry są stringami, zgodnie z wymaganiami Bybit API v5
        response = session.place_order(
            category="linear", symbol=TICKER, side=side, orderType="Market",
            qty=str(qty), takeProfit=str(tp_price), stopLoss=str(sl_price)
        )
        print("Odpowiedź zlecenia od Bybit:")
        print(response)
        return response['retCode'] == 0
    except Exception as e:
        print(f"Wyjątek podczas składania zlecenia: {e}")
        return False


def close_market_order(position_info, reason="UNKNOWN"):
    side = "Sell" if position_info['side'] == "Buy" else "Buy"
    qty = position_info['size']
    try:
        response = session.place_order(
            category="linear", symbol=TICKER, side=side,
            orderType="Market", qty=str(qty), reduceOnly=True
        )
        print(f"Zamknięto pozycję z powodu: {reason}. Odpowiedź z Bybit:")
        print(response)
        return response['retCode'] == 0
    except Exception as e:
        print(f"Wyjątek podczas zamykania pozycji: {e}")
        return False


# --- Główna Logika Bota (DOSTOSOWANA DO NOWEGO API) ---
def manage_open_position(position):
    """Logika zarządzania już otwartą pozycją."""
    position_side = position['side']  # 'Buy' for long, 'Sell' for short
    strategy_to_check = 'long' if position_side == 'Buy' else 'short'
    exit_action_needed = 'EXIT_LONG' if position_side == 'Buy' else 'EXIT_SHORT'

    print(f"[TRADING_BOT] Zarządzanie otwartą pozycją {position_side}. Sprawdzanie sygnału wyjścia...")
    try:
        response = requests.post(API_URL, json={"symbol": TICKER, "strategy": strategy_to_check})
        signal = response.json()
        print(f"[TRADING_BOT] Otrzymano odpowiedź dla zarządzania pozycją: {json.dumps(signal, indent=2)}")

        if signal.get("action") == exit_action_needed and signal.get("confidence", 0) >= MIN_CONFIDENCE:
            print(
                f"[TRADING_BOT] Otrzymano sygnał ZAMKNIĘCIA POZYCJI z pewnością {signal['confidence']:.2%}. Zamykanie...")
            close_market_order(position, reason=f"Signal {exit_action_needed}")
            return
    except requests.exceptions.RequestException as e:
        print(f"[TRADING_BOT] Błąd połączenia z API podczas zarządzania pozycją: {e}")

    # Sprawdzenie wyjścia czasowego jako zabezpieczenie
    entry_time = pd.to_datetime(int(position['createdTime']), unit='ms', utc=True)
    position_age_hours = (pd.Timestamp.now(tz='UTC') - entry_time).total_seconds() / 3600
    print(f"[TRADING_BOT] Pozycja otwarta od {position_age_hours:.2f} godzin.")
    if position_age_hours > POSITION_TIME_EXIT_HOURS:
        print(f"[TRADING_BOT] Pozycja przekroczyła maksymalny czas {POSITION_TIME_EXIT_HOURS}h. Zamykanie...")
        close_market_order(position, reason="Time Exit")
    else:
        print("[TRADING_BOT] Brak sygnału wyjścia. Pozycja pozostaje otwarta.")


def look_for_new_trade():
    """Logika poszukiwania nowej transakcji, gdy nie ma otwartej pozycji."""
    print("[TRADING_BOT] Brak otwartych pozycji. Odpytywanie modeli o sygnał wejścia...")
    try:
        response_long = requests.post(API_URL, json={"symbol": TICKER, "strategy": "long"})
        pred_long = response_long.json()
        print(f"[TRADING_BOT] Odpowiedź dla LONG: {json.dumps(pred_long, indent=2)}")

        response_short = requests.post(API_URL, json={"symbol": TICKER, "strategy": "short"})
        pred_short = response_short.json()
        print(f"[TRADING_BOT] Odpowiedź dla SHORT: {json.dumps(pred_short, indent=2)}")

    except requests.exceptions.RequestException as e:
        print(f"[TRADING_BOT] KRYTYCZNY BŁĄD połączenia z serwerem API: {e}")
        return

    long_signal = pred_long if pred_long.get("action") == "ENTER_LONG" and pred_long.get("confidence",
                                                                                         0) >= MIN_CONFIDENCE else None
    short_signal = pred_short if pred_short.get("action") == "ENTER_SHORT" and pred_short.get("confidence",
                                                                                              0) >= MIN_CONFIDENCE else None

    best_signal = None
    trade_side = None

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
        print(f"[TRADING_BOT] Wybrano sygnał: {best_signal['action']} z pewnością {best_signal['confidence']:.2%}")
        tp_price = best_signal.get('take_profit_price')
        sl_price = best_signal.get('stop_loss_price')

        if trade_side and tp_price and sl_price:
            balance = get_wallet_balance()
            price = get_current_price()
            if balance and price:
                position_value = balance * RISK_PER_TRADE * float(LEVERAGE)
                qty = round(position_value / price, 3)

                print(f"[TRADING_BOT] DECYZJA: Otwieranie pozycji {trade_side}.")
                print(f"Kapitał: ${balance:.2f} | Cena: ${price:.4f} | Wielkość: {qty} {TICKER}")
                print(f"TP={tp_price:.4f}, SL={sl_price:.4f}")

                session.set_leverage(category="linear", symbol=TICKER, buyLeverage=str(LEVERAGE),
                                     sellLeverage=str(LEVERAGE))
                place_market_order(trade_side, qty, tp_price, sl_price)
    else:
        print(f"[TRADING_BOT] Brak sygnału wejścia o wystarczającej pewności (próg: {MIN_CONFIDENCE * 100}%).")


def check_for_signal_and_trade():
    print(f"\n--- {pd.Timestamp.now()} | Uruchamianie cyklu bota ---")
    if not session:
        print("[TRADING_BOT] Sesja Bybit nieaktywna, pomijam cykl.")
        return

    position = get_open_position()
    if position:
        manage_open_position(position)
    else:
        look_for_new_trade()


# --- Testy Startowe (bez zmian) ---
def run_startup_smoke_tests():
    print("\n--- Przeprowadzanie testów startowych (smoke tests) ---")
    print("1. Testowanie połączenia z Bybit API...")
    balance = get_wallet_balance()
    if balance is not None:
        print(f"   ✅ Połączenie z Bybit OK. Saldo: {balance:.2f} USDT.")
    else:
        print("   ❌ KRYTYCZNY BŁĄD: Nie udało się połączyć z Bybit lub klucze API są nieprawidłowe.")
        sys.exit(1)
    print("2. Testowanie połączenia z serwerem predykcyjnym ML...")
    time.sleep(5)
    try:
        response = requests.post(API_URL, json={"symbol": TICKER, "strategy": "long"})
        if response.status_code == 200:
            print(f"   ✅ Połączenie z serwerem ML API OK (status: {response.status_code}).")
        else:
            print(f"   ❌ KRYTYCZNY BŁĄD: Serwer ML API odpowiedział ze statusem {response.status_code}.")
            sys.exit(1)
    except requests.exceptions.ConnectionError as e:
        print(f"   ❌ KRYTYCZNY BŁĄD: Nie można połączyć się z serwerem ML API pod adresem {API_URL}.")
        sys.exit(1)
    print("--- Wszystkie testy startowe zakończone pomyślnie. Bot jest gotowy. ---\n")


# --- Harmonogram (bez zmian) ---
if __name__ == "__main__":
    run_startup_smoke_tests()
    print("Uruchamianie bota tradingowego...")
    check_for_signal_and_trade()
    schedule.every().hour.at(":02").do(check_for_signal_and_trade)
    while True:
        schedule.run_pending()
        time.sleep(1)