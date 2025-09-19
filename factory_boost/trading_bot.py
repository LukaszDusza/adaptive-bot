import os
import time
import schedule
import requests
import pandas as pd
import sys
from pybit.unified_trading import HTTP

# --- Konfiguracja Bota (bez zmian) ---
TICKER = "ICPUSDT"
RISK_PER_TRADE = 0.02
LEVERAGE = "10"
API_URL = "http://api_service:8080/predict"
POSITION_TIME_EXIT_HOURS = 12

# --- Konfiguracja API Bybit (bez zmian) ---
api_key = os.getenv("BYBIT_API_KEY")
api_secret = os.getenv("BYBIT_API_SECRET")
if not api_key or not api_secret:
    print("OSTRZEŻENIE: Używam kluczy demonstracyjnych.")
    api_key = "pvXUTTIFpBuHDndmpf"
    api_secret = "uZnjNGSE4OZ3uHNyNU53XFMr2q9X2dlEJk46"
try:
    session = HTTP(testnet=True, api_key=api_key, api_secret=api_secret)
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


# ... (reszta funkcji pomocniczych: get_open_position, get_current_price, place_market_order, close_market_order - bez zmian) ...
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
        response = session.place_order(
            category="linear", symbol=TICKER, side=side, orderType="Market",
            qty=str(qty), takeProfit=str(tp_price), stopLoss=str(sl_price)
        )
        print("Odpowiedź zlecenia od Bybit:");
        print(response)
        return response['retCode'] == 0
    except Exception as e:
        print(f"Wyjątek podczas składania zlecenia: {e}");
        return False


def close_market_order(position_info):
    side = "Sell" if position_info['side'] == "Buy" else "Buy"
    qty = position_info['size']
    try:
        response = session.place_order(
            category="linear", symbol=TICKER, side=side,
            orderType="Market", qty=str(qty), reduceOnly=True
        )
        print(f"Zamknięto pozycję z powodu Time Exit. Odpowiedź z Bybit:");
        print(response)
        return response['retCode'] == 0
    except Exception as e:
        print(f"Wyjątek podczas zamykania pozycji: {e}");
        return False


# --- Główna Logika Bota (bez zmian) ---
def check_for_signal_and_trade():
    # ... (cała ta funkcja pozostaje bez zmian) ...
    print(f"\n--- {pd.Timestamp.now()} | Uruchamianie cyklu bota ---")
    if not session: print("Sesja Bybit nieaktywna, pomijam cykl."); return
    position = get_open_position()
    if position:
        entry_time = pd.to_datetime(int(position['createdTime']), unit='ms', utc=True)
        position_age_hours = (pd.Timestamp.now(tz='UTC') - entry_time).total_seconds() / 3600
        print(f"Pozycja otwarta od {position_age_hours:.2f} godzin.")
        if position_age_hours > POSITION_TIME_EXIT_HOURS:
            print(f"Pozycja przekroczyła maksymalny czas {POSITION_TIME_EXIT_HOURS}h. Zamykanie...")
            close_market_order(position)
        else:
            print("Pozycja jest aktywna i w limicie czasowym. Czekam na TP/SL.")
        return
    print("Brak otwartych pozycji. Odpytywanie modeli o sygnał...")
    try:
        response_long = requests.post(API_URL, json={"symbol": TICKER, "strategy": "long"})
        pred_long = response_long.json()
        response_short = requests.post(API_URL, json={"symbol": TICKER, "strategy": "short"})
        pred_short = response_short.json()
    except Exception as e:
        print(f"Błąd połączenia z serwerem API predykcji: {e}");
        return
    signal_side = None;
    tp_price = None;
    sl_price = None
    if pred_long.get("prediction") == "BUY":
        print(f"Otrzymano sygnał BUY dla LONG z pewnością {pred_long.get('confidence_buy') * 100:.2f}%")
        signal_side = "Buy";
        tp_price = pred_long.get('take_profit_price');
        sl_price = pred_long.get('stop_loss_price')
    elif pred_short.get("prediction") == "BUY":
        print(f"Otrzymano sygnał BUY dla SHORT z pewnością {pred_short.get('confidence_buy') * 100:.2f}%")
        signal_side = "Sell";
        tp_price = pred_short.get('take_profit_price');
        sl_price = pred_short.get('stop_loss_price')
    else:
        print("Brak wyraźnego sygnału BUY od modeli. Czekam na kolejną godzinę.");
        return
    if signal_side and tp_price and sl_price:
        balance = get_wallet_balance();
        price = get_current_price()
        if balance and price:
            position_value = balance * RISK_PER_TRADE * float(LEVERAGE)
            qty = round(position_value / price, 3)
            print(
                f"Sygnał: {signal_side} | Kapitał: ${balance:.2f} | Cena: ${price:.4f} | Wielkość pozycji: {qty} {TICKER}")
            print(f"Składanie zlecenia: TP={tp_price:.4f}, SL={sl_price:.4f}")
            session.set_leverage(category="linear", symbol=TICKER, buyLeverage=LEVERAGE, sellLeverage=LEVERAGE)
            place_market_order(signal_side, qty, tp_price, sl_price)


# --- NOWA FUNKCJA: Testy Startowe ---
def run_startup_smoke_tests():
    """Przeprowadza serię testów przy starcie, aby upewnić się, że system jest gotowy."""
    print("\n--- Przeprowadzanie testów startowych (smoke tests) ---")

    # Test 1: Połączenie z Bybit i weryfikacja kluczy
    print("1. Testowanie połączenia z Bybit API...")
    balance = get_wallet_balance()
    if balance is not None:
        print(f"   ✅ Połączenie z Bybit OK. Saldo: {balance:.2f} USDT.")
    else:
        print("   ❌ KRYTYCZNY BŁĄD: Nie udało się połączyć z Bybit lub klucze API są nieprawidłowe.")
        sys.exit(1)  # Zakończ działanie skryptu z kodem błędu

    # Test 2: Połączenie z serwerem predykcyjnym (api_service)
    print("2. Testowanie połączenia z serwerem predykcyjnym ML...")
    # Dajemy serwerowi API kilka sekund na pełne uruchomienie się w środowisku Docker
    time.sleep(5)
    try:
        response = requests.post(API_URL, json={"symbol": TICKER, "strategy": "long"})
        if response.status_code == 200:
            print(f"   ✅ Połączenie z serwerem ML API OK (status: {response.status_code}).")
        else:
            print(f"   ❌ KRYTYCZNY BŁĄD: Serwer ML API odpowiedział ze statusem {response.status_code}.")
            print(f"   Odpowiedź: {response.text}")
            sys.exit(1)
    except requests.exceptions.ConnectionError as e:
        print(f"   ❌ KRYTYCZNY BŁĄD: Nie można połączyć się z serwerem ML API pod adresem {API_URL}.")
        print(f"   Upewnij się, że kontener 'api_service' jest uruchomiony. Szczegóły: {e}")
        sys.exit(1)

    print("--- Wszystkie testy startowe zakończone pomyślnie. Bot jest gotowy. ---\n")


# --- Harmonogram ---
if __name__ == "__main__":
    # Krok 1: Uruchom testy startowe
    run_startup_smoke_tests()

    # Krok 2: Jeśli testy przejdą, uruchom główną logikę bota
    print("Uruchamianie bota tradingowego...")
    check_for_signal_and_trade()

    # Krok 3: Ustaw harmonogram na przyszłe cykle
    schedule.every().hour.at(":02").do(check_for_signal_and_trade)

    while True:
        schedule.run_pending()
        time.sleep(1)