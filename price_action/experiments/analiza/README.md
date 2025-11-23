# Analiza Historii Tradingu z Bybit (1-21 listopada 2025)

Ten katalog zawiera kompletne dane o wszystkich pozycjach handlowych z Bybit dla 16 tickerów.

## Struktura Plików

Dla każdego tickera (np. SOLUSDT) są 3 pliki CSV:

### 1. `{TICKER}_closed_pnl.csv` - Zamknięte Pozycje
**Najważniejsze kolumny:**
- `closedPnl` - Zysk/strata netto (po opłatach)
- `avgEntryPrice` / `avgExitPrice` - Średnie ceny wejścia/wyjścia
- `side` - Kierunek (Buy/Sell)
- `leverage` - Zastosowana dźwignia
- `openFee` / `closeFee` - Opłaty
- `qty` - Wielkość pozycji
- `createdTime` / `updatedTime` - Timestampy
- `fillCount` - Liczba egzekucji składających się na trade

**Zastosowanie:** Analiza rentowności poszczególnych tradów, win rate, średni zysk/strata

### 2. `{TICKER}_executions.csv` - Egzekucje
**Najważniejsze kolumny:**
- `execPrice` / `execQty` - Cena i wielkość egzekucji
- `execFee` - Opłata za egzekucję
- `isMaker` - Czy egzekucja była maker (true) czy taker (false)
- `feeRate` - Stawka opłaty (0.02% maker, 0.055% taker)
- `execTime` - Timestamp egzekucji
- `orderType` - Market/Limit
- `side` - Buy/Sell
- `markPrice` - Mark price w momencie egzekucji

**Zastosowanie:** Analiza slippage'u, maker/taker ratio, timing egzekucji

### 3. `{TICKER}_orders.csv` - Ordery
**Najważniejsze kolumny:**
- `orderStatus` - Status (Filled, Cancelled, Rejected)
- `avgPrice` - Średnia cena wypełnienia
- `takeProfit` / `stopLoss` - Poziomy TP/SL
- `cumExecQty` / `cumExecValue` - Skumulowana wielkość/wartość
- `cumExecFee` - Łączne opłaty
- `createdTime` / `updatedTime` - Timestampy
- `timeInForce` - GTC/IOC/FOK
- `reduceOnly` - Czy order tylko redukuje pozycję

**Zastosowanie:** Analiza skuteczności SL/TP, cancelled orders, order execution quality

## Szybka Analiza

### Uruchom skrypt podsumowania:
```bash
python analyze_trading_data.py
```

### Ręczna analiza w Pythonie:
```python
import pandas as pd

# Wczytaj dane dla tickera
pnl = pd.read_csv('SOLUSDT_closed_pnl.csv')
exec_df = pd.read_csv('SOLUSDT_executions.csv')
orders = pd.read_csv('SOLUSDT_orders.csv')

# Podstawowe statystyki
print(f"Total PnL: {pnl['closedPnl'].astype(float).sum():.2f} USDT")
print(f"Win Rate: {(pnl['closedPnl'].astype(float) > 0).sum() / len(pnl) * 100:.1f}%")
print(f"Avg Win: {pnl[pnl['closedPnl'].astype(float) > 0]['closedPnl'].astype(float).mean():.2f} USDT")
print(f"Avg Loss: {pnl[pnl['closedPnl'].astype(float) < 0]['closedPnl'].astype(float).mean():.2f} USDT")
```

## Kluczowe Pytania do Analizy

### 1. Gdzie bot traci pieniądze?
- **Kierunek:** Które strony (LONG vs SHORT) są bardziej rentowne?
- **Wielkość pozycji:** Czy większe pozycje mają gorszy win rate?
- **Czas trwania:** Czy długie/krótkie tray są bardziej zyskowne?
- **Leverage:** Jaki leverage daje najlepsze wyniki?

### 2. Warunki rynkowe
- **Volatility:** Porównaj ATR z zyskownością
- **Trend:** Czy bot lepiej radzi sobie w trendach czy konsolidacjach?
- **Time of day:** Czy pewne godziny są bardziej zyskowne?
- **Day of week:** Czy weekend/weekdays mają znaczenie?

### 3. Execution Quality
- **Slippage:** Porównaj `orderPrice` vs `avgPrice` (z orders.csv)
- **Maker/Taker ratio:** Ile % egzekucji to maker vs taker?
- **Fill rate:** Jaki % orderów zostaje cancelled vs filled?
- **Partial fills:** Czy partial fills wpływają na PnL?

### 4. Risk Management
- **SL effectiveness:** Jak często SL jest triggered?
- **TP effectiveness:** Jak często TP jest triggered?
- **Max drawdown per ticker:** Jaki jest największy ciąg strat?
- **MAE/MFE:** Maximum Adverse/Favorable Excursion

## Przykładowe Analizy

### Analiza per strona (LONG vs SHORT):
```python
pnl['side_clean'] = pnl['side'].str.strip()
print(pnl.groupby('side_clean')['closedPnl'].agg(['sum', 'mean', 'count']))
```

### Top 10 najgorszych tradów:
```python
worst_trades = pnl.nsmallest(10, 'closedPnl')[['symbol', 'side', 'closedPnl', 'avgEntryPrice', 'avgExitPrice', 'createdTime_readable']]
print(worst_trades)
```

### Analiza opłat:
```python
total_fees = pnl['openFee'].astype(float) + pnl['closeFee'].astype(float)
print(f"Total fees paid: {total_fees.sum():.2f} USDT")
print(f"Avg fee per trade: {total_fees.mean():.4f} USDT")
```

## Podsumowanie (1-21 listopada 2025)

**Ogólne statystyki:**
- Total PnL: **-22.17 USDT**
- Total Trades: **880**
- Overall Win Rate: **65.9%**

**TOP 3 Najlepsze tickery:**
1. **1000PEPEUSDT**: +3.34 USDT (65.2% WR, 46 trades)
2. **ETHUSDT**: +3.04 USDT (67.3% WR, 49 trades)
3. **NEARUSDT**: +1.24 USDT (69.9% WR, 93 trades)

**TOP 3 Najgorsze tickery:**
1. **PONKEUSDT**: -5.21 USDT (66.3% WR, 104 trades)
2. **KASUSDT**: -4.09 USDT (59.4% WR, 64 trades)
3. **ONDOUSDT**: -3.71 USDT (64.3% WR, 28 trades)

## Wnioski Wstępne

⚠️ **Pomimo wysokiego win rate (65.9%), bot stracił -22.17 USDT**

To wskazuje na problemy z:
1. **Risk/Reward Ratio** - Średnia wygrana < Średnia strata
2. **Position Sizing** - Możliwe że przegrywające tray mają większe pozycje
3. **Opłaty** - Taker fees mogą zjadać zyski
4. **Slippage** - Market orders mogą powodować gorsze ceny

## Następne Kroki Analizy

1. **Oblicz Risk/Reward Ratio** dla każdego tickera
2. **Analizuj timeframe** - czy pewne godziny dnia są lepsze?
3. **Sprawdź correlation** między tickerami - czy tracą jednocześnie?
4. **Zbadaj SL/TP triggers** - czy są dobrze ustawione?
5. **Porównaj z backtestem** - czy live results odpowiadają oczekiwaniom?

---

*Dane wygenerowane przez: `fetch_bybit_history.py`*
*Data pobrania: 21 listopada 2025*
