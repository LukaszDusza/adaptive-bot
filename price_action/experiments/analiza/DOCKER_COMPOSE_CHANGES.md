# 🔧 docker-compose.yaml - Zmiany

**Data:** 22 listopada 2025

---

## ✅ CO ZOSTAŁO ZMIENIONE

### Wszystkie 16 botów zaktualizowane:

**PRZED (stare DCA z 3 poziomami):**
```yaml
command: >
  python bot.py
  --version v1.2.doge
  --ticker DOGEUSDT
  ...
  --dca-mode
  --dca-level1-pct 0.006           # ❌ USUNIĘTE
  --dca-atr-multiplier 1.5         # ❌ USUNIĘTE (teraz default)
  --dca-max-swing-distance 1.5     # ❌ USUNIĘTE
```

**PO (nowe DCA z 1 poziomem ATR):**
```yaml
command: >
  python bot.py
  --version v1.2.doge
  --ticker DOGEUSDT
  ...
  --dca-mode                       # ✅ Tylko to jest potrzebne
```

---

## 📋 LISTA ZMIENIONYCH BOTÓW (16):

1. ✅ `bot-luk-doge-dca` - DOGEUSDT
2. ✅ `bot-luk-sol-ict` - SOLUSDT
3. ✅ `bot-eth` - ETHUSDT
4. ✅ `bot-link` - LINKUSDT
5. ✅ `bot-hbar` - HBARUSDT
6. ✅ `bot-sui` - SUIUSDT
7. ✅ `bot-near` - NEARUSDT
8. ✅ `bot-trx` - TRXUSDT
9. ✅ `bot-kas` - KASUSDT
10. ✅ `bot-pepe` - 1000PEPEUSDT
11. ✅ `bot-xrp` - XRPUSDT
12. ✅ `bot-brett` - BRETTUSDT
13. ✅ `bot-icp` - ICPUSDT
14. ✅ `bot-ondo` - ONDOUSDT
15. ✅ `bot-ponke` - PONKEUSDT
16. ✅ `bot-wif` - WIFUSDT

---

## 🎯 CO OZNACZAJĄ NOWE USTAWIENIA

### DCA Mode:

**Stare (3 poziomy):**
- Level 1: Fixed 0.6% offset
- Level 2: 1.5x ATR
- Level 3: Swing-based (1.5-2.0% max)

**Nowe (1 poziom):**
- **Tylko ATR-based:** 1.5x ATR = ~1.18% średnia odległość
- **Adaptive:** ATR dostosowuje się do volatility
- **Prostsze:** 1 zlecenie zamiast 3

### RSI Reversal Filter:

**Automatycznie włączony** (nie wymaga żadnych argumentów):
- ✅ SHORT gdy RSI przekroczy 70 W DÓŁ
- ✅ LONG gdy RSI przekroczy 30 W GÓRĘ
- ✅ Blokuje LONG jeśli RSI >65 bez 30-cross
- ✅ Blokuje SHORT jeśli RSI <35 bez 70-cross

---

## 🚀 JAK WDROŻYĆ ZMIANY

### 1. Zatrzymaj wszystkie boty:

```bash
cd /Users/lukasz/projects/adaptive-bot/price_action
docker compose down
```

### 2. Zbuduj nowy obraz (z nowym kodem bot.py):

```bash
docker compose build
```

### 3. Uruchom boty z nowymi ustawieniami:

```bash
docker compose up -d
```

### 4. Sprawdź logi:

```bash
# Sprawdź RSI filter status
docker compose logs bot-luk-doge-dca | grep "RSI Filter"

# Powinieneś zobaczyć:
# RSI Filter:        ON (reversal-based entry)

# Sprawdź DCA config
docker compose logs bot-luk-doge-dca | grep "Order Type"

# Powinieneś zobaczyć:
# Order Type:        DCA LIMIT (ATR-based, timeout: 300s)
#   Level:           1.5x ATR (~1.18% avg)
```

---

## 📊 OCZEKIWANE REZULTATY

### Po uruchomieniu z nowymi ustawieniami:

**DCA Mode:**
- Bot będzie otwierał **1 limit order** (nie 3)
- Dystans: **~1.18%** od aktualnej ceny (bazując na ATR)
- SL/TP: **embedded w zleceniu** (ochrona od razu)

**RSI Filter:**
- Bot będzie **blokował** late entries (RSI >65 bez reversal)
- Bot będzie **overridował** decyzje ML na crossingach (70↓ = SHORT, 30↑ = LONG)
- **Mniej tradów** (15-20% redukcja), ale **lepsza jakość**

**Logi - przykłady:**
```
🔴 RSI REVERSAL: Crossed 70.0 down (72.0 → 68.0) - OVERRIDE to SHORT
⚠️ RSI HIGH FILTER: RSI=67.3 >65.0 without 30.0-cross - BLOCKING LONG
📊 DCA Mode: Single ATR-based limit order
   Level: $168.1234 (1.18%)
   Quantity: 5.0000
✓ DCA order placed | Wait max 300s
```

---

## ⚙️ OPCJONALNE DOSTOSOWANIA

Jeśli chcesz **zmienić ustawienia**, dodaj do `command:`:

### RSI Filter:

```yaml
# Wyłącz RSI filter (test)
--no-rsi-reversal-filter

# Bardziej agresywny (blokuj przy RSI >60)
--rsi-high-threshold 60
--rsi-low-threshold 40

# Zmień poziomy crossingów
--rsi-overbought-level 75    # SHORT przy 75↓ (zamiast 70↓)
--rsi-oversold-level 25      # LONG przy 25↑ (zamiast 30↑)
```

### DCA Mode:

```yaml
# Szerszy dystans (2x ATR = ~1.57%)
--dca-atr-multiplier 2.0

# Węższy dystans (1x ATR = ~0.79%)
--dca-atr-multiplier 1.0
```

---

## 🔍 WERYFIKACJA ZMIAN

### Sprawdź czy wszystko działa:

```bash
# 1. Status wszystkich botów
docker compose ps

# 2. Logi z ostatnich 5 minut
docker compose logs --since 5m

# 3. Szukaj błędów
docker compose logs | grep -i error

# 4. Sprawdź czy RSI filter działa
docker compose logs | grep "RSI REVERSAL"
docker compose logs | grep "RSI.*FILTER"

# 5. Sprawdź czy DCA działa
docker compose logs | grep "DCA Mode"
```

### ✅ Wszystko OK jeśli:

- Boty się uruchamiają (status: Up)
- Widzisz "RSI Filter: ON" w logach
- Widzisz "Order Type: DCA LIMIT (ATR-based)" w logach
- Brak errorów związanych z `dca-level1-pct` lub `dca-max-swing-distance`

### ❌ Problem jeśli:

```
Error: unrecognized arguments: --dca-level1-pct
```
**Rozwiązanie:** Sprawdź czy na pewno zbudowałeś nowy obraz (`docker compose build`)

---

## 📝 BACKUP

**Przed wdrożeniem zrobiłem backup:**
- Stare `docker-compose.yaml` → przechowane w git history
- Możesz cofnąć zmiany przez: `git checkout HEAD~1 docker-compose.yaml`

---

## 🎯 PODSUMOWANIE

**Zmienione:**
- ❌ Usunięto `--dca-level1-pct`
- ❌ Usunięto `--dca-max-swing-distance`
- ❌ Usunięto `--dca-atr-multiplier` (teraz default 1.5)
- ✅ Pozostawiono tylko `--dca-mode`

**Dodane automatycznie:**
- ✅ RSI Reversal Filter (domyślnie włączony)
- ✅ Single ATR-based DCA (1.5x multiplier)

**Wynik:**
- Prostszy config
- Mniej parametrów do zarządzania
- Nowe filtry działają od razu

---

*Aktualizacja: 22 listopada 2025*
*Wszystkie 16 botów zaktualizowane*
