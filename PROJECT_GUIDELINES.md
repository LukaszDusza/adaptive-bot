# Wytyczne Projektu: Adaptacyjny Bot Tradingowy - Kompletny Przewodnik

## 1. Wprowadzenie i Cele Projektu

### 1.1 Opis Projektu
Niniejszy dokument stanowi kompleksowe wytyczne dla rozwoju zaawansowanego, adaptacyjnego bota tradingowego w Pythonie. System wykracza poza prosty, binarny model przełączania strategii, implementując wielostanowy model operacyjny oparty na dynamicznej analizie reżimów rynkowych.

### 1.2 Główne Cele
- **Adaptacja do Reżimu Rynkowego:** Automatyczne przełączanie między strategią mean-reversion (konsolidacja) a trend-following (trend)
- **Minimalizacja Ryzyka:** Zaawansowane techniki zarządzania ryzykiem z dynamicznym stop-loss i wielkością pozycji opartą na ATR
- **Redukcja Fałszywych Sygnałów:** Wielowarstwowa filtracja warunków rynkowych
- **Wysoka Efektywność:** Docelowo 50-60% wygranych transakcji z stosunkiem zysku do ryzyka 1:2
- **Maksymalne ryzyko na transakcję:** 2% kapitału

## 2. Architektura Systemu i Klasyfikacja Reżimów Rynkowych

### 2.1 Czterostanowa Macierz Reżimów Rynkowych

| Reżim | Warunek ADX(14) | Warunek Zmienności (ATR vs SMA) | Aktywny Moduł | Dozwolone Działania |
|-------|-----------------|--------------------------------|---------------|-------------------|
| **Trendowy** | ADX > 25 | 0.5 < ATR/SMA < 3.0 | Trend-Following | Pełne operacje |
| **Konsolidacji** | ADX < 25 | 0.5 < ATR/SMA < 3.0 | Mean-Reversion | Pełne operacje |
| **Stagnacyjny** | Dowolny | ATR/SMA < 0.5 | Brak | Tylko zarządzanie pozycjami |
| **Paniki** | Dowolny | ATR/SMA > 3.0 | Brak | Zakaz nowych pozycji + zacieśnianie SL |

### 2.2 Wskaźnik ADX - Podstawowy Mechanizm Przełączania
- **Reżim Konsolidacji:** ADX < 25 (brak wyraźnego trendu)
- **Reżim Trendowy:** ADX > 25 (silny trend kierunkowy)  
- **Strefa Przejściowa:** ADX 20-25 (bez nowych pozycji, tylko zarządzanie)
- **Analiza nachylenia ADX:** Rosnące nachylenie sygnalizuje wzmacniający się reżim

### 2.3 Filtr Zmienności ATR - Główny Wyłącznik
- **Stagnacja:** ATR(14) < 0.5 × SMA(ATR(14), 100) → Dezaktywacja nowych pozycji
- **Panika:** ATR(14) > 3.0 × SMA(ATR(14), 100) → Dezaktywacja + zacieśnianie SL
- **Optymalny zakres:** 0.5 < ATR/SMA < 3.0 → Normalny handel

## 3. Główny Cykl Decyzyjny (Co 15 minut)

### 3.1 Sekwencja Działań
1. **Pobranie Danych:** Aktualne dane OHLCV dla nowej świecy 15-minutowej
2. **Ocena Zmienności (ATR):**
   - Oblicz ATR(14) i porównaj z SMA(ATR(14), 100)
   - Zastosuj odpowiedni filtr (Stagnacja/Panika/Normalny)
3. **Ocena Reżimu (ADX):**
   - Oblicz ADX(14) 
   - Określ aktywny reżim i moduł strategii
4. **Wykonanie Strategii:** Zgodnie z aktywnym modułem

## 4. Moduł Strategii Konsolidacji (Mean-Reversion)

### 4.1 Warunki Aktywacji
- ADX(14) < 25
- ATR w optymalnym zakresie zmienności
- Oscylator Stochastyczny (5, 3, 3) jako główny wskaźnik

### 4.2 Sygnały Wejścia

#### Pozycja Długa (Kupno):
1. %K i %D poniżej 20 (strefa wyprzedania)
2. %K przecina %D od dołu
3. Cena blisko poziomu wsparcia (< 0.25 × ATR)
4. Potwierdzenie wzrostu wolumenu

#### Pozycja Krótka (Sprzedaż):
1. %K i %D powyżej 80 (strefa wykupienia)
2. %K przecina %D od góry  
3. Cena blisko poziomu oporu (< 0.25 × ATR)
4. Potwierdzenie wzrostu wolumenu

### 4.3 Zarządzanie Pozycją
- **Take Profit:** Stały cel blisko przeciwległego ograniczenia zakresu
- **Stop-Loss:** 2 × ATR(14) za poziomem wsparcia/oporu

## 5. Moduł Strategii Trendowej (Trend-Following)

### 5.1 Warunki Aktywacji
- ADX(14) > 25 z dodatnim nachyleniem
- ATR w optymalnym zakresie
- System DMI (ADX, +DI, -DI) jako główny wskaźnik

### 5.2 Sygnały Wejścia

#### Pozycja Długa (Kupno):
1. ADX(14) > 25 i rosnące nachylenie
2. +DI przecina -DI od dołu
3. Opcjonalnie: cena powyżej 50-okresowej EMA

#### Pozycja Krótka (Sprzedaż):
1. ADX(14) > 25 i rosnące nachylenie
2. -DI przecina +DI od dołu
3. Opcjonalnie: cena poniżej 50-okresowej EMA

### 5.3 Zarządzanie Pozycją
- **Take Profit:** Otwarty (bez stałego TP)
- **Stop-Loss:** Początkowy 2 × ATR(14), następnie kroczący stop-loss (Chandelier Exit)
- **Chandelier Exit:** N × ATR(14) od szczytu/dołka (N = 2.5-3.5)

## 6. Zarządzanie Ryzykiem i Pozycjami

### 6.1 Dynamiczne Zarządzanie Wielkością Pozycji
- **Stałe ryzyko:** 2% kapitału na transakcję
- **Formuła:** `Wielkość Pozycji = (Kapitał × Ryzyko%) / (Odległość Stop-Loss w $)`
- **Samoregulacja:** Wyższa zmienność → szerszy SL → mniejsza pozycja

### 6.2 Inteligentny Mechanizm Ponownego Wejścia

#### Warunki Ponownego Wejścia (po aktywacji Stop-Loss):
1. Pozycja zamknięta przez SL w ciągu ostatnich 3 świec
2. Pierwotny reżim rynkowy (wg ADX) nie uległ zmianie
3. Ruch cenowy nie przekroczył "progu błędu" (1.5 × pierwotna odległość SL)
4. Pojawienie się nowego, ważnego sygnału wejścia
5. Maksymalnie jedna próba ponownego wejścia na sygnał

### 6.3 Zarządzanie podczas Zmiany Reżimu

#### Przejście Konsolidacja → Trend (ADX > 25):
- Anuluj stały Take Profit
- Konwertuj Stop Loss na kroczący (Chandelier Exit)

#### Przejście Trend → Konsolidacja (ADX < 25):
- Utrzymaj kroczący SL, ale zablokuj przesuwanie
- Ustaw nowy stały TP na najbliższym oporze/wsparciu

## 7. Implementacja Techniczna

### 7.1 Stos Technologiczny
- **Język:** Python 3.9+
- **Backtesting:** `vectorbt` (optymalizacja), `backtesting.py` (prototypowanie)
- **Live Trading:** `ccxt` (API giełd kryptowalut)
- **Analiza:** `pandas` (dane), `pandas_ta` (wskaźniki)
- **Asynchroniczność:** `asyncio`, `aiohttp`, `websockets`

### 7.2 Struktura Modułowa
```
adaptive-bot/
├── main.py                 # Główny punkt wejścia
├── core/
│   ├── regime_detector.py  # Detekcja reżimów (ADX/ATR)
│   ├── consolidation_strategy.py  # Strategia konsolidacji
│   ├── trend_strategy.py   # Strategia trendowa
│   └── risk_manager.py     # Zarządzanie ryzykiem
├── indicators/
│   ├── technical.py        # Wskaźniki techniczne
│   └── filters.py          # Filtry sygnałów
├── backtesting/
│   └── validator.py        # Walidacja strategii
└── data/
    └── provider.py         # Dostawca danych
```

## 8. Protokół Backtestingu i Walidacji

### 8.1 Rygorystyczne Testowanie
- **Unikanie Lookahead Bias:** Tylko dane historyczne w decyzjach
- **Analiza Walk-Forward:** Segmenty in-sample/out-of-sample
- **Realistyczne Koszty:** Prowizje, opłaty, slippage
- **Istotność Statystyczna:** Min. 100-200 transakcji

### 8.2 Kluczowe Metryki
- **Zwrot skorygowany o ryzyko:** Sharpe, Sortino
- **Metryki ryzyka:** Maximum Drawdown, Calmar Ratio
- **Analiza krzywej kapitału:** Stabilność wzrostu
- **Dezagregacja wyników:** Osobne raporty dla każdego reżimu

### 8.3 Wymagane Analizy
- Transakcje w reżimie konsolidacji
- Transakcje w reżimie trendowym  
- Transakcje podczas zmiany reżimu
- Skuteczność klasyfikatora reżimów

## 9. Parametry Docelowe

### 9.1 Cele Wydajnościowe
- **Trafność:** 50-60% wygranych transakcji
- **Stosunek Zysku do Ryzyka:** 1:2
- **Maksymalne ryzyko:** 2% kapitału na transakcję
- **Maximum Drawdown:** < 20%
- **Sharpe Ratio:** > 1.5

### 9.2 Parametry Wskaźników
- **ADX:** okres 14, próg 25
- **ATR:** okres 14, mnożniki 0.5/3.0 dla filtrów
- **Stochastic:** (5, 3, 3), progi 20/80
- **Stop-Loss:** 2 × ATR(14)
- **Trailing Stop:** 2.5-3.5 × ATR(14)

## 10. Plan Wdrożenia

### 10.1 Fazy Rozwoju
1. **Faza 1:** Implementacja wskaźników i detektora reżimów
2. **Faza 2:** Rozwój modułów strategii (konsolidacja/trend)
3. **Faza 3:** System zarządzania ryzykiem i pozycjami
4. **Faza 4:** Backtesting i optymalizacja parametrów
5. **Faza 5:** Implementacja live trading z API giełd

### 10.2 Kryteria Sukcesu
- Pozytywne wyniki w analizie walk-forward
- Stabilna krzywa kapitału bez długich drawdownów
- Skuteczne przełączanie między reżimami
- Rentowność po uwzględnieniu kosztów transakcyjnych

---

## Uwagi Końcowe

Ten dokument stanowi kompletny przewodnik dla implementacji adaptacyjnego bota tradingowego. Kluczem do sukcesu jest rygorystyczne testowanie każdego komponentu i stopniowe budowanie systemu z naciskiem na zarządzanie ryzykiem. System musi być nie tylko rentowny, ale przede wszystkim odporny na różne warunki rynkowe i zdolny do adaptacji w czasie rzeczywistym.

**Ważne:** Wszystkie parametry podane w tym dokumencie stanowią punkt wyjścia i wymagają optymalizacji przez backtesting na konkretnych danych rynkowych przed wdrożeniem w środowisku produkcyjnym.