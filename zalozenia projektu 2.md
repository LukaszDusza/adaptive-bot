# Wytyczne Projektu: Adaptacyjny Bot Tradingowy Python

## 1. Wprowadzenie

Niniejszy dokument stanowi kompleksowe wytyczne dla rozwoju adaptacyjnego bota tradingowego w Pythonie. Celem bota jest dynamiczne dostosowywanie strategii handlowej do panujących reżimów rynkowych, wykorzystując wskaźniki ADX i ATR do oceny warunków rynkowych. Bot będzie przełączał się między strategią `mean-reversion` (konsolidacja) a strategią `trend-following` (trend), zarządzając jednocześnie ryzykiem i minimalizując wpływ szumu rynkowego.

## 2. Cele i Założenia Strategii

* **Adaptacja do Reżimu Rynkowego:** Bot będzie automatycznie przełączał strategie w zależności od stanu rynku (konsolidacja vs. trend).
* **Minimalizacja Ryzyka:** Stosowanie zaawansowanych technik zarządzania ryzykiem, w tym dynamicznego stop-loss i wielkości pozycji opartej na ATR.
* **Redukcja Fałszywych Sygnałów:** Filtracja warunków rynkowych za pomocą ATR i inteligentne podejście do ponownego wejścia w pozycję.
* **Wysoka Trafność i Stosunek Zysku do Ryzyka:** Dążenie do wysokiej trafności w konsolidacji i wysokiego RRR w trendzie.
* **Docelowa Skuteczność:** Minimalny próg rentowności strategii to 50-60% wygranych transakcji.
* **Stosunek Zysku do Ryzyka (RRR):** Docelowo 1:2.
* **Zarządzanie Kapitałem:** Maksymalne ryzyko na pojedynczą transakcję to **2%** kapitału.

## 3. Architektura Systemu i Moduły

Bot będzie działał w oparciu o modułową architekturę, co ułatwi rozwój, testowanie i utrzymanie.

### 3.1. Główny Cykl Decyzyjny (Co 15 minut)

1.  **Pobranie Danych:** Na początku każdej nowej świecy 15-minutowej pobierz aktualne dane OHLCV.
2.  **Ocena Zmienności (ATR):**
    * Oblicz `ATR(14)`.
    * Porównaj `ATR(14)` z jej długoterminową średnią kroczącą (`SMA(ATR(14), 100)`).
    * **Filtr "Stagnacja":** Jeśli `ATR(14) < 0.5 * SMA(ATR(14), 100)`:
        * **Akcja:** Dezaktywuj otwieranie nowych pozycji. Zarządzaj istniejącymi pozycjami (nie zamykaj, ale nie przesuwaj SL na BE).
    * **Filtr "Panika":** Jeśli `ATR(14) > 3.0 * SMA(ATR(14), 100)`:
        * **Akcja:** Dezaktywuj otwieranie nowych pozycji. Zacieśnij stop-lossy dla istniejących pozycji (np. do $1.5 \times ATR$).
    * **Optymalny Zakres:** W przeciwnym razie, kontynuuj proces decyzyjny.
3.  **Ocena Reżimu Rynku (ADX):**
    * Oblicz `ADX(14)`.
    * **Reżim Konsolidacji:** Jeśli `ADX < 25`:
        * **Akcja:** Aktywuj `Moduł Strategii Konsolidacji`.
    * **Reżim Trendowy:** Jeśli `ADX > 25`:
        * **Akcja:** Aktywuj `Moduł Strategii Trendowej`.
    * **Strefa Przejściowa (20-25):** W tym zakresie bot nie otwiera nowych pozycji, ale monitoruje nachylenie ADX jako wczesny sygnał zmiany. Zarządza istniejącymi pozycjami.

### 3.2. Moduł Strategii Konsolidacji (Mean-Reversion)

* **Aktywacja:** `ADX(14) < 25` i rynek w optymalnym zakresie zmienności.
* **Wskaźnik:** Oscylator Stochastyczny `(5, 3, 3)`.
* **Sygnał Wejścia (Kupno):**
    1.  %K i %D poniżej 20 (strefa wyprzedania).
    2.  %K przecina %D od dołu.
    3.  Cena blisko poziomu wsparcia (np. w odległości $< 0.25 \times ATR$).
    4.  Potwierdzenie wzrostu wolumenu.
* **Sygnał Wejścia (Sprzedaż):**
    1.  %K i %D powyżej 80 (strefa wykupienia).
    2.  %K przecina %D od góry.
    3.  Cena blisko poziomu oporu (np. w odległości $< 0.25 \times ATR$).
    4.  Potwierdzenie wzrostu wolumenu.
* **Cel Zysku (Take Profit):** Stały, blisko przeciwległego ograniczenia zakresu (np. najbliższy opór dla pozycji długiej).
* **Stop-Loss:** Początkowy, dynamicznie obliczony na podstawie `2 * ATR(14)` poniżej wsparcia (kupno) lub powyżej oporu (sprzedaż).

### 3.3. Moduł Strategii Trendowej (Trend-Following)

* **Aktywacja:** `ADX(14) > 25` (i rosnące nachylenie ADX) oraz rynek w optymalnym zakresie zmienności.
* **Wskaźnik:** System DMI (ADX, +DI, -DI).
* **Sygnał Wejścia (Kupno):**
    1.  `ADX(14) > 25` i `ADX` ma dodatnie nachylenie (wzrasta).
    2.  `+DI` przecina `-DI` od dołu.
    3.  Cena powyżej 50-okresowej EMA (opcjonalny filtr sentymentu).
* **Sygnał Wejścia (Sprzedaż):**
    1.  `ADX(14) > 25` i `ADX` ma dodatnie nachylenie.
    2.  `-DI` przecina `+DI` od dołu.
    3.  Cena poniżej 50-okresowej EMA (opcjonalny filtr sentymentu).
* **Cel Zysku (Take Profit):** Otwarty (brak stałego TP).
* **Stop-Loss:** Początkowy `2 * ATR(14)`. Następnie **kroczący stop-loss (Trailing Stop)** oparty na **Chandelier Exit** (`N * ATR(14)` od szczytu/dołka, gdzie `N` jest optymalizowanym mnożnikiem, np. 2.5-3.5).

## 4. Zarządzanie Ryzykiem i Pozycjami

### 4.1. Dynamiczne Zarządzanie Wielkością Pozycji

* **Stałe Ryzyko:** Ryzyko na transakcję to **2% kapitału**.
* **Kalkulacja Wielkości Pozycji:**
    $$ \text{Wielkość Pozycji} = \frac{\text{Kapitał} \times \text{Ryzyko \%}}{\text{Odległość Stop-Loss w \$}} $$
    Odległość Stop-Loss jest zawsze obliczana dynamicznie na podstawie `ATR(14)`.

### 4.2. Inteligentny Mechanizm Ponownego Wejścia

* **Warunki Ponownego Wejścia (po aktywacji Stop-Loss):**
    1.  Pozycja zamknięta przez SL w ciągu ostatnich `N=3` świec.
    2.  Pierwotny reżim rynkowy (wg ADX) nie uległ zmianie.
    3.  Ruch cenowy, który aktywował SL, nie przekroczył "progu błędu" (np. $1.5 \times$ pierwotna odległość SL). To ma odróżnić szum od błędu w analizie.
    4.  Pojawienie się **nowego, ważnego sygnału** wejścia, zgodnego z logiką aktywnego modułu.
    5.  Maksymalnie **jedna** próba ponownego wejścia na jeden pierwotny sygnał.

### 4.3. Zarządzanie Pozycjami w Trakcie Zmiany Reżimu

Każda otwarta pozycja musi być "otagowana" reżimem, w którym została otwarta.

* **Przejście Konsolidacja -> Trend (ADX rośnie > 25):**
    * **Akcja:** Anuluj stały Take Profit. Konwertuj Stop Loss na **kroczący stop-loss (Chandelier Exit)** zgodny z logiką strategii trendowej.
* **Przejście Trend -> Konsolidacja (ADX spada < 25):**
    * **Akcja:** Utrzymaj kroczący stop-loss, ale **zablokuj jego przesuwanie** (tj. staje się statycznym SL na ostatnim "high-water mark"). Zidentyfikuj i ustaw nowy, stały cel zysku na najbliższym istotnym oporze/wsparciu.

## 5. Stos Technologiczny i Implementacja

### 5.1. Język Programowania

* **Python 3.9+**

### 5.2. Frameworki i Biblioteki

* **Backtesting i Optymalizacja:**
    * **`vectorbt`**: Zalecany do masowej optymalizacji parametrów i szybkiego testowania odporności.
    * **`backtesting.py`**: Alternatywa dla prototypowania i wizualizacji, jeśli preferowana jest bardziej "świeca po świecy" logika.
    * (`backtrader` jako opcja do szczegółowego, realistycznego testowania finalnej wersji).
* **Live Trading i Połączenie z Giełdą:**
    * **`ccxt`**: Ujednolicone API do giełd kryptowalut (Bybit).
* **Analiza Danych i Wskaźniki:**
    * **`pandas`**: Do manipulacji danymi OHLCV.
    * **`pandas_ta`**: Do łatwego obliczania wskaźników technicznych (ADX, ATR, Stochastic, +DI, -DI).
* **Obsługa Asynchroniczna (opcjonalnie, dla skalowalności):**
    * **`asyncio`**
    * **`aiohttp`** (dla zapypań HTTP)
    * **`websockets`** (dla subskrypcji danych w czasie rzeczywistym)

## 6. Protokół Backtestingu i Walidacji

### 6.1. Rygorystyczne Testowanie

* **Unikanie Lookahead Bias:** Należy bezwzględnie upewnić się, że w momencie podejmowania decyzji algorytm ma dostęp tylko do danych historycznych.
* **Analiza Walk-Forward:** Podziel dane na segmenty "in-sample" (optymalizacja) i "out-of-sample" (testowanie), powtarzając proces krocząco.
* **Realistyczna Symulacja Kosztów:** Włącz prowizje, opłaty i realistyczne poślizgi cenowe (slippage) w backtestingu.
* **Istotność Statystyczna:** Upewnij się, że strategia wygenerowała wystarczającą liczbę transakcji (min. 100-200) dla wiarygodnych wyników.

### 6.2. Metryki Wydajności

* **Zwrot Skorygowany o Ryzyko:**
    * Współczynnik Sharpe'a
    * Współczynnik Sortino
* **Metryki Ryzyka:**
    * Maksymalne Obsunięcie Kapitału (Maximum Drawdown)
    * Współczynnik Calmara
* **Analiza Krzywej Kapitału:** Wizualna inspekcja stabilności i wzrostu.
* **Dezagregacja Wyników:** Kluczowe dla adaptacyjnego bota. Należy generować osobne raporty dla:
    * Transakcji w reżimie konsolidacji.
    * Transakcji w reżimie trendowym.
    * Transakcji utrzymywanych podczas zmiany reżimu.
    * Skuteczności samego klasyfikatora reżimów.

## 7. Struktura Projektu (Proponowana)