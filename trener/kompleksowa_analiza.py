"""
Główny skrypt analityczny - Master Analyzer.

Uruchom ten skrypt, aby przeprowadzić kompletną, wieloaspektową analizę
wytrenowanego modelu. Skrypt wymaga istnienia plików wyjściowych z procesu
treningu: `final_predictions.csv`, `feature_importances.csv` oraz
pliku cache z cechami `.parquet`.

Wyniki (pliki .png i .csv) zostaną zapisane w nowym folderze `analizy_wynikow`.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import sys
import os
import glob
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm

# --- Globalna Konfiguracja ---
OUTPUT_DIR = "analizy_wynikow"
FEATURES_CACHE_DIR = "data_cache/features"


def analyze_returns_distribution(df_trades):
    """Analizuje rozkład zwrotów z pojedynczych transakcji."""
    print("\n[Nowa Analiza] Rozkład Zwrotów i Ryzyko Tłustych Ogonów...")

    if 'pnl' not in df_trades.columns:
        df_trades['pnl'] = df_trades['is_correct'].apply(lambda x: 2 if x else -1)

    # Obliczenia statystyczne
    skewness = df_trades['pnl'].skew()
    kurtosis = df_trades['pnl'].kurtosis()

    print(f"  > Skośność (Skewness) rozkładu zwrotów: {skewness:.3f}")
    print(f"  > Kurtoza (Kurtosis) rozkładu zwrotów: {kurtosis:.3f}")

    # Zapis statystyk do pliku
    pd.DataFrame({'metryka': ['skewness', 'kurtosis'], 'wartosc': [skewness, kurtosis]}).to_csv(
        os.path.join(OUTPUT_DIR, "9_dane_statystyki_zwrotow.csv"), index=False
    )

    # Wizualizacja
    plt.figure(figsize=(12, 7))
    # W naszym przypadku mamy tylko dwie wartości, więc barplot będzie lepszy niż histogram
    pnl_counts = df_trades['pnl'].value_counts().sort_index()
    pnl_counts.plot(kind='bar', color=['crimson', 'forestgreen'], alpha=0.8)

    plt.title('Rozkład Zysków i Strat z Pojedynczych Transakcji', fontsize=16)
    plt.xlabel('Wynik Transakcji (w "R")', fontsize=12)
    plt.ylabel('Liczba Transakcji', fontsize=12)
    plt.xticks(ticks=[0, 1], labels=['Strata (-1R)', 'Zysk (+2R)'], rotation=0)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "9_rozkład_zwrotow.png"));
    plt.close()
    print("-> Pliki PNG i CSV zostały zapisane.")


# W pliku kompleksowa_analiza.py

def analyze_bootstrap(df_trades, num_simulations=5000):
    """
    Przeprowadza analizę Monte Carlo (bootstraping) na wynikach strategii,
    aby ocenić jej statystyczną solidność.
    """
    print(f"\n[Nowa Analiza] Testy Monte Carlo / Bootstraping ({num_simulations} symulacji)...")

    if 'pnl' not in df_trades.columns:
        df_trades['pnl'] = df_trades['is_correct'].apply(lambda x: 2 if x else -1)

    trade_returns = df_trades['pnl'].values
    n_trades = len(trade_returns)

    final_equities = []
    max_drawdowns = []
    example_paths = []

    for i in tqdm(range(num_simulations), desc="Uruchamianie symulacji", leave=False, ncols=100):
        bootstrapped_returns = np.random.choice(trade_returns, size=n_trades, replace=True)
        equity_curve = np.cumsum(bootstrapped_returns)

        final_equities.append(equity_curve[-1])

        running_max = np.maximum.accumulate(np.insert(equity_curve, 0, 0))
        drawdown = equity_curve - running_max[1:]
        max_drawdowns.append(np.min(drawdown) if len(drawdown) > 0 else 0)

        if i < 5:
            example_paths.append(equity_curve)

    final_equities = np.array(final_equities)
    prob_of_loss = np.mean(final_equities <= 0)
    mean_final_equity = np.mean(final_equities)
    percentile_5 = np.percentile(final_equities, 5)
    percentile_95 = np.percentile(final_equities, 95)

    print("\n--- Wyniki Symulacji Monte Carlo ---")
    print(f"Średni końcowy kapitał: {mean_final_equity:.2f} R")
    print(f"Prawdopodobieństwo straty (końcowy kapitał <= 0): {prob_of_loss:.2%}")
    print(f"Przedział ufności 90% dla końcowego kapitału: [{percentile_5:.2f} R, {percentile_95:.2f} R]")
    print(f"Najgorszy prawdopodobny scenariusz (5 percentyl): {percentile_5:.2f} R")
    print("-------------------------------------\n")

    pd.DataFrame({
        'final_equity': final_equities,
        'max_drawdown': max_drawdowns
    }).to_csv(os.path.join(OUTPUT_DIR, "10_dane_monte_carlo.csv"), index=False)

    # === ZMIANA: Rozbicie na dwa osobne wykresy ===

    # 1. Wykres przykładowych ścieżek
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(14, 7))
    original_equity = df_trades['equity'].values
    plt.plot(original_equity, color='black', linewidth=3, label='Oryginalna Krzywa Kapitału')
    for i, path in enumerate(example_paths):
        plt.plot(path, alpha=0.5, label=f'Symulacja #{i + 1}')
    plt.title('Oryginalna vs. Symulowane Krzywe Kapitału', fontsize=16)
    plt.xlabel('Liczba Transakcji');
    plt.ylabel('Kapitał (w "R")');
    plt.legend()
    plt.tight_layout()
    output_filename_paths = "10a_symulacje_monte_carlo_sciezki.png"
    plt.savefig(os.path.join(OUTPUT_DIR, output_filename_paths));
    plt.close()

    # 2. Histogram końcowych wyników
    plt.figure(figsize=(12, 7))
    sns.histplot(final_equities, kde=True, color='navy')
    plt.axvline(mean_final_equity, color='red', linestyle='--', label=f'Średnia ({mean_final_equity:.2f} R)')
    plt.axvline(0, color='black', linestyle='-')
    plt.title('Rozkład Końcowych Wyników Kapitału', fontsize=16)
    plt.xlabel('Końcowy Kapitał (w "R")');
    plt.ylabel('Częstość');
    plt.legend()
    plt.tight_layout()
    output_filename_hist = "10b_symulacje_monte_carlo_histogram.png"
    plt.savefig(os.path.join(OUTPUT_DIR, output_filename_hist));
    plt.close()

    print(f"-> Pliki PNG ('{output_filename_paths}', '{output_filename_hist}') i CSV zostały zapisane.")


def analyze_returns_distribution(df_trades):
    """Analizuje rozkład zwrotów z pojedynczych transakcji."""
    print("\n[Nowa Analiza] Rozkład Zwrotów i Ryzyko Tłustych Ogonów...")

    # Nie musimy już obliczać 'pnl'
    skewness = df_trades['pnl'].skew()
    # ... reszta funkcji bez zmian ...

def analyze_rolling_performance(df_trades, window=252):
    """Oblicza i wizualizuje kluczowe metryki wydajności w oknie kroczącym."""
    print(f"\n[Nowa Analiza] Wydajność w Oknie Kroczącym (okno={window} transakcji)...")

    # Nie musimy już obliczać 'pnl'
    df_trades['rolling_accuracy'] = df_trades['is_correct'].rolling(window=window).mean()
    # ... reszta funkcji bez zmian ...


def analyze_performance(df_preds):
    """Orkiestruje wszystkie analizy związane z wydajnością predykcji."""
    print("\n--- Rozpoczynanie Analizy Wydajności Modelu ---")

    df_preds['timestamp'] = pd.to_datetime(df_preds['timestamp'])

    if set(df_preds['target'].unique()) <= {0, 1}:
        print("Wykryto wyniki z modelu binarnego. Normalizowanie etykiet do 0 (SPADEK) i 2 (WZROST)...")
        df_preds['target'] = df_preds['target'].map({0: 0, 1: 2})
        df_preds['prediction'] = df_preds['prediction'].map({0: 0, 1: 2})
        df_preds.rename(columns={'proba_UP(1)': 'proba_UP(2)'}, inplace=True)

    df_trades = df_preds[df_preds['prediction'] != 1].copy()
    df_trades['is_correct'] = (df_preds['target'] == df_preds['prediction'])

    proba_cols = [col for col in ['proba_DOWN(0)', 'proba_UP(2)'] if col in df_trades.columns]

    if df_trades.empty:
        print("W pliku nie znaleziono żadnych sygnałów transakcyjnych. Pomijanie analiz wydajności.")
        return

    if proba_cols:
        df_trades['proba_max'] = df_trades[proba_cols].max(axis=1)
    else:
        print("OSTRZEŻENIE: Brak kolumn z prawdopodobieństwami. Niektóre analizy zostaną pominięte.")
        df_trades['proba_max'] = 0.5

        # === ZMIANA: Centralne obliczanie PnL i Equity ===
    # Obliczamy te kolumny raz, tutaj, aby były dostępne dla wszystkich poniższych funkcji.
    df_trades['pnl'] = df_trades['is_correct'].apply(lambda x: 2 if x else -1)
    df_trades['equity'] = df_trades['pnl'].cumsum()
    # =================================================

    # Uruchomienie analiz
    analyze_accuracy_vs_confidence(df_trades.copy())
    analyze_equity_and_drawdowns(df_trades.copy())  # Ta funkcja teraz otrzyma dane z kolumną 'equity'
    analyze_confusion_matrix(df_preds)
    analyze_performance_by_time(df_trades.copy())
    analyze_long_vs_short(df_trades.copy())
    analyze_performance_vs_volatility(df_trades.copy())
    analyze_probability_distribution(df_trades.copy())
    analyze_rolling_performance(df_trades.copy())
    analyze_returns_distribution(df_trades.copy())
    analyze_bootstrap(df_trades.copy())
    analyze_returns_distribution(df_trades.copy())


def analyze_accuracy_vs_confidence(df_trades):
    print("\n[Analiza 1/7] Skuteczność vs. Próg Pewności...")
    # ... (kod funkcji bez zmian, tylko dodany zapis do CSV i zmiana ścieżki zapisu)
    thresholds = np.arange(0.50, 1.0, 0.01)
    results = []
    for thresh in thresholds:
        subset = df_trades[df_trades['proba_max'] >= thresh]
        trade_count = len(subset)
        accuracy = subset['is_correct'].mean() if trade_count > 0 else np.nan
        results.append({'threshold': thresh, 'accuracy': accuracy, 'trade_count': trade_count})
    results_df = pd.DataFrame(results)

    # Zapis CSV
    results_df.to_csv(os.path.join(OUTPUT_DIR, "1_dane_skutecznosc_vs_pewnosc.csv"), index=False)

    # Wykres
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(14, 7))
    color = 'royalblue';
    ax1.set_xlabel('Próg Pewności Modelu', fontsize=12);
    ax1.set_ylabel('Skuteczność (Accuracy)', color=color, fontsize=12)
    line1 = ax1.plot(results_df['threshold'], results_df['accuracy'], color=color, marker='o', markersize=4,
                     label='Skuteczność')
    ax1.tick_params(axis='y', labelcolor=color);
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    if not results_df['accuracy'].dropna().empty: ax1.set_ylim(bottom=max(0, results_df['accuracy'].min() * 0.95))
    ax2 = ax1.twinx();
    color = 'crimson';
    ax2.set_ylabel('Liczba Transakcji', color=color, fontsize=12)
    line2 = ax2.plot(results_df['threshold'], results_df['trade_count'], color=color, marker='x', markersize=4,
                     linestyle='--', label='Liczba Transakcji')
    ax2.tick_params(axis='y', labelcolor=color)
    plt.title('Skuteczność Modelu vs. Próg Pewności', fontsize=16, pad=20)
    lines = line1 + line2;
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    plt.savefig(os.path.join(OUTPUT_DIR, "1_skutecznosc_vs_pewnosc.png"));
    plt.close()
    print("-> Pliki PNG i CSV zostały zapisane.")


def analyze_equity_and_drawdowns(df_trades):
    """Generuje wykres krzywej kapitału oraz analizę obsunięć."""
    print("\n[Analiza 2/7] Krzywa Kapitału i Obsunięcia...")

    # Nie musimy już obliczać 'pnl' i 'equity', bo są już w df_trades
    running_max = df_trades['equity'].cummax()
    df_trades['drawdown'] = df_trades['equity'] - running_max
    df_trades['drawdown_pct'] = (df_trades['drawdown'] / running_max).replace([np.inf, -np.inf], 0)

    # ... reszta funkcji bez zmian ...


def analyze_confusion_matrix(df):
    print("\n[Analiza 3/7] Macierz Pomyłek...")
    # ... (kod funkcji bez zmian, tylko zmiana ścieżki zapisu)
    y_true = df['target'];
    y_pred = df['prediction']
    cm = confusion_matrix(y_true, y_pred, labels=[2, 1, 0])
    labels = ['FAKTYCZNY WZROST', 'FAKTYCZNY BOK', 'FAKTYCZNY SPADEK']
    columns = ['PREDYKCJA WZROST', 'PREDYKCJA BOK', 'PREDYKCJA SPADEK']
    row_sums = cm.sum(axis=1);
    row_sums[row_sums == 0] = 1
    cm_normalized = cm.astype('float') / row_sums[:, np.newaxis]
    cm_df = pd.DataFrame(cm_normalized, index=labels, columns=columns)

    # Zapis CSV
    cm_df.to_csv(os.path.join(OUTPUT_DIR, "3_dane_macierz_pomylek.csv"))

    # Wykres
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', xticklabels=columns, yticklabels=labels)
    plt.title('Macierz Pomyłek', fontsize=16)
    plt.ylabel('Faktyczna Klasa')
    plt.xlabel('Przewidywana Klasa')
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, "3_macierz_pomylek.png"));
    plt.close()
    print("-> Pliki PNG i CSV zostały zapisane.")


def analyze_performance_by_time(df_trades):
    print("\n[Analiza 4/7] Analiza Czasowa...")
    # ... (dodany zapis do CSV i zmiana ścieżki zapisu) ...
    hourly_analysis = df_trades.groupby(df_trades['timestamp'].dt.hour).agg(accuracy=('is_correct', 'mean'),
                                                                            trade_count=('is_correct', 'size'))
    daily_analysis = df_trades.groupby(df_trades['timestamp'].dt.dayofweek).agg(accuracy=('is_correct', 'mean'),
                                                                                trade_count=('is_correct', 'size'))
    day_names = ['Pon', 'Wt', 'Śr', 'Czw', 'Pt', 'Sob', 'Niedz'];
    daily_analysis.index = daily_analysis.index.map(lambda x: day_names[x])

    hourly_analysis.to_csv(os.path.join(OUTPUT_DIR, "4a_dane_analiza_godzinowa.csv"))
    daily_analysis.to_csv(os.path.join(OUTPUT_DIR, "4b_dane_analiza_dzienna.csv"))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    hourly_analysis['accuracy'].plot(kind='bar', ax=ax1, color='teal', alpha=0.7);
    ax1_twin = ax1.twinx()
    ax1_twin.plot(ax1.get_xticks(), hourly_analysis['trade_count'], color='darkorange', linestyle='--', marker='o')
    ax1.set_title('Analiza wg Godziny');
    ax1.set_xlabel('');
    ax1.set_ylabel('Skuteczność')
    daily_analysis['accuracy'].plot(kind='bar', ax=ax2, color='purple', alpha=0.7)
    ax2_twin = ax2.twinx();
    ax2_twin.plot(ax2.get_xticks(), daily_analysis['trade_count'], color='darkorange', linestyle='--', marker='o')
    ax2.set_title('Analiza wg Dnia Tygodnia');
    ax2.set_xlabel('');
    ax2.set_ylabel('Skuteczność')

    plt.tight_layout(h_pad=4)
    plt.savefig(os.path.join(OUTPUT_DIR, "4_analiza_czasowa.png"));
    plt.close()
    print("-> Pliki PNG i CSV zostały zapisane.")


def analyze_long_vs_short(df_trades):
    print("\n[Analiza 5/7] Long vs. Short...")
    # ... (dodany zapis do CSV i zmiana ścieżki zapisu) ...
    longs = df_trades[df_trades['prediction'] == 2]
    shorts = df_trades[df_trades['prediction'] == 0]
    stats = {
        'LONG': {'Skuteczność': longs['is_correct'].mean(), 'Liczba Sygnałów': len(longs)},
        'SHORT': {'Skuteczność': shorts['is_correct'].mean(), 'Liczba Sygnałów': len(shorts)}
    }
    stats_df = pd.DataFrame(stats).T
    print("\n--- Statystyki LONG vs SHORT ---");
    print(stats_df.to_string(float_format="{:.2%}".format));
    print("--------------------------------")
    stats_df.to_csv(os.path.join(OUTPUT_DIR, "5_dane_long_vs_short.csv"))

    stats_df.plot(kind='bar', y='Skuteczność', figsize=(10, 6), color=['forestgreen', 'crimson'], alpha=0.8,
                  legend=False)
    plt.title('Porównanie Skuteczności: Sygnały LONG vs. SHORT');
    plt.xlabel('Typ Sygnału');
    plt.ylabel('Skuteczność')
    plt.xticks(rotation=0);
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "5_analiza_long_vs_short.png"));
    plt.close()
    print("-> Pliki PNG i CSV zostały zapisane.")


def analyze_performance_vs_volatility(df_trades):
    print("\n[Analiza 6/7] Skuteczność vs. Zmienność...")
    # ... (dodany zapis do CSV i zmiana ścieżki zapisu) ...
    df_trades['volatility'] = (df_trades['high'] - df_trades['low']) / df_trades['close']
    df_trades['vol_quantile'] = pd.qcut(df_trades['volatility'], 5,
                                        labels=['1. Najniższa', '2. Niska', '3. Średnia', '4. Wysoka', '5. Najwyższa'],
                                        duplicates='drop')
    vol_analysis = df_trades.groupby('vol_quantile', observed=False).agg(accuracy=('is_correct', 'mean'),
                                                                         trade_count=('is_correct', 'size'))
    vol_analysis.to_csv(os.path.join(OUTPUT_DIR, "6_dane_skutecznosc_vs_zmiennosc.csv"))

    fig, ax = plt.subplots(figsize=(12, 7))
    vol_analysis['accuracy'].plot(kind='bar', ax=ax, color='darkcyan', alpha=0.8);
    ax.set_ylabel('Skuteczność')
    ax2 = ax.twinx();
    ax2.plot(ax.get_xticks(), vol_analysis['trade_count'], color='darkorange', marker='o');
    ax2.set_ylabel('Liczba Transakcji')
    ax.set_title('Skuteczność w Różnych Reżimach Zmienności');
    ax.set_xlabel('Kwintyl Zmienności')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "6_skutecznosc_vs_zmiennosc.png"));
    plt.close()
    print("-> Pliki PNG i CSV zostały zapisane.")


def analyze_probability_distribution(df_trades):
    print("\n[Analiza 7/7] Rozkład Prawdopodobieństw...")
    # ... (dodany zapis do CSV i zmiana ścieżki zapisu) ...
    correct_trades = df_trades[df_trades['is_correct'] == 1]['proba_max']
    incorrect_trades = df_trades[df_trades['is_correct'] == 0]['proba_max']
    pd.DataFrame({'correct_proba': correct_trades, 'incorrect_proba': incorrect_trades}).to_csv(
        os.path.join(OUTPUT_DIR, "7_dane_rozkład_prawdopodobienstw.csv"), index=False)

    plt.figure(figsize=(12, 7))
    sns.kdeplot(correct_trades, color='forestgreen', fill=True, label='Poprawne Predykcje')
    sns.kdeplot(incorrect_trades, color='crimson', fill=True, label='Błędne Predykcje')
    plt.title('Rozkład Pewności Modelu');
    plt.xlabel('Maksymalne Prawdopodobieństwo');
    plt.ylabel('Gęstość');
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, "7_rozkład_prawdopodobienstw.png"));
    plt.close()
    print("-> Pliki PNG i CSV zostały zapisane.")


def analyze_features():
    """Orkiestruje wszystkie analizy związane z cechami."""
    print("\n--- Rozpoczynanie Analizy Cech i Jakości Danych ---")

    # 1. Analiza ważności cech
    try:
        df_imp = pd.read_csv("feature_importances.csv")
        analyze_feature_importance(df_imp)
    except FileNotFoundError:
        print("OSTRZEŻENIE: Nie znaleziono pliku 'feature_importances.csv'. Pomijanie analizy ważności cech.")

    # 2. Analiza jakości danych (korelacja, dystrybucja, PCA)
    try:
        list_of_files = glob.glob(f'{FEATURES_CACHE_DIR}/*.parquet')
        if not list_of_files: raise FileNotFoundError
        latest_file = max(list_of_files, key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0)
        df_full_features = pd.read_parquet(latest_file)

        # ZMIANA: Usuwamy globalne .dropna()
        # df_full_features.dropna(inplace=True)

        feature_cols = [col for col in df_full_features.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'turnover', 'target']]
        df_features_only = df_full_features[feature_cols]

        # Wywołujemy analizy na ramce danych, która może zawierać NaN
        analyze_correlation(df_features_only)
        analyze_feature_distribution(df_features_only)
        analyze_pca_redundancy(df_features_only)

    except (ValueError, FileNotFoundError):
        print("OSTRZEŻENIE: Nie znaleziono plików cache z cechami. Pomijanie analiz jakości danych.")
        return

def analyze_feature_importance(df_imp):
    print("\n[Analiza 8/10] Ważność Cech...")
    zero_importance_features = df_imp[df_imp['importance'] == 0]
    print(f"  > Znaleziono {len(zero_importance_features)} cech o zerowej ważności (szum).")
    df_plot = df_imp[df_imp['importance'] > 0]
    plt.figure(figsize=(12, 10)); sns.barplot(x="importance", y="feature", data=df_plot.head(30), palette="viridis", hue="feature", legend=False)
    plt.title('Top 30 najważniejszych cech'); plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "8a_top_30_cech.png")); plt.close()
    plt.figure(figsize=(12, 10)); sns.barplot(x="importance", y="feature", data=df_plot.tail(30), palette="rocket_r", hue="feature", legend=False)
    plt.title('30 cech o najniższej (ale niezerowej) ważności'); plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "8b_ostatnie_30_cech.png")); plt.close()
    print("-> Pliki PNG zostały zapisane.")

def analyze_correlation(df_features, top_n=30):
    print("\n[Analiza 9/10] Korelacja Cech...")
    # ZMIANA: Lokalna obsługa NaN
    df_subset = df_features.iloc[:, :top_n].dropna()

    if df_subset.empty:
        print("  > OSTRZEŻENIE: Po usunięciu NaN nie ma danych do analizy korelacji. Pomijanie.")
        return

    correlation_matrix = df_subset.corr()
    correlation_matrix.to_csv(os.path.join(OUTPUT_DIR, "9_dane_korelacja.csv"))
    plt.figure(figsize=(18, 15)); sns.heatmap(correlation_matrix, cmap='coolwarm')
    plt.title(f'Macierz Korelacji dla Top {top_n} Cech'); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "9_macierz_korelacji.png")); plt.close()
    print("-> Pliki PNG i CSV zostały zapisane.")

def analyze_feature_distribution(df_features, top_n=12):
    print("\n[Analiza 10/10] Dystrybucja Cech...")
    # ZMIANA: Lokalna obsługa NaN
    df_subset = df_features.iloc[:, :top_n].dropna()

    if df_subset.empty:
        print("  > OSTRZEŻENIE: Po usunięciu NaN nie ma danych do analizy dystrybucji. Pomijanie.")
        return

    fig, axes = plt.subplots(int(np.ceil(top_n / 4)), 4, figsize=(20, 5 * int(np.ceil(top_n / 4))))
    axes = axes.flatten()
    for i, col in enumerate(df_subset.columns):
        sns.histplot(df_subset[col], kde=True, ax=axes[i], bins=50); axes[i].set_title(col, fontsize=10)
    for j in range(i + 1, len(axes)): axes[j].set_visible(False)
    fig.suptitle(f'Dystrybucja Wartości dla Top {top_n} Cech'); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "10_dystrybucja_cech.png")); plt.close()
    print("-> Plik PNG został zapisany.")

def analyze_pca_redundancy(df_features):
    print("\n[Dodatkowe] Redundancja Cech (PCA)...")
    # ZMIANA: Lokalna obsługa NaN
    df_ready = df_features.dropna()

    if df_ready.empty:
        print("  > OSTRZEŻENIE: Po usunięciu NaN nie ma danych do analizy PCA. Pomijanie.")
        return

    scaler = StandardScaler(); scaled_features = scaler.fit_transform(df_ready); pca = PCA(); pca.fit(scaled_features)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(12, 7)); plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='.', linestyle='--')
    plt.title('Skumulowana Wyjaśniona Wariancja (Analiza PCA)'); plt.xlabel('Liczba Składowych Głównych'); plt.ylabel('Procent Wyjaśnionej Wariancji')
    plt.axhline(y=0.95, color='r', linestyle=':', label='95% progu'); plt.axhline(y=0.90, color='g', linestyle=':', label='90% progu'); plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "11_analiza_pca.png")); plt.close()
    print("-> Plik PNG został zapisany.")


def analyze_feature_importance(df_imp):
    print("\n[Analiza 8/10] Ważność Cech...")
    zero_importance_features = df_imp[df_imp['importance'] == 0]
    print(f"  > Znaleziono {len(zero_importance_features)} cech o zerowej ważności (szum).")

    df_plot = df_imp[df_imp['importance'] > 0]

    plt.figure(figsize=(12, 10));
    sns.barplot(x="importance", y="feature", data=df_plot.head(30), palette="viridis")
    plt.title('Top 30 najważniejszych cech');
    plt.tight_layout();
    plt.savefig(os.path.join(OUTPUT_DIR, "8a_top_30_cech.png"));
    plt.close()

    plt.figure(figsize=(12, 10));
    sns.barplot(x="importance", y="feature", data=df_plot.tail(30), palette="rocket_r")
    plt.title('30 cech o najniższej (ale niezerowej) ważności');
    plt.tight_layout();
    plt.savefig(os.path.join(OUTPUT_DIR, "8b_ostatnie_30_cech.png"));
    plt.close()
    print("-> Pliki PNG zostały zapisane.")


def analyze_correlation(df_features, top_n=30):
    print("\n[Analiza 9/10] Korelacja Cech...")
    df_subset = df_features.iloc[:, :top_n];
    correlation_matrix = df_subset.corr()
    correlation_matrix.to_csv(os.path.join(OUTPUT_DIR, "9_dane_korelacja.csv"))

    plt.figure(figsize=(18, 15));
    sns.heatmap(correlation_matrix, cmap='coolwarm')
    plt.title(f'Macierz Korelacji dla Top {top_n} Cech');
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "9_macierz_korelacji.png"));
    plt.close()
    print("-> Pliki PNG i CSV zostały zapisane.")


def analyze_feature_distribution(df_features, top_n=12):
    print("\n[Analiza 10/10] Dystrybucja Cech...")
    df_subset = df_features.iloc[:, :top_n]
    fig, axes = plt.subplots(int(np.ceil(top_n / 4)), 4, figsize=(20, 5 * int(np.ceil(top_n / 4))))
    axes = axes.flatten()
    for i, col in enumerate(df_subset.columns):
        sns.histplot(df_subset[col], kde=True, ax=axes[i], bins=50);
        axes[i].set_title(col, fontsize=10)
    for j in range(i + 1, len(axes)): axes[j].set_visible(False)
    fig.suptitle(f'Dystrybucja Wartości dla Top {top_n} Cech');
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, "10_dystrybucja_cech.png"));
    plt.close()
    print("-> Plik PNG został zapisany.")


def analyze_pca_redundancy(df_features):
    # Ta analiza jest na tyle specyficzna, że nie generuje łatwo intepretowalnego CSV, zostawiamy PNG
    print("\n[Dodatkowe] Redundancja Cech (PCA)...")
    scaler = StandardScaler();
    scaled_features = scaler.fit_transform(df_features);
    pca = PCA();
    pca.fit(scaled_features)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    plt.figure(figsize=(12, 7));
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='.', linestyle='--')
    plt.title('Skumulowana Wyjaśniona Wariancja (Analiza PCA)');
    plt.xlabel('Liczba Składowych Głównych');
    plt.ylabel('Procent Wyjaśnionej Wariancji')
    plt.axhline(y=0.95, color='r', linestyle=':', label='95% progu');
    plt.axhline(y=0.90, color='g', linestyle=':', label='90% progu');
    plt.legend()

    plt.savefig(os.path.join(OUTPUT_DIR, "11_analiza_pca.png"));
    plt.close()
    print("-> Plik PNG został zapisany.")


def main():
    """Główna funkcja uruchamiająca wszystkie analizy."""
    print("--- Uruchamianie Głównego Skryptu Analitycznego ---")

    # Tworzenie folderu na wyniki
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Wczytanie danych predykcji
    try:
        df_preds = pd.read_csv("final_predictions.csv")
        print(f"\n> Pomyślnie wczytano dane predykcji z 'final_predictions.csv'.")
        # Uruchomienie analiz wydajności
        analyze_performance(df_preds)
    except FileNotFoundError:
        print("\nOSTRZEŻENIE: Nie znaleziono pliku 'final_predictions.csv'. Pomijanie analiz wydajności modelu.")

    # Uruchomienie analiz cech
    analyze_features()

    print("\n--- Wszystkie Analizy Zakończone ---")
    print(f"Sprawdź wygenerowane pliki w folderze: '{OUTPUT_DIR}'")


if __name__ == "__main__":
    main()