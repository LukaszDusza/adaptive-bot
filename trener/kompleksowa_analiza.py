"""
Skrypt do kompleksowej analizy wyników predykcji modelu tradingowego.

Po uruchomieniu generuje w folderze zestaw plików .png z wizualizacjami
oraz wyświetla kluczowe statystyki w konsoli.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import sys
import seaborn as sns
from sklearn.metrics import confusion_matrix


# --- Definicje Funkcji Analitycznych ---

def analyze_accuracy_vs_confidence(df, df_trades):
    """Generuje wykres skuteczności w zależności od progu pewności."""
    print("Analiza 1/7: Skuteczność vs. Próg Pewności...")

    proba_cols = [col for col in ['proba_DOWN(0)', 'proba_UP(2)'] if col in df_trades.columns]
    if 'proba_max' not in df_trades.columns and proba_cols:
        df_trades['proba_max'] = df_trades[proba_cols].max(axis=1)

    thresholds = np.arange(0.50, 1.0, 0.01)
    results = []
    for thresh in thresholds:
        subset = df_trades[df_trades['proba_max'] >= thresh]
        trade_count = len(subset)
        accuracy = subset['is_correct'].mean() if trade_count > 0 else np.nan
        results.append({'threshold': thresh, 'accuracy': accuracy, 'trade_count': trade_count})
    results_df = pd.DataFrame(results)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(14, 7))
    color = 'royalblue'
    ax1.set_xlabel('Próg Pewności Modelu (Confidence Threshold)', fontsize=12)
    ax1.set_ylabel('Skuteczność (Accuracy)', color=color, fontsize=12)
    line1 = ax1.plot(results_df['threshold'], results_df['accuracy'], color=color, marker='o', markersize=4,
                     label='Skuteczność')
    ax1.tick_params(axis='y', labelcolor=color, labelsize=10)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    if not results_df['accuracy'].dropna().empty:
        ax1.set_ylim(bottom=max(0, results_df['accuracy'].min() * 0.95))

    ax2 = ax1.twinx()
    color = 'crimson'
    ax2.set_ylabel('Liczba Transakcji (Number of Trades)', color=color, fontsize=12)
    line2 = ax2.plot(results_df['threshold'], results_df['trade_count'], color=color, marker='x', markersize=4,
                     linestyle='--', label='Liczba Transakcji')
    ax2.tick_params(axis='y', labelcolor=color, labelsize=10)

    plt.title('Skuteczność Modelu vs. Próg Pewności', fontsize=16, pad=20)
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    output_filename = "1_skutecznosc_vs_pewnosc.png"
    plt.savefig(output_filename)
    plt.close()
    print(f"-> Wykres '{output_filename}' został zapisany.")


def analyze_equity_and_drawdowns(df_trades):
    """Generuje wykres krzywej kapitału oraz analizę obsunięć."""
    print("Analiza 2/7: Krzywa Kapitału i Obsunięcia Kapitału...")

    df_trades['pnl'] = df_trades['is_correct'].apply(lambda x: 2 if x else -1)
    df_trades['equity'] = df_trades['pnl'].cumsum()

    running_max = df_trades['equity'].cummax()
    drawdown = df_trades['equity'] - running_max
    drawdown_pct = (drawdown / running_max).replace([np.inf, -np.inf], 0)

    max_drawdown = drawdown.min()
    max_drawdown_pct = drawdown_pct.min()

    print(f"  > Maksymalne obsunięcie (Max Drawdown): {max_drawdown:.2f} R")
    print(f"  > Maksymalne obsunięcie procentowe: {max_drawdown_pct:.2%}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(df_trades['timestamp'], df_trades['equity'], color='mediumseagreen',
             label='Krzywa kapitału (Equity Curve)')
    ax1.set_title('Krzywa Kapitału i Analiza Obsunięć (Drawdowns)', fontsize=16)
    ax1.set_ylabel('Skumulowany Zysk/Strata (w "R")', fontsize=12)
    ax1.legend()

    ax2.fill_between(df_trades['timestamp'], drawdown, 0, color='crimson', alpha=0.3)
    ax2.plot(df_trades['timestamp'], drawdown, color='crimson', linewidth=1.0,
             label=f'Obsunięcia (Max: {max_drawdown_pct:.2%})')
    ax2.set_ylabel('Obsunięcie (Drawdown)', fontsize=12)
    ax2.set_xlabel('Data', fontsize=12)
    ax2.legend()

    plt.tight_layout(h_pad=2)
    output_filename = "2_krzywa_kapitalu_i_drawdowns.png"
    plt.savefig(output_filename)
    plt.close()
    print(f"-> Wykres '{output_filename}' został zapisany.")


def analyze_confusion_matrix(df):
    """Generuje macierz pomyłek."""
    print("Analiza 3/7: Macierz Pomyłek...")

    y_true = df['target']
    y_pred = df['prediction']
    cm = confusion_matrix(y_true, y_pred, labels=[2, 1, 0])
    labels = ['FAKTYCZNY WZROST', 'FAKTYCZNY BOK', 'FAKTYCZNY SPADEK']
    columns = ['PREDYKCJA WZROST', 'PREDYKCJA BOK', 'PREDYKCJA SPADEK']
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', xticklabels=columns, yticklabels=labels)
    plt.title('Macierz Pomyłek (pokazuje, jak model się myli)', fontsize=16)
    plt.ylabel('Faktyczna Klasa', fontsize=12)
    plt.xlabel('Przewidywana Klasa', fontsize=12)
    plt.tight_layout()

    output_filename = "3_macierz_pomylek.png"
    plt.savefig(output_filename)
    plt.close()
    print(f"-> Wykres '{output_filename}' został zapisany.")


def analyze_performance_by_time(df_trades):
    """Generuje analizę skuteczności wg godziny i dnia tygodnia."""
    print("Analiza 4/7: Analiza Czasowa...")

    hourly_accuracy = df_trades.groupby(df_trades['timestamp'].dt.hour)['is_correct'].mean()
    hourly_counts = df_trades.groupby(df_trades['timestamp'].dt.hour).size()
    daily_accuracy = df_trades.groupby(df_trades['timestamp'].dt.dayofweek)['is_correct'].mean()
    daily_counts = df_trades.groupby(df_trades['timestamp'].dt.dayofweek).size()
    day_names = ['Pon', 'Wt', 'Śr', 'Czw', 'Pt', 'Sob', 'Niedz']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    color = 'teal'
    ax1_twin = ax1.twinx()
    ax1.bar(hourly_accuracy.index, hourly_accuracy, color=color, alpha=0.7, label='Skuteczność')
    ax1_twin.plot(hourly_counts.index, hourly_counts, color='darkorange', linestyle='--', marker='o',
                  label='Liczba sygnałów')
    ax1.set_title('Średnia Skuteczność i Liczba Sygnałów wg Godziny', fontsize=14)
    ax1.set_xlabel('Godzina Dnia');
    ax1.set_ylabel('Skuteczność');
    ax1_twin.set_ylabel('Liczba Sygnałów')
    ax1.legend(loc='upper left');
    ax1_twin.legend(loc='upper right')

    color = 'purple'
    ax2_twin = ax2.twinx()
    ax2.bar(daily_accuracy.index, daily_accuracy, color=color, alpha=0.7, label='Skuteczność')
    ax2_twin.plot(daily_counts.index, daily_counts, color='darkorange', linestyle='--', marker='o',
                  label='Liczba sygnałów')
    ax2.set_title('Średnia Skuteczność i Liczba Sygnałów wg Dnia Tygodnia', fontsize=14)
    ax2.set_xlabel('Dzień Tygodnia');
    ax2.set_xticks(range(7));
    ax2.set_xticklabels(day_names)
    ax2.set_ylabel('Skuteczność');
    ax2_twin.set_ylabel('Liczba Sygnałów')
    ax2.legend(loc='upper left');
    ax2_twin.legend(loc='upper right')

    plt.tight_layout(h_pad=4)
    output_filename = "4_analiza_czasowa.png"
    plt.savefig(output_filename)
    plt.close()
    print(f"-> Wykres '{output_filename}' został zapisany.")


def analyze_long_vs_short(df_trades):
    """Generuje analizę porównawczą dla sygnałów LONG i SHORT."""
    print("Analiza 5/7: Long vs. Short...")

    longs = df_trades[df_trades['prediction'] == 2]
    shorts = df_trades[df_trades['prediction'] == 0]
    stats = {
        'LONG': {'Skuteczność': longs['is_correct'].mean(), 'Liczba Sygnałów': len(longs)},
        'SHORT': {'Skuteczność': shorts['is_correct'].mean(), 'Liczba Sygnałów': len(shorts)}
    }
    stats_df = pd.DataFrame(stats).T

    print("\n--- Statystyki LONG vs SHORT ---")
    print(stats_df.to_string(float_format="{:.2%}".format))
    print("--------------------------------")

    stats_df.plot(kind='bar', y='Skuteczność', figsize=(10, 6), color=['forestgreen', 'crimson'], alpha=0.8,
                  legend=False)
    plt.title('Porównanie Skuteczności: Sygnały LONG vs. SHORT', fontsize=16)
    plt.xlabel('Typ Sygnału', fontsize=12);
    plt.ylabel('Skuteczność', fontsize=12)
    plt.xticks(rotation=0);
    plt.ylim(0, max(1.0, stats_df['Skuteczność'].max() * 1.1 if not stats_df['Skuteczność'].isnull().all() else 1.0))
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    plt.tight_layout()
    output_filename = "5_analiza_long_vs_short.png"
    plt.savefig(output_filename)
    plt.close()
    print(f"-> Wykres '{output_filename}' został zapisany.")


def analyze_performance_vs_volatility(df_trades):
    """Analizuje skuteczność modelu w zależności od zmienności rynku."""
    print("Analiza 6/7: Skuteczność vs. Zmienność...")

    df_trades['volatility'] = (df_trades['high'] - df_trades['low']) / df_trades['close']
    df_trades['vol_quantile'] = pd.qcut(df_trades['volatility'], 5,
                                        labels=['1. Najniższa', '2. Niska', '3. Średnia', '4. Wysoka', '5. Najwyższa'])
    vol_analysis = df_trades.groupby('vol_quantile', observed=False).agg(accuracy=('is_correct', 'mean'),
                                                                         trade_count=('is_correct', 'size'))

    fig, ax = plt.subplots(figsize=(12, 7))
    vol_analysis['accuracy'].plot(kind='bar', ax=ax, color='darkcyan', alpha=0.8, label='Skuteczność')
    ax.set_ylabel('Skuteczność');
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    ax2 = ax.twinx()
    ax2.plot(ax.get_xticks(), vol_analysis['trade_count'], color='darkorange', marker='o', label='Liczba transakcji')
    ax2.set_ylabel('Liczba Transakcji')

    ax.set_title('Skuteczność Modelu w Różnych Reżimach Zmienności', fontsize=16)
    ax.set_xlabel('Kwintyl Zmienności Rynku', fontsize=12)
    plt.xticks(rotation=0);
    fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))

    plt.tight_layout()
    output_filename = "6_skutecznosc_vs_zmiennosc.png"
    plt.savefig(output_filename)
    plt.close()
    print(f"-> Wykres '{output_filename}' został zapisany.")


def analyze_probability_distribution(df_trades):
    """Generuje wykres gęstości prawdopodobieństw dla poprawnych i błędnych predykcji."""
    print("Analiza 7/7: Rozkład Prawdopodobieństw...")

    if 'proba_max' not in df_trades.columns:
        proba_cols = [col for col in ['proba_DOWN(0)', 'proba_UP(2)'] if col in df_trades.columns]
        df_trades['proba_max'] = df_trades[proba_cols].max(axis=1)

    correct_trades = df_trades[df_trades['is_correct'] == 1]
    incorrect_trades = df_trades[df_trades['is_correct'] == 0]

    plt.figure(figsize=(12, 7))
    sns.kdeplot(correct_trades['proba_max'], color='forestgreen', fill=True, label='Poprawne Predykcje')
    sns.kdeplot(incorrect_trades['proba_max'], color='crimson', fill=True, label='Błędne Predykcje')
    plt.title('Rozkład Pewności Modelu dla Poprawnych vs. Błędnych Transakcji', fontsize=16)
    plt.xlabel('Maksymalne Prawdopodobieństwo (Pewność Modelu)', fontsize=12)
    plt.ylabel('Gęstość', fontsize=12)
    plt.legend()
    plt.tight_layout()

    output_filename = "7_rozkład_prawdopodobienstw.png"
    plt.savefig(output_filename)
    plt.close()
    print(f"-> Wykres '{output_filename}' został zapisany.")


def main():
    """Główna funkcja uruchamiająca wszystkie analizy."""
    print("--- Uruchamianie Kompleksowej Analizy Modelu ---")

    # Krok 1: Wczytanie danych
    input_filename = "final_predictions.csv"
    try:
        df = pd.read_csv(input_filename)
    except FileNotFoundError:
        print(f"\nBŁĄD: Nie znaleziono pliku '{input_filename}'.")
        print("Upewnij się, że plik z wynikami znajduje się w tym samym folderze co skrypt.")
        sys.exit(1)

    print(f"\n> Pomyślnie wczytano dane z '{input_filename}'. Liczba rekordów: {len(df)}.")

    # Krok 2: Przygotowanie danych do analiz
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df_trades = df[df['prediction'] != 1].copy()
    df_trades['is_correct'] = (df_trades['target'] == df_trades['prediction'])

    if df_trades.empty:
        print("\nW pliku nie znaleziono żadnych sygnałów transakcyjnych (predykcji WZROST/SPADEK). Analiza niemożliwa.")
        sys.exit(0)

    # Krok 3: Uruchomienie analiz
    print("\n--- Rozpoczynanie generowania raportów ---")

    analyze_accuracy_vs_confidence(df, df_trades.copy())
    analyze_equity_and_drawdowns(df_trades.copy())
    analyze_confusion_matrix(df)
    analyze_performance_by_time(df_trades.copy())
    analyze_long_vs_short(df_trades.copy())
    analyze_performance_vs_volatility(df_trades.copy())
    analyze_probability_distribution(df_trades.copy())

    print("\n--- Zakończono ---")
    print("Kompleksowa analiza została zakończona. Sprawdź wygenerowane pliki PNG w folderze.")


if __name__ == "__main__":
    main()