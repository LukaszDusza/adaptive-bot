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
    print("Generowanie wykresu: Skuteczność vs. Próg Pewności...")

    # Obliczamy maksymalne prawdopodobieństwo (pewność) dla każdej predykcji
    proba_cols = [col for col in ['proba_DOWN(0)', 'proba_UP(2)'] if col in df_trades.columns]
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

    output_filename = "skutecznosc_vs_pewnosc.png"
    plt.savefig(output_filename)
    plt.close()  # Zamknij figurę, aby zwolnić pamięć
    print(f"-> Wykres '{output_filename}' został pomyślnie wygenerowany.")


def analyze_equity_curve(df_trades):
    """Generuje wykres krzywej kapitału."""
    print("Generowanie wykresu: Krzywa Kapitału...")

    df_trades['pnl'] = df_trades['is_correct'].apply(lambda x: 2 if x else -1)
    df_trades['equity'] = df_trades['pnl'].cumsum()

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(14, 7))
    plt.plot(df_trades['timestamp'], df_trades['equity'], color='mediumseagreen',
             label='Krzywa kapitału (Equity Curve)')
    plt.title('Krzywa Kapitału Symulowanej Strategii (RR=2:1)', fontsize=16)
    plt.xlabel('Data', fontsize=12)
    plt.ylabel('Skumulowany Zysk/Strata (w jednostkach ryzyka "R")', fontsize=12)
    plt.legend()
    plt.tight_layout()

    output_filename = "krzywa_kapitalu.png"
    plt.savefig(output_filename)
    plt.close()
    print(f"-> Wykres '{output_filename}' został pomyślnie wygenerowany.")


def analyze_confusion_matrix(df):
    """Generuje macierz pomyłek."""
    print("Generowanie wykresu: Macierz Pomyłek...")

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

    output_filename = "macierz_pomylek.png"
    plt.savefig(output_filename)
    plt.close()
    print(f"-> Wykres '{output_filename}' został pomyślnie wygenerowany.")


def analyze_performance_by_time(df_trades):
    """Generuje analizę skuteczności wg godziny i dnia tygodnia."""
    print("Generowanie wykresu: Analiza Czasowa...")

    hourly_accuracy = df_trades.groupby(df_trades['timestamp'].dt.hour)['is_correct'].mean()
    hourly_counts = df_trades.groupby(df_trades['timestamp'].dt.hour).size()
    daily_accuracy = df_trades.groupby(df_trades['timestamp'].dt.dayofweek)['is_correct'].mean()
    daily_counts = df_trades.groupby(df_trades['timestamp'].dt.dayofweek).size()
    day_names = ['Pon', 'Wt', 'Śr', 'Czw', 'Pt', 'Sob', 'Niedz']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    # Godziny
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

    # Dni
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
    output_filename = "analiza_czasowa.png"
    plt.savefig(output_filename)
    plt.close()
    print(f"-> Wykres '{output_filename}' został pomyślnie wygenerowany.")


def analyze_long_vs_short(df_trades):
    """Generuje analizę porównawczą dla sygnałów LONG i SHORT."""
    print("Generowanie wykresu i statystyk: Long vs. Short...")

    longs = df_trades[df_trades['prediction'] == 2]
    shorts = df_trades[df_trades['prediction'] == 0]
    stats = {
        'LONG': {'Skuteczność': longs['is_correct'].mean(), 'Liczba Sygnałów': len(longs)},
        'SHORT': {'Skuteczność': shorts['is_correct'].mean(), 'Liczba Sygnałów': len(shorts)}
    }
    stats_df = pd.DataFrame(stats).T

    print("\n--- Statystyki LONG vs SHORT ---")
    print(stats_df.to_string(float_format="{:.2%}".format))
    print("--------------------------------\n")

    stats_df.plot(kind='bar', y='Skuteczność', figsize=(10, 6), color=['forestgreen', 'crimson'], alpha=0.8,
                  legend=False)
    plt.title('Porównanie Skuteczności: Sygnały LONG vs. SHORT', fontsize=16)
    plt.xlabel('Typ Sygnału', fontsize=12);
    plt.ylabel('Skuteczność', fontsize=12)
    plt.xticks(rotation=0);
    plt.ylim(0, max(1.0, stats_df['Skuteczność'].max() * 1.1 if not stats_df['Skuteczność'].isnull().all() else 1.0))
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    plt.tight_layout()
    output_filename = "analiza_long_vs_short.png"
    plt.savefig(output_filename)
    plt.close()
    print(f"-> Wykres '{output_filename}' został pomyślnie wygenerowany.")


def main():
    """Główna funkcja uruchamiająca wszystkie analizy."""
    # --- Krok 1: Wczytanie danych ---
    input_filename = "final_predictions.csv"
    try:
        df = pd.read_csv(input_filename)
    except FileNotFoundError:
        print(f"BŁĄD: Nie znaleziono pliku '{input_filename}'.")
        print("Upewnij się, że plik z wynikami znajduje się w tym samym folderze co skrypt.")
        sys.exit(1)

    print(f"Pomyślnie wczytano dane z '{input_filename}'. Liczba rekordów: {len(df)}.")

    # --- Krok 2: Przygotowanie danych do analiz ---
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df_trades = df[df['prediction'] != 1].copy()
    df_trades['is_correct'] = (df_trades['target'] == df_trades['prediction'])

    if df_trades.empty:
        print("W pliku nie znaleziono żadnych sygnałów transakcyjnych (predykcji WZROST/SPADEK). Analiza niemożliwa.")
        sys.exit(0)

    # --- Krok 3: Uruchomienie analiz ---
    print("\nRozpoczynanie kompleksowej analizy modelu...")

    analyze_accuracy_vs_confidence(df, df_trades)
    analyze_equity_curve(df_trades)
    analyze_confusion_matrix(df)
    analyze_performance_by_time(df_trades)
    analyze_long_vs_short(df_trades)

    print("\nKompleksowa analiza została zakończona. Sprawdź wygenerowane pliki PNG w folderze.")


if __name__ == "__main__":
    main()