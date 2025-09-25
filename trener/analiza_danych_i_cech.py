"""
Skrypt do zaawansowanej analizy jakości danych i dystrybucji cech.
Pomaga zidentyfikować redundantne cechy i zrozumieć strukturę danych.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import glob

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- Konfiguracja ---
# Upewnij się, że te ścieżki są zgodne z Twoim projektem
FEATURES_CACHE_DIR = "data_cache/features"


# --- Funkcje Analityczne ---

def analyze_correlation(df_features, top_n=30):
    """Oblicza i wizualizuje macierz korelacji dla najważniejszych cech."""
    print(f"\nAnaliza 1/3: Korelacja Cech (Top {top_n})...")

    # Bierzemy podzbiór cech, aby wykres był czytelny
    df_subset = df_features.iloc[:, :top_n]
    correlation_matrix = df_subset.corr()

    plt.figure(figsize=(18, 15))
    sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False)  # annot=True może być zbyt gęste
    plt.title(f'Macierz Korelacji dla Top {top_n} Cech', fontsize=16)
    plt.tight_layout()

    output_filename = "A1_macierz_korelacji.png"
    plt.savefig(output_filename)
    plt.close()
    print(f"-> Wykres '{output_filename}' został zapisany.")


def analyze_feature_distribution(df_features, top_n=12):
    """Wizualizuje rozkład (dystrybucję) wartości dla najważniejszych cech."""
    print(f"Analiza 2/3: Dystrybucja Cech (Top {top_n})...")

    df_subset = df_features.iloc[:, :top_n]

    num_plots = len(df_subset.columns)
    num_cols = 4
    num_rows = int(np.ceil(num_plots / num_cols))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 5 * num_rows))
    axes = axes.flatten()

    for i, col in enumerate(df_subset.columns):
        sns.histplot(df_subset[col], kde=True, ax=axes[i], bins=50)
        axes[i].set_title(col, fontsize=10)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')

    # Ukryj puste subploty, jeśli istnieją
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'Dystrybucja Wartości dla Top {top_n} Cech', fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    output_filename = "A2_dystrybucja_cech.png"
    plt.savefig(output_filename)
    plt.close()
    print(f"-> Wykres '{output_filename}' został zapisany.")


def analyze_pca_redundancy(df_features):
    """Używa PCA do oceny redundancji w zbiorze cech."""
    print("Analiza 3/3: Redundancja Cech (Analiza PCA)...")

    # Skalujemy dane przed PCA
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_features)

    # Uruchamiamy PCA
    pca = PCA()
    pca.fit(scaled_features)

    # Obliczamy skumulowaną wariancję
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # Wizualizacja
    plt.figure(figsize=(12, 7))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='.', linestyle='--')
    plt.title('Skumulowana Wyjaśniona Wariancja (Analiza PCA)', fontsize=16)
    plt.xlabel('Liczba Składowych Głównych (Cech)', fontsize=12)
    plt.ylabel('Procent Wyjaśnionej Wariancji', fontsize=12)
    plt.grid(True)
    plt.axhline(y=0.95, color='r', linestyle=':', label='95% progu wariancji')
    plt.axhline(y=0.90, color='g', linestyle=':', label='90% progu wariancji')
    plt.legend()

    output_filename = "A3_analiza_pca.png"
    plt.savefig(output_filename)
    plt.close()
    print(f"-> Wykres '{output_filename}' został zapisany.")


def main():
    """Główna funkcja skryptu."""
    print("--- Uruchamianie Zaawansowanej Analizy Danych i Cech ---")

    # Znajdź najnowszy plik z cechami w cache
    try:
        list_of_files = glob.glob(f'{FEATURES_CACHE_DIR}/*.parquet')
        latest_file = max(list_of_files, key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0)
    except (ValueError, FileNotFoundError):
        print(f"\nBŁĄD: Nie znaleziono żadnych plików z cechami w folderze '{FEATURES_CACHE_DIR}'.")
        print("Uruchom najpierw 'model_trainer.py', aby wygenerować te pliki.")
        sys.exit(1)

    print(f"\n> Wczytywanie danych z pliku: {os.path.basename(latest_file)}")
    df = pd.read_parquet(latest_file)

    # Usunięcie wierszy z NaN, które powstały na początku obliczeń
    df.dropna(inplace=True)

    # Wybieramy tylko kolumny z cechami (bez cen OHLC i targetu)
    feature_cols = [col for col in df.columns if
                    col not in ['open', 'high', 'low', 'close', 'volume', 'turnover', 'target']]
    df_features = df[feature_cols]

    # Uruchomienie analiz
    analyze_correlation(df_features)
    analyze_feature_distribution(df_features)
    analyze_pca_redundancy(df_features)

    print("\n--- Zakończono ---")
    print("Analiza danych i cech została zakończona. Sprawdź wygenerowane pliki PNG.")


if __name__ == "__main__":
    # Potrzebujemy tych importów w bloku main, aby glob mógł znaleźć plik
    import os
    import glob

    main()