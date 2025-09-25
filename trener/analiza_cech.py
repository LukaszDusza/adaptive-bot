import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

def analyze_feature_importance():
    """
    Wczytuje plik z ważnością cech i generuje wizualizacje,
    aby zidentyfikować najważniejsze cechy oraz te, które są "szumem".
    """
    input_filename = "feature_importances.csv"
    try:
        df = pd.read_csv(input_filename)
    except FileNotFoundError:
        print(f"BŁĄD: Nie znaleziono pliku '{input_filename}'.")
        print("Uruchom najpierw skrypt 'model_trainer.py', aby wygenerować ten plik.")
        sys.exit(1)

    # --- Analiza Cech o Zerowej Ważności (Szum Informacyjny) ---
    zero_importance_features = df[df['importance'] == 0]
    print("\n--- Cechy o zerowej ważności (całkowicie zignorowane przez model) ---")
    if not zero_importance_features.empty:
        print(f"Znaleziono {len(zero_importance_features)} cech, które są czystym szumem informacyjnym:")
        for feature_name in zero_importance_features['feature']:
            print(f"- {feature_name}")
    else:
        print("Nie znaleziono cech o zerowej ważności. Model wykorzystał każdą cechę.")
    print("--------------------------------------------------------------------")

    # Przygotowanie danych do wizualizacji (bez cech o zerowej ważności)
    df_plot = df[df['importance'] > 0]

    # --- Wizualizacja: Top 30 najważniejszych cech ---
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 10))
    sns.barplot(
        x="importance",
        y="feature",
        data=df_plot.head(30),
        palette="viridis"
    )
    plt.title('Top 30 najważniejszych cech', fontsize=16)
    plt.xlabel('Ważność (wg LightGBM)', fontsize=12)
    plt.ylabel('Cecha', fontsize=12)
    plt.tight_layout()
    plt.savefig("top_30_cech.png")
    print("\n> Wykres 'top_30_cech.png' został zapisany.")

    # --- Wizualizacja: Ostatnie 30 cech o najniższej (ale niezerowej) ważności ---
    plt.figure(figsize=(12, 10))
    sns.barplot(
        x="importance",
        y="feature",
        data=df_plot.tail(30),
        palette="rocket_r"
    )
    plt.title('30 cech o najniższej (ale niezerowej) ważności', fontsize=16)
    plt.xlabel('Ważność (wg LightGBM)', fontsize=12)
    plt.ylabel('Cecha', fontsize=12)
    plt.tight_layout()
    plt.savefig("ostatnie_30_cech.png")
    print("> Wykres 'ostatnie_30_cech.png' został zapisany.")


if __name__ == "__main__":
    analyze_feature_importance()