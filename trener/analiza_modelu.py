import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import sys # ZMIANA: Dodano import 'sys' do obsługi wyjścia z programu

# --- Krok 1: Wczytanie danych ---
input_filename = "final_predictions.csv"
try:
    df = pd.read_csv(input_filename)
except FileNotFoundError:
    # ZMIANA: Usunięto wadliwą obsługę błędów. Teraz program wyświetli jasny komunikat i zakończy działanie.
    print(f"BŁĄD: Nie znaleziono pliku '{input_filename}'.")
    print("Upewnij się, że plik z wynikami znajduje się w tym samym folderze co skrypt.")
    sys.exit(1) # Zakończ program, jeśli nie ma danych

# --- Krok 2: Przygotowanie danych do analizy ---

# Bierzemy pod uwagę tylko sygnały na WZROST (2) lub SPADEK (0), ignorujemy ruch boczny
df_trades = df[df['prediction'] != 1].copy()

# Obliczamy maksymalne prawdopodobieństwo dla każdej predykcji (pewność modelu)
# Sprawdzamy, które kolumny z prawdopodobieństwami istnieją, aby uniknąć błędu
proba_cols = [col for col in ['proba_DOWN(0)', 'proba_UP(2)'] if col in df_trades.columns]
if not proba_cols:
    print("BŁĄD: W pliku CSV brakuje kolumn z prawdopodobieństwami ('proba_DOWN(0)', 'proba_UP(2)')")
    sys.exit(1)
df_trades['proba_max'] = df_trades[proba_cols].max(axis=1)

# Sprawdzamy, czy predykcja była poprawna (1 jeśli tak, 0 jeśli nie)
df_trades['is_correct'] = (df_trades['target'] == df_trades['prediction']).astype(int)

# --- Krok 3: Obliczenia dla wykresu ---

# Definiujemy progi pewności, które chcemy przetestować (od 50% do 99% co 1%)
thresholds = np.arange(0.50, 1.0, 0.01)

results = []
for thresh in thresholds:
    # Filtrujemy transakcje, które spełniają dany próg pewności
    subset = df_trades[df_trades['proba_max'] >= thresh]

    trade_count = len(subset)
    if trade_count > 0:
        # Obliczamy skuteczność (średnią z kolumny 'is_correct')
        accuracy = subset['is_correct'].mean()
    else:
        # Jeśli nie ma transakcji, skuteczność jest nieokreślona
        accuracy = np.nan

    results.append({'threshold': thresh, 'accuracy': accuracy, 'trade_count': trade_count})

results_df = pd.DataFrame(results)

# --- Krok 4: Generowanie wykresu ---

# Ustawienie stylu wykresu
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax1 = plt.subplots(figsize=(14, 7))

# Wykres skuteczności (lewa oś Y)
color = 'royalblue'
ax1.set_xlabel('Próg Pewności Modelu (Confidence Threshold)', fontsize=12)
ax1.set_ylabel('Skuteczność (Accuracy)', color=color, fontsize=12)
line1 = ax1.plot(results_df['threshold'], results_df['accuracy'], color=color, marker='o', markersize=4, label='Skuteczność')
ax1.tick_params(axis='y', labelcolor=color, labelsize=10)
ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

# Ustawiamy limity osi Y w sposób bezpieczny, nawet jeśli nie ma danych
if not results_df['accuracy'].dropna().empty:
    ax1.set_ylim(bottom=max(0, results_df['accuracy'].min() * 0.95))

# Druga oś Y dla liczby transakcji
ax2 = ax1.twinx()
color = 'crimson'
ax2.set_ylabel('Liczba Transakcji (Number of Trades)', color=color, fontsize=12)
line2 = ax2.plot(results_df['threshold'], results_df['trade_count'], color=color, marker='x', markersize=4, linestyle='--',
                 label='Liczba Transakcji')
ax2.tick_params(axis='y', labelcolor=color, labelsize=10)

# Tytuł i legenda
plt.title('Skuteczność Modelu vs. Próg Pewności', fontsize=16, pad=20)

# ZMIANA: Poprawiono obsługę legendy dla obu osi Y
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, fontsize=10)

fig.tight_layout(rect=[0, 0, 1, 0.96]) # Dopasowanie układu, aby tytuł się zmieścił

# Zapis do pliku
output_filename = "skutecznosc_vs_pewnosc.png"
plt.savefig(output_filename)

print(f"\nWykres '{output_filename}' został pomyślnie wygenerowany.")
print("Sprawdź plik w folderze, w którym uruchomiłeś ten skrypt.")