import optuna
import sys
import config  # Importujemy config, aby mieć dostęp do nazwy bazy danych


def analyze_optuna_study():
    """
    Wczytuje studium Optuny z bazy danych i generuje raporty wizualne.
    """
    print("--- Uruchamianie Analizy Studium Optuny ---")

    # Tworzymy nazwę bazy danych w ten sam sposób, co w skrypcie treningowym
    study_name = f"optimization_binary_{config.TICKER}_{config.BASE_TIMEFRAME}"
    storage_name = f"sqlite:///{study_name}.db"

    # Wczytujemy istniejące studium
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_name)
    except KeyError:
        print(f"\nBŁĄD: Nie znaleziono studium o nazwie '{study_name}' w pliku '{storage_name}'.")
        print("Upewnij się, że przynajmniej raz uruchomiłeś optymalizację w 'model_trainer.py'.")
        sys.exit(1)

    # Sprawdzamy, czy są jakiekolwiek ukończone triale
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        print("\nBŁĄD: W wczytanym studium nie ma żadnych ukończonych traiali. Analiza niemożliwa.")
        sys.exit(1)

    print(f"\n> Pomyślnie wczytano studium. Liczba ukończonych traiali: {len(completed_trials)}")
    print(f"> Najlepszy wynik (F1-score): {study.best_value:.4f}")
    print(f"> Najlepsze parametry: {study.best_params}")

    # --- Generowanie Wykresów ---

    # 1. Wykres historii optymalizacji
    try:
        fig_history = optuna.visualization.plot_optimization_history(study)
        output_filename_hist = "optuna_historia_optymalizacji.html"
        fig_history.write_html(output_filename_hist)
        print(f"\n> Wykres historii optymalizacji został zapisany do: '{output_filename_hist}'")
    except Exception as e:
        print(f"Nie udało się wygenerować wykresu historii optymalizacji: {e}")

    # 2. Wykres ważności hiperparametrów
    try:
        fig_importance = optuna.visualization.plot_param_importances(study)
        output_filename_imp = "optuna_waznosc_parametrow.html"
        fig_importance.write_html(output_filename_imp)
        print(f"> Wykres ważności parametrów został zapisany do: '{output_filename_imp}'")
    except Exception as e:
        print(f"Nie udało się wygenerować wykresu ważności parametrów: {e}")

    print("\n--- Analiza Zakończona ---")
    print("Otwórz wygenerowane pliki .html w przeglądarce internetowej, aby zobaczyć interaktywne wykresy.")


if __name__ == "__main__":
    analyze_optuna_study()