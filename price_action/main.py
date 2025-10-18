import argparse
import os
from dotenv import load_dotenv


def main():
    parser = argparse.ArgumentParser(description="Główny skrypt do zarządzania potokiem ML dla bota tradingowego.")

    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--train', action='store_true', help="Uruchom potok treningowy dla modelu 'long' lub 'short'.")
    action.add_argument('--backtest', action='store_true',
                        help="Uruchom backtest dla połączonej strategii 'long/short'.")
    action.add_argument('--report', action='store_true', help="Wygeneruj raport HTML z ostatniego backtestu.")
    action.add_argument('--run-bot', action='store_true', help="Uruchom bota tradingowego na żywo.")

    parser.add_argument('--ticker', type=str, default="SOLUSDT", help="Ticker do analizy (np. SOLUSDT).")
    parser.add_argument('--timeframe', type=str, default="1h", help="Główny interwał czasowy (np. 1h).")

    parser.add_argument('--helper-timeframes', nargs='*', default=None,
                        help="Pomocnicze interwały czasowe.")

    parser.add_argument('--side', type=str, choices=['long', 'short'],
                        help="Wymagane dla --train: 'long' (BUY/HOLD) lub 'short' (SELL/HOLD).")
    parser.add_argument('--version', type=str, default='v1.0',
                        help="Wersja modelu (np. v1.0, v1.1, v2.0). Domyślnie: v1.0")
    parser.add_argument('--label-trials', type=int, default=50)
    parser.add_argument('--model-trials', type=int, default=100)

    parser.add_argument('--limit', type=int, default=3000, help="Liczba świec do pobrania.")
    parser.add_argument('--date-from', type=str, default=None, 
                        help="Data końcowa dla danych treningowych (YYYY-MM-DD). Jeśli podana, dane będą pobrane wstecz od tej daty.")
    parser.add_argument('--prob-threshold', type=float, default=0.8)
    parser.add_argument('--min-proba-diff', type=float, default=0.0,
                        help="Minimum probability difference between BUY and SELL (confidence gap)")
    parser.add_argument('--tp-pct', type=float, default=0.07)
    parser.add_argument('--tsl-pct', type=float, default=0.04)
    parser.add_argument('--trade-size', type=float, default=100.0)
    parser.add_argument('--leverage', type=int, default=10)
    parser.add_argument('--partial-tp', action='store_true',
                        help='Enable old partial TP mechanism (50%% at halfway to TP)')
    parser.add_argument('--dynamic-tp', action='store_true',
                        help='Enable new dynamic TP mechanism (25%% at each of 4 levels: 25%%, 50%%, 75%%, 100%%)')
    parser.add_argument('--hedge-mode', action='store_true',
                        help='Enable Hedge Mode (positionIdx: 1=Long, 2=Short). Default is One-Way Mode (positionIdx: 0).')

    args = parser.parse_args()

    if args.train:

        from data_preparer_pa import fetch_and_prepare_data


        from model_pipeline import run_training_pipeline

        if not args.side:
            parser.error("--side jest wymagany z flagą --train.")
        print_header(f"Trening modelu: {args.side.upper()}")
        df_features = fetch_and_prepare_data(ticker=args.ticker, timeframe=args.timeframe, limit=args.limit,
                                             helper_timeframes=args.helper_timeframes, side=args.side,
                                             date_from=args.date_from, version=args.version)
        if not df_features.empty:
            run_training_pipeline(df_features=df_features, n_label_trials=args.label_trials,
                                  n_model_trials=args.model_trials,
                                  ticker=args.ticker, timeframe=args.timeframe,
                                  helper_timeframes=args.helper_timeframes, side=args.side, version=args.version)

    elif args.backtest:
        from backtester import run_backtester_with_args
        print_header("Uruchamianie Backtestu")
        run_backtester_with_args(args)

    elif args.report:
        from report_generator import run_report_generator_with_args
        print_header("Generowanie Raportu")
        run_report_generator_with_args(args)

    elif args.run_bot:
        from bot import launch_bot
        print_header("Uruchamianie Bota Tradingowego")
        launch_bot(args)


def print_header(title):
    print("\n" + "=" * 60)
    print(f"--- {title.upper()} ---")
    print("=" * 60)


if __name__ == "__main__":
    load_dotenv()
    main()