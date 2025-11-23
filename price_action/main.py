import argparse
import sys
from dotenv import load_dotenv
from pydantic import ValidationError


def main():
    parser = argparse.ArgumentParser(description="Główny skrypt do zarządzania potokiem ML dla bota tradingowego.")

    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--train', action='store_true', help="Uruchom potok treningowy dla modelu 'long' lub 'short'.")
    action.add_argument('--backtest', action='store_true',
                        help="Uruchom backtest dla połączonej strategii 'long/short'.")
    action.add_argument('--run-bot', action='store_true', help="Uruchom bota tradingowego na żywo.")

    parser.add_argument('--ticker', type=str, default="SOLUSDT", help="Ticker do analizy (np. SOLUSDT).")
    parser.add_argument('--timeframe', type=str, default="1h", help="Główny interwał czasowy (np. 1h).")

    parser.add_argument('--helper-timeframes', nargs='*', default=None,
                        help="Pomocnicze interwały czasowe.")

    parser.add_argument('--side', type=str, choices=['long', 'short'],
                        help="Wymagane dla --train: 'long' (BUY/HOLD) lub 'short' (SELL/HOLD).")
    parser.add_argument('--version', type=str, default='v1.0',
                        help="Wersja modelu (np. v1.0, v1.1, v2.0). Domyślnie: v1.0")
    parser.add_argument('--label-trials', type=int, default=50,
                        help="Label optimization trials. Label space jest prosty (tylko 2 params), więc 150-200 często wystarcza. Default: 50")
    parser.add_argument('--model-trials', type=int, default=100,
                        help="Model hyperparameter optimization trials. Higher values = better optimization ale dłuższy training. Default: 100")
    parser.add_argument('--top-n-features', type=int, default=None,
                        help="Feature pruning: zachowaj tylko top N features (ignores importance threshold). Zalecane: 60-80 dla SOLUSDT")

    parser.add_argument('--limit', type=int, default=3000, help="Liczba świec do pobrania.")
    parser.add_argument('--date-from', type=str, default=None,
                        help="Data końcowa dla danych treningowych (YYYY-MM-DD). Jeśli podana, dane będą pobrane wstecz od tej daty.")
    parser.add_argument('--fetch-max-history', action='store_true',
                        help="Fetch MAXIMUM available history from Bybit API (ignores --limit parameter)")
    parser.add_argument('--prob-threshold', type=float, default=0.8)
    parser.add_argument('--min-confidence-ratio', type=float, default=1.5,
                        help="Minimum confidence ratio: stronger_signal/weaker_signal (1.5 = 50% more confident, 2.0 = 100% more)")
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
    parser.add_argument('--limit-order', action='store_true',
                        help='Use limit orders instead of market orders for better entry prices')
    parser.add_argument('--max-waiting-limit-order', type=int, default=300,
                        help='Maximum seconds to wait for limit order execution before cancelling (default: 300s = 5min)')
    parser.add_argument('--limit-offset-pct', type=float, default=0.005,
                        help='Price offset for limit orders as percentage (default: 0.005 = 0.5%%). LONG: -offset (buy cheaper), SHORT: +offset (sell higher)')
    parser.add_argument('--protect-profit', action='store_true',
                        help='Enable profit protection: move SL to breakeven if profit peaks >0.25%% but declines before hitting partial TP')

    args = parser.parse_args()

    # Validate configuration based on mode
    try:
        if args.train:
            from config_validation import TrainingConfigValidated
            validated_config = TrainingConfigValidated(
                ticker=args.ticker,
                timeframe=args.timeframe,
                helper_timeframes=args.helper_timeframes or [],
                side=args.side,
                version=args.version,
                label_trials=args.label_trials,
                model_trials=args.model_trials,
                top_n_features=args.top_n_features,
                limit=args.limit,
                date_from=args.date_from,
                fetch_max_history=args.fetch_max_history
            )
        elif args.backtest:
            from config_validation import BacktestConfigValidated
            validated_config = BacktestConfigValidated(
                ticker=args.ticker,
                timeframe=args.timeframe,
                helper_timeframes=args.helper_timeframes or [],
                version=args.version,
                probability_threshold=args.prob_threshold,
                min_confidence_ratio=args.min_confidence_ratio,
                tp_pct=args.tp_pct,
                tsl_pct=args.tsl_pct,
                trade_size_usd=args.trade_size,
                leverage=args.leverage,
                partial_tp_enabled=args.partial_tp,
                dynamic_tp_enabled=args.dynamic_tp,
                limit=args.limit
            )
        elif args.run_bot:
            from config_validation import BotConfigValidated
            validated_config = BotConfigValidated(
                ticker=args.ticker,
                timeframe=args.timeframe,
                helper_timeframes=args.helper_timeframes or [],
                trade_size_usd=args.trade_size,
                leverage=args.leverage,
                tp_pct=args.tp_pct,
                tsl_pct=args.tsl_pct,
                probability_threshold=args.prob_threshold,
                min_confidence_ratio=args.min_confidence_ratio,
                partial_tp_enabled=args.partial_tp,
                dynamic_tp_enabled=args.dynamic_tp,
                hedge_mode=args.hedge_mode,
                limit_order_mode=args.limit_order,
                max_waiting_limit_order=args.max_waiting_limit_order,
                limit_offset_pct=args.limit_offset_pct,
                protect_profit_enabled=args.protect_profit
            )
    except ValidationError as e:
        print("\n❌ BŁĄD KONFIGURACJI - Nieprawidłowe parametry:\n")
        for error in e.errors():
            field = ' → '.join(str(loc) for loc in error['loc'])
            msg = error['msg']
            print(f"  • {field}: {msg}")
        print("\n")
        sys.exit(1)

    if args.train:

        from data_preparer_pa import fetch_and_prepare_data


        from model_pipeline import run_training_pipeline

        if not args.side:
            parser.error("--side jest wymagany z flagą --train.")
        print_header(f"Trening modelu: {args.side.upper()}")
        df_features = fetch_and_prepare_data(ticker=args.ticker, timeframe=args.timeframe, limit=args.limit,
                                             helper_timeframes=args.helper_timeframes, side=args.side,
                                             date_from=args.date_from, version=args.version,
                                             fetch_max_history=args.fetch_max_history)
        if not df_features.empty:
            run_training_pipeline(df_features=df_features, n_label_trials=args.label_trials,
                                  n_model_trials=args.model_trials,
                                  ticker=args.ticker, timeframe=args.timeframe,
                                  helper_timeframes=args.helper_timeframes, side=args.side, version=args.version,
                                  top_n_features=args.top_n_features)

    elif args.backtest:
        from backtester import run_backtester_with_args
        print_header("Uruchamianie Backtestu")
        run_backtester_with_args(args)

    elif args.run_bot:
        from bot import launch_bot
        print_header("Uruchamianie Bota Tradingowego")
        launch_bot(args)


def print_header(title):
    print("\n" + "=" * 60)
    print(f"--- {title.upper()} ---")
    print("=" * 60)


if __name__ == "__main__":
    # Load .env_demo for training/backtesting (demo API for fetching historical data)
    load_dotenv('.env_demo')
    main()