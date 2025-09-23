import asyncio
import config
from async_data_fetcher import fetch_data_for_trainer_async
from data_preparer import prepare_feature_set_for_timeframe
from model_trainer import calculate_multiclass_target


async def check_class_balance():
    print("Wczytywanie i przygotowywanie danych...")
    df_raw = await fetch_data_for_trainer_async(
        ticker=config.TICKER, start_date=config.TRAIN_START_DATE, end_date=config.TRAIN_END_DATE
    )
    df_features = prepare_feature_set_for_timeframe(df_raw, base_tf=config.BASE_TIMEFRAME)

    # === EDYTUJ WARTOŚCI PONIŻEJ, ABY TESTOWAĆ RÓŻNE SCENARIUSZE ===
    test_params = {
        'PRICE_TARGET_PCT': 0.01,
        'HORIZON_BARS': 32
    }
    # =================================================================

    print(f"\nSprawdzanie rozkładu klas dla parametrów: {test_params}")

    df = df_features.copy()
    targets = calculate_multiclass_target(
        df,
        target_pct=test_params['PRICE_TARGET_PCT'],
        horizon=test_params['HORIZON_BARS']
    )

    df['target'] = targets.map({-1: 0, 0: 1, 1: 2})

    df.dropna(inplace=True)

    print("\n--- WYNIK DIAGNOZY ---")
    print("Procentowy rozkład klas:")
    print(df['target'].value_counts(normalize=True).map('{:.2%}'.format))
    print("0 = RUCH W DÓŁ, 1 = BRAK RUCHU (SIDEWAYS), 2 = RUCH W GÓRĘ")


if __name__ == "__main__":
    asyncio.run(check_class_balance())