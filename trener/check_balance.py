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

    print(f"\nSprawdzanie rozkładu klas dla parametrów z config.py:")
    print(f"TARGET_TYPE: {config.TARGET_TYPE}, HORIZON_BARS: {config.HORIZON_BARS}")
    if config.TARGET_TYPE == 'DYNAMIC_ATR':
        print(f"ATR_TP_MULTIPLIER: {config.ATR_TP_MULTIPLIER}, ATR_SL_MULTIPLIER: {config.ATR_SL_MULTIPLIER}")
    else:
        print(f"PRICE_TARGET_PCT: {config.PRICE_TARGET_PCT}")

    df = df_features.copy()

    # Wywołanie funkcji jest teraz prostsze
    targets = calculate_multiclass_target(
        df,
        horizon=config.HORIZON_BARS
    )

    df['target'] = targets.map({-1: 0, 0: 1, 1: 2})
    df.dropna(inplace=True)

    print("\n--- WYNIK DIAGNOZY ---")
    print("Liczba etykiet w każdej klasie:")
    print(df['target'].value_counts())
    print("\nProcentowy rozkład klas:")
    print(df['target'].value_counts(normalize=True).map('{:.2%}'.format))
    print("\n0 = RUCH W DÓŁ (SHORT), 1 = BRAK RUCHU (SIDEWAYS), 2 = RUCH W GÓRĘ (LONG)")


if __name__ == "__main__":
    asyncio.run(check_class_balance())
