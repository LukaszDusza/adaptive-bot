# test_pta.py
import pandas as pd

# Tworzymy małą, przykładową tabelę danych
df = pd.DataFrame({
    "open": [1, 2, 3, 4, 5],
    "high": [1, 2, 3, 4, 5],
    "low": [1, 2, 3, 4, 5],
    "close": [1, 2, 3, 4, 5],
    "volume": [10, 20, 30, 40, 50],
})

# Używamy funkcji dir(), aby wyświetlić listę wszystkich dostępnych metod
print("Dostępne metody w obiekcie '.ta':")
print(dir(df.ta))
