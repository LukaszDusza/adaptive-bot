# Używamy pełnego obrazu dla maksymalnej kompatybilności
FROM python:3.12-slim

# Ustaw folder roboczy wewnątrz kontenera
WORKDIR /app

# Skopiuj plik z zależnościami
COPY requirements.txt requirements.txt

# Zaktualizuj pip i zainstaluj wszystkie pakiety z pliku requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Skopiuj resztę kodu aplikacji
COPY . .