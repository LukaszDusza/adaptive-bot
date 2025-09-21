# Używamy pełnego obrazu dla maksymalnej kompatybilności
FROM python:3.12-slim

# --- FIX: Zainstaluj brakującą bibliotekę systemową libgomp1 ---
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Ustaw folder roboczy wewnątrz kontenera
WORKDIR /app

# Skopiuj plik z zależnościami
COPY requirements.txt requirements.txt

# Zaktualizuj pip i zainstaluj wszystkie pakiety z pliku requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Skopiuj resztę kodu aplikacji
COPY . .