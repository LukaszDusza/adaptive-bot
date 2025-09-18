Twoja Finałowa Instrukcja:

Umieść wszystkie pliki (.py, .joblib, .json, Dockerfile, requirements.txt, pliki .csv) w jednym folderze.

Mając zainstalowanego Dockera, otwórz terminal w tym folderze i zbuduj obraz:
docker build -t trading-bot-api .

Uruchom kontener: docker run -p 5000:5000 trading-bot-api

Teraz Twoja aplikacja w Javie może wysyłać zapytania POST na adres http://localhost:5000/predict z danymi w formacie JSON, a w odpowiedzi otrzyma decyzję modelu.