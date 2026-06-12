# NBA Win Predictor v5: FastAPI + Streamlit + SQLAlchemy

Проект состоит из:

```text
frontend/streamlit_app.py  -> пользовательский интерфейс
backend/main.py            -> FastAPI backend
SQLite + SQLAlchemy        -> история прогнозов
artifacts/                 -> ML-артефакты из v5 ноутбука
```

## Что делает приложение

Пользователь вводит домашнюю команду, гостевую команду, дату матча и составы по 7 игроков. Backend рассчитывает team rolling features, player rolling features, Elo, matchup features и возвращает прогноз ensemble-модели.

Если матч есть в датасете, приложение показывает фактический результат и сравнивает его с прогнозом. Все прогнозы сохраняются в SQLite.

## Артефакты

Положи в папку `artifacts/`:

```text
nba_v5_model_bundle.pkl
nba_v5_app_data.pkl
nba_v5_mlp_model.keras
```

## Локальный запуск

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Backend:

```powershell
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

Frontend во втором терминале:

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run frontend/streamlit_app.py
```

## API

```text
GET  /health
GET  /teams
GET  /players?team_name=Lakers&game_date=2026-09-15&allow_transfers=false
POST /predict
GET  /predictions
```

## Ограничение

Если дата позже последней даты датасета, прогноз строится по последнему историческому срезу. Модель не знает будущие травмы, трансферы и изменения состава.
