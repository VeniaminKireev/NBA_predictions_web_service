import requests
import streamlit as st
import os

st.set_page_config(page_title="NBA Win Predictor v5", page_icon="🏀", layout="wide")
st.title("🏀 NBA Win Predictor v5")
st.caption("Streamlit frontend + FastAPI backend + SQLAlchemy history")

DEFAULT_API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

API_URL = st.sidebar.text_input(
    "FastAPI URL",
    value=DEFAULT_API_URL,
)

try:
    r = requests.get(f"{API_URL}/health", timeout=10)
    r.raise_for_status()
except Exception as exc:
    st.error(f"Backend недоступен: {exc}")
    st.stop()

@st.cache_data(ttl=300)
def load_teams(api_url):
    r = requests.get(f"{api_url}/teams", timeout=30)
    r.raise_for_status()
    return r.json()["teams"]

def load_players(api_url, team_name, game_date, allow_transfers):
    r = requests.get(
        f"{api_url}/players",
        params={"team_name": team_name, "game_date": str(game_date), "allow_transfers": allow_transfers},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["players"]

teams = load_teams(API_URL)

st.sidebar.header("Матч")
home_team_name = st.sidebar.selectbox("Домашняя команда", teams)
away_team_name = st.sidebar.selectbox("Гостевая команда", teams)
game_date = st.sidebar.date_input("Дата матча")
allow_transfers = st.sidebar.checkbox("Разрешить выбирать игроков из других команд", value=False)

if home_team_name == away_team_name:
    st.error("Домашняя и гостевая команда не могут совпадать.")
    st.stop()

try:
    home_players = load_players(API_URL, home_team_name, game_date, allow_transfers)
    away_players = load_players(API_URL, away_team_name, game_date, allow_transfers)
except Exception as exc:
    st.error(f"Не удалось загрузить игроков: {exc}")
    st.stop()

tab_predict, tab_history = st.tabs(["🔮 Прогноз", "📜 История прогнозов"])

with tab_predict:
    col_home, col_away = st.columns(2)
    with col_home:
        st.markdown(f"### {home_team_name}")
        home_lineup = st.multiselect("Выберите 7 игроков", home_players, default=home_players[:7], max_selections=7)
    with col_away:
        st.markdown(f"### {away_team_name}")
        away_lineup = st.multiselect("Выберите 7 игроков", away_players, default=away_players[:7], max_selections=7)

    if len(home_lineup) != 7 or len(away_lineup) != 7:
        st.warning("Нужно выбрать ровно по 7 игроков для каждой команды.")
        st.stop()

    if st.button("Сделать прогноз", type="primary"):
        payload = {
            "home_team_name": home_team_name,
            "away_team_name": away_team_name,
            "game_date": str(game_date),
            "home_lineup": home_lineup,
            "away_lineup": away_lineup,
            "allow_transfers": allow_transfers,
        }
        with st.spinner("Считаем прогноз..."):
            r = requests.post(f"{API_URL}/predict", json=payload, timeout=180)
        if r.status_code != 200:
            st.error(r.text)
            st.stop()
        p = r.json()
        st.subheader("Результат прогноза")
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Вероятность победы {home_team_name}", f"{p['home_win_probability']:.1%}")
        c2.metric(f"Вероятность победы {away_team_name}", f"{p['away_win_probability']:.1%}")
        c3.metric("Уверенность модели", f"{p['confidence']:.1%}")

        if p["confidence"] >= 0.75:
            st.success(f"Прогноз: **{p['predicted_winner']}**. Прогноз уверенный.")
        elif p["confidence"] >= 0.65:
            st.info(f"Прогноз: **{p['predicted_winner']}**. Прогноз умеренно уверенный.")
        else:
            st.warning(f"Прогноз: **{p['predicted_winner']}**. Матч выглядит близким.")

        if p.get("warning"):
            st.warning(p["warning"])

        st.subheader("Вероятности отдельных моделей")
        st.dataframe([
            {"model": "Logistic Regression", "home_win_probability": p["logreg_proba"]},
            {"model": "HistGradientBoosting", "home_win_probability": p["hgb_proba"]},
            {"model": "CatBoost", "home_win_probability": p["catboost_proba"]},
            {"model": "MLP", "home_win_probability": p["nn_proba"]},
            {"model": "Ensemble", "home_win_probability": p["home_win_probability"]},
        ], use_container_width=True)

        st.subheader("Фактический результат")
        if p.get("actual_winner") is not None:
            st.write(f"Реальный победитель: **{p['actual_winner']}**")
            st.write(f"Счёт: **{int(p['home_score'])} : {int(p['away_score'])}**")
            st.success("Модель угадала.") if p["model_correct"] else st.error("Модель ошиблась.")
        else:
            st.info("Матч не найден в датасете. Показываем только прогноз модели.")

with tab_history:
    st.subheader("Последние прогнозы")
    try:
        r = requests.get(f"{API_URL}/predictions", params={"limit": 50}, timeout=30)
        r.raise_for_status()
        hist = r.json()
        st.dataframe(hist, use_container_width=True) if hist else st.info("История пока пустая.")
    except Exception as exc:
        st.error(f"Не удалось загрузить историю: {exc}")
