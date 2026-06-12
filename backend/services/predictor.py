from collections import defaultdict
from typing import Dict, List, Optional
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from backend.core.config import APP_DATA_PATH, MLP_MODEL_PATH, MODEL_BUNDLE_PATH

class PredictorService:
    def __init__(self):
        self.loaded = False

    def load(self):
        if self.loaded:
            return
        self.model_bundle = joblib.load(MODEL_BUNDLE_PATH)
        app_data = joblib.load(APP_DATA_PATH)
        self.nn_model = tf.keras.models.load_model(MLP_MODEL_PATH)
        self.teams = app_data["teams"].copy()
        self.players_played = app_data["players_played"].copy()
        self.games_model = app_data["games_model"].copy()
        for df in [self.teams, self.players_played, self.games_model]:
            df["gameDateTimeEst"] = pd.to_datetime(df["gameDateTimeEst"], errors="coerce")
        self.loaded = True

    def get_teams(self) -> List[str]:
        self.load()
        return sorted(self.teams["teamName"].dropna().unique().tolist())

    def get_players_for_team_before_date(self, team_name: str, game_date: str, allow_transfers: bool = False, limit: int = 80) -> List[str]:
        self.load()
        game_date = pd.to_datetime(game_date)
        if allow_transfers:
            temp = self.players_played[self.players_played["gameDateTimeEst"] < game_date].copy()
        else:
            temp = self.players_played[(self.players_played["playerteamName"] == team_name) & (self.players_played["gameDateTimeEst"] < game_date)].copy()
        if temp.empty:
            return []
        summary = (
            temp.groupby("playerName")
            .agg(last_game=("gameDateTimeEst", "max"), avg_minutes=("numMinutes", "mean"), games=("gameId", "nunique"))
            .reset_index()
            .sort_values(["last_game", "avg_minutes"], ascending=False)
        )
        return summary["playerName"].head(limit).tolist()

    def find_actual_game_result(self, home_team_name: str, away_team_name: str, game_date: str) -> Optional[Dict]:
        self.load()
        game_date_only = pd.to_datetime(game_date).date()
        temp = self.games_model.copy()
        temp["date_only"] = temp["gameDateTimeEst"].dt.date
        found = temp[(temp["home_team_name"] == home_team_name) & (temp["away_team_name"] == away_team_name) & (temp["date_only"] == game_date_only)]
        if found.empty:
            return None
        row = found.iloc[0]
        actual_winner = row["home_team_name"] if int(row["home_win"]) == 1 else row["away_team_name"]
        return {"actual_winner": actual_winner, "home_score": float(row["home_score"]), "away_score": float(row["away_score"]), "home_win": int(row["home_win"])}

    def latest_team_row(self, team_name: str, game_date: str):
        self.load()
        game_date = pd.to_datetime(game_date)
        hist = self.teams[(self.teams["teamName"] == team_name) & (self.teams["gameDateTimeEst"] < game_date)].sort_values(["gameDateTimeEst", "gameId"])
        if hist.empty:
            raise ValueError(f"Нет истории для команды {team_name} до даты {game_date.date()}")
        return hist.iloc[-1]

    def elo_before_date(self, home_team_id: int, away_team_id: int, game_date: str, k: int = 20, home_advantage: int = 65) -> Dict:
        self.load()
        game_date = pd.to_datetime(game_date)
        games = self.games_model[self.games_model["gameDateTimeEst"] < game_date].sort_values(["gameDateTimeEst", "gameId"])
        ratings = defaultdict(lambda: 1500.0)
        for _, row in games.iterrows():
            h, a = int(row["home_team_id"]), int(row["away_team_id"])
            h_elo, a_elo = ratings[h], ratings[a]
            p = 1 / (1 + 10 ** (-(h_elo + home_advantage - a_elo) / 400))
            actual = int(row["home_win"])
            margin = abs(float(row["home_score"]) - float(row["away_score"]))
            change = k * np.log(margin + 1) * (actual - p)
            ratings[h] += change
            ratings[a] -= change
        home_pre_elo, away_pre_elo = ratings[int(home_team_id)], ratings[int(away_team_id)]
        elo_diff = home_pre_elo - away_pre_elo
        elo_diff_with_home = home_pre_elo + home_advantage - away_pre_elo
        elo_home_win_prob = 1 / (1 + 10 ** (-elo_diff_with_home / 400))
        return {
            "home_pre_elo": home_pre_elo,
            "away_pre_elo": away_pre_elo,
            "elo_diff": elo_diff,
            "elo_diff_with_home": elo_diff_with_home,
            "elo_home_win_prob": elo_home_win_prob,
        }

    def latest_player_row(self, player_name: str, team_id: int, game_date: str, allow_transfers: bool = False):
        self.load()
        game_date = pd.to_datetime(game_date)
        mask = (self.players_played["playerName"].str.lower() == str(player_name).lower()) & (self.players_played["gameDateTimeEst"] < game_date)
        if not allow_transfers:
            mask = mask & (self.players_played["playerteamId"].astype(int) == int(team_id))
        hist = self.players_played[mask].sort_values(["gameDateTimeEst", "gameId"])
        if hist.empty:
            return None
        return hist.iloc[-1]

    def lineup_features(self, lineup_names: List[str], team_id: int, game_date: str, prefix: str, allow_transfers: bool = False):
        rows, missing = [], []
        for name in lineup_names:
            row = self.latest_player_row(name, team_id, game_date, allow_transfers)
            if row is None:
                missing.append(name)
            else:
                rows.append(row)
        if not rows:
            raise ValueError(f"Не удалось найти историю игроков для team_id={team_id}")
        df = pd.DataFrame(rows)
        features = {}
        for w in [3, 5, 10]:
            power_col = f"player_power_{w}"
            if power_col not in df.columns:
                continue
            df[power_col] = pd.to_numeric(df[power_col], errors="coerce").fillna(0)
            features[f"{prefix}_player_w{w}_top7_sum_power"] = df[power_col].sum()
            features[f"{prefix}_player_w{w}_top7_avg_power"] = df[power_col].mean()
            features[f"{prefix}_player_w{w}_top7_count"] = len(df)
            for base in ["numMinutes","points","assists","reboundsTotal","turnovers","steals","blocks","netRating","trueShootingPercentage","usagePercentage","playerImpactEstimate","percentTeamPoints","percentTeamAssists","percentTeamRebounds"]:
                col = f"player_roll_{w}_{base}"
                if col in df.columns:
                    vals = pd.to_numeric(df[col], errors="coerce")
                    features[f"{prefix}_player_w{w}_top7_sum_{base}"] = vals.sum()
                    features[f"{prefix}_player_w{w}_top7_avg_{base}"] = vals.mean()
            top3 = df.sort_values(power_col, ascending=False).head(3)
            features[f"{prefix}_player_w{w}_top3_sum_power"] = top3[power_col].sum()
            features[f"{prefix}_player_w{w}_top3_avg_power"] = top3[power_col].mean()
            features[f"{prefix}_player_w{w}_top3_count"] = len(top3)
            starters = df.sort_values(power_col, ascending=False).head(5)
            bench = df.sort_values(power_col, ascending=False).iloc[5:]
            features[f"{prefix}_player_w{w}_starter_sum_power"] = starters[power_col].sum()
            features[f"{prefix}_player_w{w}_starter_avg_power"] = starters[power_col].mean()
            features[f"{prefix}_player_w{w}_starter_count"] = len(starters)
            features[f"{prefix}_player_w{w}_bench_sum_power"] = bench[power_col].sum()
            features[f"{prefix}_player_w{w}_bench_avg_power"] = bench[power_col].mean() if len(bench) else np.nan
            features[f"{prefix}_player_w{w}_bench_count"] = len(bench)
            top7_power, top3_power = df[power_col].sum(), top3[power_col].sum()
            features[f"{prefix}_player_w{w}_star_power_ratio"] = top3_power / top7_power if top7_power else np.nan
        return features, missing

    def predict(self, home_team_name: str, away_team_name: str, game_date: str, home_lineup: List[str], away_lineup: List[str], allow_transfers: bool = False) -> Dict:
        self.load()
        feature_cols = self.model_bundle["feature_cols"]
        team_context_cols = self.model_bundle["team_context_cols"]
        rolling_windows = self.model_bundle["rolling_windows"]
        home_row, away_row = self.latest_team_row(home_team_name, game_date), self.latest_team_row(away_team_name, game_date)
        home_team_id, away_team_id = int(home_row["teamId"]), int(away_row["teamId"])
        features = {}
        for col in team_context_cols:
            features[f"diff_team_{col}"] = pd.to_numeric(home_row.get(col, np.nan), errors="coerce") - pd.to_numeric(away_row.get(col, np.nan), errors="coerce")
        for col in ["rest_days","back_to_back","games_before","season_win_rate_before"]:
            features[f"home_team_{col}"] = home_row.get(col, np.nan)
            features[f"away_team_{col}"] = away_row.get(col, np.nan)
        for w in rolling_windows:
            home_off, home_def = home_row.get(f"team_roll_{w}_offensiveRating", np.nan), home_row.get(f"team_roll_{w}_defensiveRating", np.nan)
            away_off, away_def = away_row.get(f"team_roll_{w}_offensiveRating", np.nan), away_row.get(f"team_roll_{w}_defensiveRating", np.nan)
            features[f"expected_home_rating_{w}"] = (home_off + away_def) / 2
            features[f"expected_away_rating_{w}"] = (away_off + home_def) / 2
            features[f"expected_matchup_net_{w}"] = features[f"expected_home_rating_{w}"] - features[f"expected_away_rating_{w}"]
            features[f"expected_pace_{w}"] = (home_row.get(f"team_roll_{w}_pace", np.nan) + away_row.get(f"team_roll_{w}_pace", np.nan)) / 2
            features[f"expected_home_efg_adv_{w}"] = home_row.get(f"team_roll_{w}_effectiveFieldGoalPercentage", np.nan) - away_row.get(f"team_roll_{w}_opponentEffectiveFieldGoalPercentage", np.nan)
            features[f"expected_home_tov_adv_{w}"] = away_row.get(f"team_roll_{w}_teamTurnoverPercentage", np.nan) - home_row.get(f"team_roll_{w}_teamTurnoverPercentage", np.nan)
            features[f"expected_home_ftr_adv_{w}"] = home_row.get(f"team_roll_{w}_freeThrowAttemptRate", np.nan) - away_row.get(f"team_roll_{w}_opponentFreeThrowAttemptRate", np.nan)
        features.update(self.elo_before_date(home_team_id, away_team_id, game_date))
        home_pf, _ = self.lineup_features(home_lineup, home_team_id, game_date, "home", allow_transfers)
        away_pf, _ = self.lineup_features(away_lineup, away_team_id, game_date, "away", allow_transfers)
        for hk, hv in home_pf.items():
            ak = hk.replace("home_", "away_")
            if ak in away_pf:
                features["diff_" + hk.replace("home_", "")] = hv - away_pf[ak]
        X = pd.DataFrame([features])
        for col in feature_cols:
            if col not in X.columns:
                X[col] = np.nan
        X = X[feature_cols]
        logreg = float(self.model_bundle["logreg_model"].predict_proba(X)[:, 1][0])
        hgb = float(self.model_bundle["hgb_model"].predict_proba(X)[:, 1][0])
        Xcb = X.copy()
        Xcb["home_team_name"] = home_team_name
        Xcb["away_team_name"] = away_team_name
        Xcb["gameType"] = "Regular Season"
        cat = float(self.model_bundle["catboost_model"].predict_proba(Xcb)[:, 1][0])
        Xnn = self.model_bundle["nn_preprocessor"].transform(X)
        nn = float(self.nn_model.predict(Xnn, verbose=0).ravel()[0])
        weights = self.model_bundle.get("ensemble_weights", {"catboost":0.35,"hgb":0.30,"logreg":0.20,"nn":0.15})
        proba = weights["catboost"]*cat + weights["hgb"]*hgb + weights["logreg"]*logreg + weights["nn"]*nn
        predicted = home_team_name if proba >= 0.5 else away_team_name
        confidence = proba if proba >= 0.5 else 1-proba
        actual = self.find_actual_game_result(home_team_name, away_team_name, game_date)
        max_date = max(self.teams["gameDateTimeEst"].max(), self.players_played["gameDateTimeEst"].max(), self.games_model["gameDateTimeEst"].max())
        warning = None
        if pd.to_datetime(game_date) > max_date:
            warning = f"Дата позже последней даты в данных: {max_date.date()}. Прогноз построен по последнему историческому срезу."
        result = {
            "home_team_name": home_team_name, "away_team_name": away_team_name, "game_date": str(pd.to_datetime(game_date).date()),
            "home_win_probability": float(proba), "away_win_probability": float(1-proba), "confidence": float(confidence),
            "predicted_winner": predicted, "logreg_proba": logreg, "hgb_proba": hgb, "catboost_proba": cat, "nn_proba": nn,
            "actual_winner": None, "home_score": None, "away_score": None, "model_correct": None, "warning": warning
        }
        if actual:
            result["actual_winner"] = actual["actual_winner"]
            result["home_score"] = actual["home_score"]
            result["away_score"] = actual["away_score"]
            result["model_correct"] = predicted == actual["actual_winner"]
        return result

predictor_service = PredictorService()
