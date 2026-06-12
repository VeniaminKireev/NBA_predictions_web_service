from typing import List, Optional
from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    home_team_name: str
    away_team_name: str
    game_date: str
    home_lineup: List[str] = Field(min_length=7, max_length=7)
    away_lineup: List[str] = Field(min_length=7, max_length=7)
    allow_transfers: bool = False

class PredictResponse(BaseModel):
    home_team_name: str
    away_team_name: str
    game_date: str

    home_win_probability: float
    away_win_probability: float
    confidence: float
    predicted_winner: str

    logreg_proba: float
    hgb_proba: float
    catboost_proba: float
    nn_proba: float

    actual_winner: Optional[str] = None
    home_score: Optional[float] = None
    away_score: Optional[float] = None
    model_correct: Optional[bool] = None
    warning: Optional[str] = None

class TeamListResponse(BaseModel):
    teams: List[str]

class PlayerListResponse(BaseModel):
    players: List[str]

class PredictionHistoryItem(BaseModel):
    id: int
    created_at: str
    home_team_name: str
    away_team_name: str
    game_date: str
    predicted_winner: str
    confidence: float
    home_win_probability: float
    actual_winner: Optional[str] = None
    model_correct: Optional[bool] = None
