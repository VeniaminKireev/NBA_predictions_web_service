import json
from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from backend.db.session import get_db
from backend.models.prediction import PredictionRecord
from backend.schemas.prediction import PlayerListResponse, PredictRequest, PredictResponse, PredictionHistoryItem, TeamListResponse
from backend.services.predictor import predictor_service

router = APIRouter()

@router.get("/health")
def health():
    return {"status": "ok"}

@router.get("/teams", response_model=TeamListResponse)
def get_teams():
    try:
        return {"teams": predictor_service.get_teams()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@router.get("/players", response_model=PlayerListResponse)
def get_players(team_name: str = Query(...), game_date: str = Query(...), allow_transfers: bool = Query(False)):
    try:
        return {"players": predictor_service.get_players_for_team_before_date(team_name, game_date, allow_transfers)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest, db: Session = Depends(get_db)):
    try:
        pred = predictor_service.predict(
            request.home_team_name, request.away_team_name, request.game_date,
            request.home_lineup, request.away_lineup, request.allow_transfers
        )
        record = PredictionRecord(
            home_team_name=request.home_team_name,
            away_team_name=request.away_team_name,
            game_date=request.game_date,
            home_lineup=json.dumps(request.home_lineup, ensure_ascii=False),
            away_lineup=json.dumps(request.away_lineup, ensure_ascii=False),
            allow_transfers=request.allow_transfers,
            home_win_probability=pred["home_win_probability"],
            away_win_probability=pred["away_win_probability"],
            confidence=pred["confidence"],
            predicted_winner=pred["predicted_winner"],
            actual_winner=pred.get("actual_winner"),
            home_score=pred.get("home_score"),
            away_score=pred.get("away_score"),
            model_correct=pred.get("model_correct"),
        )
        db.add(record)
        db.commit()
        return pred
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@router.get("/predictions", response_model=List[PredictionHistoryItem])
def get_prediction_history(db: Session = Depends(get_db), limit: int = Query(20, ge=1, le=100)):
    rows = db.query(PredictionRecord).order_by(PredictionRecord.id.desc()).limit(limit).all()
    return [
        PredictionHistoryItem(
            id=row.id,
            created_at=str(row.created_at),
            home_team_name=row.home_team_name,
            away_team_name=row.away_team_name,
            game_date=row.game_date,
            predicted_winner=row.predicted_winner,
            confidence=row.confidence,
            home_win_probability=row.home_win_probability,
            actual_winner=row.actual_winner,
            model_correct=row.model_correct,
        )
        for row in rows
    ]
