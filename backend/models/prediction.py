from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text, func
from backend.db.session import Base

class PredictionRecord(Base):
    __tablename__ = "prediction_records"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    home_team_name = Column(String, nullable=False)
    away_team_name = Column(String, nullable=False)
    game_date = Column(String, nullable=False)

    home_lineup = Column(Text, nullable=False)
    away_lineup = Column(Text, nullable=False)
    allow_transfers = Column(Boolean, default=False)

    home_win_probability = Column(Float, nullable=False)
    away_win_probability = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    predicted_winner = Column(String, nullable=False)

    actual_winner = Column(String, nullable=True)
    home_score = Column(Float, nullable=True)
    away_score = Column(Float, nullable=True)
    model_correct = Column(Boolean, nullable=True)
