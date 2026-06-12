from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes import router
from backend.db.session import Base, engine
from backend.models.prediction import PredictionRecord
from backend.services.predictor import predictor_service

Base.metadata.create_all(bind=engine)

app = FastAPI(title="NBA Win Predictor API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    predictor_service.load()

app.include_router(router)
