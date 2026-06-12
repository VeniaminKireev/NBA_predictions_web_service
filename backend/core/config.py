from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

MODEL_BUNDLE_PATH = ARTIFACTS_DIR / "nba_v5_model_bundle.pkl"
APP_DATA_PATH = ARTIFACTS_DIR / "nba_v5_app_data.pkl"
MLP_MODEL_PATH = ARTIFACTS_DIR / "nba_v5_mlp_model.keras"

DATABASE_URL = f"sqlite:///{PROJECT_ROOT / 'nba_predictions.db'}"
