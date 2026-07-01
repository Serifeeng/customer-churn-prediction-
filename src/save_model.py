import json
from pathlib import Path

import joblib

from src.config import MODELS_DIR

MANIFEST_PATH = MODELS_DIR / "manifest.json"


def save_artifacts(model, preprocessor, model_name: str) -> Path:
    """Persist the best model and preprocessing pipeline."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    safe_name = model_name.lower().replace(" ", "_")
    model_path = MODELS_DIR / "best_model.pkl"
    preprocessor_path = MODELS_DIR / "preprocessor.pkl"

    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)
    MANIFEST_PATH.write_text(
        json.dumps({"model_name": model_name, "artifact_name": safe_name}, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved model to {model_path}")
    print(f"Saved preprocessor to {preprocessor_path}")

    return model_path


def load_artifacts():
    """Load the saved best model and preprocessor."""
    model_path = MODELS_DIR / "best_model.pkl"
    preprocessor_path = MODELS_DIR / "preprocessor.pkl"

    if not model_path.exists() or not preprocessor_path.exists():
        raise FileNotFoundError(
            "Saved model artifacts not found. Run `python main.py` first."
        )

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor
