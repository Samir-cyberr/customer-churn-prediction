from pathlib import Path

import joblib
import pandas as pd


MODEL_PATH = Path("models/churn_model.joblib")


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run `python -m src.train` first."
        )
    return joblib.load(MODEL_PATH)


def predict_churn(model, customer: dict) -> tuple[int, float]:
    frame = pd.DataFrame([customer])
    probability = float(model.predict_proba(frame)[0, 1])
    prediction = int(probability >= 0.5)
    return prediction, probability
