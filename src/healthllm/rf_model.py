import pickle
from pathlib import Path

import pandas as pd

from healthllm.prompts import FEATURE_COLS


def default_model_path():
    return Path(__file__).resolve().parents[2] / "models" / "readiness_rf.pkl"


def load_rf_artifact(model_path=None):
    model_path = Path(model_path or default_model_path())

    with model_path.open("rb") as f:
        return pickle.load(f)


def predict_readiness_rf(row, artifact):
    model = artifact["model"]
    feature_cols = artifact.get("feature_columns", FEATURE_COLS)

    X = pd.DataFrame([{col: row.get(col) for col in feature_cols}])
    prediction = model.predict(X)[0]

    return {
        "predicted_readiness": float(prediction),
        "parse_success": True,
        "prompt_version": "rf_baseline_v1",
        "error_message": None,
        "raw_response": None,
    }
