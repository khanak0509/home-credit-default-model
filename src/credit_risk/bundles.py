from typing import Any

import joblib
import numpy as np
import pandas as pd

from credit_risk.artifacts import require
from credit_risk.config import get_paths

MODEL_LABELS = {
    "legacy": "Legacy logistic baseline",
    "scorecard": "WoE scorecard",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "lightgbm_monotone": "LightGBM monotonic",
}

MODEL_ORDER = list(MODEL_LABELS)


def save_bundle(name: str, estimator: Any, features: list[str], calibrator: Any, calibration_scores: dict[str, float], kind: str) -> None:
    joblib.dump(
        {"name": name, "label": MODEL_LABELS[name], "kind": kind, "estimator": estimator, "features": features,
         "calibrator": calibrator, "calibration_scores": calibration_scores},
        get_paths().models / f"{name}.joblib",
        compress=3,
    )


def load_bundle(name: str) -> dict[str, Any]:
    return joblib.load(require(get_paths().models / f"{name}.joblib", "train"))


def predict_raw(bundle: dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    estimator = bundle["estimator"]
    X = X[bundle["features"]]
    if bundle["kind"] == "sklearn":
        return estimator.predict_proba(X)[:, 1]
    return estimator.predict_proba(X)


def predict_calibrated(bundle: dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    return bundle["calibrator"].predict(predict_raw(bundle, X))
