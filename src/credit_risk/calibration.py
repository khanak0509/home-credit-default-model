import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

EPSILON = 1e-6


def logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, EPSILON, 1 - EPSILON)
    return np.log(clipped / (1 - clipped))


class PlattCalibrator:
    method = "platt"

    def fit(self, probabilities: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        self.model = LogisticRegression(C=1e6, max_iter=1000)
        self.model.fit(logit(probabilities).reshape(-1, 1), y)
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(logit(probabilities).reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    method = "isotonic"

    def fit(self, probabilities: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        self.model = IsotonicRegression(out_of_bounds="clip", y_min=EPSILON, y_max=1 - EPSILON)
        self.model.fit(probabilities, y)
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        return self.model.predict(probabilities)


CALIBRATORS = {"isotonic": IsotonicCalibrator, "platt": PlattCalibrator}


def cross_fitted(method: str, probabilities: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    calibrated = np.zeros_like(probabilities, dtype=float)
    for fold in np.unique(folds):
        fit = folds != fold
        calibrated[~fit] = CALIBRATORS[method]().fit(probabilities[fit], y[fit]).predict(probabilities[~fit])
    return calibrated


def choose_calibrator(probabilities: np.ndarray, y: np.ndarray, folds: np.ndarray, methods: list[str]) -> tuple[object, dict[str, float]]:
    scores = {"uncalibrated": float(brier_score_loss(y, probabilities))}
    for method in methods:
        calibrated = cross_fitted(method, probabilities, y, folds)
        scores[method] = float(np.mean([brier_score_loss(y[folds == k], calibrated[folds == k]) for k in np.unique(folds)]))
    best = min(methods, key=lambda method: scores[method])
    return CALIBRATORS[best]().fit(probabilities, y), scores
