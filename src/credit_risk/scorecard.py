import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from credit_risk.woe import WoEBinner


def points_scaling(base_score: float, base_odds: float, points_to_double_odds: float) -> tuple[float, float]:
    factor = points_to_double_odds / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    return factor, offset


class ScorecardModel:
    def __init__(self, binner: WoEBinner, C: float, seed: int, base_score: float, base_odds: float, points_to_double_odds: float):
        self.binner = binner
        self.C = C
        self.seed = seed
        self.factor, self.offset = points_scaling(base_score, base_odds, points_to_double_odds)

    def fit_regression(self, encoded: pd.DataFrame, y: np.ndarray) -> "ScorecardModel":
        self.regression = LogisticRegression(C=self.C, solver="lbfgs", max_iter=2000, random_state=self.seed)
        self.regression.fit(encoded.to_numpy(), y)
        self.features_ = list(encoded.columns)
        return self

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "ScorecardModel":
        self.binner.fit(X, y)
        return self.fit_regression(self.binner.transform(X), y)

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        return self.regression.decision_function(self.binner.transform(X)[self.features_].to_numpy())

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.regression.predict_proba(self.binner.transform(X)[self.features_].to_numpy())[:, 1]

    def points(self, X: pd.DataFrame) -> np.ndarray:
        return self.offset - self.factor * self.decision_function(X)

    def table(self) -> pd.DataFrame:
        table = self.binner.binning_table(self.features_)
        coefficients = dict(zip(self.features_, self.regression.coef_[0]))
        n_features = len(self.features_)
        intercept = self.regression.intercept_[0]
        table["coefficient"] = table["feature"].map(coefficients)
        table["points"] = self.offset / n_features - self.factor * (table["coefficient"] * table["woe"] + intercept / n_features)
        return table

    def attribute_points(self, X: pd.DataFrame) -> pd.DataFrame:
        encoded = self.binner.transform(X)[self.features_]
        coefficients = self.regression.coef_[0]
        n_features = len(self.features_)
        intercept = self.regression.intercept_[0]
        contributions = self.offset / n_features - self.factor * (encoded * coefficients + intercept / n_features)
        return contributions
