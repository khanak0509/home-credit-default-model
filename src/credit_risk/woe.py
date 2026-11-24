from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

MISSING = "MISSING"
OTHER = "OTHER"


@dataclass
class FeatureBinning:
    name: str
    kind: str
    cuts: np.ndarray = field(default_factory=lambda: np.array([]))
    category_map: dict[str, str] = field(default_factory=dict)
    labels: list[str] = field(default_factory=list)
    woe: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    event_rates: dict[str, float] = field(default_factory=dict)
    iv: float = 0.0
    direction: str = ""

    def assign(self, values: pd.Series) -> np.ndarray:
        if self.kind == "numeric":
            numbers = values.to_numpy(dtype=float)
            labels = np.array(interval_labels(self.cuts), dtype=object)[np.searchsorted(self.cuts, numbers, side="right")]
            labels[np.isnan(numbers)] = MISSING
            return labels
        text = values.astype(object).where(values.notna(), MISSING).astype(str)
        return text.map(self.category_map).fillna(OTHER).to_numpy(dtype=object)

    def transform(self, values: pd.Series) -> np.ndarray:
        if self.kind == "numeric":
            numbers = values.to_numpy(dtype=float)
            lookup = np.array([self.woe.get(label, 0.0) for label in interval_labels(self.cuts)])
            encoded = lookup[np.searchsorted(self.cuts, numbers, side="right")]
            encoded[np.isnan(numbers)] = self.woe.get(MISSING, 0.0)
            return encoded
        labels = pd.Series(self.assign(values))
        return labels.map(self.woe).fillna(0.0).to_numpy(dtype=float)


def interval_labels(cuts: np.ndarray) -> list[str]:
    bounds = [-np.inf, *cuts, np.inf]
    return [f"[{bounds[i]:.6g}, {bounds[i + 1]:.6g})" for i in range(len(bounds) - 1)]


def woe_and_iv(goods: np.ndarray, bads: np.ndarray, smoothing: float) -> tuple[np.ndarray, float]:
    good_share = (goods + smoothing) / (goods.sum() + smoothing * goods.size)
    bad_share = (bads + smoothing) / (bads.sum() + smoothing * bads.size)
    woe = np.log(good_share / bad_share)
    return woe, float(np.sum((good_share - bad_share) * woe))


def prebin_cuts(values: np.ndarray, n_prebins: int) -> np.ndarray:
    quantiles = np.quantile(values, np.linspace(0, 1, n_prebins + 1)[1:-1])
    return np.unique(quantiles)


def bin_counts(values: np.ndarray, y: np.ndarray, cuts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positions = np.searchsorted(cuts, values, side="right")
    bads = np.bincount(positions, weights=y, minlength=cuts.size + 1)
    totals = np.bincount(positions, minlength=cuts.size + 1)
    return totals - bads, bads


def merge_until_monotonic(cuts: np.ndarray, goods: np.ndarray, bads: np.ndarray, increasing: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cuts, goods, bads = list(cuts), list(goods), list(bads)
    while len(goods) > 1:
        rates = np.array(bads) / np.maximum(np.array(goods) + np.array(bads), 1)
        steps = np.diff(rates)
        violations = np.flatnonzero(steps < 0) if increasing else np.flatnonzero(steps > 0)
        if violations.size == 0:
            break
        i = int(violations[np.argmin(np.abs(steps[violations]))])
        goods[i] += goods.pop(i + 1)
        bads[i] += bads.pop(i + 1)
        cuts.pop(i)
    return np.array(cuts), np.array(goods), np.array(bads)


def merge_small_bins(cuts: np.ndarray, goods: np.ndarray, bads: np.ndarray, min_count: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cuts, goods, bads = list(cuts), list(goods), list(bads)
    while len(goods) > 1:
        totals = np.array(goods) + np.array(bads)
        smallest = int(np.argmin(totals))
        if totals[smallest] >= min_count:
            break
        if smallest == 0:
            neighbor = 1
        elif smallest == len(goods) - 1:
            neighbor = smallest - 1
        else:
            rates = np.array(bads) / np.maximum(totals, 1)
            left_gap = abs(rates[smallest] - rates[smallest - 1])
            right_gap = abs(rates[smallest] - rates[smallest + 1])
            neighbor = smallest - 1 if left_gap <= right_gap else smallest + 1
        keep, drop = min(smallest, neighbor), max(smallest, neighbor)
        goods[keep] += goods.pop(drop)
        bads[keep] += bads.pop(drop)
        cuts.pop(keep)
    return np.array(cuts), np.array(goods), np.array(bads)


class WoEBinner(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        n_prebins: int = 20,
        min_bin_share: float = 0.05,
        iv_threshold: float = 0.02,
        max_features: int | None = None,
        correlation_threshold: float | None = None,
        smoothing: float = 0.5,
    ):
        self.n_prebins = n_prebins
        self.min_bin_share = min_bin_share
        self.iv_threshold = iv_threshold
        self.max_features = max_features
        self.correlation_threshold = correlation_threshold
        self.smoothing = smoothing

    def _finalize(self, binning: FeatureBinning, labels: np.ndarray, y: np.ndarray) -> FeatureBinning:
        frame = pd.DataFrame({"label": labels, "y": y})
        grouped = frame.groupby("label", sort=False)["y"].agg(["sum", "count"])
        ordered = [label for label in binning.labels if label in grouped.index]
        bads = grouped.loc[ordered, "sum"].to_numpy(dtype=float)
        goods = grouped.loc[ordered, "count"].to_numpy(dtype=float) - bads
        woe, iv = woe_and_iv(goods, bads, self.smoothing)
        binning.labels = ordered
        binning.woe = dict(zip(ordered, woe.tolist()))
        binning.counts = dict(zip(ordered, (goods + bads).astype(int).tolist()))
        binning.event_rates = dict(zip(ordered, (bads / np.maximum(goods + bads, 1)).tolist()))
        binning.iv = iv
        return binning

    def _fit_numeric(self, name: str, values: pd.Series, y: np.ndarray) -> FeatureBinning:
        numbers = values.to_numpy(dtype=float)
        present = ~np.isnan(numbers)
        min_count = self.min_bin_share * y.size
        best = None
        if present.sum() > 0:
            cuts = prebin_cuts(numbers[present], self.n_prebins)
            goods, bads = bin_counts(numbers[present], y[present], cuts)
            for increasing in (True, False):
                merged = merge_until_monotonic(cuts, goods, bads, increasing)
                merged = merge_small_bins(*merged, min_count)
                merged = merge_until_monotonic(*merged, increasing)
                _, iv = woe_and_iv(merged[1], merged[2], self.smoothing)
                if best is None or iv > best[0]:
                    best = (iv, merged[0], "increasing" if increasing else "decreasing")
        cuts = best[1] if best else np.array([])
        binning = FeatureBinning(name=name, kind="numeric", cuts=cuts, direction=best[2] if best else "")
        binning.labels = [*interval_labels(cuts), MISSING]
        return self._finalize(binning, binning.assign(values), y)

    def _fit_categorical(self, name: str, values: pd.Series, y: np.ndarray) -> FeatureBinning:
        text = values.astype(object).where(values.notna(), MISSING).astype(str)
        shares = text.value_counts(normalize=True)
        category_map = {value: (value if share >= self.min_bin_share or value == MISSING else OTHER) for value, share in shares.items()}
        binning = FeatureBinning(name=name, kind="categorical", category_map=category_map)
        binning.labels = sorted(set(category_map.values()) | {OTHER})
        return self._finalize(binning, binning.assign(values), y)

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "WoEBinner":
        y = np.asarray(y, dtype=float)
        self.binnings_ = {}
        for column in X.columns:
            values = X[column]
            is_categorical = isinstance(values.dtype, pd.CategoricalDtype) or values.dtype == object
            fitter = self._fit_categorical if is_categorical else self._fit_numeric
            self.binnings_[column] = fitter(column, values, y)
        self.information_values_ = {name: binning.iv for name, binning in self.binnings_.items()}
        self.selected_ = self._select(X)
        return self

    def _select(self, X: pd.DataFrame) -> list[str]:
        ranked = sorted((name for name, iv in self.information_values_.items() if iv >= self.iv_threshold),
                        key=lambda name: -self.information_values_[name])
        if self.correlation_threshold is None:
            return ranked[: self.max_features]
        encoded = pd.DataFrame({name: self.binnings_[name].transform(X[name]) for name in ranked})
        correlations = encoded.corr().abs() if ranked else pd.DataFrame()
        selected: list[str] = []
        for name in ranked:
            if all(correlations.loc[name, kept] < self.correlation_threshold for kept in selected):
                selected.append(name)
            if self.max_features and len(selected) >= self.max_features:
                break
        return selected

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({name: self.binnings_[name].transform(X[name]) for name in self.selected_}, index=X.index)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(self.selected_, dtype=object)

    def binning_table(self, features: list[str] | None = None) -> pd.DataFrame:
        rows = []
        for name in features or self.selected_:
            binning = self.binnings_[name]
            for label in binning.labels:
                rows.append({
                    "feature": name,
                    "bin": label,
                    "count": binning.counts.get(label, 0),
                    "event_rate": binning.event_rates.get(label, float("nan")),
                    "woe": binning.woe.get(label, 0.0),
                    "iv": binning.iv,
                })
        return pd.DataFrame(rows)
