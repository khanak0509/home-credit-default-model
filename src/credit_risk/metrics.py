import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

EPSILON = 1e-7


def fast_auc(y: np.ndarray, scores: np.ndarray) -> float:
    positives = y == 1
    n_pos = positives.sum()
    n_neg = y.size - n_pos
    ranks = rankdata(scores)
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def ks_statistic(y: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_y = y[order]
    tpr = np.cumsum(sorted_y) / sorted_y.sum()
    fpr = np.cumsum(1 - sorted_y) / (sorted_y.size - sorted_y.sum())
    last_of_tie = np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    return float(np.max(tpr[last_of_tie] - fpr[last_of_tie]))


def ks_curve(y: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-scores, kind="mergesort")
    sorted_y = y[order]
    population = np.arange(1, y.size + 1) / y.size
    tpr = np.cumsum(sorted_y) / sorted_y.sum()
    fpr = np.cumsum(1 - sorted_y) / (sorted_y.size - sorted_y.sum())
    return population, tpr, fpr


def expected_calibration_error(y: np.ndarray, probabilities: np.ndarray, n_bins: int) -> float:
    edges = np.quantile(probabilities, np.linspace(0, 1, n_bins + 1))
    bins = np.clip(np.searchsorted(edges[1:-1], probabilities, side="right"), 0, n_bins - 1)
    error = 0.0
    for b in np.unique(bins):
        mask = bins == b
        error += mask.mean() * abs(probabilities[mask].mean() - y[mask].mean())
    return float(error)


def calibration_table(y: np.ndarray, probabilities: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    edges = np.quantile(probabilities, np.linspace(0, 1, n_bins + 1))
    bins = np.clip(np.searchsorted(edges[1:-1], probabilities, side="right"), 0, n_bins - 1)
    predicted = np.array([probabilities[bins == b].mean() for b in np.unique(bins)])
    observed = np.array([y[bins == b].mean() for b in np.unique(bins)])
    return predicted, observed


def classification_metrics(y: np.ndarray, probabilities: np.ndarray, ece_bins: int = 10) -> dict[str, float]:
    clipped = np.clip(probabilities, EPSILON, 1 - EPSILON)
    auc = roc_auc_score(y, probabilities)
    return {
        "roc_auc": float(auc),
        "pr_auc": float(average_precision_score(y, probabilities)),
        "ks": ks_statistic(y, probabilities),
        "gini": float(2 * auc - 1),
        "brier": float(brier_score_loss(y, probabilities)),
        "log_loss": float(log_loss(y, clipped)),
        "ece": expected_calibration_error(y, probabilities, ece_bins),
        "mean_predicted": float(probabilities.mean()),
        "observed_rate": float(y.mean()),
    }


def approval_at_bad_rate(y: np.ndarray, probabilities: np.ndarray, target_bad_rate: float) -> dict[str, float]:
    order = np.argsort(probabilities, kind="mergesort")
    sorted_probabilities = probabilities[order]
    cumulative_bad_rate = np.cumsum(y[order]) / np.arange(1, y.size + 1)
    last_of_tie = np.r_[sorted_probabilities[1:] != sorted_probabilities[:-1], True]
    eligible = np.flatnonzero((cumulative_bad_rate <= target_bad_rate) & last_of_tie)
    if eligible.size == 0:
        return {"approval_rate": 0.0, "threshold": float(sorted_probabilities[0]), "realized_bad_rate": float("nan")}
    cut = eligible.max()
    return {
        "approval_rate": float((cut + 1) / y.size),
        "threshold": float(sorted_probabilities[cut]),
        "realized_bad_rate": float(cumulative_bad_rate[cut]),
    }


def population_stability_index(expected: np.ndarray, actual: np.ndarray, n_bins: int) -> float:
    expected = expected[~np.isnan(expected)]
    actual_missing = np.isnan(actual)
    actual = actual[~actual_missing]
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    inner = edges[1:-1]
    expected_share = np.bincount(np.searchsorted(inner, expected, side="right"), minlength=inner.size + 1) / expected.size
    actual_share = np.bincount(np.searchsorted(inner, actual, side="right"), minlength=inner.size + 1) / actual.size
    expected_share = np.clip(expected_share, 1e-6, None)
    actual_share = np.clip(actual_share, 1e-6, None)
    return float(np.sum((actual_share - expected_share) * np.log(actual_share / expected_share)))


def paired_bootstrap(y: np.ndarray, scores: dict[str, np.ndarray], n_resamples: int, seed: int) -> dict[str, dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    draws = {name: {"roc_auc": np.empty(n_resamples), "ks": np.empty(n_resamples)} for name in scores}
    for b in range(n_resamples):
        index = rng.integers(0, y.size, y.size)
        sampled_y = y[index]
        for name, values in scores.items():
            draws[name]["roc_auc"][b] = fast_auc(sampled_y, values[index])
            draws[name]["ks"][b] = ks_statistic(sampled_y, values[index])
    return draws


def interval(values: np.ndarray, level: float = 0.95) -> list[float]:
    tail = (1 - level) / 2
    return [float(np.quantile(values, tail)), float(np.quantile(values, 1 - tail))]
