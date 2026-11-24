import numpy as np
import pytest
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

from credit_risk.calibration import choose_calibrator
from credit_risk.metrics import (
    approval_at_bad_rate,
    expected_calibration_error,
    fast_auc,
    ks_statistic,
    paired_bootstrap,
    population_stability_index,
)


@pytest.fixture(scope="module")
def scored():
    rng = np.random.default_rng(1)
    n = 5000
    y = (rng.random(n) < 0.1).astype(int)
    scores = rng.normal(size=n) + y * 1.0
    return y, scores


def test_fast_auc_matches_sklearn(scored):
    y, scores = scored
    assert fast_auc(y, scores) == pytest.approx(roc_auc_score(y, scores))


def test_fast_auc_handles_ties():
    y = np.array([0, 1, 0, 1, 0, 1])
    scores = np.array([0.1, 0.1, 0.5, 0.5, 0.9, 0.9])
    assert fast_auc(y, scores) == pytest.approx(roc_auc_score(y, scores))


def test_ks_matches_two_sample_statistic(scored):
    y, scores = scored
    assert ks_statistic(y, scores) == pytest.approx(ks_2samp(scores[y == 1], scores[y == 0]).statistic)


def test_approval_respects_target_bad_rate(scored):
    y, scores = scored
    result = approval_at_bad_rate(y, scores, 0.05)
    approved = scores <= result["threshold"]
    assert y[approved].mean() <= 0.05
    assert approved.mean() == pytest.approx(result["approval_rate"])
    assert approval_at_bad_rate(y, scores, 1.0)["approval_rate"] == 1.0


def test_psi_is_small_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(2)
    reference = rng.normal(size=20000)
    assert population_stability_index(reference, rng.normal(size=20000), 10) < 0.01
    assert population_stability_index(reference, rng.normal(loc=1.0, size=20000), 10) > 0.25


def test_ece_near_zero_for_calibrated_probabilities():
    rng = np.random.default_rng(3)
    probabilities = rng.uniform(0, 0.3, size=50000)
    y = (rng.random(50000) < probabilities).astype(int)
    assert expected_calibration_error(y, probabilities, 10) < 0.01
    assert expected_calibration_error(y, np.clip(probabilities * 3, 0, 1), 10) > 0.1


def test_bootstrap_is_paired_and_reproducible(scored):
    y, scores = scored
    first = paired_bootstrap(y, {"a": scores, "b": scores}, 20, seed=42)
    second = paired_bootstrap(y, {"a": scores, "b": scores}, 20, seed=42)
    assert np.array_equal(first["a"]["roc_auc"], first["b"]["roc_auc"])
    assert np.array_equal(first["a"]["roc_auc"], second["a"]["roc_auc"])


def test_calibrator_fixes_inflated_probabilities():
    rng = np.random.default_rng(4)
    n = 30000
    truth = rng.uniform(0.01, 0.2, size=n)
    y = (rng.random(n) < truth).astype(int)
    inflated = np.clip(truth * 3, 0, 0.99)
    folds = np.arange(n) % 5
    calibrator, scores = choose_calibrator(inflated, y, folds, ["isotonic", "platt"])
    assert min(scores["isotonic"], scores["platt"]) < scores["uncalibrated"]
    assert abs(calibrator.predict(inflated).mean() - y.mean()) < 0.01
