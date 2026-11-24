import numpy as np
import pandas as pd
import pytest

from credit_risk.scorecard import ScorecardModel, points_scaling
from credit_risk.woe import MISSING, OTHER, WoEBinner


@pytest.fixture(scope="module")
def sample():
    rng = np.random.default_rng(0)
    n = 20000
    risk_driver = rng.normal(size=n)
    noise = rng.normal(size=n)
    category = rng.choice(["a", "b", "c", "rare"], size=n, p=[0.5, 0.3, 0.18, 0.02])
    logit = -2.5 + 1.2 * risk_driver + np.where(category == "a", 0.5, 0.0)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    driver = risk_driver.copy()
    driver[rng.random(n) < 0.1] = np.nan
    X = pd.DataFrame({"driver": driver, "noise": noise, "segment": pd.Categorical(category)})
    return X, y


def test_numeric_bins_are_monotonic(sample):
    X, y = sample
    binner = WoEBinner(iv_threshold=0.0).fit(X, y)
    binning = binner.binnings_["driver"]
    rates = [binning.event_rates[label] for label in binning.labels if label != MISSING]
    assert len(rates) >= 3
    assert all(np.diff(rates) > 0) or all(np.diff(rates) < 0)
    assert binning.direction == "increasing"


def test_bins_respect_minimum_share(sample):
    X, y = sample
    binner = WoEBinner(min_bin_share=0.05, iv_threshold=0.0).fit(X, y)
    binning = binner.binnings_["driver"]
    counts = [binning.counts[label] for label in binning.labels if label != MISSING]
    assert min(counts) >= 0.05 * len(y)


def test_iv_selection_drops_noise(sample):
    X, y = sample
    binner = WoEBinner(iv_threshold=0.02).fit(X, y)
    assert "driver" in binner.selected_
    assert "noise" not in binner.selected_
    assert binner.information_values_["driver"] > binner.information_values_["noise"]


def test_rare_and_unseen_categories_map_to_other(sample):
    X, y = sample
    binner = WoEBinner(iv_threshold=0.0).fit(X, y)
    binning = binner.binnings_["segment"]
    assert binning.category_map["rare"] == OTHER
    unseen = pd.Series(pd.Categorical(["unseen"]))
    assert binning.assign(unseen)[0] == OTHER


def test_missing_values_get_their_own_woe(sample):
    X, y = sample
    binner = WoEBinner(iv_threshold=0.0).fit(X, y)
    encoded = binner.transform(pd.DataFrame({"driver": [np.nan], "noise": [0.0], "segment": pd.Categorical(["a"])}))
    assert encoded.loc[0, "driver"] == pytest.approx(binner.binnings_["driver"].woe[MISSING])


def test_higher_risk_bins_have_lower_woe(sample):
    X, y = sample
    binner = WoEBinner(iv_threshold=0.0).fit(X, y)
    binning = binner.binnings_["driver"]
    labels = [label for label in binning.labels if label != MISSING]
    woe = [binning.woe[label] for label in labels]
    rates = [binning.event_rates[label] for label in labels]
    assert np.corrcoef(woe, rates)[0, 1] < -0.9


def test_points_scaling_matches_definition():
    factor, offset = points_scaling(600, 50, 20)
    assert offset + factor * np.log(50) == pytest.approx(600)
    assert offset + factor * np.log(100) == pytest.approx(620)


def test_scorecard_table_sums_to_total_points(sample):
    X, y = sample
    model = ScorecardModel(WoEBinner(iv_threshold=0.0), C=1.0, seed=42, base_score=600, base_odds=50, points_to_double_odds=20).fit(X, y)
    table = model.table()
    row = X.iloc[[5]]
    total = 0.0
    for feature in model.features_:
        label = model.binner.binnings_[feature].assign(row[feature])[0]
        total += table[(table.feature == feature) & (table.bin == label)]["points"].iloc[0]
    assert total == pytest.approx(model.points(row)[0], abs=1e-6)
    probability = model.predict_proba(row)[0]
    assert model.points(row)[0] == pytest.approx(model.offset + model.factor * np.log((1 - probability) / probability))
