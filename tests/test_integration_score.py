import numpy as np
import polars as pl
import pytest

from credit_risk.score import UnknownApplicantError, score_applicant

EXPECTED_KEYS = {"sk_id_curr", "model", "in_holdout", "probability_of_default", "raw_model_probability",
                 "scorecard_points", "risk_band", "reason_codes"}


@pytest.fixture(scope="module")
def known_ids(paths):
    split = pl.read_parquet(paths.processed / "split.parquet").sort("SK_ID_CURR")
    holdout = split.filter(pl.col("is_holdout"))["SK_ID_CURR"].head(3).to_list()
    train = split.filter(~pl.col("is_holdout"))["SK_ID_CURR"].head(2).to_list()
    return holdout + train


def test_score_schema_and_ranges(known_ids, config):
    bands = {band[0] for band in config["scoring"]["risk_bands"]}
    for applicant_id in known_ids:
        result = score_applicant(applicant_id)
        assert set(result) == EXPECTED_KEYS
        assert result["sk_id_curr"] == applicant_id
        assert 0.0 <= result["probability_of_default"] <= 1.0
        assert 0.0 <= result["raw_model_probability"] <= 1.0
        assert np.isfinite(result["scorecard_points"])
        assert result["risk_band"] in bands
        assert len(result["reason_codes"]) == config["scoring"]["reason_codes"]
        for reason in result["reason_codes"]:
            assert set(reason) == {"feature", "value", "direction", "contribution"}
            assert reason["direction"] in {"increases risk", "decreases risk"}
            assert (reason["contribution"] > 0) == (reason["direction"] == "increases risk")
        contributions = [abs(r["contribution"]) for r in result["reason_codes"]]
        assert contributions == sorted(contributions, reverse=True)


def test_unknown_applicant_raises():
    with pytest.raises(UnknownApplicantError):
        score_applicant(-1)
