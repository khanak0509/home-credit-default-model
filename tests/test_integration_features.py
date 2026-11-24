import polars as pl
import pytest

from credit_risk.artifacts import read_json
from credit_risk.db import connect
from credit_risk.features import prepare_views


@pytest.fixture(scope="module")
def features(paths):
    return pl.read_parquet(paths.processed / "features.parquet")


@pytest.fixture(scope="module")
def manifest(paths):
    return read_json(paths.processed / "features_manifest.json", "features")


def test_one_row_per_applicant(features):
    assert features["SK_ID_CURR"].n_unique() == features.height


def test_manifest_covers_every_feature(features, manifest):
    feature_columns = set(features.columns) - {"SK_ID_CURR", "TARGET"}
    assert feature_columns == set(manifest["groups"])
    assert set(manifest["groups"].values()) == {"application", "bureau", "installments", "bureau_balance", "previous_application", "pos_cash", "credit_card"}


def test_gender_is_not_a_feature(features, manifest):
    assert "CODE_GENDER" not in features.columns
    assert "CODE_GENDER" not in manifest["groups"]


def test_sentinel_removed(features):
    assert features.filter(pl.col("DAYS_EMPLOYED") == 365243).height == 0
    assert features["DAYS_EMPLOYED_ANOMALY"].sum() > 0


def test_filtered_views_have_no_future_offsets():
    con = connect()
    prepare_views(con)
    checks = {
        "installments_valid": "greatest(max(DAYS_INSTALMENT), max(DAYS_ENTRY_PAYMENT))",
        "bureau_valid": "greatest(max(DAYS_CREDIT), max(DAYS_ENDDATE_FACT), max(DAYS_CREDIT_UPDATE))",
        "bureau_balance_valid": "max(MONTHS_BALANCE)",
        "previous_application_valid": "greatest(max(DAYS_DECISION), max(DAYS_FIRST_DUE), max(DAYS_LAST_DUE), max(DAYS_TERMINATION))",
        "pos_cash_valid": "max(MONTHS_BALANCE)",
        "credit_card_valid": "max(MONTHS_BALANCE)",
    }
    for view, expression in checks.items():
        assert con.execute(f"SELECT {expression} FROM {view}").fetchone()[0] <= 0
