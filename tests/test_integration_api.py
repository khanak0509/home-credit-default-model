import polars as pl
from fastapi.testclient import TestClient

from credit_risk.api import app


def test_api_scores_a_real_applicant(paths):
    applicant_id = pl.read_parquet(paths.processed / "split.parquet")["SK_ID_CURR"].min()
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert health["status"] == "ok"
        body = client.get(f"/applicants/{applicant_id}/score").json()
        assert body["sk_id_curr"] == applicant_id
        assert 0.0 <= body["probability_of_default"] <= 1.0
        assert client.get("/applicants/-1/score").status_code == 404
