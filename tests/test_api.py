import pytest
from fastapi.testclient import TestClient

from credit_risk import api
from credit_risk.score import UnknownApplicantError


def fake_score(applicant_id: int) -> dict:
    if applicant_id == 1:
        raise UnknownApplicantError(applicant_id)
    return {
        "sk_id_curr": applicant_id,
        "model": "lightgbm",
        "in_holdout": True,
        "probability_of_default": 0.05,
        "raw_model_probability": 0.04,
        "scorecard_points": 580.0,
        "risk_band": "C",
        "reason_codes": [{"feature": "EXT_SOURCE_MEAN", "value": 0.31, "direction": "increases risk", "contribution": 0.4}],
    }


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "score_applicant", fake_score)
    monkeypatch.setattr(api, "scoring_artifacts", lambda: {"model_name": "lightgbm", "applicant_count": 3})
    with TestClient(api.app) as test_client:
        yield test_client


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model": "lightgbm", "applicants": 3}


def test_score_one(client):
    body = client.get("/applicants/100002/score").json()
    assert body["sk_id_curr"] == 100002
    assert body["risk_band"] == "C"
    assert body["reason_codes"][0]["feature"] == "EXT_SOURCE_MEAN"


def test_unknown_applicant_is_404(client):
    response = client.get("/applicants/1/score")
    assert response.status_code == 404


def test_batch_splits_known_and_unknown(client):
    body = client.post("/score", json={"sk_id_curr": [5, 1, 6]}).json()
    assert [s["sk_id_curr"] for s in body["scores"]] == [5, 6]
    assert body["unknown"] == [1]


def test_batch_rejects_empty_list(client):
    assert client.post("/score", json={"sk_id_curr": []}).status_code == 422
