from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from credit_risk.score import UnknownApplicantError, score_applicant, scoring_artifacts


class ReasonCode(BaseModel):
    feature: str
    value: float | str | None
    direction: str
    contribution: float


class Score(BaseModel):
    sk_id_curr: int
    model: str
    in_holdout: bool
    probability_of_default: float
    raw_model_probability: float
    scorecard_points: float
    risk_band: str
    reason_codes: list[ReasonCode]


class BatchRequest(BaseModel):
    sk_id_curr: list[int] = Field(min_length=1, max_length=500)


class BatchResponse(BaseModel):
    scores: list[Score]
    unknown: list[int]


class Health(BaseModel):
    status: str
    model: str
    applicants: int


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    scoring_artifacts()
    yield


app = FastAPI(title="credit-risk-lab", description="Probability of default, risk band and reason codes for Home Credit applicants", lifespan=lifespan)


@app.get("/health", response_model=Health)
def health() -> Health:
    artifacts = scoring_artifacts()
    return Health(status="ok", model=artifacts["model_name"], applicants=artifacts["applicant_count"])


@app.get("/applicants/{sk_id_curr}/score", response_model=Score)
def score_one(sk_id_curr: int) -> Score:
    try:
        return Score(**score_applicant(sk_id_curr))
    except UnknownApplicantError:
        raise HTTPException(status_code=404, detail=f"unknown SK_ID_CURR {sk_id_curr}") from None


@app.post("/score", response_model=BatchResponse)
def score_batch(request: BatchRequest) -> BatchResponse:
    scores, unknown = [], []
    for applicant_id in request.sk_id_curr:
        try:
            scores.append(Score(**score_applicant(applicant_id)))
        except UnknownApplicantError:
            unknown.append(applicant_id)
    return BatchResponse(scores=scores, unknown=unknown)
