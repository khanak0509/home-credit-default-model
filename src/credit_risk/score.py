import argparse
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from credit_risk.artifacts import read_json, require, write_json
from credit_risk.bundles import load_bundle, predict_raw
from credit_risk.config import get_paths, load_config
from credit_risk.dataset import load_feature_frame, load_manifest, select_features, to_model_frame
from credit_risk.explain import fold_shap_values


class UnknownApplicantError(KeyError):
    pass


@lru_cache(maxsize=1)
def scoring_artifacts() -> dict[str, Any]:
    paths = get_paths()
    training = read_json(paths.reports / "training.json", "train")
    manifest = load_manifest()
    split = pl.read_parquet(require(paths.processed / "split.parquet", "features"))
    return {
        "model_name": training["best_lightgbm"],
        "model": load_bundle(training["best_lightgbm"]),
        "scorecard": load_bundle("scorecard"),
        "manifest": manifest,
        "columns": select_features(manifest),
        "holdout_ids": set(split.filter(pl.col("is_holdout"))["SK_ID_CURR"].to_list()),
        "applicant_count": split.height,
    }


def risk_band(probability: float, bands: list[list[Any]]) -> str:
    for name, lower, upper in bands:
        if lower <= probability < upper:
            return name
    raise ValueError(f"probability {probability} outside configured bands")


def applicant_frame(applicant_id: int) -> pd.DataFrame:
    artifacts = scoring_artifacts()
    frame = load_feature_frame(artifacts["columns"], ids=[applicant_id])
    if frame.height == 0:
        raise UnknownApplicantError(applicant_id)
    return to_model_frame(frame, artifacts["columns"], artifacts["manifest"])


def display_value(value: Any) -> Any:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (np.floating, float)):
        return round(float(value), 6)
    if isinstance(value, np.integer):
        return int(value)
    return str(value)


def reason_codes(X: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    ensemble = scoring_artifacts()["model"]["estimator"]
    contributions = np.mean(fold_shap_values(ensemble, X), axis=0)[0]
    order = np.argsort(-np.abs(contributions))[:limit]
    return [
        {
            "feature": ensemble.features[i],
            "value": display_value(X.iloc[0][ensemble.features[i]]),
            "direction": "increases risk" if contributions[i] > 0 else "decreases risk",
            "contribution": round(float(contributions[i]), 6),
        }
        for i in order
    ]


def score_applicant(applicant_id: int) -> dict[str, Any]:
    config = load_config()
    artifacts = scoring_artifacts()
    X = applicant_frame(applicant_id)
    model = artifacts["model"]
    raw = predict_raw(model, X)
    probability = float(model["calibrator"].predict(raw)[0])
    points = float(artifacts["scorecard"]["estimator"].points(X[artifacts["scorecard"]["features"]])[0])
    return {
        "sk_id_curr": int(applicant_id),
        "model": artifacts["model_name"],
        "in_holdout": applicant_id in artifacts["holdout_ids"],
        "probability_of_default": round(probability, 6),
        "raw_model_probability": round(float(raw[0]), 6),
        "scorecard_points": round(points, 1),
        "risk_band": risk_band(probability, config["scoring"]["risk_bands"]),
        "reason_codes": reason_codes(X, config["scoring"]["reason_codes"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score Home Credit applicants by SK_ID_CURR")
    parser.add_argument("ids", nargs="*", type=int)
    parser.add_argument("--example", action="store_true", help="score the first holdout applicant")
    parser.add_argument("--output", help="also write the result to this JSON file")
    args = parser.parse_args()
    ids = args.ids or ([min(scoring_artifacts()["holdout_ids"])] if args.example else [])
    if not ids:
        parser.error("give at least one SK_ID_CURR or --example")
    results = []
    for applicant_id in ids:
        try:
            results.append(score_applicant(applicant_id))
        except UnknownApplicantError:
            results.append({"sk_id_curr": applicant_id, "error": "unknown SK_ID_CURR"})
    payload = results[0] if len(results) == 1 else results
    if args.output:
        write_json(Path(args.output), payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
