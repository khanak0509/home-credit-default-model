import argparse
import hashlib
import json
import re
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from credit_risk.artifacts import markdown_table, write_json
from credit_risk.boosting import cross_validate, lightgbm_base_params, tune, xgboost_base_params
from credit_risk.bundles import MODEL_LABELS, save_bundle
from credit_risk.calibration import choose_calibrator
from credit_risk.config import get_paths, load_config
from credit_risk.dataset import Dataset, load_dataset
from credit_risk.metrics import classification_metrics, fast_auc
from credit_risk.scorecard import ScorecardModel
from credit_risk.woe import WoEBinner


def balanced_weights(y: np.ndarray) -> np.ndarray:
    positive_share = y.mean()
    return np.where(y == 1, 0.5 / positive_share, 0.5 / (1 - positive_share))


def fold_aucs(y: np.ndarray, oof: np.ndarray, folds: np.ndarray) -> list[float]:
    return [fast_auc(y[folds == k], oof[folds == k]) for k in np.unique(folds)]


def train_legacy(data: Dataset, features: list[str], seed: int) -> tuple[np.ndarray, Any]:
    def build() -> Any:
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))

    X = data.X_train[features]
    oof = np.zeros(data.y_train.size)
    for train_index, valid_index in data.fold_indices():
        model = build().fit(X.iloc[train_index], data.y_train[train_index])
        oof[valid_index] = model.predict_proba(X.iloc[valid_index])[:, 1]
    return oof, build().fit(X, data.y_train)


def scorecard_factory(config: dict[str, Any], C: float) -> ScorecardModel:
    settings = config["scorecard"]
    binner = WoEBinner(
        n_prebins=settings["n_prebins"],
        min_bin_share=settings["min_bin_share"],
        iv_threshold=settings["iv_threshold"],
        max_features=settings["max_features"],
        correlation_threshold=settings["correlation_threshold"],
    )
    return ScorecardModel(binner, C, config["seed"], settings["base_score"], settings["base_odds"], settings["points_to_double_odds"])


def train_scorecard(data: Dataset, config: dict[str, Any]) -> tuple[np.ndarray, ScorecardModel, dict[str, Any]]:
    grid = config["scorecard"]["c_grid"]
    oof_by_c = {C: np.zeros(data.y_train.size) for C in grid}
    selected_per_fold = []
    for train_index, valid_index in data.fold_indices():
        X_fit, y_fit = data.X_train.iloc[train_index], data.y_train[train_index]
        model = scorecard_factory(config, grid[0])
        model.binner.fit(X_fit, y_fit)
        encoded_fit = model.binner.transform(X_fit)
        encoded_valid = model.binner.transform(data.X_train.iloc[valid_index])
        selected_per_fold.append(len(model.binner.selected_))
        for C in grid:
            model.C = C
            model.fit_regression(encoded_fit, y_fit)
            oof_by_c[C][valid_index] = model.regression.predict_proba(encoded_valid.to_numpy())[:, 1]
    cv_auc = {C: float(np.mean(fold_aucs(data.y_train, oof_by_c[C], data.folds))) for C in grid}
    best_C = max(grid, key=lambda C: cv_auc[C])
    final = scorecard_factory(config, best_C).fit(data.X_train, data.y_train)
    summary = {"cv_auc_by_C": cv_auc, "best_C": best_C, "selected_features_per_fold": selected_per_fold,
               "final_features": final.features_, "information_values": final.binner.information_values_}
    return oof_by_c[best_C], final, summary


def feature_signature(features: list[str], extra: dict[str, Any]) -> str:
    payload = json.dumps({"features": features, **extra}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def tuned_params(kind: str, data: Dataset, config: dict[str, Any], tuning: dict[str, Any], retune: bool) -> dict[str, Any]:
    paths = get_paths()
    base = lightgbm_base_params(config["seed"], tuning["n_jobs"]) if kind == "lightgbm" else xgboost_base_params(config["seed"], tuning["n_jobs"])
    signature = feature_signature(data.features, {"tuning": tuning, "seed": config["seed"], "base": base})
    cache = paths.models / f"tuning_{kind}.json"
    if cache.exists() and not retune:
        cached = json.loads(cache.read_text())
        if cached.get("signature") == signature:
            print(f"{kind} tuning loaded from cache")
            return cached
    result = tune(kind, data, base, tuning["n_trials"], tuning["max_boost_rounds"], tuning["early_stopping_rounds"], config["seed"])
    result["signature"] = signature
    write_json(cache, result)
    return result


def monotone_vector(features: list[str], config: dict[str, Any], categorical: list[str]) -> list[int]:
    decreasing = set(config["monotone"]["decreasing_risk"])
    patterns = [re.compile(p) for p in config["monotone"]["increasing_risk_patterns"]]
    vector = []
    for name in features:
        if name in categorical:
            vector.append(0)
        elif name in decreasing:
            vector.append(-1)
        elif any(p.search(name) for p in patterns):
            vector.append(1)
        else:
            vector.append(0)
    return vector


def imbalance_study(data: Dataset, params: dict[str, Any], baseline_oof: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    tuning = config["tuning"]
    ece_bins = config["evaluation"]["ece_bins"]
    ratio = float((data.y_train == 0).sum() / (data.y_train == 1).sum())
    variants = {"no_weighting": baseline_oof}
    variants["scale_pos_weight"], _, _ = cross_validate(
        "lightgbm", {**params, "scale_pos_weight": ratio}, data, tuning["max_boost_rounds"], tuning["early_stopping_rounds"])
    variants["balanced_class_weight"], _, _ = cross_validate(
        "lightgbm", params, data, tuning["max_boost_rounds"], tuning["early_stopping_rounds"], weight_fn=balanced_weights)
    results = {name: classification_metrics(data.y_train, oof, ece_bins) for name, oof in variants.items()}
    for name, oof in variants.items():
        calibrated_brier = choose_calibrator(oof, data.y_train, data.folds, config["calibration"]["methods"])[1]
        results[name]["brier_after_recalibration"] = min(v for k, v in calibrated_brier.items() if k != "uncalibrated")
    return {"scale_pos_weight": ratio, "variants": results}


def write_imbalance_report(study: dict[str, Any]) -> None:
    paths = get_paths()
    write_json(paths.reports / "imbalance.json", study)
    variants = study["variants"]
    rows = [[name, f"{m['roc_auc']:.4f}", f"{m['pr_auc']:.4f}", f"{m['brier']:.4f}", f"{m['log_loss']:.4f}", f"{m['ece']:.4f}",
             f"{m['mean_predicted']:.4f}", f"{m['observed_rate']:.4f}", f"{m['brier_after_recalibration']:.4f}"] for name, m in variants.items()]
    base = variants["no_weighting"]
    lines = [
        "# Class imbalance study",
        "LightGBM with the tuned parameters, 5-fold out-of-fold predictions on the training split. "
        f"scale_pos_weight = negatives / positives = {study['scale_pos_weight']:.3f}. Balanced class weights give each class half the total weight.",
        markdown_table(["variant", "ROC-AUC", "PR-AUC", "Brier", "log loss", "ECE", "mean predicted", "observed rate", "Brier after recalibration"], rows),
    ]
    for name in ("scale_pos_weight", "balanced_class_weight"):
        m = variants[name]
        lines.append(
            f"{name}: ROC-AUC change {m['roc_auc'] - base['roc_auc']:+.4f}, Brier {base['brier']:.4f} to {m['brier']:.4f}, "
            f"mean predicted probability {m['mean_predicted']:.4f} against an observed rate of {m['observed_rate']:.4f}."
        )
    (paths.reports / "imbalance.md").write_text("\n\n".join(lines) + "\n")


def write_scorecard_table(model: ScorecardModel, summary: dict[str, Any]) -> None:
    paths = get_paths()
    table = model.table()
    table.to_csv(paths.reports / "scorecard.csv", index=False, float_format="%.6g")
    iv = pd.Series(summary["information_values"]).sort_values(ascending=False)
    pd.DataFrame({"feature": iv.index, "iv": iv.values, "selected": iv.index.isin(model.features_)}).to_csv(
        paths.reports / "scorecard_iv.csv", index=False, float_format="%.6g")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retune", action="store_true")
    parser.add_argument("--n-trials", type=int)
    args = parser.parse_args()
    config = load_config()
    paths = get_paths()
    tuning = {**config["tuning"], "n_trials": args.n_trials or config["tuning"]["n_trials"]}
    calibration_methods = config["calibration"]["methods"]
    data = load_dataset()
    print(f"training rows {data.y_train.size:,}  features {len(data.features)}")
    oof: dict[str, np.ndarray] = {}
    summary: dict[str, Any] = {"models": {}}
    timings = {}

    def finish(name: str, estimator: Any, features: list[str], kind: str, extra: dict[str, Any]) -> None:
        calibrator, scores = choose_calibrator(oof[name], data.y_train, data.folds, calibration_methods)
        save_bundle(name, estimator, features, calibrator, scores, kind)
        aucs = fold_aucs(data.y_train, oof[name], data.folds)
        summary["models"][name] = {
            "label": MODEL_LABELS[name], "n_features": len(features), "cv_auc_mean": float(np.mean(aucs)),
            "cv_auc_std": float(np.std(aucs)), "cv_fold_auc": aucs, "oof_auc": fast_auc(data.y_train, oof[name]),
            "calibration_brier_cv": scores, "calibration_method": calibrator.method, **extra,
        }
        print(f"{name:<20}cv auc {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}  calibration {calibrator.method}")

    started = time.time()
    legacy_features = [f for f in config["legacy_features"] if f in data.features]
    oof["legacy"], legacy_model = train_legacy(data, legacy_features, config["seed"])
    finish("legacy", legacy_model, legacy_features, "sklearn", {})
    timings["legacy"] = time.time() - started

    started = time.time()
    oof["scorecard"], scorecard_model, scorecard_summary = train_scorecard(data, config)
    write_scorecard_table(scorecard_model, scorecard_summary)
    finish("scorecard", scorecard_model, data.features, "scorecard",
           {"best_C": scorecard_summary["best_C"], "cv_auc_by_C": scorecard_summary["cv_auc_by_C"],
            "selected_features": scorecard_summary["final_features"], "selected_features_per_fold": scorecard_summary["selected_features_per_fold"]})
    timings["scorecard"] = time.time() - started

    tuning_results = {}
    for kind in ("lightgbm", "xgboost"):
        started = time.time()
        tuning_results[kind] = tuned_params(kind, data, config, tuning, args.retune)
        params = tuning_results[kind]["best_params"]
        oof[kind], ensemble, _ = cross_validate(kind, params, data, tuning["max_boost_rounds"], tuning["early_stopping_rounds"])
        finish(kind, ensemble, data.features, "ensemble",
               {"params": params, "best_iterations": ensemble.best_iterations, "tuning_best_cv_auc": tuning_results[kind]["best_value"],
                "tuning_trials": tuning_results[kind]["n_trials"], "tuning_pruned": tuning_results[kind]["n_pruned"]})
        timings[kind] = time.time() - started

    started = time.time()
    constraints = monotone_vector(data.features, config, data.categorical)
    monotone_params = {**tuning_results["lightgbm"]["best_params"], "monotone_constraints": constraints,
                       "monotone_constraints_method": "advanced"}
    oof["lightgbm_monotone"], monotone_ensemble, _ = cross_validate(
        "lightgbm", monotone_params, data, tuning["max_boost_rounds"], tuning["early_stopping_rounds"])
    constrained = {name: direction for name, direction in zip(data.features, constraints) if direction != 0}
    finish("lightgbm_monotone", monotone_ensemble, data.features, "ensemble",
           {"params": {k: v for k, v in monotone_params.items() if k != "monotone_constraints"}, "constraints": constrained,
            "best_iterations": monotone_ensemble.best_iterations})
    summary["monotone_cv_auc_cost"] = summary["models"]["lightgbm"]["cv_auc_mean"] - summary["models"]["lightgbm_monotone"]["cv_auc_mean"]
    timings["lightgbm_monotone"] = time.time() - started

    started = time.time()
    study = imbalance_study(data, tuning_results["lightgbm"]["best_params"], oof["lightgbm"], config)
    write_imbalance_report(study)
    timings["imbalance"] = time.time() - started

    tree_models = ("lightgbm", "xgboost", "lightgbm_monotone")
    summary["best_tree_model"] = max(("lightgbm", "xgboost"), key=lambda n: summary["models"][n]["cv_auc_mean"])
    summary["best_lightgbm"] = max(("lightgbm", "lightgbm_monotone"), key=lambda n: summary["models"][n]["cv_auc_mean"])
    summary["selection_basis"] = "mean 5-fold CV AUC on the training split"
    summary["timings_seconds"] = timings
    summary["tree_models"] = list(tree_models)
    write_json(paths.reports / "training.json", summary)
    frame = pd.DataFrame({"SK_ID_CURR": data.ids_train, "TARGET": data.y_train, "fold": data.folds, **{f"oof_{k}": v for k, v in oof.items()}})
    frame.to_parquet(paths.processed / "oof_predictions.parquet", index=False)
    print(f"best tree model {summary['best_tree_model']}  monotone auc cost {summary['monotone_cv_auc_cost']:+.4f}")


if __name__ == "__main__":
    main()
