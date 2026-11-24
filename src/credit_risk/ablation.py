from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from credit_risk.artifacts import markdown_table, read_json, signed, write_json
from credit_risk.boosting import cross_validate
from credit_risk.config import get_paths, load_config
from credit_risk.dataset import load_dataset, load_manifest
from credit_risk.metrics import fast_auc, interval, ks_statistic, paired_bootstrap
from credit_risk.plotting import ACCENT, INK_SECONDARY, NEUTRAL, apply_style, save

GROUP_LABELS = {
    "application": "application",
    "bureau": "bureau",
    "bureau_balance": "bureau balance",
    "previous_application": "previous applications",
    "installments": "installments",
    "pos_cash": "POS and cash loan balances",
    "credit_card": "credit card balances",
}


def ablation_steps(manifest: dict[str, Any], order: list[str]) -> list[dict[str, Any]]:
    if sorted(order) != sorted(manifest["group_order"]):
        raise ValueError("ablation_order must list every feature group exactly once")
    steps = [{"key": "application_no_external", "label": "application without external scores", "groups": ["application"],
              "drop": manifest["external_score"], "kind": "external"}]
    for i, group in enumerate(order):
        label = GROUP_LABELS[group] if i == 0 else f"+ {GROUP_LABELS[group]}"
        steps.append({"key": "_".join(order[: i + 1]), "label": label, "groups": order[: i + 1], "drop": [], "kind": "cumulative"})
    return steps


def run_step(step: dict[str, Any], kind: str, params: dict[str, Any], config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    tuning = config["tuning"]
    data = load_dataset(step["groups"], step["drop"])
    oof, ensemble, fold_aucs = cross_validate(kind, params, data, tuning["max_boost_rounds"], tuning["early_stopping_rounds"])
    holdout = ensemble.predict_proba(data.X_holdout)
    info = {"n_features": len(data.features), "cv_auc_mean": float(np.mean(fold_aucs)), "oof_auc": fast_auc(data.y_train, oof)}
    return data.y_holdout, holdout, info


def plot_ablation(results: list[dict[str, Any]], path) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 0.62 * len(results) + 1.2))
    labels = [r["label"] for r in results]
    aucs = np.array([r["roc_auc"] for r in results])
    lower = aucs - np.array([r["roc_auc_ci"][0] for r in results])
    upper = np.array([r["roc_auc_ci"][1] for r in results]) - aucs
    colors = [NEUTRAL if r["kind"] == "external" else ACCENT for r in results]
    positions = np.arange(len(results))[::-1]
    axis.barh(positions, aucs, color=colors, height=0.6, xerr=[lower, upper], error_kw={"ecolor": INK_SECONDARY, "elinewidth": 1, "capsize": 3})
    for position, value in zip(positions, aucs):
        axis.text(value + upper.max() + 0.002, position, f"{value:.4f}", va="center", fontsize=8, color=INK_SECONDARY)
    axis.set_yticks(positions, labels)
    low = min(r["roc_auc_ci"][0] for r in results)
    axis.set_xlim(max(0.5, low - 0.02), max(r["roc_auc_ci"][1] for r in results) + 0.02)
    axis.set(xlabel="Holdout ROC-AUC with 95% bootstrap CI", title="Feature group ablation")
    axis.grid(axis="y", visible=False)
    save(figure, path)


def describe(results: list[dict[str, Any]]) -> list[str]:
    cumulative = [r for r in results if r["kind"] == "cumulative"]
    ranked = sorted(cumulative[1:], key=lambda r: -r["delta_auc"])
    lines = []
    for r in ranked:
        significant = "excludes zero" if r["delta_auc_ci"][0] > 0 or r["delta_auc_ci"][1] < 0 else "includes zero"
        lines.append(f"- {r['label'].lstrip('+ ')}: AUC {signed(r['delta_auc'])} over the previous step, 95% CI [{signed(r['delta_auc_ci'][0])}, {signed(r['delta_auc_ci'][1])}] ({significant}).")
    external = next(r for r in results if r["kind"] == "external")
    with_external = cumulative[0]
    lines.append(f"- external scores: application with external scores {with_external['roc_auc']:.4f} versus without {external['roc_auc']:.4f}, "
                 f"difference {signed(with_external['roc_auc'] - external['roc_auc'])}, 95% CI [{signed(external['external_delta_ci'][0])}, {signed(external['external_delta_ci'][1])}].")
    return lines


def main() -> None:
    config = load_config()
    paths = get_paths()
    training = read_json(paths.reports / "training.json", "train")
    kind = training["best_tree_model"]
    params = training["models"][kind]["params"]
    manifest = load_manifest()
    steps = ablation_steps(manifest, config["ablation_order"])
    holdout_scores, infos, y = {}, {}, None
    for step in steps:
        y, holdout_scores[step["key"]], infos[step["key"]] = run_step(step, kind, params, config)
        print(f"{step['label']:<40}features {infos[step['key']]['n_features']:>4}  cv auc {infos[step['key']]['cv_auc_mean']:.4f}")
    draws = paired_bootstrap(y, holdout_scores, config["evaluation"]["n_bootstrap"], config["seed"])
    results = []
    previous = None
    for step in steps:
        key = step["key"]
        result = {**step, **infos[key], "roc_auc": fast_auc(y, holdout_scores[key]), "ks": ks_statistic(y, holdout_scores[key]),
                  "roc_auc_ci": interval(draws[key]["roc_auc"]), "ks_ci": interval(draws[key]["ks"])}
        if step["kind"] == "cumulative" and previous is not None:
            delta = draws[key]["roc_auc"] - draws[previous]["roc_auc"]
            result["delta_auc"] = result["roc_auc"] - fast_auc(y, holdout_scores[previous])
            result["delta_auc_ci"] = interval(delta)
        if step["kind"] == "external":
            first_cumulative = next(s["key"] for s in steps if s["kind"] == "cumulative")
            result["external_delta_ci"] = interval(draws[first_cumulative]["roc_auc"] - draws[key]["roc_auc"])
        if step["kind"] == "cumulative":
            previous = key
        results.append(result)
    apply_style()
    plot_ablation(results, paths.figures / "ablation.png")
    write_json(paths.reports / "ablation.json", {"model": kind, "n_bootstrap": config["evaluation"]["n_bootstrap"], "steps": results})
    rows = [[r["label"], r["n_features"], f"{r['cv_auc_mean']:.4f}", f"{r['roc_auc']:.4f} [{r['roc_auc_ci'][0]:.4f}, {r['roc_auc_ci'][1]:.4f}]",
             f"{r['ks']:.4f} [{r['ks_ci'][0]:.4f}, {r['ks_ci'][1]:.4f}]",
             f"{signed(r['delta_auc'])} [{signed(r['delta_auc_ci'][0])}, {signed(r['delta_auc_ci'][1])}]" if "delta_auc" in r else ""] for r in results]
    text = "\n\n".join([
        "# Feature group ablation",
        f"Model: {training['models'][kind]['label']} with the tuned parameters, retrained with 5-fold CV on each feature set; "
        f"holdout predictions average the fold models. Intervals come from {config['evaluation']['n_bootstrap']} paired bootstrap resamples of the holdout.",
        markdown_table(["feature set", "features", "CV AUC", "holdout AUC [95% CI]", "holdout KS [95% CI]", "AUC gain over previous step [95% CI]"], rows),
        "![ablation](figures/ablation.png)",
        "## Contribution by data source, largest first",
        "\n".join(describe(results)),
    ])
    (paths.reports / "ablation.md").write_text(text + "\n")


if __name__ == "__main__":
    main()
