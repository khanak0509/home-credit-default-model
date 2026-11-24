from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from credit_risk.artifacts import markdown_table, read_json, signed, write_json
from credit_risk.bundles import MODEL_LABELS, MODEL_ORDER, load_bundle, predict_raw
from credit_risk.config import get_paths, load_config
from credit_risk.dataset import load_dataset
from credit_risk.metrics import (
    approval_at_bad_rate,
    calibration_table,
    classification_metrics,
    interval,
    ks_curve,
    paired_bootstrap,
)
from credit_risk.plotting import INK_SECONDARY, MODEL_COLORS, NEUTRAL, apply_style, save

RANKING_METRICS = ("roc_auc", "pr_auc", "ks", "gini")
CALIBRATION_METRICS = ("brier", "log_loss", "ece", "mean_predicted")


def holdout_scores() -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, str]]:
    data = load_dataset()
    raw, calibrated, methods = {}, {}, {}
    for name in MODEL_ORDER:
        bundle = load_bundle(name)
        raw[name] = predict_raw(bundle, data.X_holdout)
        calibrated[name] = bundle["calibrator"].predict(raw[name])
        methods[name] = bundle["calibrator"].method
    return data.ids_holdout, data.y_holdout, raw, calibrated, methods


def model_metrics(y: np.ndarray, raw: np.ndarray, calibrated: np.ndarray, ece_bins: int) -> dict[str, Any]:
    ranking = classification_metrics(y, raw, ece_bins)
    after = classification_metrics(y, calibrated, ece_bins)
    combined = {key: ranking[key] for key in RANKING_METRICS}
    combined.update({key: after[key] for key in CALIBRATION_METRICS})
    combined["observed_rate"] = after["observed_rate"]
    return {"metrics": combined, "uncalibrated": {key: ranking[key] for key in CALIBRATION_METRICS}}


def bootstrap_summary(y: np.ndarray, raw: dict[str, np.ndarray], n_resamples: int, seed: int) -> dict[str, Any]:
    draws = paired_bootstrap(y, raw, n_resamples, seed)
    baseline = draws["legacy"]["roc_auc"]
    summary = {}
    for name in raw:
        auc_difference = draws[name]["roc_auc"] - baseline
        summary[name] = {
            "roc_auc_ci": interval(draws[name]["roc_auc"]),
            "ks_ci": interval(draws[name]["ks"]),
            "gini_ci": [2 * value - 1 for value in interval(draws[name]["roc_auc"])],
            "auc_diff_vs_legacy": float(np.mean(auc_difference)),
            "auc_diff_vs_legacy_ci": interval(auc_difference),
            "auc_diff_share_below_zero": float(np.mean(auc_difference <= 0)),
        }
    return summary


def cost_analysis(y: np.ndarray, raw: np.ndarray, calibrated: np.ndarray, margin: float, loss_given_default: float) -> dict[str, Any]:
    order = np.argsort(raw, kind="mergesort")
    bads = np.cumsum(y[order])
    goods = np.arange(1, y.size + 1) - bads
    profit = (margin * goods - loss_given_default * bads) / y.size
    best = int(np.argmax(profit))
    break_even = margin / (margin + loss_given_default)
    approved = calibrated < break_even
    policy_profit = (margin * np.sum(approved & (y == 0)) - loss_given_default * np.sum(approved & (y == 1))) / y.size
    return {
        "cost_ratio": loss_given_default / margin,
        "loss_given_default": loss_given_default,
        "best_profit_per_applicant": float(profit[best]),
        "best_approval_rate": float((best + 1) / y.size),
        "break_even_pd": break_even,
        "policy_profit_per_applicant": float(policy_profit),
        "policy_approval_rate": float(approved.mean()),
        "approve_all_profit_per_applicant": float(profit[-1]),
        "curve": profit,
    }


def plot_roc(y: np.ndarray, raw: dict[str, np.ndarray], metrics: dict[str, Any], path) -> None:
    figure, axis = plt.subplots(figsize=(5.6, 5))
    for name, scores in raw.items():
        fpr, tpr, _ = roc_curve(y, scores)
        axis.plot(fpr, tpr, color=MODEL_COLORS[name], label=f"{MODEL_LABELS[name]}  {metrics[name]['metrics']['roc_auc']:.3f}")
    axis.plot([0, 1], [0, 1], color=NEUTRAL, linewidth=1, linestyle="--")
    axis.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve, holdout")
    axis.legend(loc="lower right", title="model, AUC")
    save(figure, path)


def plot_pr(y: np.ndarray, raw: dict[str, np.ndarray], metrics: dict[str, Any], path) -> None:
    figure, axis = plt.subplots(figsize=(5.6, 5))
    for name, scores in raw.items():
        precision, recall, _ = precision_recall_curve(y, scores)
        axis.plot(recall, precision, color=MODEL_COLORS[name], label=f"{MODEL_LABELS[name]}  {metrics[name]['metrics']['pr_auc']:.3f}")
    axis.axhline(y.mean(), color=NEUTRAL, linewidth=1, linestyle="--")
    axis.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve, holdout")
    axis.legend(loc="upper right", title="model, PR-AUC")
    save(figure, path)


def plot_ks(y: np.ndarray, raw: dict[str, np.ndarray], names: list[str], path) -> None:
    figure, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 4.2), sharey=True)
    for axis, name in zip(np.atleast_1d(axes), names):
        population, tpr, fpr = ks_curve(y, raw[name])
        gap = tpr - fpr
        peak = int(np.argmax(gap))
        axis.plot(population, tpr, color=MODEL_COLORS[name], label="cumulative share of defaults")
        axis.plot(population, fpr, color=NEUTRAL, label="cumulative share of non-defaults")
        axis.vlines(population[peak], fpr[peak], tpr[peak], color=INK_SECONDARY, linewidth=1.2)
        axis.annotate(f"KS {gap[peak]:.3f}", (population[peak], (tpr[peak] + fpr[peak]) / 2), xytext=(6, 0),
                      textcoords="offset points", fontsize=8, color=INK_SECONDARY, va="center")
        axis.set(xlabel="Share of applicants, riskiest first", title=MODEL_LABELS[name])
        axis.legend(loc="lower right")
    np.atleast_1d(axes)[0].set_ylabel("Cumulative share")
    save(figure, path)


def plot_calibration(y: np.ndarray, raw: dict[str, np.ndarray], calibrated: dict[str, np.ndarray], n_bins: int, path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharex=True, sharey=True)
    for axis, scores, title in ((axes[0], raw, "Before calibration"), (axes[1], calibrated, "After calibration")):
        upper = 0.0
        for name, values in scores.items():
            predicted, observed = calibration_table(y, values, n_bins)
            upper = max(upper, predicted.max(), observed.max())
            axis.plot(predicted, observed, color=MODEL_COLORS[name], marker="o", markersize=4, label=MODEL_LABELS[name])
        axis.plot([0, upper], [0, upper], color=NEUTRAL, linewidth=1, linestyle="--")
        axis.set(xlabel="Mean predicted probability (decile)", title=f"{title}, holdout")
    axes[0].set_ylabel("Observed default rate")
    axes[1].legend(loc="upper left")
    save(figure, path)


def plot_cost(results: dict[str, dict[str, Any]], y_size: int, path) -> None:
    figure, axis = plt.subplots(figsize=(6.4, 4.6))
    approval = np.arange(1, y_size + 1) / y_size
    for name, result in results.items():
        axis.plot(approval, result["curve"], color=MODEL_COLORS[name], label=MODEL_LABELS[name], linewidth=1.6)
    first = next(iter(results.values()))
    axis.axhline(0, color=NEUTRAL, linewidth=1)
    axis.set(xlabel="Approval rate (safest applicants first)", ylabel="Profit per applicant (share of loan amount)",
             title=f"Illustrative profit curve, LGD {first['loss_given_default']:.2f}")
    axis.legend(loc="upper left")
    save(figure, path)


def percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def render_report(report: dict[str, Any], margin: float) -> str:
    models = report["models"]
    headers = ["model", "ROC-AUC [95% CI]", "PR-AUC", "KS [95% CI]", "Gini", "Brier", "log loss", "ECE", "calibration"]
    rows = []
    for name, m in models.items():
        metric = m["metrics"]
        rows.append([
            m["label"], f"{metric['roc_auc']:.4f} [{m['roc_auc_ci'][0]:.4f}, {m['roc_auc_ci'][1]:.4f}]", f"{metric['pr_auc']:.4f}",
            f"{metric['ks']:.4f} [{m['ks_ci'][0]:.4f}, {m['ks_ci'][1]:.4f}]", f"{metric['gini']:.4f}", f"{metric['brier']:.5f}",
            f"{metric['log_loss']:.4f}", f"{metric['ece']:.4f}", m["calibration_method"],
        ])
    diff_rows = [[m["label"], signed(m["auc_diff_vs_legacy"]), f"[{signed(m['auc_diff_vs_legacy_ci'][0])}, {signed(m['auc_diff_vs_legacy_ci'][1])}]",
                  f"{m['auc_diff_share_below_zero']:.3f}"] for name, m in models.items() if name != "legacy"]
    calibration_rows = [[m["label"], m["calibration_method"], f"{m['uncalibrated']['brier']:.5f}", f"{m['metrics']['brier']:.5f}",
                         f"{m['uncalibrated']['ece']:.4f}", f"{m['metrics']['ece']:.4f}", f"{m['metrics']['mean_predicted']:.4f}"] for m in models.values()]
    rates = report["target_bad_rates"]
    approval_rows = [[m["label"], *[percent(m["approval"][str(rate)]["approval_rate"]) for rate in rates]] for m in models.values()]
    cost_sections = []
    for by_model in report["cost"].values():
        first = next(iter(by_model.values()))
        cost_sections.append(
            f"Loss given default {first['loss_given_default']:.2f}, margin {margin:.2f}, cost ratio {first['cost_ratio']:.1f}, break-even PD {first['break_even_pd']:.4f}\n\n"
            + markdown_table(
                ["model", "best profit per applicant", "approval at best cut-off", "profit with break-even PD policy", "approval under policy"],
                [[MODEL_LABELS[name], f"{r['best_profit_per_applicant']:.5f}", percent(r["best_approval_rate"]),
                  f"{r['policy_profit_per_applicant']:.5f}", percent(r["policy_approval_rate"])] for name, r in by_model.items()],
            )
        )
    return "\n\n".join([
        "# Model comparison",
        f"Holdout set: {report['holdout_rows']:,} applications, observed default rate {report['observed_rate']:.4f}. "
        f"The holdout was split off once before any modeling and never used for tuning, feature selection or calibration fitting. "
        f"Confidence intervals come from {report['n_bootstrap']} paired bootstrap resamples (same indices for every model).",
        "Ranking metrics (ROC-AUC, PR-AUC, KS, Gini) use the raw model score. Probability metrics (Brier, log loss, ECE) use the calibrated probability.",
        "## Holdout metrics",
        markdown_table(headers, rows),
        "## AUC difference versus the legacy baseline",
        markdown_table(["model", "AUC difference", "95% CI", "share of resamples at or below zero"], diff_rows),
        "## Calibration",
        f"Calibrator chosen per model by cross-validated Brier score on out-of-fold training predictions. ECE uses {report['ece_bins']} equal-frequency bins.",
        markdown_table(["model", "method", "Brier before", "Brier after", "ECE before", "ECE after", "mean predicted after"], calibration_rows),
        "![calibration](figures/calibration.png)",
        "## Approval rate at a fixed bad rate",
        "Applicants are approved from lowest to highest predicted risk until the bad rate among approved applicants reaches the target.",
        markdown_table(["model", *[f"bad rate {percent(rate)}" for rate in rates]], approval_rows),
        "## Illustrative cost analysis",
        "**These figures use illustrative assumptions, not data.** Profit per approved good loan is the margin, loss per approved bad loan is the loss given default, "
        "both as a share of the loan amount. The break-even PD policy approves applicants whose calibrated PD is below margin / (margin + LGD).",
        *cost_sections,
        "![cost curve](figures/cost_curve.png)",
        "## Curves",
        "![roc](figures/roc.png)\n\n![pr](figures/pr.png)\n\n![ks](figures/ks.png)",
    ]) + "\n"


def main() -> None:
    config = load_config()
    paths = get_paths()
    evaluation = config["evaluation"]
    training = read_json(paths.reports / "training.json", "train")
    ids, y, raw, calibrated, methods = holdout_scores()
    pd.DataFrame({"SK_ID_CURR": ids, "TARGET": y, **{f"raw_{k}": v for k, v in raw.items()}, **{f"calibrated_{k}": v for k, v in calibrated.items()}}).to_parquet(
        paths.processed / "holdout_predictions.parquet", index=False)
    bootstrap = bootstrap_summary(y, raw, evaluation["n_bootstrap"], config["seed"])
    models = {}
    for name in raw:
        models[name] = {"label": MODEL_LABELS[name], "calibration_method": methods[name],
                        **model_metrics(y, raw[name], calibrated[name], evaluation["ece_bins"]), **bootstrap[name]}
        models[name]["approval"] = {str(rate): approval_at_bad_rate(y, raw[name], rate) for rate in evaluation["target_bad_rates"]}
    margin = evaluation["cost"]["margin_per_good_loan"]
    cost = {}
    for lgd in evaluation["cost"]["loss_given_default"]:
        cost[f"lgd_{lgd}"] = {name: cost_analysis(y, raw[name], calibrated[name], margin, lgd) for name in raw}
    apply_style()
    figures = paths.figures
    plot_roc(y, raw, models, figures / "roc.png")
    plot_pr(y, raw, models, figures / "pr.png")
    plot_ks(y, raw, ["legacy", training["best_tree_model"]], figures / "ks.png")
    plot_calibration(y, raw, calibrated, evaluation["ece_bins"], figures / "calibration.png")
    middle = list(cost)[len(cost) // 2]
    plot_cost(cost[middle], y.size, figures / "cost_curve.png")
    for by_model in cost.values():
        for result in by_model.values():
            result.pop("curve")
    report = {
        "holdout_rows": int(y.size), "observed_rate": float(y.mean()), "n_bootstrap": evaluation["n_bootstrap"],
        "ece_bins": evaluation["ece_bins"], "target_bad_rates": evaluation["target_bad_rates"],
        "best_tree_model": training["best_tree_model"], "models": models, "cost": cost,
        "cost_assumptions": {"margin_per_good_loan": margin, "loss_given_default": evaluation["cost"]["loss_given_default"], "illustrative": True},
    }
    write_json(paths.reports / "metrics.json", report)
    (paths.reports / "model_comparison.md").write_text(render_report(report, margin))
    for name, m in models.items():
        metric = m["metrics"]
        print(f"{name:<20}auc {metric['roc_auc']:.4f} [{m['roc_auc_ci'][0]:.4f}, {m['roc_auc_ci'][1]:.4f}]  ks {metric['ks']:.4f}  "
              f"brier {metric['brier']:.5f}  ece {metric['ece']:.4f}  auc vs legacy {m['auc_diff_vs_legacy']:+.4f}")


if __name__ == "__main__":
    main()
