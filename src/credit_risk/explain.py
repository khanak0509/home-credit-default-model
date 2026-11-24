import warnings
from itertools import combinations
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import shap
from scipy.stats import spearmanr

from credit_risk.artifacts import markdown_table, read_json, require, write_json
from credit_risk.boosting import FoldEnsemble
from credit_risk.bundles import load_bundle
from credit_risk.config import get_paths, load_config
from credit_risk.dataset import Dataset, load_dataset
from credit_risk.metrics import approval_at_bad_rate, fast_auc, population_stability_index
from credit_risk.plotting import ACCENT, INK_SECONDARY, NEUTRAL, apply_style, save

warnings.filterwarnings("ignore", message="LightGBM binary classifier with TreeExplainer")


def plot_frame(X: pd.DataFrame) -> pd.DataFrame:
    frame = X.copy()
    for column in frame.columns:
        if isinstance(frame[column].dtype, pd.CategoricalDtype):
            frame[column] = frame[column].cat.codes.replace(-1, np.nan).astype(float)
    return frame


def fold_shap_values(ensemble: FoldEnsemble, X: pd.DataFrame) -> list[np.ndarray]:
    values = []
    for model in ensemble.models:
        explanation = shap.TreeExplainer(model).shap_values(X[ensemble.features])
        values.append(explanation[1] if isinstance(explanation, list) else explanation)
    return values


def shap_stability(fold_values: list[np.ndarray], features: list[str], top_n: int = 20) -> dict[str, Any]:
    importance = np.array([np.abs(values).mean(axis=0) for values in fold_values])
    correlations = [spearmanr(importance[a], importance[b]).statistic for a, b in combinations(range(len(fold_values)), 2)]
    top_sets = [set(np.argsort(-row)[:top_n]) for row in importance]
    overlap = [len(top_sets[a] & top_sets[b]) / top_n for a, b in combinations(range(len(top_sets)), 2)]
    return {
        "pairwise_spearman": [float(c) for c in correlations],
        "mean_spearman": float(np.mean(correlations)),
        "min_spearman": float(np.min(correlations)),
        f"mean_top{top_n}_overlap": float(np.mean(overlap)),
        "fold_importance": {name: importance[:, i].tolist() for i, name in enumerate(features)},
    }


def permutation_importance(ensemble: FoldEnsemble, X: pd.DataFrame, y: np.ndarray, repeats: int, seed: int) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(seed)
    baseline = fast_auc(y, ensemble.predict_margin(X))
    results = {}
    for column in ensemble.features:
        drops = []
        for _ in range(repeats):
            shuffled = X.copy()
            shuffled[column] = shuffled[column].to_numpy()[rng.permutation(len(shuffled))]
            if isinstance(X[column].dtype, pd.CategoricalDtype):
                shuffled[column] = pd.Categorical(shuffled[column], categories=X[column].cat.categories)
            drops.append(baseline - fast_auc(y, ensemble.predict_margin(shuffled)))
        results[column] = {"mean": float(np.mean(drops)), "std": float(np.std(drops))}
    return results


def categorical_psi(expected: pd.Series, actual: pd.Series) -> float:
    expected_share = expected.astype(object).fillna("MISSING").value_counts(normalize=True)
    actual_share = actual.astype(object).fillna("MISSING").value_counts(normalize=True)
    categories = expected_share.index.union(actual_share.index)
    e = np.clip(expected_share.reindex(categories, fill_value=0).to_numpy(), 1e-6, None)
    a = np.clip(actual_share.reindex(categories, fill_value=0).to_numpy(), 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_label(value: float, thresholds: list[float]) -> str:
    if value < thresholds[0]:
        return "stable"
    if value < thresholds[1]:
        return "moderate shift"
    return "significant shift"


def plot_summary(values: np.ndarray, X: pd.DataFrame, path) -> None:
    shap.summary_plot(values, plot_frame(X), max_display=20, show=False, plot_size=(8, 7.5))
    figure = plt.gcf()
    figure.axes[0].set_title("SHAP summary, holdout sample", loc="left")
    save(figure, path)


def plot_dependence(feature: str, values: np.ndarray, X: pd.DataFrame, features: list[str], path) -> None:
    figure, axis = plt.subplots(figsize=(5.4, 4.2))
    index = features.index(feature)
    x = plot_frame(X[[feature]])[feature].to_numpy()
    present = ~np.isnan(x)
    axis.scatter(x[present], values[present, index], s=6, alpha=0.35, color=ACCENT, linewidths=0)
    axis.axhline(0, color=NEUTRAL, linewidth=1)
    missing_share = 1 - present.mean()
    axis.set(xlabel=feature, ylabel="SHAP value (log-odds of default)", title=f"Dependence: {feature}")
    if missing_share > 0:
        axis.annotate(f"{100 * missing_share:.1f}% missing, mean SHAP {values[~present, index].mean():+.3f}",
                      (0.02, 0.96), xycoords="axes fraction", fontsize=8, color=INK_SECONDARY, va="top")
    save(figure, path)


def plot_importance_comparison(table: pd.DataFrame, path) -> None:
    top = table.nsmallest(20, "shap_rank").iloc[::-1]
    figure, axes = plt.subplots(1, 2, figsize=(10, 6.4), sharey=True)
    positions = np.arange(len(top))
    axes[0].barh(positions, top["mean_abs_shap"], color=ACCENT, height=0.6)
    axes[0].set(title="Mean |SHAP|", xlabel="log-odds")
    axes[1].barh(positions, top["permutation_auc_drop"], color=NEUTRAL, height=0.6)
    axes[1].set(title="Permutation importance", xlabel="holdout AUC drop")
    axes[0].set_yticks(positions, top.index)
    for axis in axes:
        axis.grid(axis="y", visible=False)
    save(figure, path)


def fairness_table(groups: pd.Series, y: np.ndarray, raw: np.ndarray, calibrated: np.ndarray, threshold: float,
                   order: list[str] | None = None) -> list[dict[str, Any]]:
    rows = []
    approved = raw <= threshold
    present = set(groups.dropna().unique().tolist())
    for value in [v for v in order if v in present] if order else sorted(present):
        mask = (groups == value).to_numpy()
        has_both = np.unique(y[mask]).size == 2
        rows.append({
            "group": str(value),
            "count": int(mask.sum()),
            "observed_default_rate": float(y[mask].mean()),
            "mean_predicted_pd": float(calibrated[mask].mean()),
            "roc_auc": fast_auc(y[mask], raw[mask]) if has_both else None,
            "approval_rate": float(approved[mask].mean()),
            "bad_rate_among_approved": float(y[mask & approved].mean()) if (mask & approved).any() else None,
        })
    return rows


def age_band_labels(edges: list[int]) -> list[str]:
    labels = [f"{edges[i]}-{edges[i + 1]}" if i < len(edges) - 2 else f"{edges[i]}+" for i in range(len(edges) - 1)]
    labels[0] = f"under {edges[1]}"
    return labels


def age_band(age: pd.Series, edges: list[int]) -> pd.Series:
    return pd.cut(age, bins=edges, labels=age_band_labels(edges), right=False).astype(str)


def run_fairness(data: Dataset, name: str, config: dict[str, Any]) -> dict[str, Any]:
    paths = get_paths()
    predictions = pl.read_parquet(require(paths.processed / "holdout_predictions.parquet", "evaluate"))
    audit = pl.read_parquet(require(paths.processed / "audit_attributes.parquet", "features"))
    joined = predictions.join(audit.drop("TARGET"), on="SK_ID_CURR", how="left").to_pandas()
    y = joined["TARGET"].to_numpy()
    raw = joined[f"raw_{name}"].to_numpy()
    calibrated = joined[f"calibrated_{name}"].to_numpy()
    target = config["fairness"]["operating_bad_rate"]
    operating = approval_at_bad_rate(y, raw, target)
    return {
        "model": name,
        "operating_bad_rate": target,
        "operating_point": operating,
        "gender": fairness_table(joined["CODE_GENDER"], y, raw, calibrated, operating["threshold"]),
        "gender_missing": int(joined["CODE_GENDER"].isna().sum()),
        "age_band": fairness_table(age_band(joined["AGE_YEARS"], config["fairness"]["age_bands"]), y, raw, calibrated, operating["threshold"],
                                   age_band_labels(config["fairness"]["age_bands"])),
    }


def ordering_note(entries: list[dict[str, Any]]) -> str:
    correlation = spearmanr([e["observed_default_rate"] for e in entries], [e["approval_rate"] for e in entries]).statistic
    return (f"Across all groups above, the Spearman rank correlation between observed default rate and approval rate is {correlation:.3f}; "
            "a value of -1 means groups with higher observed default rates are always approved less often under the single cut-off.")


def fairness_markdown(result: dict[str, Any]) -> str:
    def rows(entries: list[dict[str, Any]]) -> list[list[Any]]:
        return [[e["group"], f"{e['count']:,}", f"{e['observed_default_rate']:.4f}", f"{e['mean_predicted_pd']:.4f}",
                 f"{e['roc_auc']:.4f}" if e["roc_auc"] is not None else "n/a", f"{100 * e['approval_rate']:.1f}%",
                 f"{e['bad_rate_among_approved']:.4f}" if e["bad_rate_among_approved"] is not None else "n/a"] for e in entries]

    def gaps(entries: list[dict[str, Any]], dimension: str) -> str:
        approval = [e["approval_rate"] for e in entries]
        aucs = [e["roc_auc"] for e in entries if e["roc_auc"] is not None]
        defaults = [e["observed_default_rate"] for e in entries]
        return (f"{dimension}: approval rate ranges from {100 * min(approval):.1f}% to {100 * max(approval):.1f}% "
                f"(gap {100 * (max(approval) - min(approval)):.1f} points); observed default rate ranges from {min(defaults):.4f} to {max(defaults):.4f}; "
                f"ROC-AUC ranges from {min(aucs):.4f} to {max(aucs):.4f}.")

    headers = ["group", "applicants", "observed default rate", "mean predicted PD", "ROC-AUC", "approval rate", "bad rate among approved"]
    point = result["operating_point"]
    return "\n\n".join([
        "# Fairness audit",
        f"Model: {result['model']}. Holdout set only. CODE_GENDER is not a model feature; it is used here for auditing only. "
        f"Operating point: a single score cut-off chosen so that the bad rate among all approved holdout applicants is at most "
        f"{100 * result['operating_bad_rate']:.0f}% (approval rate {100 * point['approval_rate']:.1f}%, realized bad rate {point['realized_bad_rate']:.4f}).",
        "## By CODE_GENDER",
        markdown_table(headers, rows(result["gender"])),
        f"Applicants with unknown gender (XNA in the raw data) in the holdout: {result['gender_missing']}.",
        "## By age band",
        markdown_table(headers, rows(result["age_band"])),
        "## Observed gaps",
        gaps(result["gender"], "Gender"),
        gaps(result["age_band"], "Age band"),
        ordering_note(result["gender"] + result["age_band"]),
        "This audit reports the gaps only; it does not test whether they are justified, and it does not cover other protected attributes.",
    ]) + "\n"


def main() -> None:
    config = load_config()
    paths = get_paths()
    settings = config["explain"]
    training = read_json(paths.reports / "training.json", "train")
    name = training["best_lightgbm"]
    bundle = load_bundle(name)
    ensemble: FoldEnsemble = bundle["estimator"]
    data = load_dataset()
    rng = np.random.default_rng(config["seed"])
    sample_index = rng.choice(data.y_holdout.size, size=min(settings["shap_sample_size"], data.y_holdout.size), replace=False)
    X_sample = data.X_holdout.iloc[sample_index][ensemble.features]
    fold_values = fold_shap_values(ensemble, X_sample)
    values = np.mean(fold_values, axis=0)
    mean_abs = pd.Series(np.abs(values).mean(axis=0), index=ensemble.features).sort_values(ascending=False)
    stability = shap_stability(fold_values, ensemble.features)

    perm_index = rng.choice(data.y_holdout.size, size=min(settings["permutation_sample_size"], data.y_holdout.size), replace=False)
    permutation = permutation_importance(ensemble, data.X_holdout.iloc[perm_index][ensemble.features], data.y_holdout[perm_index],
                                         settings["permutation_repeats"], config["seed"])
    table = pd.DataFrame({"mean_abs_shap": mean_abs})
    table["permutation_auc_drop"] = pd.Series({k: v["mean"] for k, v in permutation.items()})
    table["permutation_std"] = pd.Series({k: v["std"] for k, v in permutation.items()})
    table["shap_rank"] = table["mean_abs_shap"].rank(ascending=False, method="first").astype(int)
    table["permutation_rank"] = table["permutation_auc_drop"].rank(ascending=False, method="first").astype(int)
    gap = settings["disagreement_rank_gap"]
    in_top = (table["shap_rank"] <= 20) | (table["permutation_rank"] <= 20)
    table["disagreement"] = in_top & (((table["shap_rank"] - table["permutation_rank"]).abs() > gap) | (table["permutation_auc_drop"] <= 0))
    table.index.name = "feature"
    table.sort_values("shap_rank").to_csv(paths.reports / "feature_importance.csv", float_format="%.6g")

    apply_style()
    plot_summary(values, X_sample, paths.figures / "shap_summary.png")
    top_features = mean_abs.index[: settings["dependence_top_n"]].tolist()
    for feature in top_features:
        plot_dependence(feature, values, X_sample, ensemble.features, paths.figures / f"shap_dependence_{feature}.png")
    plot_importance_comparison(table, paths.figures / "importance_comparison.png")

    oof = pl.read_parquet(require(paths.processed / "oof_predictions.parquet", "train"))[f"oof_{name}"].to_numpy()
    holdout = pl.read_parquet(require(paths.processed / "holdout_predictions.parquet", "evaluate"))[f"raw_{name}"].to_numpy()
    thresholds = settings["psi_thresholds"]
    psi_rows = [{"variable": "model score", "psi": population_stability_index(oof, holdout, settings["psi_bins"])}]
    for feature in mean_abs.index[: settings["psi_top_n"]]:
        if feature in data.categorical:
            value = categorical_psi(data.X_train[feature], data.X_holdout[feature])
        else:
            value = population_stability_index(data.X_train[feature].to_numpy(dtype=float), data.X_holdout[feature].to_numpy(dtype=float), settings["psi_bins"])
        psi_rows.append({"variable": feature, "psi": value})
    for row in psi_rows:
        row["status"] = psi_label(row["psi"], thresholds)

    fairness = run_fairness(data, name, config)
    disagreements = table[table["disagreement"]].sort_values("shap_rank")
    write_json(paths.reports / "explainability.json", {
        "model": name, "shap_sample_size": int(len(sample_index)), "top_features": mean_abs.head(20).to_dict(),
        "stability": {k: v for k, v in stability.items() if k != "fold_importance"},
        "permutation_sample_size": int(len(perm_index)), "permutation_repeats": settings["permutation_repeats"],
        "disagreements": disagreements.index.tolist(), "dependence_features": top_features,
    })
    write_json(paths.reports / "psi.json", {"thresholds": thresholds, "rows": psi_rows, "reference": "training split (score: out-of-fold)", "comparison": "holdout"})
    write_json(paths.reports / "fairness.json", fairness)
    (paths.reports / "fairness.md").write_text(fairness_markdown(fairness))
    (paths.reports / "explainability.md").write_text(explainability_markdown(name, table, stability, disagreements, top_features, psi_rows, thresholds,
                                                                             len(sample_index), len(perm_index), settings["permutation_repeats"]))
    print(f"model {name}")
    print(f"shap fold rank correlation mean {stability['mean_spearman']:.4f} min {stability['min_spearman']:.4f}")
    print(f"top features {', '.join(mean_abs.index[:5])}")
    print(f"importance disagreements {len(disagreements)}")
    print(f"score psi {psi_rows[0]['psi']:.5f}  max feature psi {max(r['psi'] for r in psi_rows[1:]):.5f}")


def explainability_markdown(name: str, table: pd.DataFrame, stability: dict[str, Any], disagreements: pd.DataFrame, top_features: list[str],
                            psi_rows: list[dict[str, Any]], thresholds: list[float], shap_rows: int, perm_rows: int, repeats: int) -> str:
    top = table.sort_values("shap_rank").head(20)
    rows = [[int(r.shap_rank), feature, f"{r.mean_abs_shap:.4f}", int(r.permutation_rank), f"{r.permutation_auc_drop:.5f}", "yes" if r.disagreement else ""]
            for feature, r in top.iterrows()]
    disagreement_rows = [[feature, int(r.shap_rank), int(r.permutation_rank), f"{r.permutation_auc_drop:.5f}"] for feature, r in disagreements.iterrows()]
    pairwise = ", ".join(f"{value:.3f}" for value in stability["pairwise_spearman"])
    return "\n\n".join([
        "# Explainability and stability",
        f"Model: {name}, an average of 5 fold models. SHAP values come from TreeExplainer on each fold model and are averaged "
        f"(averaging is exact because the ensemble averages log-odds). SHAP sample: {shap_rows:,} holdout rows.",
        "![shap summary](figures/shap_summary.png)",
        "## Dependence plots for the top 3 features",
        "\n\n".join(f"![{f}](figures/shap_dependence_{f}.png)" for f in top_features),
        "## SHAP stability across CV folds",
        f"Spearman rank correlation of mean absolute SHAP between each pair of fold models: mean {stability['mean_spearman']:.4f}, "
        f"minimum {stability['min_spearman']:.4f}. Pairwise values: {pairwise}. Mean overlap of the top 20 features between folds: {stability['mean_top20_overlap']:.2f}.",
        "## SHAP versus permutation importance",
        f"Permutation importance is the drop in holdout ROC-AUC when one feature is shuffled ({perm_rows:,} holdout rows, {repeats} repeats). "
        "A feature is flagged when it is in the top 20 of either ranking and the ranks differ by more than 10 places, or when shuffling it does not reduce AUC.",
        markdown_table(["SHAP rank", "feature", "mean abs SHAP", "permutation rank", "AUC drop", "flagged"], rows),
        "![importance comparison](figures/importance_comparison.png)",
        "### Flagged disagreements",
        markdown_table(["feature", "SHAP rank", "permutation rank", "AUC drop"], disagreement_rows) if disagreement_rows else "None.",
        "Disagreements usually come from correlated features: SHAP splits credit among correlated inputs while permuting one of them leaves the others to carry the signal.",
        "## Population stability index",
        f"Reference: training split (the score uses out-of-fold predictions). Comparison: holdout. Thresholds: below {thresholds[0]} stable, "
        f"{thresholds[0]} to {thresholds[1]} moderate shift, above {thresholds[1]} significant shift. "
        "The holdout is a random split of the same population, so low values are expected; this table is the template for monitoring later vintages.",
        markdown_table(["variable", "PSI", "status"], [[r["variable"], f"{r['psi']:.5f}", r["status"]] for r in psi_rows]),
    ]) + "\n"


if __name__ == "__main__":
    main()
