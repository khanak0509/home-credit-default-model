import numpy as np
import pandas as pd

from credit_risk.artifacts import markdown_table, write_json
from credit_risk.config import get_paths, load_config
from credit_risk.dataset import Dataset, load_dataset, load_manifest
from credit_risk.metrics import fast_auc


def out_of_fold_category_rate(values: pd.Series, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    encoded = np.full(len(values), y.mean())
    codes = values.cat.codes.to_numpy()
    for fold in np.unique(folds):
        fit = folds != fold
        rates = pd.Series(y[fit]).groupby(codes[fit]).mean()
        encoded[~fit] = pd.Series(codes[~fit]).map(rates).fillna(y[fit].mean()).to_numpy()
    return encoded


def single_feature_auc(data: Dataset, column: str) -> dict[str, float]:
    values = data.X_train[column]
    y = data.y_train
    if column in data.categorical:
        scores = out_of_fold_category_rate(values, y, data.folds)
        mask = np.ones(len(scores), dtype=bool)
    else:
        scores = values.to_numpy(dtype=float)
        mask = ~np.isnan(scores)
    coverage = float(mask.mean())
    if mask.sum() == 0 or np.unique(y[mask]).size < 2 or np.unique(scores[mask]).size < 2:
        return {"auc": 0.5, "direction": 0, "coverage": coverage}
    auc = fast_auc(y[mask], scores[mask])
    return {"auc": max(auc, 1 - auc), "direction": 1 if auc >= 0.5 else -1, "coverage": coverage}


def main() -> None:
    config = load_config()
    paths = get_paths()
    manifest = load_manifest()
    data = load_dataset()
    threshold = config["leakage_audit"]["auc_flag_threshold"]
    rows = []
    for column in data.features:
        result = single_feature_auc(data, column)
        rows.append({"feature": column, "group": manifest["groups"][column], **result, "flagged": result["auc"] > threshold})
    rows.sort(key=lambda row: -row["auc"])
    flagged = [row for row in rows if row["flagged"]]
    write_json(paths.data_reports / "leakage_audit.json", {"threshold": threshold, "features": rows, "flagged": [r["feature"] for r in flagged]})
    top_n = config["leakage_audit"]["top_n"]
    table = markdown_table(
        ["rank", "feature", "group", "single-feature AUC", "direction", "non-null coverage"],
        [[i + 1, r["feature"], r["group"], f"{r['auc']:.4f}", "higher value, higher risk" if r["direction"] > 0 else "higher value, lower risk",
          f"{100 * r['coverage']:.1f}%"] for i, r in enumerate(rows[:top_n])],
    )
    verdict = (
        f"{len(flagged)} features exceed the review threshold of {threshold}: {', '.join(r['feature'] for r in flagged)}."
        if flagged else f"No feature exceeds the review threshold of {threshold}."
    )
    text = "\n\n".join([
        "# Leakage audit",
        "Single-feature ROC-AUC on the training split (holdout excluded). AUC is direction-free (max of AUC and 1 - AUC). "
        "Numeric features are scored on their non-null rows; categorical features use out-of-fold default rate per category.",
        verdict,
        f"## Top {top_n}",
        table,
        f"Features audited: {len(rows)}.",
    ])
    (paths.data_reports / "leakage_audit.md").write_text(text + "\n")
    for i, r in enumerate(rows[:top_n]):
        print(f"{i + 1:>3} {r['feature']:<36}{r['group']:<16}{r['auc']:.4f}{'  FLAG' if r['flagged'] else ''}")
    print(f"flagged above {threshold}: {len(flagged)}")


if __name__ == "__main__":
    main()
