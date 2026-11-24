import json
from typing import Any

from credit_risk.artifacts import markdown_table, read_json, signed
from credit_risk.bundles import MODEL_ORDER
from credit_risk.config import PROJECT_ROOT, get_paths, load_config


def load_results() -> dict[str, Any]:
    paths = get_paths()
    data_sources = {
        "summary": ("data_summary.json", "data"),
        "quality": ("data_quality.json", "quality"),
        "build": ("feature_build.json", "features"),
        "audit": ("leakage_audit.json", "audit"),
    }
    model_sources = {
        "training": ("training.json", "train"),
        "imbalance": ("imbalance.json", "train"),
        "metrics": ("metrics.json", "evaluate"),
        "ablation": ("ablation.json", "ablation"),
        "explain": ("explainability.json", "explain"),
        "psi": ("psi.json", "explain"),
        "fairness": ("fairness.json", "explain"),
        "score_example": ("score_example.json", "readme"),
    }
    results = {key: read_json(paths.data_reports / name, target) for key, (name, target) in data_sources.items()}
    results.update({key: read_json(paths.reports / name, target) for key, (name, target) in model_sources.items()})
    return results


def ci(values: list[float], digits: int = 4) -> str:
    return f"[{values[0]:.{digits}f}, {values[1]:.{digits}f}]"


def signed_ci(values: list[float]) -> str:
    return f"[{signed(values[0])}, {signed(values[1])}]"


def percent(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}%"


def results_table(metrics: dict[str, Any]) -> str:
    rows = []
    for name in MODEL_ORDER:
        m = metrics["models"][name]
        metric = m["metrics"]
        rows.append([
            m["label"], f"{metric['roc_auc']:.4f} {ci(m['roc_auc_ci'])}", f"{metric['pr_auc']:.4f}",
            f"{metric['ks']:.4f} {ci(m['ks_ci'])}", f"{metric['gini']:.4f} {ci(m['gini_ci'])}", f"{metric['brier']:.5f}",
            "" if name == "legacy" else f"{signed(m['auc_diff_vs_legacy'])} {signed_ci(m['auc_diff_vs_legacy_ci'])}",
        ])
    return markdown_table(["model", "ROC-AUC [95% CI]", "PR-AUC", "KS [95% CI]", "Gini [95% CI]", "Brier", "AUC vs legacy [95% CI]"], rows)


def approval_table(metrics: dict[str, Any]) -> str:
    rates = metrics["target_bad_rates"]
    rows = [[metrics["models"][name]["label"], *[percent(metrics["models"][name]["approval"][str(rate)]["approval_rate"]) for rate in rates]]
            for name in MODEL_ORDER]
    return markdown_table(["model", *[f"approval at {percent(rate, 0)} bad rate" for rate in rates]], rows)


def ablation_lines(ablation: dict[str, Any]) -> list[str]:
    steps = ablation["steps"]
    cumulative = [s for s in steps if s["kind"] == "cumulative"]
    external = next(s for s in steps if s["kind"] == "external")
    gains = sorted(cumulative[1:], key=lambda s: -s["delta_auc"])
    lines = [f"Holdout AUC rises from {cumulative[0]['roc_auc']:.4f} (application only) to {cumulative[-1]['roc_auc']:.4f} (all {len(cumulative)} sources)."]
    lines.append("Gain from adding each source, largest first: " + "; ".join(
        f"{s['label'].lstrip('+ ')} {signed(s['delta_auc'])} {signed_ci(s['delta_auc_ci'])}" for s in gains) + ".")
    lines.append(f"Removing the three external scores and their aggregates from the application set drops AUC from "
                 f"{cumulative[0]['roc_auc']:.4f} to {external['roc_auc']:.4f}.")
    return lines


def key_findings(results: dict[str, Any]) -> list[str]:
    metrics, training, imbalance = results["metrics"], results["training"], results["imbalance"]
    models = metrics["models"]
    best = metrics["best_tree_model"]
    monotone = models["lightgbm_monotone"]["metrics"]["roc_auc"]
    unconstrained = models["lightgbm"]["metrics"]["roc_auc"]
    base, weighted = imbalance["variants"]["no_weighting"], imbalance["variants"]["scale_pos_weight"]
    stability = results["explain"]["stability"]
    flagged = results["audit"]["flagged"]
    scorecard_features = len(training["models"]["scorecard"]["selected_features"])
    lines = [
        f"- The best tree model by CV AUC is {models[best]['label']}. On the holdout it beats the legacy baseline by "
        f"{signed(models[best]['auc_diff_vs_legacy'])} AUC {signed_ci(models[best]['auc_diff_vs_legacy_ci'])}.",
        f"- The {scorecard_features}-feature WoE scorecard reaches {models['scorecard']['metrics']['roc_auc']:.4f} AUC with a fully transparent points table "
        f"([reports/scorecard.csv](reports/scorecard.csv)).",
        f"- Monotonic constraints cost {signed(unconstrained - monotone)} holdout AUC ({signed(training['monotone_cv_auc_cost'])} in CV) on {len(training['models']['lightgbm_monotone']['constraints'])} constrained features.",
        f"- Class weighting changes out-of-fold AUC by {signed(weighted['roc_auc'] - base['roc_auc'])} but raises the Brier score from {base['brier']:.4f} to "
        f"{weighted['brier']:.4f}; the mean predicted PD becomes {weighted['mean_predicted']:.3f} against an observed rate of {weighted['observed_rate']:.3f}.",
        f"- SHAP importance is stable across CV folds: mean pairwise Spearman correlation {stability['mean_spearman']:.3f}, minimum {stability['min_spearman']:.3f}.",
        f"- Leakage audit: {len(flagged)} of {len(results['audit']['features'])} features exceed the single-feature AUC review threshold of {results['audit']['threshold']}"
        + (f" ({', '.join(flagged)})." if flagged else "."),
    ]
    return lines


def problem_statement(results: dict[str, Any]) -> str:
    summary, build = results["summary"], results["build"]
    future = sum(t["dropped_future_offset"] for t in build["leakage_filter"].values())
    attached = build["leakage_filter"]["bureau_balance"]["dropped_parent_filtered"]
    planned = build["leakage_filter"]["previous_application"]["masked_planned_dates"]
    return (
        f"The target is whether a Home Credit applicant has payment difficulties (TARGET = 1, a late payment on one of the first installments), "
        f"which happens for {percent(summary['target_rate'])} of {summary['applications']:,} approved applications. "
        f"Features come from all {len(summary['row_counts'])} tables: the application itself, credit bureau records and their monthly balances, previous Home Credit applications, "
        "and the monthly POS, cash loan and credit card balances and installment repayments of earlier loans, aggregated in DuckDB SQL into one row per applicant. "
        f"Leakage is prevented by aggregating only rows dated at or before the application ({future:,} future-dated rows and {attached:,} rows attached to them removed, "
        f"{planned:,} previous applications with planned dates after the application masked, "
        f"enforced by a test that fails if any feature query can see them), tuning only on 5-fold CV inside an 80% training split, and scoring once on an untouched 20% holdout."
    )


def render(results: dict[str, Any]) -> str:
    metrics, ablation, fairness, psi = results["metrics"], results["ablation"], results["fairness"], results["psi"]
    shap_model = results["explain"]["model"]
    score_psi = psi["rows"][0]["psi"]
    fairness_rows = [[f"gender {r['group']}", f"{r['count']:,}", f"{r['observed_default_rate']:.4f}", f"{r['roc_auc']:.4f}", percent(r["approval_rate"])]
                     for r in fairness["gender"]]
    fairness_rows += [[f"age {r['group']}", f"{r['count']:,}", f"{r['observed_default_rate']:.4f}", f"{r['roc_auc']:.4f}", percent(r["approval_rate"])]
                      for r in fairness["age_band"]]
    sections = [
        "# home-credit-default-model",
        "[![ci](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)",
        problem_statement(results),
        f"## Results on the holdout ({metrics['holdout_rows']:,} applications)",
        f"95% intervals from {metrics['n_bootstrap']} paired bootstrap resamples. Brier is after calibration.",
        results_table(metrics),
        "\n".join(key_findings(results)),
        "### What each data source adds",
        f"{results['training']['models'][ablation['model']]['label']} retrained on cumulative feature groups.",
        "![feature group ablation](reports/figures/ablation.png)",
        " ".join(ablation_lines(ablation)),
        "### What drives the score",
        f"SHAP values for {results['training']['models'][shap_model]['label']}, averaged over the 5 fold models.",
        "![shap summary](reports/figures/shap_summary.png)",
        "### Calibration",
        "![calibration](reports/figures/calibration.png)",
        "### Approval rate at a fixed bad rate",
        approval_table(metrics),
        "### Stability and fairness",
        f"Score PSI between the training split and the holdout is {score_psi:.5f} (random split, so near zero is expected). "
        f"Fairness audit at the single cut-off that gives a {percent(fairness['operating_bad_rate'], 0)} bad rate on the holdout; gender is not a model input.",
        markdown_table(["group", "applicants", "observed default rate", "ROC-AUC", "approval rate"], fairness_rows),
        "### Scoring an applicant",
        "```bash\nmake score ID=<SK_ID_CURR>\n```",
        "```json\n" + json.dumps(results["score_example"], indent=2) + "\n```",
        "The same scorer runs as an HTTP service. `make serve` starts it locally; OpenAPI docs are at `/docs`.",
        "\n".join([
            "```bash",
            "make serve",
            f"curl localhost:8000/applicants/{results['score_example']['sk_id_curr']}/score",
            "curl -X POST localhost:8000/score -H 'content-type: application/json' -d '{\"sk_id_curr\": [100002, 100003]}'",
            "```",
        ]),
        "Or in Docker, with the trained models and the feature store mounted:",
        "\n".join([
            "```bash",
            "make docker",
            "docker run -p 8000:8000 -v $PWD/models:/app/models -v $PWD/data/processed:/app/data/processed credit-risk-lab",
            "```",
        ]),
        "## Reports",
        "\n".join([
            "- [Data quality](reports/data_quality.md)",
            "- [Feature build and leakage filter](reports/feature_build.md)",
            "- [Leakage audit](reports/leakage_audit.md)",
            "- [Model comparison](reports/model_comparison.md) with ROC, PR, KS, calibration and illustrative cost curves",
            "- [Class imbalance study](reports/imbalance.md)",
            "- [Feature group ablation](reports/ablation.md)",
            "- [Explainability and PSI](reports/explainability.md)",
            "- [Fairness audit](reports/fairness.md)",
            "- [Scorecard table](reports/scorecard.csv)",
            "- [Methodology](METHODOLOGY.md)",
        ]),
        "## Reproduce",
        "\n".join([
            "Needs Python 3.11, a Kaggle API token in `~/.kaggle/kaggle.json` and the competition rules accepted at "
            "kaggle.com/competitions/home-credit-default-risk (the downloader falls back to a public mirror of the same files).",
            "",
            "```bash",
            "make setup      # virtualenv and pinned dependencies",
            "make all        # download, quality, features, audit, train, evaluate, ablation, explain, README",
            "make test       # unit and integration tests",
            "make quick      # the whole modeling pipeline with a small tuning budget, written to build/quick",
            "make score ID=100002",
            "```",
            "",
            "Individual stages: `make data`, `make quality`, `make features`, `make audit`, `make train`, `make evaluate`, `make ablation`, "
            "`make explain`, `make readme`. `make test-unit` runs the tests that need no data, which is what CI runs on every push together with `ruff` and a Docker build. "
            "The full run tunes 40 trials per library and takes hours on a laptop; `make quick` finishes in a fraction of that and never touches `reports/`. "
            "All settings, including seeds, live in `config.yaml`.",
        ]),
        "## Project layout",
        "\n".join([
            "```",
            "sql/                 cleaning views, leakage filter views, feature queries",
            "src/credit_risk/     pipeline modules (one per make target), model code and the FastAPI service",
            "tests/               unit tests and data-dependent integration tests",
            "reports/             generated reports, figures and JSON results",
            "config.yaml          every setting and assumption, plus the quick profile",
            "Dockerfile           scoring service image",
            ".github/workflows/   lint, unit tests and Docker build on every push",
            "```",
        ]),
        "## Limitations",
        "\n".join([
            "- The dataset has no real timestamps, only offsets relative to each application, so validation is stratified CV plus a random holdout, not an out-of-time test. Performance on future vintages is likely to be lower.",
            "- The cost curve and profit tables use illustrative margin and loss-given-default assumptions from `config.yaml`, not observed economics.",
            "- No reject inference was done. The data only contains approved applicants, so the models are fit on the accepted population and may rank rejected-type applicants differently.",
            "- All results come from one public dataset from one lender in one period.",
        ]),
    ]
    return "\n\n".join(sections) + "\n"


def main() -> None:
    results = load_results()
    target = PROJECT_ROOT / "README.md" if load_config()["profile"] == "full" else get_paths().reports / "README.md"
    text = render(results)
    if target.parent != PROJECT_ROOT:
        text = text.replace("](reports/", "](")
    target.write_text(text)
    print(f"{target.relative_to(PROJECT_ROOT)} written")


if __name__ == "__main__":
    main()
