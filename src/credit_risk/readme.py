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
            m["label"], f"{metric['roc_auc']:.4f} {ci(m['roc_auc_ci'])}", f"{metric['ks']:.4f}", f"{metric['gini']:.4f}",
            f"{metric['pr_auc']:.4f}", f"{metric['brier']:.5f}", "" if name == "legacy" else signed(m["auc_diff_vs_legacy"]),
        ])
    return markdown_table(["model", "AUC", "KS", "Gini", "PR-AUC", "Brier", "AUC gain vs old model"], rows)


def approval_table(metrics: dict[str, Any]) -> str:
    rates = metrics["target_bad_rates"]
    rows = [[metrics["models"][name]["label"], *[percent(metrics["models"][name]["approval"][str(rate)]["approval_rate"]) for rate in rates]]
            for name in MODEL_ORDER]
    return markdown_table(["model", *[f"{percent(rate, 0)} bad rate" for rate in rates]], rows)


def ablation_text(ablation: dict[str, Any]) -> str:
    steps = ablation["steps"]
    cumulative = [s for s in steps if s["kind"] == "cumulative"]
    external = next(s for s in steps if s["kind"] == "external")
    gains = sorted(cumulative[1:], key=lambda s: -s["delta_auc"])
    best, worst = gains[0], gains[-1]
    return (
        f"The application table alone gets {cumulative[0]['roc_auc']:.4f}, all {len(cumulative)} tables get {cumulative[-1]['roc_auc']:.4f}. "
        f"Biggest single jump is {best['label'].lstrip('+ ')} ({signed(best['delta_auc'])}), and {worst['label'].lstrip('+ ')} adds basically nothing "
        f"({signed(worst['delta_auc'])}, interval {signed_ci(worst['delta_auc_ci'])}). "
        f"The three external bureau scores carry a lot: drop them and the application-only model falls to {external['roc_auc']:.4f}. "
        "Full numbers with intervals are in [reports/ablation.md](reports/ablation.md)."
    )


def notes(results: dict[str, Any]) -> list[str]:
    metrics, training, imbalance = results["metrics"], results["training"], results["imbalance"]
    models = metrics["models"]
    best = metrics["best_tree_model"]
    monotone = models["lightgbm_monotone"]["metrics"]["roc_auc"]
    unconstrained = models["lightgbm"]["metrics"]["roc_auc"]
    base, weighted = imbalance["variants"]["no_weighting"], imbalance["variants"]["scale_pos_weight"]
    flagged = results["audit"]["flagged"]
    middle = str(metrics["target_bad_rates"][len(metrics["target_bad_rates"]) // 2])
    approve_best = models[best]["approval"][middle]["approval_rate"]
    approve_legacy = models["legacy"]["approval"][middle]["approval_rate"]
    return [
        f"- {models[best]['label']} beats the old logistic model by {signed(models[best]['auc_diff_vs_legacy'])} AUC "
        f"{signed_ci(models[best]['auc_diff_vs_legacy_ci'])}. Holding the bad rate at {percent(float(middle), 0)}, "
        f"that means approving {percent(approve_best)} of applicants instead of {percent(approve_legacy)}.",
        f"- The scorecard is weaker ({models['scorecard']['metrics']['roc_auc']:.4f}) but every point is readable, see "
        "[reports/scorecard.csv](reports/scorecard.csv).",
        f"- Forcing monotonic behaviour on {len(training['models']['lightgbm_monotone']['constraints'])} features costs only "
        f"{signed(unconstrained - monotone)} AUC, cheap for a model you have to defend.",
        f"- Class weighting didn't help: AUC moved {signed(weighted['roc_auc'] - base['roc_auc'])} and Brier went from {base['brier']:.4f} to "
        f"{weighted['brier']:.4f} (it predicts a {weighted['mean_predicted']:.0%} average default rate when the real one is {weighted['observed_rate']:.0%}).",
        f"- No feature scores above {results['audit']['threshold']} AUC on its own, so nothing looked like a leak"
        + (f" except {', '.join(flagged)}." if flagged else "."),
    ]


def intro(results: dict[str, Any]) -> str:
    summary = results["summary"]
    return (
        f"Predicting which Home Credit applicants will have trouble repaying. {summary['applications']:,} loans, "
        f"{percent(summary['target_rate'])} of them went bad, and all {len(summary['row_counts'])} tables from the Kaggle competition are used. "
        "Features are built in DuckDB SQL, only from records dated before each application, with tests that fail if anything from the future gets in."
    )


def details(summary: str, body: str) -> str:
    return f"<details>\n<summary>{summary}</summary>\n\n{body}\n\n</details>"


def pipeline_diagram(results: dict[str, Any]) -> str:
    tables = len(results["summary"]["row_counts"])
    features = sum(results["build"]["feature_counts"].values())
    holdout = results["metrics"]["holdout_rows"]
    return "\n".join([
        "```mermaid",
        "flowchart LR",
        f"    raw[{tables} raw CSVs] --> sql[DuckDB SQL<br/>past rows only]",
        f"    sql --> feat[{features} features<br/>one row per applicant]",
        "    feat --> cv[5-fold CV<br/>Optuna tuning]",
        "    cv --> models[logistic, scorecard,<br/>LightGBM, XGBoost]",
        f"    models --> hold[holdout<br/>{holdout:,} applicants]",
        "    hold --> out[reports and this README]",
        "    models --> api[FastAPI scorer]",
        "```",
    ])


def render(results: dict[str, Any]) -> str:
    metrics, ablation, fairness, psi = results["metrics"], results["ablation"], results["fairness"], results["psi"]
    score_psi = psi["rows"][0]["psi"]
    fairness_rows = [[f"{r['group']}", f"{r['count']:,}", f"{r['observed_default_rate']:.3f}", f"{r['roc_auc']:.4f}", percent(r["approval_rate"])]
                     for r in fairness["gender"] + fairness["age_band"]]
    example_id = results["score_example"]["sk_id_curr"]
    best = metrics["models"][metrics["best_tree_model"]]
    sections = [
        "# home-credit-default-model",
        " ".join([
            "[![ci](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)",
            "![python](https://img.shields.io/badge/python-3.11-3776ab)",
            f"![auc](https://img.shields.io/badge/holdout%20AUC-{best['metrics']['roc_auc']:.4f}-2a78d6)",
            "![duckdb](https://img.shields.io/badge/features-DuckDB%20SQL-fff000)",
        ]),
        intro(results),
        "**Jump to:** [results](#results) · [what each table adds](#what-each-table-adds) · [what drives the score](#what-drives-the-score) · "
        "[calibration](#calibration) · [fairness](#fairness-and-drift) · [scoring](#scoring) · [running it](#running-it)",
        pipeline_diagram(results),
        "## Results",
        f"On a {metrics['holdout_rows']:,} applicant holdout that was never used for tuning. AUC brackets are 95% bootstrap intervals.",
        results_table(metrics),
        "\n".join(notes(results)),
        details("Approval rate at a fixed bad rate", approval_table(metrics)),
        "## What each table adds",
        "![feature group ablation](reports/figures/ablation.png)",
        ablation_text(ablation),
        "## What drives the score",
        "![shap summary](reports/figures/shap_summary.png)",
        details("ROC, KS and cost curves", "\n\n".join([
            "![roc](reports/figures/roc.png)", "![ks](reports/figures/ks.png)", "![cost](reports/figures/cost_curve.png)"])),
        "## Calibration",
        "![calibration](reports/figures/calibration.png)",
        "## Fairness and drift",
        f"One cut-off, set so the approved book has a {percent(fairness['operating_bad_rate'], 0)} bad rate. "
        "Gender is not a model input. Approval rates mostly follow each group's real default rate. "
        f"Score PSI between train and holdout is {score_psi:.4f}, as expected for a random split.",
        details("By gender and age", markdown_table(["group", "applicants", "default rate", "AUC", "approved"], fairness_rows)),
        "## Scoring",
        "```bash\nmake score ID=<SK_ID_CURR>\n```",
        details(f"Example output for applicant {example_id}", "```json\n" + json.dumps(results["score_example"], indent=2) + "\n```"),
        details("As an API or in Docker", "\n".join([
            "```bash",
            "make serve",
            f"curl localhost:8000/applicants/{example_id}/score",
            "curl -X POST localhost:8000/score -H 'content-type: application/json' -d '{\"sk_id_curr\": [100002, 100003]}'",
            "",
            "make docker",
            "docker run -p 8000:8000 -v $PWD/models:/app/models -v $PWD/data/processed:/app/data/processed credit-risk-lab",
            "```",
            "",
            "Swagger docs at `localhost:8000/docs`.",
        ])),
        "## Running it",
        "\n".join([
            "Python 3.11 and a Kaggle token in `~/.kaggle/kaggle.json`.",
            "",
            "```bash",
            "make setup",
            "make all",
            "make test",
            "```",
            "",
            "`make all` goes from raw CSVs to this README. The full run tunes 40 trials per model and takes a few hours on a laptop, "
            "`make quick` does the same with a tiny budget and writes to `build/quick` instead. Settings are in `config.yaml`.",
        ]),
        details("More results", "\n".join([
            "- [model comparison](reports/model_comparison.md)",
            "- [data quality](reports/data_quality.md)",
            "- [feature build](reports/feature_build.md) and [leakage audit](reports/leakage_audit.md)",
            "- [class imbalance](reports/imbalance.md)",
            "- [explainability and PSI](reports/explainability.md)",
            "- [fairness](reports/fairness.md)",
            "- [scorecard points](reports/scorecard.csv)",
        ])),
        "## Caveats",
        "\n".join([
            "- No real dates in the data, so this is a random holdout, not an out-of-time test. Expect it to do a bit worse on new loans.",
            "- Only approved applicants are in the data, no reject inference.",
            "- The cost curve uses made-up margin and loss numbers from `config.yaml`.",
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
