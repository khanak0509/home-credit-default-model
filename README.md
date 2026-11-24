# home-credit-default-model

[![ci](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)

The target is whether a Home Credit applicant has payment difficulties (TARGET = 1, a late payment on one of the first installments), which happens for 8.1% of 307,511 approved applications. Features come from all 7 tables: the application itself, credit bureau records and their monthly balances, previous Home Credit applications, and the monthly POS, cash loan and credit card balances and installment repayments of earlier loans, aggregated in DuckDB SQL into one row per applicant. Leakage is prevented by aggregating only rows dated at or before the application (17 future-dated rows and 102 rows attached to them removed, 224,392 previous applications with planned dates after the application masked, enforced by a test that fails if any feature query can see them), tuning only on 5-fold CV inside an 80% training split, and scoring once on an untouched 20% holdout.

## Results on the holdout (61,503 applications)

95% intervals from 1000 paired bootstrap resamples. Brier is after calibration.

| model | ROC-AUC [95% CI] | PR-AUC | KS [95% CI] | Gini [95% CI] | Brier | AUC vs legacy [95% CI] |
|---|---|---|---|---|---|---|
| Legacy logistic baseline | 0.7288 [0.7209, 0.7364] | 0.2120 | 0.3421 [0.3286, 0.3573] | 0.4575 [0.4419, 0.4729] | 0.06939 |  |
| WoE scorecard | 0.7581 [0.7504, 0.7656] | 0.2388 | 0.3896 [0.3773, 0.4054] | 0.5162 [0.5009, 0.5312] | 0.06799 | +0.0293 [+0.0252, +0.0336] |
| XGBoost | 0.7947 [0.7880, 0.8012] | 0.2993 | 0.4465 [0.4349, 0.4608] | 0.5894 [0.5760, 0.6024] | 0.06503 | +0.0660 [+0.0608, +0.0709] |
| LightGBM | 0.7946 [0.7880, 0.8011] | 0.2996 | 0.4449 [0.4339, 0.4589] | 0.5891 [0.5760, 0.6021] | 0.06504 | +0.0659 [+0.0606, +0.0708] |
| LightGBM monotonic | 0.7940 [0.7873, 0.8006] | 0.2984 | 0.4437 [0.4320, 0.4575] | 0.5880 [0.5745, 0.6011] | 0.06508 | +0.0653 [+0.0601, +0.0700] |

- The best tree model by CV AUC is LightGBM. On the holdout it beats the legacy baseline by +0.0659 AUC [+0.0606, +0.0708].
- The 40-feature WoE scorecard reaches 0.7581 AUC with a fully transparent points table ([reports/scorecard.csv](reports/scorecard.csv)).
- Monotonic constraints cost +0.0006 holdout AUC (+0.0009 in CV) on 29 constrained features.
- Class weighting changes out-of-fold AUC by -0.0029 but raises the Brier score from 0.0656 to 0.1644; the mean predicted PD becomes 0.359 against an observed rate of 0.081.
- SHAP importance is stable across CV folds: mean pairwise Spearman correlation 0.960, minimum 0.955.
- Leakage audit: 0 of 287 features exceed the single-feature AUC review threshold of 0.75.

### What each data source adds

LightGBM retrained on cumulative feature groups.

![feature group ablation](reports/figures/ablation.png)

Holdout AUC rises from 0.7716 (application only) to 0.7946 (all 7 sources). Gain from adding each source, largest first: previous applications +0.0073 [+0.0054, +0.0094]; bureau +0.0066 [+0.0049, +0.0084]; installments +0.0056 [+0.0038, +0.0076]; POS and cash loan balances +0.0018 [+0.0008, +0.0029]; credit card balances +0.0017 [+0.0007, +0.0027]; bureau balance -0.0001 [-0.0008, +0.0006]. Removing the three external scores and their aggregates from the application set drops AUC from 0.7716 to 0.7260.

### What drives the score

SHAP values for LightGBM, averaged over the 5 fold models.

![shap summary](reports/figures/shap_summary.png)

### Calibration

![calibration](reports/figures/calibration.png)

### Approval rate at a fixed bad rate

| model | approval at 3% bad rate | approval at 5% bad rate | approval at 8% bad rate |
|---|---|---|---|
| Legacy logistic baseline | 41.9% | 76.3% | 99.8% |
| WoE scorecard | 53.3% | 81.4% | 99.9% |
| XGBoost | 63.7% | 86.8% | 99.9% |
| LightGBM | 63.9% | 86.9% | 99.9% |
| LightGBM monotonic | 64.0% | 86.8% | 99.9% |

### Stability and fairness

Score PSI between the training split and the holdout is 0.00049 (random split, so near zero is expected). Fairness audit at the single cut-off that gives a 5% bad rate on the holdout; gender is not a model input.

| group | applicants | observed default rate | ROC-AUC | approval rate |
|---|---|---|---|---|
| gender F | 40,561 | 0.0699 | 0.7900 | 88.8% |
| gender M | 20,940 | 0.1017 | 0.7929 | 83.1% |
| age under 30 | 8,905 | 0.1130 | 0.7686 | 77.3% |
| age 30-40 | 16,487 | 0.0957 | 0.7991 | 82.5% |
| age 40-50 | 15,303 | 0.0758 | 0.8024 | 88.1% |
| age 50-60 | 13,617 | 0.0635 | 0.7810 | 92.4% |
| age 60+ | 7,191 | 0.0496 | 0.7499 | 95.7% |

### Scoring an applicant

```bash
make score ID=<SK_ID_CURR>
```

```json
{
  "sk_id_curr": 100009,
  "model": "lightgbm",
  "in_holdout": true,
  "probability_of_default": 0.004604,
  "raw_model_probability": 0.004371,
  "scorecard_points": 620.1,
  "risk_band": "A",
  "reason_codes": [
    {
      "feature": "ANNUITY_TO_CREDIT",
      "value": 0.026463,
      "direction": "decreases risk",
      "contribution": -0.321203
    },
    {
      "feature": "DOCUMENT_COUNT",
      "value": 2.0,
      "direction": "decreases risk",
      "contribution": -0.280091
    },
    {
      "feature": "EXT_SOURCE_MEAN",
      "value": 0.663607,
      "direction": "decreases risk",
      "contribution": -0.200478
    },
    {
      "feature": "AMT_ANNUITY",
      "value": 41301.0,
      "direction": "increases risk",
      "contribution": 0.174802
    },
    {
      "feature": "OCCUPATION_TYPE",
      "value": "Accountants",
      "direction": "decreases risk",
      "contribution": -0.174623
    }
  ]
}
```

The same scorer runs as an HTTP service. `make serve` starts it locally; OpenAPI docs are at `/docs`.

```bash
make serve
curl localhost:8000/applicants/100009/score
curl -X POST localhost:8000/score -H 'content-type: application/json' -d '{"sk_id_curr": [100002, 100003]}'
```

Or in Docker, with the trained models and the feature store mounted:

```bash
make docker
docker run -p 8000:8000 -v $PWD/models:/app/models -v $PWD/data/processed:/app/data/processed credit-risk-lab
```

## Reports

- [Data quality](reports/data_quality.md)
- [Feature build and leakage filter](reports/feature_build.md)
- [Leakage audit](reports/leakage_audit.md)
- [Model comparison](reports/model_comparison.md) with ROC, PR, KS, calibration and illustrative cost curves
- [Class imbalance study](reports/imbalance.md)
- [Feature group ablation](reports/ablation.md)
- [Explainability and PSI](reports/explainability.md)
- [Fairness audit](reports/fairness.md)
- [Scorecard table](reports/scorecard.csv)
- [Methodology](METHODOLOGY.md)

## Reproduce

Needs Python 3.11, a Kaggle API token in `~/.kaggle/kaggle.json` and the competition rules accepted at kaggle.com/competitions/home-credit-default-risk (the downloader falls back to a public mirror of the same files).

```bash
make setup      # virtualenv and pinned dependencies
make all        # download, quality, features, audit, train, evaluate, ablation, explain, README
make test       # unit and integration tests
make quick      # the whole modeling pipeline with a small tuning budget, written to build/quick
make score ID=100002
```

Individual stages: `make data`, `make quality`, `make features`, `make audit`, `make train`, `make evaluate`, `make ablation`, `make explain`, `make readme`. `make test-unit` runs the tests that need no data, which is what CI runs on every push together with `ruff` and a Docker build. The full run tunes 40 trials per library and takes hours on a laptop; `make quick` finishes in a fraction of that and never touches `reports/`. All settings, including seeds, live in `config.yaml`.

## Project layout

```
sql/                 cleaning views, leakage filter views, feature queries
src/credit_risk/     pipeline modules (one per make target), model code and the FastAPI service
tests/               unit tests and data-dependent integration tests
reports/             generated reports, figures and JSON results
config.yaml          every setting and assumption, plus the quick profile
Dockerfile           scoring service image
.github/workflows/   lint, unit tests and Docker build on every push
```

## Limitations

- The dataset has no real timestamps, only offsets relative to each application, so validation is stratified CV plus a random holdout, not an out-of-time test. Performance on future vintages is likely to be lower.
- The cost curve and profit tables use illustrative margin and loss-given-default assumptions from `config.yaml`, not observed economics.
- No reject inference was done. The data only contains approved applicants, so the models are fit on the accepted population and may rank rejected-type applicants differently.
- All results come from one public dataset from one lender in one period.
