# Methodology

This document covers definitions and design choices. Every result lives in `reports/` and is written by a run; nothing here is a measured number. Settings quoted below come from `config.yaml`.

## Target definition

`TARGET` in `application_train` is 1 when the client had payment difficulties: a payment more than X days late on at least one of the first Y installments of the loan (Home Credit does not publish X and Y), and 0 otherwise. Every applicant in `application_train` is a loan that was approved and observed, so the population is accepted applicants only. See the limitations on reject inference in the README.

## Data and cleaning

All seven tables of the competition training data are used: `application_train` (one row per application), `bureau` (credits reported by other institutions), `bureau_balance` (monthly status of those credits), `previous_application` (earlier applications to Home Credit, approved or not), `installments_payments` (repayment history of earlier Home Credit loans), `POS_CASH_balance` (monthly snapshots of earlier point of sale and cash loans) and `credit_card_balance` (monthly snapshots of earlier Home Credit cards). The raw CSVs are registered as DuckDB views and staged once to parquet. DuckDB runs with an 8 GB memory limit and spills to disk, so the large behavioral tables are aggregated in SQL and never loaded into pandas.

Two application anomalies are fixed in `sql/01_application_clean.sql`:

- `DAYS_EMPLOYED = 365243` is a sentinel (about 1000 years of employment) used for pensioners and unemployed applicants. It is set to null and an indicator `DAYS_EMPLOYED_ANOMALY` is added, because the sentinel group has a different default rate and the information should not be lost.
- `CODE_GENDER = XNA` is set to null. Gender is excluded from model features and kept only for the fairness audit.

Other anomalies (placeholder categories, extreme incomes, implausible bureau end dates) are reported in `reports/data_quality.md` with how each is handled.

## Leakage controls

The dataset has no calendar dates. All behavioral tables express time as an offset relative to the application: negative values are before the application, positive values after it. A feature built from a positive offset would use information that did not exist when the decision was made.

`sql/02_valid_views.sql` defines filtered views that keep only rows with every relevant offset at or before zero:

- installments: `DAYS_INSTALMENT <= 0` and `DAYS_ENTRY_PAYMENT <= 0`
- bureau: `DAYS_CREDIT <= 0`, `DAYS_ENDDATE_FACT <= 0`, `DAYS_CREDIT_UPDATE <= 0` (null end dates are kept: an open credit has no actual end date)
- bureau balance: `MONTHS_BALANCE <= 0` and the parent bureau record must survive the bureau filter
- previous applications: `DAYS_DECISION <= 0`; the five schedule dates (`DAYS_FIRST_DRAWING`, `DAYS_FIRST_DUE`, `DAYS_LAST_DUE_1ST_VERSION`, `DAYS_LAST_DUE`, `DAYS_TERMINATION`) are set to null when positive, which covers both planned dates after the application and the 365243 placeholder for events that never happened
- POS and cash balances, credit card balances: `MONTHS_BALANCE <= 0`

`DAYS_CREDIT_ENDDATE` (the scheduled end date of an open credit) is legitimately positive at application time but is not used by any feature.

Feature queries may only read the filtered views. `tests/test_leakage_sql.py` enforces this three ways: a static check of every relation each feature query reads, a check that the filter conditions exist, and a synthetic test that inserts future-dated rows and asserts the features are identical to features computed with those rows removed by hand (plus the converse: without the filter the features change). Rows dropped per table are reported in `reports/feature_build.md` and `reports/data_quality.md`.

A separate leakage audit (`reports/leakage_audit.md`) computes single-feature AUC on the training split and flags anything above the review threshold of 0.75 for manual review.

## Features

One row per `SK_ID_CURR`, written to `data/processed/features.parquet`, with each feature tagged by group in `features_manifest.json`.

**Application.** Raw fields after cleaning (gender excluded) plus credit to income, annuity to income, annuity to credit, credit to goods price, income per family member, age in years, employment years, employment to age, counts of submitted documents, contact flags and address mismatches, and the mean, min, max, standard deviation and product of the three external scores.

**Installments.** Payments are first rolled up to one row per installment (`SK_ID_PREV`, version, number), because a single installment can be paid in several transactions. Then:

- days past due = `greatest(last payment day - due day, 0)`
- payment ratio = amount paid / amount due
- underpayment = `greatest(amount due - amount paid, 0)`

For windows of the last 3, 6, 12 and 24 months (`DAYS_INSTALMENT >= -90, -180, -365, -730`) and the full history: installment count, late count, late share, max DPD, mean DPD, mean payment ratio, total underpayment and count of installments more than 30 DPD. Recency: days since the last late installment and since the last installment.

**Bureau.** Active, closed and bad-debt counts, active share, total and max current overdue, max historical overdue, max days overdue, debt to credit ratio (all credits and active only), active credit, debt and limit sums, number of distinct credit types, days since the most recent and the oldest bureau credit, credits opened in the last year and prolongation count.

**Previous applications.** Counts of applications, approvals, refusals and cancellations, refused share, applications and refusals in the last year, days since the last decision, approval and refusal, mean requested amount, mean and max approved credit and annuity, credit to requested amount (mean and min), down payment rate, mean term, payback ratio (term x annuity / credit, a proxy for the price of the loan), shares of high-yield, revolving and new-client applications, approved loans still open at the application date, and the mean gap between the actual and the originally planned last due date.

**POS and cash balances.** Month count, max DPD and max DPD with tolerance (overall and last 12 months), share of months with any DPD, months since the last record and the last DPD month, and from each loan's latest month: loan, active and completed counts, remaining installments on active loans and the remaining share of their term.

**Credit card balances.** Card and month counts, utilization (balance / limit) mean, max and over the last 3 and 12 months, max and recent balance, max limit, ATM drawings, drawing count, payment to minimum payment ratio, share of months paid below the minimum, max DPD (overall, with tolerance and last 12 months), share of months with DPD and months since the last record.

**Bureau balance.** For the last 6, 12 and 24 months, the share of months in each status (0 to 5 are DPD buckets, C closed, X unknown). Worst bucket ever and in the last 12 months. Roll rate: the share of consecutive month pairs where the bucket got worse, with C treated as bucket 0 and X as unknown.

## Validation design

A stratified 80/20 split creates the holdout once (seed 42). The holdout is never used for tuning, feature selection, calibrator fitting or model selection. Everything else uses stratified 5-fold CV on the 80% training portion, with fold assignments stored in `data/processed/split.parquet` so every model sees the same folds.

The best tree model and the best LightGBM variant are chosen by mean CV AUC, not by holdout results.

## Models

**Legacy baseline.** Logistic regression on a short list of raw application fields (amounts, age, employment, children, region rating and the three external scores), median imputation and standardization. It stands in for the model currently in production.

**Scorecard.** A WoE transformer written as an sklearn-style estimator:

1. Numeric features are pre-binned into 20 quantile bins. Adjacent bins are merged until the event rate is monotonic, bins under 5% of the population are merged into their closest neighbor, and monotonicity is enforced again. Both directions are tried and the one with the higher information value is kept. Missing values get their own bin.
2. Categorical features keep categories with at least 5% share; smaller ones are pooled into OTHER, which also receives unseen categories.
3. WoE = ln(share of goods / share of bads) with 0.5 smoothing, so higher WoE means lower risk. IV = sum over bins of (share of goods - share of bads) x WoE.
4. Features with IV below 0.02 are dropped. The rest are ranked by IV and added greedily, skipping any whose WoE correlates above 0.8 with a feature already selected, up to 40 features.

The binner is fit inside every CV fold on that fold's training rows only, so out-of-fold predictions never see the rows they score. The L2 penalty C is chosen from {0.01, 0.1, 1} by mean CV AUC. The final scorecard is refit on the full training portion.

Points use the standard scaling: factor = 20 / ln 2, offset = 600 - factor x ln 50, so a score of 600 means good:bad odds of 50:1 and every 20 points doubles the odds. Each attribute's points are offset / n - factor x (coefficient x WoE + intercept / n), and the attribute points sum exactly to the total score. The table is exported to `reports/scorecard.csv`.

**Gradient boosting.** LightGBM and XGBoost on all engineered features, with native categorical handling. Each is tuned with Optuna: TPE sampler (seed 42), 40 trials, objective = mean AUC over the 5 folds, early stopping (150 rounds) inside each fold on that fold's validation rows, and a median pruner that stops trials whose running mean AUC falls behind after the second fold. The tuned parameters are refit across the 5 folds; the deployed model averages the 5 fold models in log-odds space, which keeps SHAP values additive.

Early stopping on the validation fold makes CV AUC slightly optimistic, which is one reason the holdout exists.

**Monotonic LightGBM.** The tuned LightGBM refit with monotonic constraints only where the direction is not in doubt: the external scores and their aggregates decrease risk; installment late counts, max DPD and counts of installments over 30 DPD, POS and credit card max DPD, and the number of refused previous applications increase risk. The AUC cost of the constraints is reported from CV and the holdout.

## Class imbalance

Defaults are a small minority of applicants (the exact rate is in `reports/data_summary.json`). Three options are compared on out-of-fold predictions from the tuned LightGBM: no weighting, `scale_pos_weight = negatives / positives`, and balanced class weights. Reweighting shifts the model's probabilities toward 50/50, which damages calibration (Brier, log loss, ECE, mean predicted vs observed rate), while AUC barely moves because reweighting mostly rescales the scores rather than reordering applicants. The results are in `reports/imbalance.md`.

SMOTE is not used. Synthetic minority points are made by interpolating between neighbors in feature space, which creates applicants that do not exist: mixed category codes, impossible combinations of counts and ratios, and interpolated values inside sentinel or missing-value regions. That distorts the joint distribution of tabular credit data, and because the model is trained on a fake base rate its probabilities stop meaning what a PD should mean. Gradient boosting with no weighting plus explicit calibration handles imbalance without either problem.

## Calibration

For each model, isotonic regression and Platt scaling (logistic regression on the logit of the score) are compared with cross-fitting over the out-of-fold predictions: each fold's calibrator is fit on the other four folds' OOF predictions and scored by Brier on the held-out fold. The method with the lower mean Brier is refit on all OOF predictions and applied to the holdout scores. Calibrators are saved inside each model bundle in `models/`.

## Metrics

- ROC-AUC, Gini = 2 x AUC - 1, and KS (max distance between the cumulative score distributions of defaulters and non-defaulters) measure ranking. These use the raw model score.
- PR-AUC (average precision) measures ranking with emphasis on the minority class.
- Brier score, log loss and expected calibration error measure probability quality. They use the calibrated probability. ECE uses 10 equal-frequency bins because PDs are concentrated near zero and equal-width bins would put almost everyone in the first bin.
- Approval rate at a fixed bad rate: applicants are approved from lowest to highest risk until the bad rate among the approved reaches 3%, 5% or 8%.

Uncertainty comes from a paired bootstrap on the holdout: 1000 resamples drawn once and shared by every model, which gives 95% percentile intervals for AUC and KS and for the AUC difference of each model against the legacy baseline.

The cost analysis uses a margin per good loan and a loss given default, both as a share of the loan amount, and evaluates three LGD values. These are **illustrative assumptions, not data**. It reports the best achievable profit per applicant over all cut-offs and the profit of a policy that approves applicants whose calibrated PD is below the break-even PD, margin / (margin + LGD). The second number depends on calibration, the first does not.

## Explainability and monitoring

SHAP values come from TreeExplainer on each of the 5 fold models of the best LightGBM, averaged. Stability is the Spearman rank correlation of mean absolute SHAP between every pair of fold models. Permutation importance (drop in holdout AUC when a feature is shuffled) is computed for every feature as a cross-check, and features whose ranks differ by more than 10 places within the top 20, or whose permutation importance is not positive, are flagged.

PSI compares the training split with the holdout for the model score (training side uses out-of-fold scores) and the top 20 features, with the usual thresholds of 0.1 and 0.25. Because the holdout is a random split, PSI here confirms the split and sets up the monitoring template rather than testing drift over time.

The fairness audit reports, by gender and by age band on the holdout: AUC, observed default rate, mean predicted PD, and approval rate and bad rate among the approved at the single cut-off that gives a 5% bad rate overall.

## Scoring

`python -m credit_risk.score <SK_ID_CURR>` returns the calibrated PD from the best LightGBM, the scorecard points, a risk band from the PD bands in `config.yaml`, and the top 5 reason codes: the features with the largest absolute SHAP contribution, their value, and whether they push risk up or down.

`make serve` exposes the same function over HTTP with FastAPI (`GET /applicants/{id}/score`, `POST /score` for a batch, `GET /health`, OpenAPI docs at `/docs`), and the Dockerfile packages that service. Scoring is by applicant ID because the features need the applicant's full bureau and repayment history, which lives in the feature store built by `make features`.

## Tuning budget and reproducibility

40 Optuna trials per boosting library, 5 folds per trial, up to 4000 boosting rounds with early stopping. Tuning results are cached in `models/tuning_<library>.json` keyed by the feature list and tuning settings; `make train` reuses the cache when nothing changed and `python -m credit_risk.train --retune` forces a fresh search. Seeds are fixed at 42 for the split, folds, Optuna, LightGBM (deterministic mode), XGBoost, the bootstrap and SHAP sampling.

`make quick` runs training, evaluation, ablation and explainability with the `quick` profile from `config.yaml` (4 trials, fewer boosting rounds, 200 bootstrap resamples, smaller SHAP and permutation samples) and writes to `build/quick/` so the published results in `reports/` are never overwritten. It is for checking that the pipeline works end to end, not for the reported numbers.
