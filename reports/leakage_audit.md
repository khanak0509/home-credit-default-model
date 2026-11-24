# Leakage audit

Single-feature ROC-AUC on the training split (holdout excluded). AUC is direction-free (max of AUC and 1 - AUC). Numeric features are scored on their non-null rows; categorical features use out-of-fold default rate per category.

No feature exceeds the review threshold of 0.75.

## Top 15

| rank | feature | group | single-feature AUC | direction | non-null coverage |
|---|---|---|---|---|---|
| 1 | EXT_SOURCE_PRODUCT | application | 0.7385 | higher value, lower risk | 35.6% |
| 2 | EXT_SOURCE_MEAN | application | 0.7165 | higher value, lower risk | 99.9% |
| 3 | EXT_SOURCE_MIN | application | 0.6910 | higher value, lower risk | 99.9% |
| 4 | EXT_SOURCE_MAX | application | 0.6863 | higher value, lower risk | 99.9% |
| 5 | EXT_SOURCE_3 | application | 0.6797 | higher value, lower risk | 80.2% |
| 6 | EXT_SOURCE_1 | application | 0.6659 | higher value, lower risk | 43.7% |
| 7 | EXT_SOURCE_2 | application | 0.6553 | higher value, lower risk | 99.8% |
| 8 | CC_MEAN_UTILIZATION_3M | credit_card | 0.6484 | higher value, higher risk | 21.3% |
| 9 | CC_MEAN_UTILIZATION_12M | credit_card | 0.6391 | higher value, higher risk | 23.3% |
| 10 | CC_MEAN_UTILIZATION | credit_card | 0.6306 | higher value, higher risk | 28.0% |
| 11 | CC_MEAN_DRAWING_COUNT | credit_card | 0.6241 | higher value, higher risk | 28.3% |
| 12 | CC_MAX_UTILIZATION | credit_card | 0.6108 | higher value, higher risk | 28.0% |
| 13 | CC_MEAN_BALANCE_3M | credit_card | 0.6035 | higher value, higher risk | 27.8% |
| 14 | POS_MONTHS_SINCE_LAST_DPD | pos_cash | 0.5977 | higher value, lower risk | 17.8% |
| 15 | BUREAU_DEBT_TO_CREDIT | bureau | 0.5937 | higher value, higher risk | 83.0% |

Features audited: 287.
