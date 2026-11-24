# Fairness audit

Model: lightgbm. Holdout set only. CODE_GENDER is not a model feature; it is used here for auditing only. Operating point: a single score cut-off chosen so that the bad rate among all approved holdout applicants is at most 5% (approval rate 86.9%, realized bad rate 0.0500).

## By CODE_GENDER

| group | applicants | observed default rate | mean predicted PD | ROC-AUC | approval rate | bad rate among approved |
|---|---|---|---|---|---|---|
| F | 40,561 | 0.0699 | 0.0731 | 0.7900 | 88.8% | 0.0455 |
| M | 20,940 | 0.1017 | 0.0931 | 0.7929 | 83.1% | 0.0593 |

Applicants with unknown gender (XNA in the raw data) in the holdout: 2.

## By age band

| group | applicants | observed default rate | mean predicted PD | ROC-AUC | approval rate | bad rate among approved |
|---|---|---|---|---|---|---|
| under 30 | 8,905 | 0.1130 | 0.1127 | 0.7686 | 77.3% | 0.0647 |
| 30-40 | 16,487 | 0.0957 | 0.0952 | 0.7991 | 82.5% | 0.0548 |
| 40-50 | 15,303 | 0.0758 | 0.0759 | 0.8024 | 88.1% | 0.0467 |
| 50-60 | 13,617 | 0.0635 | 0.0609 | 0.7810 | 92.4% | 0.0455 |
| 60+ | 7,191 | 0.0496 | 0.0486 | 0.7499 | 95.7% | 0.0404 |

## Observed gaps

Gender: approval rate ranges from 83.1% to 88.8% (gap 5.8 points); observed default rate ranges from 0.0699 to 0.1017; ROC-AUC ranges from 0.7900 to 0.7929.

Age band: approval rate ranges from 77.3% to 95.7% (gap 18.4 points); observed default rate ranges from 0.0496 to 0.1130; ROC-AUC ranges from 0.7499 to 0.8024.

Across all groups above, the Spearman rank correlation between observed default rate and approval rate is -0.964; a value of -1 means groups with higher observed default rates are always approved less often under the single cut-off.

This audit reports the gaps only; it does not test whether they are justified, and it does not cover other protected attributes.
