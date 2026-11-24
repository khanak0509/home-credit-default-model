# Class imbalance study

LightGBM with the tuned parameters, 5-fold out-of-fold predictions on the training split. scale_pos_weight = negatives / positives = 11.387. Balanced class weights give each class half the total weight.

| variant | ROC-AUC | PR-AUC | Brier | log loss | ECE | mean predicted | observed rate | Brier after recalibration |
|---|---|---|---|---|---|---|---|---|
| no_weighting | 0.7925 | 0.2849 | 0.0656 | 0.2354 | 0.0027 | 0.0798 | 0.0807 | 0.0656 |
| scale_pos_weight | 0.7896 | 0.2826 | 0.1644 | 0.4984 | 0.2785 | 0.3592 | 0.0807 | 0.0657 |
| balanced_class_weight | 0.7904 | 0.2830 | 0.1635 | 0.4967 | 0.2773 | 0.3580 | 0.0807 | 0.0657 |

scale_pos_weight: ROC-AUC change -0.0029, Brier 0.0656 to 0.1644, mean predicted probability 0.3592 against an observed rate of 0.0807.

balanced_class_weight: ROC-AUC change -0.0021, Brier 0.0656 to 0.1635, mean predicted probability 0.3580 against an observed rate of 0.0807.
