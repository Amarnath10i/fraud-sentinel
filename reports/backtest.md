# Retraining policy backtest (Jul 2019 - Dec 2020)

Every month is scored by the model each policy would have had in production at the start of that month; training only ever uses matured labels (60 days) as reported by that date. LightGBM with fixed hyperparameters and 400 rounds for every retrain.

## Summary

| policy                    |   tree retrains |   mean_pr_auc |   worst_month_pr_auc |   mean_recall |
|:--------------------------|----------------:|--------------:|---------------------:|--------------:|
| static                    |               0 |        0.9733 |               0.949  |        0.8827 |
| monthly expanding         |              17 |        0.9878 |               0.9604 |        0.8948 |
| monthly sliding 6m        |              17 |        0.9835 |               0.9604 |        0.8908 |
| drift-triggered           |               3 |        0.9845 |               0.9572 |        0.8926 |
| static GBDT + online FTRL |               0 |        0.9892 |               0.9678 |        0.8975 |

## PR-AUC by month

| month   |   static |   monthly expanding |   monthly sliding 6m |   drift-triggered |   static GBDT + online FTRL |
|:--------|---------:|--------------------:|---------------------:|------------------:|----------------------------:|
| 2019-07 |   0.9604 |              0.9604 |               0.9604 |            0.9604 |                      0.9678 |
| 2019-08 |   0.9717 |              0.9815 |               0.9815 |            0.9717 |                      0.9850 |
| 2019-09 |   0.9572 |              0.9624 |               0.9624 |            0.9572 |                      0.9689 |
| 2019-10 |   0.9897 |              0.9941 |               0.9929 |            0.9897 |                      0.9961 |
| 2019-11 |   0.9728 |              0.9846 |               0.9801 |            0.9728 |                      0.9821 |
| 2019-12 |   0.9698 |              0.9850 |               0.9799 |            0.9698 |                      0.9872 |
| 2020-01 |   0.9803 |              0.9934 |               0.9891 |            0.9934 |                      0.9937 |
| 2020-02 |   0.9848 |              0.9957 |               0.9925 |            0.9961 |                      0.9953 |
| 2020-03 |   0.9869 |              0.9938 |               0.9899 |            0.9938 |                      0.9967 |
| 2020-04 |   0.9772 |              0.9885 |               0.9834 |            0.9890 |                      0.9910 |
| 2020-05 |   0.9884 |              0.9978 |               0.9952 |            0.9964 |                      0.9989 |
| 2020-06 |   0.9730 |              0.9901 |               0.9828 |            0.9861 |                      0.9883 |
| 2020-07 |   0.9490 |              0.9830 |               0.9691 |            0.9760 |                      0.9882 |
| 2020-08 |   0.9744 |              0.9925 |               0.9833 |            0.9925 |                      0.9931 |
| 2020-09 |   0.9759 |              0.9944 |               0.9902 |            0.9936 |                      0.9933 |
| 2020-10 |   0.9789 |              0.9953 |               0.9897 |            0.9944 |                      0.9944 |
| 2020-11 |   0.9785 |              0.9934 |               0.9908 |            0.9936 |                      0.9906 |
| 2020-12 |   0.9508 |              0.9948 |               0.9889 |            0.9948 |                      0.9946 |

## Drift-triggered retrains

- 2020-01: retrained (score PSI 0.009, top-feature PSI 0.578 (`card__sum_amount_24h`), alert-rate ratio 0.85)
- 2020-08: retrained (score PSI 0.009, top-feature PSI 0.026 (`card__max_amount_24h`), alert-rate ratio 0.55)
- 2020-12: retrained (score PSI 0.022, top-feature PSI 0.002 (`card__max_amount_24h`), alert-rate ratio 0.58)
