# Class imbalance strategies (LightGBM, test period)

Training data: 977,052 rows, 5,562 fraud (0.57%).

## Ranking (95% card-level bootstrap intervals)

| strategy                             | PR-AUC               | ROC-AUC              | Recall@0.5%          | $Recall@0.5%         | Prec@Recall80        |
|:-------------------------------------|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|
| none                                 | 0.9903 (0.987-0.994) | 0.9999 (1.000-1.000) | 0.9883 (0.980-0.993) | 0.9973 (0.992-0.999) | 1.0000 (0.999-1.000) |
| class weight sqrt (13x)              | 0.9911 (0.987-0.995) | 0.9999 (1.000-1.000) | 0.9893 (0.982-0.995) | 0.9972 (0.993-1.000) | 1.0000 (1.000-1.000) |
| class weight balanced (175x)         | 0.9893 (0.986-0.993) | 0.9998 (1.000-1.000) | 0.9851 (0.978-0.991) | 0.9952 (0.991-0.999) | 1.0000 (1.000-1.000) |
| undersample negatives to 10%         | 0.9885 (0.985-0.992) | 0.9999 (1.000-1.000) | 0.9874 (0.979-0.993) | 0.9979 (0.994-0.999) | 0.9988 (0.996-1.000) |
| undersample + prior-shift correction | 0.9885 (0.985-0.992) | 0.9999 (1.000-1.000) | 0.9874 (0.979-0.993) | 0.9979 (0.994-0.999) | 0.9988 (0.996-1.000) |
| SMOTE to 1:20                        | 0.9899 (0.986-0.993) | 0.9998 (1.000-1.000) | 0.9869 (0.980-0.994) | 0.9972 (0.995-0.999) | 1.0000 (1.000-1.000) |
| focal loss (gamma=2, alpha=0.25)     | 0.8596 (0.841-0.879) | 0.9169 (0.904-0.928) | 0.8793 (0.861-0.895) | 0.9247 (0.908-0.938) | 0.9184 (0.898-0.935) |

## Calibration

The classes are almost separable here, so nearly every prediction sits close to 0 or 1 and ECE/Brier barely move between strategies. Reweighting and resampling show up where they should: in the probabilities given to legitimate transactions (true value ~0) and in log loss, which punishes confident mistakes.

| strategy                             |   fit (s) |   rounds |   median p, legit |   mean p, legit |   log loss raw |   log loss after isotonic |   ECE raw |   ECE after isotonic |
|:-------------------------------------|----------:|---------:|------------------:|----------------:|---------------:|--------------------------:|----------:|---------------------:|
| none                                 |        87 |      827 |          7.01e-08 |        3.19e-05 |        0.00148 |                   0.00221 |    0.0003 |               0.0001 |
| class weight sqrt (13x)              |       146 |     1318 |          8.26e-09 |        4.51e-05 |        0.00132 |                   0.00176 |    0.0002 |               0.0002 |
| class weight balanced (175x)         |       164 |     2973 |          2.44e-10 |        6.37e-05 |        0.00146 |                   0.00218 |    0.0001 |               0.0001 |
| undersample negatives to 10%         |        47 |      794 |          1.03e-06 |        0.000291 |        0.00118 |                   0.0022  |    0.0001 |               0.0002 |
| undersample + prior-shift correction |        47 |      794 |          1.03e-07 |        0.000105 |        0.00123 |                   0.0022  |    0.0002 |               0.0002 |
| SMOTE to 1:20                        |       133 |      936 |          6.32e-08 |        3.42e-05 |        0.00142 |                   0.00181 |    0.0003 |               0.0001 |
| focal loss (gamma=2, alpha=0.25)     |       218 |     1281 |          5.18e-05 |        0.00168  |        0.01694 |                   0.00528 |    0.0018 |               0.0002 |
