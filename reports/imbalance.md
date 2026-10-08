# Class imbalance strategies (LightGBM, test period)

Training data: 977,052 rows, 5,562 fraud (0.57%).

## Ranking (95% card-level bootstrap intervals)

| strategy                             | PR-AUC               | ROC-AUC              | Recall@0.5%          | $Recall@0.5%         | Prec@Recall80        |
|:-------------------------------------|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|
| none                                 | 0.9903 (0.987-0.994) | 0.9999 (1.000-1.000) | 0.9883 (0.980-0.993) | 0.9973 (0.992-0.999) | 1.0000 (0.999-1.000) |
| class weight sqrt (13x)              | 0.9911 (0.987-0.995) | 0.9999 (1.000-1.000) | 0.9893 (0.982-0.995) | 0.9972 (0.993-1.000) | 1.0000 (1.000-1.000) |
| class weight balanced (175x)         | 0.9893 (0.986-0.993) | 0.9999 (1.000-1.000) | 0.9851 (0.978-0.992) | 0.9952 (0.991-0.999) | 1.0000 (1.000-1.000) |
| undersample negatives to 10%         | 0.9885 (0.985-0.992) | 0.9999 (1.000-1.000) | 0.9874 (0.979-0.993) | 0.9979 (0.994-0.999) | 0.9988 (0.996-1.000) |
| undersample + prior-shift correction | 0.9885 (0.985-0.992) | 0.9999 (1.000-1.000) | 0.9874 (0.979-0.993) | 0.9979 (0.994-0.999) | 0.9988 (0.996-1.000) |
| SMOTE to 1:20                        | 0.9899 (0.986-0.993) | 0.9998 (1.000-1.000) | 0.9869 (0.980-0.994) | 0.9972 (0.995-0.999) | 1.0000 (1.000-1.000) |
| focal loss (gamma=2, alpha=0.25)     | 0.8712 (0.853-0.890) | 0.9293 (0.917-0.940) | 0.8918 (0.876-0.908) | 0.9351 (0.917-0.948) | 0.9246 (0.905-0.944) |

## Calibration

The classes are almost separable here, so nearly every prediction sits close to 0 or 1 and ECE/Brier barely move between strategies. Reweighting and resampling show up where they should: in the probabilities given to legitimate transactions (true value ~0) and in log loss, which punishes confident mistakes.

| strategy                             |   fit (s) |   rounds |   median p, legit |   mean p, legit |   log loss raw |   log loss after isotonic |   ECE raw |   ECE after isotonic |
|:-------------------------------------|----------:|---------:|------------------:|----------------:|---------------:|--------------------------:|----------:|---------------------:|
| none                                 |        58 |      827 |          7.01e-08 |        3.19e-05 |        0.00148 |                   0.00221 |    0.0003 |               0.0001 |
| class weight sqrt (13x)              |        84 |     1318 |          8.26e-09 |        4.51e-05 |        0.00132 |                   0.00176 |    0.0002 |               0.0002 |
| class weight balanced (175x)         |       159 |     2603 |          3.12e-10 |        6.36e-05 |        0.00145 |                   0.00213 |    0.0001 |               0.0001 |
| undersample negatives to 10%         |        14 |      794 |          1.03e-06 |        0.000291 |        0.00118 |                   0.0022  |    0.0001 |               0.0002 |
| undersample + prior-shift correction |        14 |      794 |          1.03e-07 |        0.000105 |        0.00123 |                   0.0022  |    0.0002 |               0.0002 |
| SMOTE to 1:20                        |        67 |      936 |          6.32e-08 |        3.42e-05 |        0.00142 |                   0.00181 |    0.0003 |               0.0001 |
| focal loss (gamma=2, alpha=0.25)     |       510 |     2988 |          3.53e-06 |        0.000793 |        0.01502 |                   0.00514 |    0.0008 |             inf      |
