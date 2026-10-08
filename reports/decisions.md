# Decisions: from probabilities to dollars

Test period: 555,719 transactions, 2,145 frauds worth $1,133,325.

Cost model: approving a fraud loses its amount; a review costs $3 of analyst time plus $1 of customer friction and stops 95% of frauds; a false decline costs $10 plus 2% of the amount in lost revenue.

## Are the probabilities trustworthy? (test period)

| scores                    |   median p, legit |   mean p, fraud |    ECE |   Brier |   log loss |
|:--------------------------|------------------:|----------------:|-------:|--------:|-----------:|
| LightGBM                  |           7e-08   |           0.914 | 0.0003 | 0.00028 |     0.0015 |
| LightGBM + isotonic       |           0       |           0.936 | 0.0001 | 0.00022 |     0.0022 |
| LightGBM balanced weights |           3.1e-10 |           0.949 | 0.0001 | 0.00021 |     0.0014 |
| balanced + Platt          |           4.3e-06 |           0.943 | 0.0001 | 0.0002  |     0.001  |
| balanced + isotonic       |           0       |           0.941 | 0.0001 | 0.00021 |     0.0021 |

## Policies (total cost with 95% card-level bootstrap interval)

| policy                                        | total cost $                    |   fraud $ approved |   reviews |   max reviews/day |   declines |   false declines |   frauds approved |
|:----------------------------------------------|:--------------------------------|-------------------:|----------:|------------------:|-----------:|-----------------:|------------------:|
| approve everything (no model)                 | 1,133,325 (1,004,102-1,266,433) |          1,133,325 |         0 |                 0 |          0 |                0 |              2145 |
| decline if p >= 0.5                           | 47,814 (37,289-61,761)          |             47,549 |         0 |                 0 |       1987 |               12 |               170 |
| decline if p >= 0.189 (best F1 on validation) | 31,860 (23,095-40,831)          |             31,279 |         0 |                 0 |       2051 |               26 |               120 |
| review top 0.5% of scores                     | 68,580 (60,869-76,502)          |              3,072 |      2779 |                46 |          0 |                0 |                25 |
| Bayes, balanced-weight scores (miscalibrated) | 13,376 (7,871-18,826)           |             12,003 |        79 |                 6 |       2070 |               29 |                80 |
| Bayes, balanced-weight scores + isotonic      | 6,963 (4,185-9,556)             |              3,928 |       362 |                10 |       2062 |               32 |                56 |
| Bayes, LightGBM + isotonic                    | 7,145 (4,199-10,603)            |              3,916 |       380 |                 7 |       2034 |               24 |                57 |
| Bayes + isotonic, max 4 reviews/day           | 7,204 (4,279-10,680)            |              4,074 |       356 |                 4 |       2042 |               27 |                60 |
| conformal sets (alpha fraud 5%, legit 0.2%)   | 9,705 (6,238-14,233)            |              5,356 |       533 |                13 |       1994 |               14 |                34 |

## Conformal guarantee check

Calibrated on validation (fraud quantile 0.2000, legit quantile 0.0273). On test, 1.59% of frauds were auto-approved (target <= 5%) and 0.003% of legitimate transactions were declined (target <= 0.2%). The guarantee assumes the test period is exchangeable with the calibration period; drift between them is the main way it can fail.

## Segment audit (Bayes, LightGBM + isotonic)

Gender and age are model inputs here because the dataset provides them. A real deployment would have to justify that legally; this table is the check that would inform the decision: who absorbs the friction, and whose fraud gets caught.

| segment   |   transactions | fraud rate   | legit reviewed or declined   | legit declined   | fraud stopped or reviewed   |
|:----------|---------------:|:-------------|:-----------------------------|:-----------------|:----------------------------|
| gender F  |         304886 | 0.382%       | 0.066%                       | 0.0053%          | 96.48%                      |
| gender M  |         250833 | 0.391%       | 0.050%                       | 0.0032%          | 98.37%                      |
| age < 30  |          93995 | 0.337%       | 0.062%                       | 0.0075%          | 100.00%                     |
| age 30-49 |         256777 | 0.331%       | 0.070%                       | 0.0039%          | 96.11%                      |
| age 50-69 |         142689 | 0.480%       | 0.046%                       | 0.0021%          | 97.37%                      |
| age 70+   |          62258 | 0.472%       | 0.040%                       | 0.0065%          | 97.96%                      |
