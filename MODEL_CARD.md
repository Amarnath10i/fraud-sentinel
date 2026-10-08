# Model card: fraud-sentinel production model

## Overview

| | |
|---|---|
| Task | Probability that a card authorization is fraudulent, at authorization time |
| Model | LightGBM (binary log loss) + isotonic calibration |
| Inputs | 39 features: 9 row features, 23 point-in-time aggregates, 7 derived ratios/rates ([spec](src/sentinel/features/spec.py)) |
| Output | Calibrated probability, an action (approve / review / decline) from a cost model, and up to 3 reason codes (TreeSHAP) for review/decline |
| Feature spec | versioned by content hash; the scorer refuses a model trained on a different spec |
| Registry | `model_registry` table in PostgreSQL; artifacts under `artifacts/models/<version>/` |

## Intended use

Ranking and routing card transactions for fraud review in a simulated card
portfolio. Not intended for decisions about individuals beyond the
transaction, and not validated on real cardholder data.

## Training data

Sparkov synthetic credit-card transactions (Kaggle `kartik2112/fraud-detection`):
1.85M transactions, 999 cards, 693 merchants, Jan 2019 - Dec 2020.

* Training: Jan 2019 - Jan 2020 (977k transactions).
* Validation (early stopping, calibration, policy fitting): Feb 1 - Apr 21 2020 (169k).
* Labels: chargebacks reported before the deploy date (2020-06-21), with a
  60-day maturity rule for legitimate labels. Ground truth is never used for
  training.

## Evaluation

Held-out deployment period Jun 21 - Dec 31 2020 (556k transactions, 2,145
frauds), scored against ground truth, 95% card-level cluster-bootstrap
intervals.

| | |
|---|---|
| PR-AUC (tuned model) | 0.992 (0.988-0.994) |
| Recall at a 0.5% alert rate | 0.990 |
| Total cost under the decision policy (decisions report, same pipeline with default LightGBM parameters) | $7,145 vs $1,133,325 approving everything |
| Online vs offline probability, all test transactions | identical (max difference 0.0) |
| Online latency, full path | p50 0.47 ms, p99 1.05 ms (single thread) |

Hyperparameters: [`configs/lightgbm_params.json`](configs/lightgbm_params.json),
from Optuna on rolling-origin CV. See [model comparison](reports/model_comparison.md),
[tuning](reports/tuning.md), [decisions](reports/decisions.md),
[replay](reports/replay.md) and [backtest](reports/backtest.md).

## Ethical considerations

* Gender and age are model inputs because the dataset provides them. Using
  them for credit/fraud decisions may be restricted by law; the
  [segment audit](reports/decisions.md) reports friction and fraud capture
  by gender and age band so the trade-off is visible.
* False declines fall on legitimate customers; the cost model prices them
  explicitly and the conformal policy bounds the declined share of
  legitimate transactions.

## Limitations

* Synthetic data: fraud is easier to separate than in real portfolios, so
  absolute metrics are optimistic.
* Chargeback delays and business costs are simulated assumptions.
* Declined transactions never receive labels in production; retraining on
  approved-only outcomes biases future models (not simulated here).
* The calibrator is fitted on Feb-Apr 2020; calibration degrades if the fraud
  base rate drifts (monitored with score PSI, see the backtest).
