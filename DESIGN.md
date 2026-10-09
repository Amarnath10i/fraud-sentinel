# Design notes

Decisions behind fraud-sentinel, the alternatives considered, and what would
change at production scale. Results live in [`reports/`](reports/).

## 1. Problem framing

Card fraud scoring at authorization time. The output is not a label but an
action (approve, send to a human reviewer, decline), and the business cost
of each mistake depends on the amount. So:

* the model's job is a **calibrated probability**, not a ranking alone;
* the decision layer turns probability + amount into the cheapest action
  under an explicit cost model;
* evaluation reports ranking quality, calibration and dollars.

## 2. Data model and time

The flat Kaggle CSVs are normalized into `customers`, `merchants`,
`transactions` (monthly range partitions), `chargebacks` and `ground_truth`
([`sql/schema.sql`](sql/schema.sql)).

The most important modelling decision is that **labels are events, not
columns**. In production nobody knows a transaction is fraudulent when it
happens; a chargeback arrives days or weeks later. The simulator gives every
fraud a log-normal reporting delay (median 12 days, about 1% after 60 days).
Consequences that the code enforces everywhere:

* a training set "as of time T" labels a transaction as fraud only if its
  chargeback was reported before T;
* a transaction counts as legitimate only after a 60-day maturity window, so
  the most recent two months can never be used for training;
* `ground_truth` is read only by evaluation code.

The main experiment deploys on 2020-06-21 (where the original Kaggle test
file starts): train on Jan 2019 - Jan 2020, validate on Feb - Apr 21 2020
(labels as known on deploy day), test on Jun 21 - Dec 31 2020 against ground
truth.

## 3. Features: one definition, two engines

Training-serving skew is the classic way a fraud model looks great offline and
underperforms in production. Features are declared once
([`features/spec.py`](src/sentinel/features/spec.py)) as aggregations over two
event streams (transactions, chargebacks) keyed by an entity (card, card x
category, merchant, category, global), and compiled twice:

| | Offline (training) | Online (serving) |
|---|---|---|
| Code | [`features/sql.py`](src/sentinel/features/sql.py) | [`features/online.py`](src/sentinel/features/online.py) |
| Runs in | PostgreSQL window functions | Python state machine |
| Windows | `RANGE BETWEEN <span> PRECEDING AND '1 microsecond' PRECEDING` | deques evicted as time advances |
| Cost | 71 s for 1.86M events | ~25k events/s, O(1) amortized per event |

**Point-in-time rule:** a feature at time t sees only events strictly before
t; events sharing a timestamp never see each other (33k Sparkov events share
a timestamp). Online, this is done by buffering state updates until the
clock moves past the current timestamp.

**Why that frame and not `EXCLUDE GROUP`:** both give the same answer, but
`EXCLUDE` disables PostgreSQL's moving-aggregate optimization and re-aggregates
every frame, which is quadratic on the 30-day merchant and global windows.
Ending the frame 1 microsecond early keeps COUNT and SUM(numeric) on their
inverse-transition fast path.

**Exactness:** money is integer cents in the engine and NUMERIC in SQL, so
sliding sums never accumulate floating error; variance comes from exact
integer moments. Window functions cannot do `COUNT(DISTINCT)`, so distinct
counts use a correlated subquery on the `(card_id, ts)` index.

**Verified three ways:**

1. every feature equals an O(n²) brute-force definition on adversarial
   synthetic streams (ties, window edges, same-second chargebacks);
2. features never change when future events are deleted;
3. SQL equals streaming for all 26 aggregates x 1.85M transactions
   ([`reports/parity.md`](reports/parity.md)). The parity test caught a real
   bug: PostgreSQL's `GREATEST` ignores NULLs, which turned "no history" into
   a standard deviation of 0.

## 4. Data structures

| Structure | Used for | Cost |
|---|---|---|
| Time-window deque with integer moments | count / sum / mean / std over 1h-30d | O(1) amortized |
| Monotonic deque | max amount in window | O(1) amortized |
| Deque + reference counts | exact distinct merchants / categories in window | O(1) amortized |
| Welford + Chan merge | numerically stable streaming variance | O(1) |
| LRU (hash map + doubly linked list) | bounded per-entity state | O(1) |
| Count-Min Sketch (conservative update) | approximate frequencies in fixed memory | O(depth) |
| Bloom filter | "has this card paid this merchant before?" | O(k) |
| Size-k min-heap | daily analyst review queue | O(log k) |

All are property-tested against brute-force references with Hypothesis.

**Picking the right sketch** ([`reports/sketches.md`](reports/sketches.md)).
28.6% of transactions are a card's first payment to that merchant. Count-Min
answers "how many?" with an additive error of eps x N over the whole stream,
which swamps the zero counts of a long tail: a 1 MB sketch flags only 12% of
first-time pairs. A 0.63 MB Bloom filter answers the actual question,
membership, and catches 99.8%, against 65 MB for the exact dict.

## 5. Models

* **Logistic regression from scratch** - Newton/IRLS with L2 (matches
  scikit-learn to 1e-6 in fewer than 20 iterations) and FTRL-Proximal for
  online learning (per-coordinate rates, exact zeros through L1).
* **Gradient boosting from scratch** - second-order (XGBoost-style gain and
  leaf values), uint8 quantile histograms, histogram subtraction,
  learned default direction for missing values, flat-array trees with
  vectorized traversal, Saabas contributions. Tested to agree with LightGBM
  and with an exhaustive split search.
* **LightGBM** - the production model, tuned with Optuna on rolling-origin
  time-series CV (never random K-fold, which leaks the future).
* **Calibration** - isotonic regression via pool-adjacent-violators and Platt
  scaling, fitted on the validation period; closed-form prior-shift
  correction after undersampling. Isotonic improves ECE but worsens log loss:
  its lowest block outputs exactly 0, and a fraud that lands there is
  penalized without bound. Platt keeps the best log loss.
* **Reproducibility** - LightGBM runs with `deterministic=True`: multithreaded
  histogram sums are otherwise order-dependent, and the same seed produced a
  model whose policy cost moved by 8% between two runs.
* **Losses** - log loss and focal loss (analytic gradient and Hessian,
  finite-difference tested), shared by the scratch GBDT and LightGBM.

## 6. Evaluation

* Headline metrics: PR-AUC, recall and dollar-recall at a 0.5% alert rate,
  precision at 80% recall. ROC-AUC is reported but barely discriminates at
  0.4% prevalence.
* Calibration: Brier, log loss and ECE with equal-mass bins (equal-width bins
  put almost everything in the first bin at this base rate).
* Uncertainty: 95% intervals from a **card-level cluster bootstrap**. Fraud
  arrives in per-card bursts, so row-level resampling would treat one
  compromised card as many independent observations and understate the
  uncertainty. Model comparisons use paired bootstrap differences.

## 7. Decisions

Given calibrated p and amount A (see [`decision.py`](src/sentinel/decision.py)):

    approve: p*A    review: c_r + (1-p)*c_f + p*(1-catch)*A    decline: (1-p)*(c_d + m*A)

The cheapest action is Bayes-optimal **only if p is calibrated**, which is why
calibration is part of the model rather than an afterthought. Two
constrained variants: a per-day review capacity enforced with a top-k heap
ordered by how much each review saves, and class-conditional conformal
prediction sets that bound the share of frauds auto-approved (valid while
the test period stays exchangeable with the calibration period).

## 8. Serving

`POST /v1/score` runs: streaming features → row and derived features (the
same NumPy functions used for training) → LightGBM → isotonic calibration →
Bayes policy → TreeSHAP reason codes (only for review/decline, where
someone will read them). Predictions are logged to PostgreSQL by a batching
background thread that drops and counts rows rather than ever blocking
authorization. Models are versioned bundles registered in `model_registry`;
a partial unique index guarantees a single production model.

Scoring is **idempotent per `txn_id`**. Payment networks retry
authorizations; without deduplication a retry would be counted twice in the
card's velocity features and could push a legitimate customer towards a
decline. Recent decisions live in an LRU keyed by the canonical transaction
id, and a retry returns the original decision without touching the online
state (`"duplicate": true` in the response).

The replay harness streams the whole test period through the online scorer
and checks that every served probability equals the offline one
([`reports/replay.md`](reports/replay.md)).

**Dashboard.** `sentinel serve` also serves a single-page dashboard at `/`
(vanilla JS, no build step) on top of the same scorer: a live replay of the
deployment period with running dollar totals, a decision inspector (reason
codes plus the card's live streaming state), a manual scoring form, the
analyst review queue (resolving a case as fraud posts a chargeback into the
online state), and monitoring (alert-rate drift, backtest, registry).
Ground truth appears only for display; the scorer never sees it.

## 9. Monitoring and retraining

Labels are weeks late, so monitoring is label-free: PSI and KS on the score
and the top features, plus adversarial validation (a classifier that tries
to tell two periods apart).

**PSI is nearly blind on a rare-event score.** About 97% of calibrated
scores are ~0. Reference-quantile bins collapse onto that spike and PSI
reads exactly 0 even when the alert volume halves; with fixed log-spaced
bins it sees the change but only moves by ~0.01, because PSI weights bins
by mass and the mass did not move. The 0.1/0.25 rules of thumb never fire.
The score is therefore monitored by its **alert-rate ratio** (live share of
scores >= 1% over the reference share), with PSI kept for features, where
it works as intended. The dashboard and the drift-triggered retraining
policy both use it. The backtest replays 18 months and compares
static, monthly expanding, monthly sliding, drift-triggered and
GBDT-leaves + online-FTRL policies ([`reports/backtest.md`](reports/backtest.md)).

## 10. What changes at scale

* **State:** one Python process holds all entity state behind a lock. At
  scale, shard by card ID so each card's state lives in one worker (Kafka
  partitions keyed by card), and move merchant, category and global
  aggregates to a shared store (Redis/Flink) or tolerate approximate values.
* **Memory:** per-entity state is LRU-bounded; high-cardinality pair
  membership (card x merchant) would move to Bloom filters, and heavy-hitter
  counts to Count-Min sketches.
* **Offline compute:** the SQL compiler targets PostgreSQL; the same
  definitions would compile to Spark/Flink SQL.
* **Model serving:** LightGBM prediction on one row costs ~0.1 ms of Python
  overhead; a compiled tree runtime (treelite) or the scratch GBDT's
  flat-array format would remove it.
* **Feedback loop:** declined transactions never get labels, so future
  training data is biased towards what the current model approves. The fix
  is a small randomized holdout or inverse-propensity weighting.

## 11. Honest limitations

* Sparkov is synthetic. Fraud patterns are cleaner than in real data, so
  tree models reach very high PR-AUC. The value of this project is in the
  machinery (point-in-time features, label delay, calibration, decisions,
  parity, monitoring), not in the headline number.
* The cost model is illustrative; real costs come from the business.
* Chargeback delays are simulated; their distribution is an assumption.
