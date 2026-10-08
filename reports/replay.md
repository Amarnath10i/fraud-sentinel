# Online replay of the deployment period

Model `lgbm-20261008-215822`; state warmed with all events before 2020-06-21 12:14 UTC, then 558,132 events (555,719 transactions, 2,413 chargebacks) streamed in time order through the in-process scorer.

## End-to-end parity (served vs offline probability)

- transactions compared: 555,719
- max |p_online - p_offline|: 0.00e+00
- decisions that differ: 0

## Latency of the online path (feature engine + model + calibration + policy + reasons)

| p50 | p95 | p99 | max | throughput |
|---|---|---|---|---|
| 0.469 ms | 0.735 ms | 1.050 ms | 17.5 ms | 1,797 txn/s (single thread) |

## Decisions

- approve: 553,325
- review: 327
- decline: 2,067

Prediction log: 555,719 rows in PostgreSQL, 0 dropped.

## Over HTTP (first 2,000 transactions, FastAPI + uvicorn, localhost)

| p50 | p95 | p99 |
|---|---|---|
| 1.90 ms | 2.43 ms | 2.79 ms |
