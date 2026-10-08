-- fraud-sentinel system of record.
--
-- Design notes
--   * `transactions` only holds what is known at authorization time. Fraud
--     labels arrive later as `chargebacks`, so any query that needs a label has
--     to say *when* it is asking, which makes label leakage a visible mistake
--     instead of a silent one.
--   * `ground_truth` exists only for offline evaluation. Training code never
--     reads it; it learns from chargebacks reported before its cutoff.
--   * `transactions` is range-partitioned by month: training windows and
--     backtests scan a contiguous time range, and retention becomes
--     DROP/DETACH PARTITION instead of a huge DELETE.
--   * Money is NUMERIC(10,2), never floating point.

CREATE TABLE IF NOT EXISTS customers (
    card_id   BIGINT PRIMARY KEY,
    gender    CHAR(1) NOT NULL CHECK (gender IN ('F', 'M')),
    dob       DATE NOT NULL,
    job       TEXT NOT NULL,
    city      TEXT NOT NULL,
    state     CHAR(2) NOT NULL,
    zip       TEXT NOT NULL,
    home_lat  DOUBLE PRECISION NOT NULL,
    home_lon  DOUBLE PRECISION NOT NULL,
    city_pop  INTEGER NOT NULL CHECK (city_pop > 0)
);

CREATE TABLE IF NOT EXISTS merchants (
    merchant_id INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE
);

-- A merchant name can appear under more than one category, so the category
-- belongs to the transaction rather than to the merchant.
CREATE TABLE IF NOT EXISTS transactions (
    txn_id      UUID NOT NULL,
    ts          TIMESTAMPTZ NOT NULL,
    card_id     BIGINT NOT NULL REFERENCES customers (card_id),
    merchant_id INTEGER NOT NULL REFERENCES merchants (merchant_id),
    category    TEXT NOT NULL,
    amount      NUMERIC(10, 2) NOT NULL CHECK (amount > 0),
    merch_lat   DOUBLE PRECISION NOT NULL,
    merch_lon   DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (txn_id, ts)
) PARTITION BY RANGE (ts);

CREATE INDEX IF NOT EXISTS transactions_card_ts ON transactions (card_id, ts);
CREATE INDEX IF NOT EXISTS transactions_merchant_ts ON transactions (merchant_id, ts);

CREATE TABLE IF NOT EXISTS chargebacks (
    txn_id      UUID PRIMARY KEY,
    txn_ts      TIMESTAMPTZ NOT NULL,
    reported_at TIMESTAMPTZ NOT NULL,
    CHECK (reported_at > txn_ts),
    FOREIGN KEY (txn_id, txn_ts) REFERENCES transactions (txn_id, ts)
);

CREATE INDEX IF NOT EXISTS chargebacks_reported_at ON chargebacks (reported_at);

CREATE TABLE IF NOT EXISTS ground_truth (
    txn_id   UUID PRIMARY KEY,
    is_fraud BOOLEAN NOT NULL
);

CREATE TABLE IF NOT EXISTS model_registry (
    version       TEXT PRIMARY KEY,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    algorithm     TEXT NOT NULL,
    feature_set   TEXT NOT NULL,
    train_start   TIMESTAMPTZ NOT NULL,
    train_end     TIMESTAMPTZ NOT NULL,
    label_cutoff  TIMESTAMPTZ NOT NULL,
    params        JSONB NOT NULL DEFAULT '{}',
    metrics       JSONB NOT NULL DEFAULT '{}',
    policy        JSONB NOT NULL DEFAULT '{}',
    artifact_path TEXT NOT NULL,
    stage         TEXT NOT NULL DEFAULT 'candidate'
                  CHECK (stage IN ('candidate', 'production', 'archived')),
    CHECK (train_end <= label_cutoff)
);

-- At most one production model at any time, enforced by the database.
CREATE UNIQUE INDEX IF NOT EXISTS model_registry_one_production
    ON model_registry (stage) WHERE stage = 'production';

CREATE TABLE IF NOT EXISTS predictions (
    txn_id        UUID NOT NULL,
    model_version TEXT NOT NULL REFERENCES model_registry (version),
    scored_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    p_fraud       REAL NOT NULL CHECK (p_fraud BETWEEN 0 AND 1),
    decision      TEXT NOT NULL CHECK (decision IN ('approve', 'review', 'decline')),
    reasons       JSONB NOT NULL DEFAULT '[]',
    latency_ms    REAL,
    PRIMARY KEY (txn_id, model_version)
);
