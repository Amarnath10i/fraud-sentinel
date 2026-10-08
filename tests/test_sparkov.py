import pandas as pd

from sentinel.config import ChargebackModel
from sentinel.data.sparkov import normalize


def _raw() -> pd.DataFrame:
    rows = [
        # ts, card, merchant, category, amount, fraud
        ("2019-01-01 00:00:10", 1, "fraud_A", "grocery_pos", 10.0, 0, "0" * 31 + "1"),
        ("2019-01-01 00:05:00", 1, "fraud_B", "misc_net", 900.0, 1, "0" * 31 + "2"),
        ("2019-01-02 10:00:00", 2, "fraud_A", "grocery_pos", 25.5, 0, "0" * 31 + "3"),
        ("2019-01-03 11:00:00", 2, "fraud_B", "shopping_net", 700.0, 1, "0" * 31 + "4"),
    ]
    raw = pd.DataFrame(
        rows, columns=["t", "cc_num", "merchant", "category", "amt", "is_fraud", "trans_num"]
    )
    raw["ts"] = pd.to_datetime(raw.pop("t")).dt.tz_localize("UTC")
    per_card = {
        1: ("F", "1980-01-01", "Teacher", "Austin", "TX", "73301", 30.0, -97.0, 1000),
        2: ("M", "1990-06-01", "Nurse", "Boise", "ID", "83701", 43.0, -116.0, 5000),
    }
    cols = ["gender", "dob", "job", "city", "state", "zip", "lat", "long", "city_pop"]
    for i, c in enumerate(cols):
        raw[c] = raw["cc_num"].map(lambda k, i=i: per_card[k][i])
    raw["merch_lat"] = raw["lat"] + 0.1
    raw["merch_long"] = raw["long"] - 0.1
    return raw


def test_normalize_splits_entities_and_moves_label_out():
    t = normalize(_raw(), ChargebackModel(seed=1))
    assert list(t.customers["card_id"]) == [1, 2]
    assert list(t.merchants["name"]) == ["A", "B"]
    assert "is_fraud" not in t.transactions.columns
    assert t.ground_truth["is_fraud"].sum() == 2
    # a merchant can appear in several categories; category stays on the transaction
    b = t.merchants.set_index("name").loc["B", "merchant_id"]
    assert set(t.transactions.loc[t.transactions.merchant_id == b, "category"]) == {
        "misc_net",
        "shopping_net",
    }


def test_chargebacks_only_for_fraud_and_always_after_the_transaction():
    model = ChargebackModel(median_days=10, sigma=0.5, min_days=1, seed=3)
    t = normalize(_raw(), model)
    fraud_ids = set(t.ground_truth.loc[t.ground_truth.is_fraud, "txn_id"])
    assert set(t.chargebacks["txn_id"]) == fraud_ids
    delay = t.chargebacks["reported_at"] - t.chargebacks["txn_ts"]
    assert (delay >= pd.Timedelta(days=1)).all()


def test_chargeback_simulation_is_independent_of_row_order():
    model = ChargebackModel(seed=5)
    raw = _raw()
    a = normalize(raw, model).chargebacks.set_index("txn_id")
    b = normalize(raw.iloc[::-1].reset_index(drop=True), model).chargebacks.set_index("txn_id")
    pd.testing.assert_series_equal(a["reported_at"], b.loc[a.index, "reported_at"])
