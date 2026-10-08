import numpy as np
import pandas as pd
import pytest

from sentinel.data.dataset import observed_labels
from sentinel.features.rows import derived_features, haversine_km, row_features


def test_haversine_known_distance():
    # New York -> London is about 5570 km
    assert haversine_km(40.7128, -74.0060, 51.5074, -0.1278) == pytest.approx(5570, rel=0.01)
    assert haversine_km(10.0, 20.0, 10.0, 20.0) == 0


def test_row_features_calendar_and_age():
    ts = pd.Timestamp("2020-03-02 23:30:00", tz="UTC").value // 10**9  # a Monday
    dob = (pd.Timestamp("1990-03-02") - pd.Timestamp("1970-01-01")).days
    f = row_features(
        {
            "ts": np.array([ts]),
            "amount": np.array([12.5]),
            "category": np.array(["grocery_pos"]),
            "gender": np.array(["M"]),
            "dob_days": np.array([dob]),
            "city_pop": np.array([1000]),
            "home_lat": np.array([40.0]),
            "home_lon": np.array([-75.0]),
            "merch_lat": np.array([40.0]),
            "merch_lon": np.array([-75.0]),
        }
    )
    assert f["hour"][0] == 23 and f["is_night"][0] == 1
    assert f["day_of_week"][0] == 0
    assert f["age_years"][0] == pytest.approx(30.0, abs=0.01)
    assert f["category_code"][0] >= 0 and f["dist_home_km"][0] == 0


def test_derived_features_handle_missing_history():
    nan = np.nan
    agg = {
        "card__mean_amount_all": np.array([nan, 50.0]),
        "card__std_amount_all": np.array([nan, 0.0]),
        "card__max_amount_7d": np.array([nan, 100.0]),
        "card_category__mean_amount_all": np.array([nan, 25.0]),
        "card__sum_amount_24h": np.array([0.0, 30.0]),
        "card__count_24h": np.array([0.0, 2.0]),
        "global__cb_count_30d": np.array([10.0, 10.0]),
        "global__count_30d": np.array([1000.0, 1000.0]),
        "merchant__cb_count_30d": np.array([0.0, 5.0]),
        "merchant__count_30d": np.array([0.0, 50.0]),
        "category__cb_count_30d": np.array([0.0, 0.0]),
        "category__count_30d": np.array([0.0, 0.0]),
    }
    d = derived_features(np.array([10.0, 100.0]), agg)
    assert np.isnan(d["amount_to_card_mean"][0]) and d["amount_to_card_mean"][1] == 2.0
    assert np.isnan(d["amount_zscore_card"][1])  # zero spread is undefined, not infinite
    assert np.isnan(d["card_avg_amount_24h"][0]) and d["card_avg_amount_24h"][1] == 15.0
    # no merchant history -> global rate; history pulls towards the merchant's own rate
    assert d["merchant_cb_rate_30d"][0] == pytest.approx(0.01)
    assert d["merchant_cb_rate_30d"][1] > 0.01


def test_observed_labels_respect_the_as_of_time():
    frame = pd.DataFrame(
        {"reported_at": pd.to_datetime(["2020-01-10", None, "2020-03-01"], utc=True)}
    )
    as_of = pd.Timestamp("2020-02-01", tz="UTC")
    assert observed_labels(frame, as_of).tolist() == [1, 0, 0]
