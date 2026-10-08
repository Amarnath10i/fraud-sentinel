"""Stateless features: functions of the current transaction and its aggregates.

These are plain NumPy functions over arrays, used unchanged for a 1.8M-row
training frame and for a single request in the API (arrays of length 1), so
there is nothing to keep in sync between the two paths.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

CATEGORIES = (
    "entertainment", "food_dining", "gas_transport", "grocery_net", "grocery_pos",
    "health_fitness", "home", "kids_pets", "misc_net", "misc_pos", "personal_care",
    "shopping_net", "shopping_pos", "travel",
)  # fmt: skip
_CATEGORY_CODE = {c: i for i, c in enumerate(CATEGORIES)}
_EARTH_KM = 6371.0088

# Smoothing strength for chargeback rates: a merchant with little volume is
# pulled towards the global rate instead of reporting 0% or 100%.
CB_RATE_PRIOR = 200.0

Arrays = Mapping[str, np.ndarray]


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    return 2 * _EARTH_KM * np.arcsin(np.sqrt(a))


def _ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    return np.where(np.isfinite(out), out, np.nan)


def row_features(c: Arrays) -> dict[str, np.ndarray]:
    """Inputs: ts (epoch s), amount ($), category, gender, dob_days (days since
    epoch), city_pop, home_lat, home_lon, merch_lat, merch_lon."""
    ts = np.asarray(c["ts"], dtype=np.int64)
    amount = np.asarray(c["amount"], dtype=np.float64)
    hour = (ts // 3600) % 24
    return {
        "amount": amount,
        "hour": hour.astype(np.float64),
        "day_of_week": ((ts // 86_400 + 3) % 7).astype(np.float64),  # 1970-01-01 was a Thursday
        "is_night": ((hour >= 22) | (hour < 4)).astype(np.float64),
        "age_years": (ts / 86_400 - np.asarray(c["dob_days"], dtype=np.float64)) / 365.25,
        "gender_m": (np.asarray(c["gender"]) == "M").astype(np.float64),
        "log_city_pop": np.log(np.asarray(c["city_pop"], dtype=np.float64)),
        "dist_home_km": haversine_km(c["home_lat"], c["home_lon"], c["merch_lat"], c["merch_lon"]),
        "category_code": np.array(
            [_CATEGORY_CODE.get(x, -1) for x in np.asarray(c["category"]).tolist()],
            dtype=np.float64,
        ),
    }


def derived_features(amount: np.ndarray, agg: Arrays) -> dict[str, np.ndarray]:
    """Comparisons of the current amount with the card's history, and smoothed
    chargeback rates around the merchant and category."""
    global_rate = _ratio(agg["global__cb_count_30d"], agg["global__count_30d"])
    global_rate = np.nan_to_num(global_rate, nan=0.0)

    def cb_rate(prefix: str) -> np.ndarray:
        cb, n = agg[f"{prefix}__cb_count_30d"], agg[f"{prefix}__count_30d"]
        return (cb + CB_RATE_PRIOR * global_rate) / (n + CB_RATE_PRIOR)

    std = agg["card__std_amount_all"]
    return {
        "amount_to_card_mean": _ratio(amount, agg["card__mean_amount_all"]),
        "amount_zscore_card": _ratio(
            amount - agg["card__mean_amount_all"], np.where(std > 0, std, np.nan)
        ),
        "amount_to_card_max_7d": _ratio(amount, agg["card__max_amount_7d"]),
        "amount_to_card_category_mean": _ratio(amount, agg["card_category__mean_amount_all"]),
        "card_avg_amount_24h": _ratio(agg["card__sum_amount_24h"], agg["card__count_24h"]),
        "merchant_cb_rate_30d": cb_rate("merchant"),
        "category_cb_rate_30d": cb_rate("category"),
    }
