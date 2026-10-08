"""Model comparison and feature ablation on the held-out deployment period.

All models train on Jan 2019 - Jan 2020 with labels as known at deploy time,
early-stop on Feb - Apr 2020, and are scored on Jun 21 - Dec 31 2020 against
ground truth. Intervals are 95% card-level cluster bootstrap.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from sentinel.data.dataset import model_features
from sentinel.eval.bootstrap import ClusterBootstrap
from sentinel.experiments.common import (
    metric_table,
    ranking_metrics,
    save_predictions,
    splits,
    write_report,
)
from sentinel.features.spec import SPARKOV
from sentinel.models.zoo import LightGBM, RulesBaseline, ScratchGBDT, ScratchLogistic

log = logging.getLogger(__name__)

STATELESS = ["amount", "hour", "day_of_week", "is_night", "age_years", "gender_m",
             "log_city_pop", "dist_home_km", "category_code"]  # fmt: skip
# everything that needs the card's own history, including derived ratios
CARD_HISTORY_DERIVED = {"amount_to_card_mean", "amount_zscore_card", "amount_to_card_max_7d",
                        "amount_to_card_category_mean", "card_avg_amount_24h"}  # fmt: skip


def no_card_history() -> list[str]:
    return [
        f for f in model_features()
        if not f.startswith(("card__", "card_category__")) and f not in CARD_HISTORY_DERIVED
    ]  # fmt: skip


def run() -> str:
    s = splits()
    tr, va, te = s["train"], s["valid"], s["test"]
    models = {
        "Rules (amount x night)": RulesBaseline(),
        "Logistic regression (scratch, IRLS)": ScratchLogistic(),
        "GBDT (scratch)": ScratchGBDT(),
        "LightGBM": LightGBM(),
        "LightGBM, stateless features only": LightGBM(features=STATELESS),
        "LightGBM, no card history (stateless + merchant/category context)": LightGBM(
            features=no_card_history()
        ),
    }
    preds, timing = {}, {}
    for name, model in models.items():
        log.info("fitting %s", name)
        model.fit(tr.frame, tr.y, va.frame, va.y)
        t0 = time.perf_counter()
        preds[name] = model.predict_proba(te.frame)
        per_1k = (time.perf_counter() - t0) / len(te.y) * 1e3 * 1e3
        timing[name] = (model.fit_seconds, per_1k)
    save_predictions("compare_test", te.frame, preds)

    table, _ = metric_table(preds, te)
    table.insert(1, "fit (s)", [f"{timing[n][0]:.0f}" for n in table["model"]])
    table.insert(2, "predict (ms / 1k rows)", [f"{timing[n][1]:.1f}" for n in table["model"]])

    boot = ClusterBootstrap(te.frame["card_id"].to_numpy(), n_boot=200, seed=0)
    amount = te.frame["amount"].to_numpy()
    diff = boot.paired_difference(
        ranking_metrics(te.y, preds["GBDT (scratch)"], amount),
        ranking_metrics(te.y, preds["LightGBM"], amount),
        len(te.y),
    )
    gbdt = models["GBDT (scratch)"].model
    lgbm = models["LightGBM"].booster
    top = np.argsort(gbdt.feature_importances_)[::-1][:10]

    lines = [
        "# Model comparison",
        "",
        f"Train {len(tr.y):,} rows ({tr.y.sum():,} fraud, labels = chargebacks reported by deploy time), "
        f"validation {len(va.y):,} rows ({va.y.sum():,} fraud), "
        f"test {len(te.y):,} rows ({te.y.sum():,} fraud, ground truth). "
        f"Feature spec `{SPARKOV.version}`, {len(model_features())} model features.",
        "",
        "Point estimate with 95% card-level cluster-bootstrap interval (200 resamples).",
        "Precision at a 0.5% alert rate is capped at "
        f"{te.y.mean() / 0.005:.3f} because only {te.y.mean():.2%} of test transactions are fraud, "
        "so recall at that rate is reported instead.",
        "",
        table.to_markdown(index=False),
        "",
        "## Scratch GBDT vs LightGBM (paired bootstrap, LightGBM minus scratch)",
        "",
        "| metric | difference | 95% interval | share of resamples <= 0 |",
        "|---|---|---|---|",
        *[
            f"| {k} | {iv.point:+.4f} | {iv.low:+.4f} to {iv.high:+.4f} | {p:.2f} |"
            for k, (iv, p) in diff.items()
        ],
        "",
        f"Scratch GBDT used {gbdt.best_iteration_} trees (early stopping); "
        f"LightGBM used {lgbm.best_iteration} rounds.",
        "",
        "## What drives the score (scratch GBDT, share of total split gain)",
        "",
        "| feature | gain share |",
        "|---|---|",
        *[f"| {model_features()[i]} | {gbdt.feature_importances_[i]:.3f} |" for i in top],
        "",
    ]
    text = "\n".join(lines)
    write_report("model_comparison", text)
    return text
