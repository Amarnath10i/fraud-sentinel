"""What each decision policy costs on the deployment period, in dollars.

Probabilities are calibrated and thresholds/conformal sets are fitted on the
validation period only (labels known at deploy time), then frozen and
applied to the test period.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from sentinel.decision import (
    APPROVE,
    DECLINE,
    REVIEW,
    ConformalPolicy,
    CostModel,
    bayes_policy,
    budget_policy,
    capacity_policy,
    threshold_policy,
)
from sentinel.eval import metrics as m
from sentinel.eval.bootstrap import ClusterBootstrap
from sentinel.experiments.common import splits, write_report
from sentinel.models.calibration import IsotonicCalibrator, PlattScaler
from sentinel.models.zoo import LightGBM

log = logging.getLogger(__name__)
COSTS = CostModel()
REVIEWS_PER_DAY = 4  # below the busiest day's demand, so the constraint binds


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    grid = np.unique(np.quantile(p, np.linspace(0.95, 0.99999, 400)))
    f1 = []
    for t in grid:
        pred = p >= t
        tp = np.sum(pred & (y == 1))
        f1.append(2 * tp / (pred.sum() + y.sum()) if pred.sum() else 0.0)
    return float(grid[int(np.argmax(f1))])


def run() -> str:
    s = splits()
    tr, va, te = s["train"], s["valid"], s["test"]
    y, amount = te.y, te.frame["amount"].to_numpy()
    day = (te.frame["ts"] - te.frame["ts"].min()).dt.days.to_numpy()

    plain = LightGBM().fit(tr.frame, tr.y, va.frame, va.y)
    neg_pos = float((tr.y == 0).sum() / tr.y.sum())
    weighted = LightGBM(params={"scale_pos_weight": neg_pos}).fit(tr.frame, tr.y, va.frame, va.y)

    pv, pt = plain.predict_proba(va.frame), plain.predict_proba(te.frame)
    wv, wt = weighted.predict_proba(va.frame), weighted.predict_proba(te.frame)
    iso = IsotonicCalibrator().fit(pv, va.y)
    pt_iso = iso.predict(pt)
    wt_iso = IsotonicCalibrator().fit(wv, va.y).predict(wt)
    wt_platt = PlattScaler().fit(wv, va.y).predict(wt)
    conformal = ConformalPolicy(alpha_fraud=0.05, alpha_legit=0.002).fit(iso.predict(pv), va.y)

    calibration = pd.DataFrame(
        [
            {"scores": name, "median p, legit": f"{np.median(p[y == 0]):.1e}",
             "mean p, fraud": f"{p[y == 1].mean():.3f}", "ECE": f"{m.expected_calibration_error(y, p):.4f}",
             "Brier": f"{m.brier(y, p):.5f}", "log loss": f"{m.log_loss(y, p):.4f}"}
            for name, p in [
                ("LightGBM", pt), ("LightGBM + isotonic", pt_iso),
                ("LightGBM balanced weights", wt), ("balanced + Platt", wt_platt),
                ("balanced + isotonic", wt_iso),
            ]
        ]
    )  # fmt: skip

    t_f1 = best_f1_threshold(va.y, pv)
    policies = {
        "approve everything (no model)": np.full(len(y), APPROVE),
        "decline if p >= 0.5": threshold_policy(pt, 0.5),
        f"decline if p >= {t_f1:.3f} (best F1 on validation)": threshold_policy(pt, t_f1),
        "review top 0.5% of scores": budget_policy(pt, 0.005),
        "Bayes, balanced-weight scores (miscalibrated)": bayes_policy(wt, amount, COSTS),
        "Bayes, balanced-weight scores + isotonic": bayes_policy(wt_iso, amount, COSTS),
        "Bayes, LightGBM + isotonic": bayes_policy(pt_iso, amount, COSTS),
        f"Bayes + isotonic, max {REVIEWS_PER_DAY} reviews/day": capacity_policy(
            pt_iso, amount, day, REVIEWS_PER_DAY, COSTS
        ),
        "conformal sets (alpha fraud 5%, legit 0.2%)": conformal.decide(pt_iso),
    }

    boot = ClusterBootstrap(te.frame["card_id"].to_numpy(), n_boot=200, seed=0)
    rows = []
    for name, action in policies.items():
        cost = COSTS.realized(action, y, amount)
        iv = boot.intervals(lambda idx, c=cost: {"cost": float(c[idx].sum())}, len(y))["cost"]
        fraud = y == 1
        rows.append(
            {
                "policy": name,
                "total cost $": f"{iv.point:,.0f} ({iv.low:,.0f}-{iv.high:,.0f})",
                "fraud $ approved": f"{amount[fraud & (action == APPROVE)].sum():,.0f}",
                "reviews": int((action == REVIEW).sum()),
                "max reviews/day": int(np.bincount(day[action == REVIEW]).max())
                if (action == REVIEW).any()
                else 0,
                "declines": int((action == DECLINE).sum()),
                "false declines": int((~fraud & (action == DECLINE)).sum()),
                "frauds approved": int((fraud & (action == APPROVE)).sum()),
            }
        )
    table = pd.DataFrame(rows)

    act = conformal.decide(pt_iso)
    fraud_auto_approved = np.mean(act[y == 1] == APPROVE)
    legit_declined = np.mean(act[y == 0] == DECLINE)
    total_fraud = amount[y == 1].sum()

    text = "\n".join(
        [
            "# Decisions: from probabilities to dollars",
            "",
            f"Test period: {len(y):,} transactions, {int(y.sum()):,} frauds worth ${total_fraud:,.0f}.",
            "",
            "Cost model: approving a fraud loses its amount; a review costs "
            f"${COSTS.review_cost:.0f} of analyst time plus ${COSTS.review_friction:.0f} of customer "
            f"friction and stops {COSTS.review_catch_rate:.0%} of frauds; a false decline costs "
            f"${COSTS.decline_fixed:.0f} plus {COSTS.decline_margin:.0%} of the amount in lost revenue.",
            "",
            "## Are the probabilities trustworthy? (test period)",
            "",
            calibration.to_markdown(index=False),
            "",
            "## Policies (total cost with 95% card-level bootstrap interval)",
            "",
            table.to_markdown(index=False),
            "",
            "## Conformal guarantee check",
            "",
            f"Calibrated on validation (fraud quantile {conformal.q_fraud:.4f}, legit quantile "
            f"{conformal.q_legit:.4f}). On test, {fraud_auto_approved:.2%} of frauds were "
            f"auto-approved (target <= 5%) and {legit_declined:.3%} of legitimate transactions were "
            "declined (target <= 0.2%). The guarantee assumes the test period is exchangeable with "
            "the calibration period; drift between them is the main way it can fail.",
            "",
            "## Segment audit (Bayes, LightGBM + isotonic)",
            "",
            "Gender and age are model inputs here because the dataset provides them. A real "
            "deployment would have to justify that legally; this table is the check that "
            "would inform the decision: who absorbs the friction, and whose fraud gets caught.",
            "",
            segment_audit(te.frame, y, policies["Bayes, LightGBM + isotonic"]).to_markdown(
                index=False
            ),
            "",
        ]
    )
    write_report("decisions", text)
    return text


def segment_audit(frame: pd.DataFrame, y: np.ndarray, action: np.ndarray) -> pd.DataFrame:
    age = frame["age_years"].to_numpy()
    groups = {
        "gender F": frame["gender_m"].to_numpy() == 0,
        "gender M": frame["gender_m"].to_numpy() == 1,
        "age < 30": age < 30,
        "age 30-49": (age >= 30) & (age < 50),
        "age 50-69": (age >= 50) & (age < 70),
        "age 70+": age >= 70,
    }
    rows = []
    for name, g in groups.items():
        legit, fraud = g & (y == 0), g & (y == 1)
        rows.append(
            {
                "segment": name,
                "transactions": int(g.sum()),
                "fraud rate": f"{y[g].mean():.3%}",
                "legit reviewed or declined": f"{np.mean(action[legit] != APPROVE):.3%}",
                "legit declined": f"{np.mean(action[legit] == DECLINE):.4%}",
                "fraud stopped or reviewed": f"{np.mean(action[fraud] != APPROVE):.2%}",
            }
        )
    return pd.DataFrame(rows)
