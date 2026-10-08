"""Train, calibrate, package and register the production model."""

from __future__ import annotations

import json
import logging

import numpy as np

from sentinel.config import settings
from sentinel.decision import CostModel
from sentinel.eval import metrics as m
from sentinel.experiments.common import splits
from sentinel.features.spec import SPARKOV
from sentinel.models.calibration import IsotonicCalibrator
from sentinel.models.zoo import LightGBM
from sentinel.serve.bundle import Bundle, new_version, register

log = logging.getLogger(__name__)


def train_production(promote: bool = True) -> Bundle:
    s = splits()
    tr, va = s["train"], s["valid"]
    params_file = settings.paths.artifacts / "best_params.json"
    params = json.loads(params_file.read_text()) if params_file.exists() else {}
    model = LightGBM(params=params).fit(tr.frame, tr.y, va.frame, va.y)

    raw = model.predict_proba(va.frame)
    cal = IsotonicCalibrator().fit(raw, va.y)
    tl = settings.timeline
    bundle = Bundle(
        version=new_version(),
        features=model.features,
        feature_spec=SPARKOV.version,
        booster=model.booster,
        calibrator=cal,
        costs=CostModel(),
        reference_scores=cal.predict(raw).astype(np.float32),
        meta={
            "train_start": tl.data_start.isoformat(),
            "train_end": tl.train_end.isoformat(),
            "label_cutoff": tl.deploy_at.isoformat(),
            "params": params,
            "best_iteration": model.booster.best_iteration,
            "metrics": {
                "valid_pr_auc": m.average_precision(va.y, raw),
                "valid_recall@0.5%": m.recall_at_rate(va.y, raw, 0.005),
            },
        },
    )
    path = bundle.save()
    register(bundle, path, promote=promote)
    return bundle
