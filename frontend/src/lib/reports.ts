// What each report answers, in interview terms.
export const REPORT_BLURBS: Record<string, string> = {
  parity: "Do training (SQL) and serving (streaming) compute identical features? All 1.85M rows checked.",
  model_comparison: "Rules vs logistic regression vs from-scratch GBDT vs LightGBM, plus what the card-history features are worth.",
  imbalance: "Class weights, undersampling + prior shift, SMOTE, focal loss: effect on ranking and on calibration.",
  decisions: "From probabilities to dollars: thresholds vs cost-based decisions, review capacity, conformal guarantees, segment audit.",
  tuning: "Optuna on rolling-origin time-series CV, judged on the untouched test period.",
  backtest: "18 months of deployment: static vs retraining vs drift-triggered vs online FTRL on frozen trees.",
  replay: "The whole test period through the live scorer: exact parity with offline scores, latency, logging.",
  sketches: "Count-Min Sketch vs Bloom filter for spotting a card's first purchase at a merchant.",
};

export const REPORT_ORDER = ["parity", "model_comparison", "imbalance", "decisions", "tuning", "backtest", "replay", "sketches"];
