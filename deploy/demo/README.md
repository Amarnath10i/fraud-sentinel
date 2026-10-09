---
title: fraud-sentinel
emoji: 🛡️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
short_description: Real-time card fraud scoring with SHAP and what-if
---

# fraud-sentinel: live demo

Real-time card fraud detection, end to end. This Space replays the 2020
deployment period through the production scorer one authorization at a time.
Every decision comes with its calibrated probability, its expected cost per
action and a TreeSHAP explanation, and a what-if explorer re-scores any card's
live state as you change the transaction.

- **Overview / Live stream:** press *Start replay* in the top bar.
- **What-if explorer:** drag amount, hour or distance and watch the decision change.
- **Experiments:** every result in the project, with bootstrap intervals.
- **API docs:** `/docs`.

Source code, methodology and results: https://github.com/Amarnath10i/fraud-sentinel

This demo runs without PostgreSQL. It ships the production model, a snapshot
of the online feature state at deploy time and the replayed transactions. The
data is the CC0 Sparkov synthetic credit-card dataset (Kaggle
`kartik2112/fraud-detection`, generated with Brandon Harris's
Sparkov_Data_Generation). Ground truth is shown for the demo only; the model
never sees it.
