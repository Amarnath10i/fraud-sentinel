# Model comparison

Train 977,052 rows (5,562 fraud, labels = chargebacks reported by deploy time), validation 169,143 rows (995 fraud), test 555,719 rows (2,145 fraud, ground truth). Feature spec `cd85031b90`, 39 model features.

Point estimate with 95% card-level cluster-bootstrap interval (200 resamples).
Precision at a 0.5% alert rate is capped at 0.772 because only 0.39% of test transactions are fraud, so recall at that rate is reported instead.

| model                                                             |   fit (s) |   predict (ms / 1k rows) | PR-AUC               | ROC-AUC              | Recall@0.5%          | $Recall@0.5%         | Prec@Recall80        |
|:------------------------------------------------------------------|----------:|-------------------------:|:---------------------|:---------------------|:---------------------|:---------------------|:---------------------|
| Rules (amount x night)                                            |         0 |                      0   | 0.2457 (0.218-0.270) | 0.8658 (0.851-0.877) | 0.4154 (0.395-0.441) | 0.7060 (0.685-0.727) | 0.0072 (0.006-0.017) |
| Logistic regression (scratch, IRLS)                               |        14 |                      2.1 | 0.8357 (0.815-0.857) | 0.9964 (0.995-0.997) | 0.8303 (0.801-0.854) | 0.9286 (0.913-0.943) | 0.7219 (0.658-0.791) |
| GBDT (scratch)                                                    |       385 |                     79.6 | 0.9886 (0.985-0.992) | 0.9999 (1.000-1.000) | 0.9879 (0.979-0.993) | 0.9959 (0.992-0.998) | 0.9983 (0.995-1.000) |
| LightGBM                                                          |        70 |                     11.7 | 0.9903 (0.987-0.994) | 0.9999 (1.000-1.000) | 0.9883 (0.980-0.993) | 0.9973 (0.992-0.999) | 1.0000 (0.999-1.000) |
| LightGBM, stateless features only                                 |         9 |                      1.6 | 0.8824 (0.864-0.902) | 0.9977 (0.997-0.998) | 0.8583 (0.836-0.885) | 0.9490 (0.935-0.966) | 0.8408 (0.761-0.900) |
| LightGBM, no card history (stateless + merchant/category context) |        12 |                      2.2 | 0.8785 (0.860-0.898) | 0.9977 (0.997-0.998) | 0.8545 (0.828-0.881) | 0.9568 (0.942-0.971) | 0.8314 (0.775-0.896) |

## Scratch GBDT vs LightGBM (paired bootstrap, LightGBM minus scratch)

| metric | difference | 95% interval | share of resamples <= 0 |
|---|---|---|---|
| PR-AUC | +0.0017 | +0.0005 to +0.0030 | 0.00 |
| ROC-AUC | +0.0000 | -0.0000 to +0.0000 | 0.36 |
| Recall@0.5% | +0.0005 | -0.0039 to +0.0047 | 0.47 |
| $Recall@0.5% | +0.0014 | -0.0021 to +0.0041 | 0.20 |
| Prec@Recall80 | +0.0017 | +0.0000 to +0.0046 | 0.06 |

Scratch GBDT used 272 trees (early stopping); LightGBM used 827 rounds.

## What drives the score (scratch GBDT, share of total split gain)

| feature | gain share |
|---|---|
| amount | 0.388 |
| card_avg_amount_24h | 0.320 |
| category_code | 0.123 |
| is_night | 0.033 |
| amount_to_card_max_7d | 0.017 |
| card__max_amount_24h | 0.015 |
| amount_to_card_mean | 0.009 |
| hour | 0.009 |
| card__sum_amount_24h | 0.008 |
| amount_zscore_card | 0.007 |
