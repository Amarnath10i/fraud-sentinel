# Offline/online feature parity

Feature spec `cd85031b90`: 26 aggregate features, 1,852,394 transactions, 1,862,045 events (transactions + chargebacks).

Streaming replay: 75s (24,955 events/s, single thread, pure Python).

Result: **all features identical** (relative tolerance 1e-9; NaN must match NaN).

| feature                        |   null_rate |   max_abs_diff |   mismatches |
|:-------------------------------|------------:|---------------:|-------------:|
| card__count_1h                 |    0.00e+00 |       0.00e+00 |            0 |
| card__count_24h                |    0.00e+00 |       0.00e+00 |            0 |
| card__count_7d                 |    0.00e+00 |       0.00e+00 |            0 |
| card__sum_amount_1h            |    0.00e+00 |       0.00e+00 |            0 |
| card__sum_amount_24h           |    0.00e+00 |       0.00e+00 |            0 |
| card__sum_amount_7d            |    0.00e+00 |       0.00e+00 |            0 |
| card__max_amount_24h           |    7.73e-02 |       0.00e+00 |            0 |
| card__max_amount_7d            |    7.12e-04 |       0.00e+00 |            0 |
| card__distinct_merchant_24h    |    0.00e+00 |       0.00e+00 |            0 |
| card__distinct_category_24h    |    0.00e+00 |       0.00e+00 |            0 |
| card__since_last               |    5.39e-04 |       0.00e+00 |            0 |
| card__count_all                |    0.00e+00 |       0.00e+00 |            0 |
| card__mean_amount_all          |    5.39e-04 |       2.27e-13 |            0 |
| card__std_amount_all           |    5.39e-04 |       9.09e-13 |            0 |
| card__since_first              |    5.39e-04 |       0.00e+00 |            0 |
| card_category__count_all       |    0.00e+00 |       0.00e+00 |            0 |
| card_category__count_7d        |    0.00e+00 |       0.00e+00 |            0 |
| card_category__mean_amount_all |    7.11e-03 |       9.09e-13 |            0 |
| merchant__count_30d            |    0.00e+00 |       0.00e+00 |            0 |
| merchant__cb_count_30d         |    0.00e+00 |       0.00e+00 |            0 |
| category__count_30d            |    0.00e+00 |       0.00e+00 |            0 |
| category__cb_count_30d         |    0.00e+00 |       0.00e+00 |            0 |
| global__count_30d              |    0.00e+00 |       0.00e+00 |            0 |
| global__cb_count_30d           |    0.00e+00 |       0.00e+00 |            0 |
| card__cb_count_all             |    0.00e+00 |       0.00e+00 |            0 |
| card__cb_since_last            |    4.69e-01 |       0.00e+00 |            0 |
