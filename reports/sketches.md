# First-time (card, merchant) pairs: exact vs Count-Min vs Bloom

1,852,394 transactions, 529,328 distinct pairs, so 28.6% of transactions are a card's first payment to that merchant.

| structure        | parameters                    | memory   |   first-time pairs detected |   false 'new' flags | mean overcount   |
|:-----------------|:------------------------------|:---------|----------------------------:|--------------------:|:-----------------|
| exact dict       | -                             | 65.4 MB  |                      1      |                   0 | 0                |
| Count-Min Sketch | eps=0.001, 2,719 x 5          | 0.11 MB  |                      0.0117 |                   0 | 279.05           |
| Count-Min Sketch | eps=0.0001, 27,183 x 5        | 1.09 MB  |                      0.1169 |                   0 | 23.31            |
| Count-Min Sketch | eps=1e-05, 271,829 x 5        | 10.87 MB |                      0.8542 |                   0 | 0.37             |
| Bloom filter     | p=0.01, 5,073,640 bits, k=7   | 0.63 MB  |                      0.9982 |                   0 | n/a              |
| Bloom filter     | p=0.001, 7,610,460 bits, k=10 | 0.95 MB  |                      0.9999 |                   0 | n/a              |

Neither structure ever flags a known pair as new (no false negatives on membership). The difference is how many genuinely new pairs slip through:

- Count-Min's error is additive, eps x N over the whole stream. With an average of 3.5 transactions per pair, a long tail of rare keys sits far below that error, so 'count == 0' is unreliable unless the sketch is almost as large as the exact table. It is the right tool for heavy hitters, not for novelty.
- A Bloom filter answers membership directly: at 1% false positives it needs ~9.6 bits per pair and catches ~99% of first-time pairs in a fraction of the exact table's memory.
