# Leaderboard check

Out-of-fold metrics (3x repeated 5-fold CV), each model's 0/1 column cut as it was submitted:

| model    |   PR-AUC (probabilities) |   PR-AUC (0/1 column) |     F1 |   recall |   flagged |   threshold |
|:---------|-------------------------:|----------------------:|-------:|---------:|----------:|------------:|
| baseline |                   0.2049 |                0.1041 | 0.2908 |   0.2323 |    0.0106 |      0.1514 |
| ours     |                   0.2262 |                0.0939 | 0.2852 |   0.3022 |    0.0198 |      0.095  |


| | PR-AUC (probabilities) | PR-AUC (0/1 column) | F1 |
|---|---|---|---|
| ours - baseline, local | +0.0213 | -0.0102 | -0.0056 |
| ours - baseline, leaderboard | | **-0.0098** | |
| paired noise (1 s.e., 12k-row test) | 0.0109 | 0.0116 | |


PR-AUC of the 0/1 column reproduces the leaderboard gap almost exactly. PR-AUC of the probabilities predicts the opposite sign; the observed gap is 2.9 paired standard errors away from it. This is strong evidence (not proof) that the leaderboard scores the submitted 0/1 labels.

Noise: one submission's score has a bootstrap s.e. of about 0.031 on 12,000 rows, and two similar submissions differ by noise of roughly 0.003-0.01, so leaderboard steps of a few thousandths are not evidence on their own.
