# Model comparison

Adversarial train-vs-test ROC-AUC: 0.668. CV = 3x repeated stratified 5-fold; shift-weighted = same OOF predictions weighted by p(test|x)/p(train|x); test-like hold-out = train on the 70% least test-like rows, validate on the 30% most test-like.

| Model                              |   CV PR-AUC |   CV ROC-AUC |   CV best F1 |   Shift-wtd PR-AUC |   Shift-wtd best F1 |   Test-like hold-out PR-AUC |   Test-like hold-out best F1 |
|:-----------------------------------|------------:|-------------:|-------------:|-------------------:|--------------------:|----------------------------:|-----------------------------:|
| Original baseline (XGBoost d2)     |      0.2049 |       0.7778 |       0.2908 |             0.3212 |              0.4309 |                      0.3043 |                       0.4251 |
| XGBoost d2, new features           |      0.2189 |       0.7770 |       0.3025 |             0.3404 |              0.4280 |                      0.3523 |                       0.4337 |
| XGBoost d2 + scale_pos_weight      |      0.1983 |       0.7708 |       0.2811 |             0.3057 |              0.4024 |                      0.2023 |                       0.2896 |
| Logistic regression (splines)      |      0.2212 |       0.7795 |       0.2950 |             0.3465 |              0.4042 |                      0.1294 |                       0.2283 |
| EBM (GAM + 10 interactions)        |      0.2163 |       0.7774 |       0.2854 |             0.3401 |              0.4195 |                      0.3270 |                       0.3922 |
| XGBoost d3 monotone                |      0.2208 |       0.7756 |       0.3022 |             0.3449 |              0.4284 |                      0.3549 |                       0.4415 |
| CatBoost d3 (3 seeds)              |      0.2196 |       0.7742 |       0.3090 |             0.3449 |              0.4368 |                      0.3468 |                       0.4749 |
| FINAL: CatBoost + monotone XGBoost |      0.2262 |       0.7790 |       0.3062 |             0.3560 |              0.4372 |                      0.3637 |                       0.4507 |
