# Data analysis

Training rows: 20000, frauds: 353 (1.76%). Test rows: 12000.

## 1. Single risk signals

Fraud rate with/without each signal, and how common the signal is in train vs test.

| signal                            |   fraud rate if present |   fraud rate if absent |    lift |   share of train |   share of test |
|:----------------------------------|------------------------:|-----------------------:|--------:|-----------------:|----------------:|
| new_device                        |                  0.075  |                 0.012  |  6.2615 |           0.09   |          0.1499 |
| young_account (<120d)             |                  0.0418 |                 0.0125 |  3.3378 |           0.1744 |          0.1206 |
| velocity_burst (1h>=5 or 24h>=16) |                  0.2491 |                 0.0144 | 17.241  |           0.0137 |          0.0398 |
| night (23:00-03:59)               |                  0.0281 |                 0.0149 |  1.8924 |           0.2097 |          0.215  |
| amount > 500                      |                  0.0568 |                 0.013  |  4.3779 |           0.1065 |          0.1737 |
| high_risk_merchant                |                  0.0449 |                 0.0117 |  3.8534 |           0.1804 |          0.1932 |
| foreign (not SG)                  |                  0.0285 |                 0.013  |  2.1873 |           0.2981 |          0.3488 |


## 2. Risk compounds

Fraud rate by the number of the first six signals present. Risk climbs steeply as signals stack up. The test set has fewer signal-free rows (38% vs 46%) and 1.5x the share of 4+ signal rows.

|    |   fraud rate |   frauds |   rows |   share of train |   share of test |
|---:|-------------:|---------:|-------:|-----------------:|----------------:|
|  0 |       0.0053 |       49 |   9204 |           0.4602 |          0.3784 |
|  1 |       0.0136 |       98 |   7193 |           0.3597 |          0.4094 |
|  2 |       0.0266 |       73 |   2742 |           0.1371 |          0.1715 |
|  3 |       0.0858 |       58 |    676 |           0.0338 |          0.027  |
|  4 |       0.3139 |       43 |    137 |           0.0069 |          0.0096 |
|  5 |       0.6667 |       32 |     48 |           0.0024 |          0.0041 |


**Irreducible noise:** 49 of 353 frauds (14%) show none of the six signals. They are indistinguishable from ordinary transactions, which caps achievable PR-AUC/recall for any model.


## 3. Train -> test distribution shift


| feature               |   train median |   test median |   train p90 |   test p90 |
|:----------------------|---------------:|--------------:|------------:|-----------:|
| transaction_amount    |          75.12 |         87.73 |      530.35 |    1706.77 |
| account_age           |         790    |       1158    |     2461    |    3250    |
| spend_last_24h        |         217.51 |        289.02 |     1400.4  |    1625.12 |
| transactions_last_24h |           4    |          4    |        7    |       7    |
| transactions_last_1h  |           1    |          1    |        2    |       2    |
| new_device (mean)     |           0.09 |          0.15 |             |            |


High-value rows (> 1500): 3.1% of train vs 10.8% of test. In test they come from old accounts (median age 2405d vs 1741d in train), rarely a new device (4.6%), mostly daytime. In train, high-value payments from calm, established accounts are close to the base fraud rate - amount is only risky *together with* other signals, which is why tree models (that learn interactions) beat additive ones here.


Fraud rate by amount, for calm established accounts vs everyone else:


| amount        |   calm & old: fraud rate |   rows |   others: fraud rate |   rows |
|:--------------|-------------------------:|-------:|---------------------:|-------:|
| [0, 200)      |                   0.0053 |   8985 |               0.0209 |   5940 |
| [200, 500)    |                   0.0102 |   1857 |               0.0376 |   1089 |
| [500, 1000)   |                   0.0353 |    652 |               0.0789 |    494 |
| [1000, 2000)  |                   0.0156 |    321 |               0.125  |    208 |
| [2000, 1e+09) |                   0.014  |    358 |               0.2396 |     96 |


## 4. Missing values

Missing values look random (fraud rate among missing rows is close to the 1.8% base rate), so we let the trees route NaNs natively and do not use 'is missing' flags as fraud signals.


| column                |   missing rows |   frauds |   fraud rate |   test missing share |
|:----------------------|---------------:|---------:|-------------:|---------------------:|
| merchant_category     |            113 |        0 |       0      |               0.0216 |
| country               |            107 |        1 |       0.0093 |               0.0223 |
| transaction_channel   |            123 |        4 |       0.0325 |               0.0237 |
| transactions_last_24h |            159 |        5 |       0.0314 |               0.0244 |
| spend_last_24h        |            132 |        4 |       0.0303 |               0.0222 |
| account_age           |            137 |        3 |       0.0219 |               0.022  |
| new_device            |            124 |        0 |       0      |               0.0244 |
| transactions_last_1h  |            142 |        4 |       0.0282 |               0.0204 |
