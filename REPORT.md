# Track 2 – Fraud Detection: Model Report

> Scores and tables marked `TBD` are filled in from `outputs/` once the final run completes.

## 1. Problem understanding

**Task.** Each row is a single card/wallet transaction with 10 attributes (amount, hour, merchant
category, country, channel, 24h/1h velocity counters, account age, new-device flag). We must output a
fraud score per transaction for a hidden test set; the organisers score **PR-AUC, F1 and Recall**.

**Why PR-AUC and not ROC-AUC.** Fraud is 1.77 % of training rows (353 of 20 000). With that
imbalance a model can reach ROC-AUC ≈ 0.77 while still ranking many legitimate transactions above
fraud. PR-AUC (average precision) only rewards *fraud ranked above legit*, which is what an analyst
queue cares about. F1 and Recall additionally test the *operating point*, i.e. where we draw the line.

**What the data told us (EDA, `experiments/e0..e3`, notebook-style scripts).**

| finding | evidence | consequence |
|---|---|---|
| Positives are rare and few | 353 fraud rows | Single 5-fold CV has PR-AUC standard error ≈ 0.02 — larger than most gains. We use repeated stratified K-fold (5×6 = 30 fits) and compare models on identical splits. |
| Signal is weak per feature, strong in *interactions* | best single raw feature PR-AUC = 0.075 (`transactions_last_24h`); but `new_device & account_age<90` → 18 % fraud, `≥15 txns/24h` → 24 %, `hi-risk merchant & new_device` → 24 % | Tree models with shallow depth (2–3 splits) capture these pairwise interactions without memorising noise. |
| Generating process looks additive in log-odds | logistic regression on engineered features is within 0.01 PR-AUC of boosted trees | A linear model is a legitimate ensemble member and a useful sanity check. |
| **Train/test covariate shift** | adversarial AUC 0.67; test has 3× the share of transactions > $1 000, older accounts (median 1 158 vs 790 days), 15 % vs 9 % new devices, more JP/AU/luxury/travel | Prefer *relative* features (amount vs. the account's own 24h baseline, amount vs. merchant-category median); keep models heavily regularised; do not trust absolute-dollar splits alone. |
| Importance-weighted base rate ≈ 2.3 % | adversarial reweighting of train rows | Hidden-set fraud rate is probably a bit above train; the F1-optimal threshold is chosen on CV *rank quantile*, which transfers more robustly than a raw probability cut-off. |
| No ID leakage | fraud rate flat across ID deciles | `id` is never used as a feature. |
| Missingness is mild and almost uninformative | 0.6 % train / 2.3 % test nulls; only `transaction_channel` missing is slightly riskier | Trees take NaN natively; `n_missing` is one weak feature; linear model uses median imputation. |

## 2. Feature engineering (`src/features.py`)

All features are per-row (there is no customer id, so no cross-row aggregation is possible).

* **Scale:** `log_amount`, `log_spend_24h`, `log_account_age`.
* **Account-relative:** `avg_ticket_24h = spend_24h / txns_24h`, `spike_ratio = amount / (avg_ticket+1)`,
  `log_spike_ratio`, **absolute** `amount_diff_avg_spend` (so a 5× spike on a $2 coffee is not confused with
  a 5× spike on a $1 000 bill), `share_of_24h_spend`.
* **Merchant-relative:** `amount_ratio_to_merchant`, `amount_diff_merchant_median`, `amount_z_in_merchant`
  (log-amount z-score within category). Medians are label-free statistics, fitted on the training fold.
* **Velocity:** `velocity_1h_share = txns_1h / txns_24h`, `high_velocity_24h (≥10)`, `high_velocity_1h (≥3)`, `txn_24h_sq`.
* **Maturity × device:** `young_account (<90 d)`, `new_device_young_account`, `new_device_x_log_amount`, `age_per_txn`.
* **Time:** `is_night (23–03h)`, `hour_sin/cos`, `night_new_device`.
* **Categorical risk:** `hi_risk_merchant` (luxury / cash_transfer / electronics), `hi_risk_merchant_new_device`,
  `is_foreign (≠SG)`, `foreign_hi_risk`.
* **Categoricals:** CatBoost members take `merchant_category`, `country`, `transaction_channel` natively
  (ordered target statistics = built-in leakage-safe target encoding). XGBoost/LightGBM members get an
  **out-of-fold m-estimate target encoding** (m = 20) of the same three columns.

**What we tried and dropped (with evidence):**

* Interaction target encodings (`country_merchant`, `merchant_channel`, `country_channel`) – cost 0.005–0.01
  PR-AUC for every model in CV (E1, E2). Too many sparse cells for 353 positives; the trees already learn the
  interaction from the two base encodings.
* Importance weighting for covariate shift – full weights hurt (−0.003 to −0.009), √-weights neutral (E3).
* Pooled (train+test) merchant medians – neutral (E3).

## 3. Models and why

| member | why it is in the ensemble |
|---|---|
| CatBoost depth 3, native categoricals (×2 feature sets) | Strongest single model in every experiment. Symmetric (oblivious) trees + ordered boosting are very resistant to over-fitting on small, imbalanced data; ordered target statistics encode categories without leakage. |
| XGBoost depth 2 | Your proven baseline on the hidden set; depth 2 = at most one interaction per tree, which matches the data's pairwise-interaction structure. |
| LightGBM, 4 leaves | Leaf-wise grower with different split heuristics → diverse errors; best F1 among singles. |
| Logistic regression (balanced) | Additive model; nearly as accurate, extremely stable under covariate shift, cheap insurance against the trees over-fitting absolute thresholds. |

Members are combined by **rank averaging** (each member's test scores are converted to ranks, then averaged).
PR-AUC is rank-based, so rank averaging keeps every member's ordering information while neutralising the
different calibrations of a linear model vs. boosted trees. Each member is itself an average over 5 seeds.

## 4. Model assumptions (what must hold for this to work)

1. **Rows are i.i.d. given the features.** No customer/merchant identifiers exist, so we cannot model
   per-customer sequences; we assume the 24h/1h counters already summarise that history.
2. **p(fraud | x) is the same in train and test** (covariate shift, not concept drift). We measured a shift in
   p(x) and designed relative features to be robust to it; if fraud *behaviour* itself changed, no supervised
   model on this data could follow it.
3. **Gradient-boosted trees:** additive ensemble of piecewise-constant functions; features need no scaling;
   monotone transforms of a feature do not matter; missing values are routed by learned default direction.
   Shallow depth assumes interactions of order ≤ 2–3 suffice (verified in EDA).
4. **CatBoost ordered target statistics** assume a random row order (we shuffle) so the encoding of a row never
   uses its own label.
5. **Logistic regression:** log-odds linear in the (engineered) features; we add the key interactions by hand
   (`new_device_young_account`, `hi_risk_merchant_new_device`, …) so the linearity assumption is approximately met.
6. **Threshold selection** assumes the hidden set's score distribution resembles CV; we therefore set the cut-off
   as a *rank quantile* (share of transactions flagged) rather than a raw probability.

## 5. Results

TBD – filled from `outputs/cv_results.csv`, `outputs/e*_*.csv`.

## 6. Reproducing

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost scipy
python3 experiments/e0_baselines.py      # model x feature-set grid
python3 experiments/e1_encodings.py      # target-encoding ablations
python3 experiments/e2_catboost_native.py
python3 experiments/e3_shift_weighting.py
python3 experiments/e4_ensemble.py 6     # blend search on identical splits
python3 run_final.py --seeds 5           # writes outputs/submission*.csv
```
