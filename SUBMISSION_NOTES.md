# Track 2 Fraud Detection: Approach, Model Choice and Assumptions

## TL;DR

* **Final model:** an equal-weight blend of **CatBoost** (depth 3, averaged over 3 seeds) and a
  **monotone-constrained XGBoost** (depth 3), trained on a small set of behavioural features.
* **The 0/1 decision is cut with a plug-in rule:** on the file being scored, we pick the threshold that
  maximises the *expected* PR-AUC of the 0/1 column, computed from the model's own calibrated
  probabilities. On the Track 2 test set this is 0.271, flagging 151 rows (1.3%).
* **What the leaderboard taught us (§7):** our first submission ranked frauds *better* than the baseline
  but scored lower (0.1844 vs 0.1942). The only metric that reproduces that gap is **PR-AUC computed on
  the submitted 0/1 labels**, which depends mostly on the threshold. Our first submission flagged 2.7% of
  rows with a recall-leaning threshold, and that is what cost the points.
* **Why this design:** the training data is small and noisy (353 frauds, 14% of which look exactly like
  normal transactions), and **the test set comes from a different distribution than the training set**.
  We chose the model that held up best when we *simulated* that shift.

| Out-of-fold metric | Original baseline | Team v2 (LB 0.2000) | **Our model, plug-in threshold** |
|---|---|---|---|
| PR-AUC of the probabilities, repeated CV | 0.205 | 0.205 | **0.226** |
| PR-AUC of the probabilities, test-like hold-out | 0.304 | n/a | **0.364** |
| PR-AUC of the 0/1 column, repeated CV | 0.104 | 0.105 | **0.117** |
| PR-AUC of the 0/1 column, weighted to the test set's segment mix | 0.122 | 0.125 | **0.156** |
| Precision / recall of the 0/1 column, repeated CV | 0.39 / 0.23 | 0.41 / 0.22 | **0.54** / 0.19 |

The baseline and v2 use their own best-F1 thresholds; v2 is our reconstruction of the team's improved
model from its description. Our model ranks best, and the plug-in threshold turns that into the best 0/1
column. The cost is recall (§5): if the judges weight recall heavily, `predict.py --rule f1` gives a
better-balanced cut.

---

## 1. Understanding the problem

**Task.** Each row is one payment. We must estimate the probability it is fraudulent and decide whether to flag it.

**Why accuracy is the wrong yardstick.** Only **1.77%** of training transactions are fraud. A model
that flags nothing would be 98.2% accurate and catch zero fraud. So we optimise the metrics that focus
on the rare positive class:

| Metric | What it measures | Why it matters here |
|---|---|---|
| **PR-AUC** (average precision) | Ranking quality across *all* thresholds, judged only on how well frauds rise to the top. A random model scores the base rate (≈0.018). | Insensitive to the huge number of easy negatives, unlike ROC-AUC. It is the right summary for heavy imbalance. |
| **Recall** | Share of actual frauds we catch | A missed fraud is a direct financial loss plus a chargeback. |
| **F1** | Harmonic mean of precision and recall at our threshold | Penalises flagging everything to get recall. Every false alarm is a blocked customer or an extra verification step. |

**Business framing.** The probability is a risk score. The threshold is a business lever: a bank would
send high scores to step-up verification (OTP or call-back) rather than block them outright. We keep
the two jobs separate: the model's only job is to *rank* well (PR-AUC of the probabilities), and the
threshold is chosen afterwards for whatever the 0/1 decision is judged on (§5).

## 2. What the data tells us

All numbers below are reproduced by `python analysis.py` (written to `reports/data_analysis.md`).

**2.1 Fraud is driven by a handful of interpretable risk signals.**

| Signal | Fraud rate if present | If absent | Lift |
|---|---|---|---|
| Velocity burst (≥5 txns in 1h or ≥16 in 24h) | 24.9% | 1.4% | **17×** |
| New device | 7.5% | 1.2% | 6× |
| Amount > 500 | 5.7% | 1.3% | 4× |
| High-risk merchant (luxury, cash transfer, electronics) | 4.5% | 1.2% | 4× |
| Young account (< 120 days) | 4.2% | 1.3% | 3× |
| Foreign country (not SG) | 2.9% | 1.3% | 2× |
| Night (23:00–03:59) | 2.8% | 1.5% | 2× |

These match well-known fraud typologies:
* **Account-testing and velocity attacks:** bursts of transactions, almost never card-present.
* **New-account and synthetic-identity fraud:** young account plus a new device.
* **Cash-out:** large amounts at easily resold or liquid merchants (electronics, luxury, cash transfer).

**2.2 Risk compounds.** The fraud rate climbs from 0.5% with no signals, to 2.7% with two, 31% with
four and 67% with five. Signals multiply rather than add, so we include a `risk_signal_count`
feature and use tree models, which learn interactions natively.

**2.3 Amount is risky only together with other signals.** For calm, established accounts (old
account, known device, normal velocity), payments above 1,000 have a fraud rate of 1.4–1.6%, near the
base rate. For everyone else the same amounts have a 12–24% fraud rate. A model that treats "big
amount" as risky on its own will raise false alarms on wealthy, legitimate customers.

**2.4 There is an irreducible noise floor.** 14% of frauds show none of the risk signals and
look exactly like ordinary purchases: median amount 65, everyday merchants (food delivery, retail,
grocery), accounts about 1.7 years old. No model can rank these above legitimate traffic, which is why
in-distribution PR-AUC tops out around 0.22. We confirmed it is a ceiling and not a modelling gap: every
model family we tried (XGBoost, LightGBM, CatBoost, logistic regression, EBM) lands between 0.20 and 0.23.

**2.5 The test set is distributed differently from the training set (covariate shift).**
An "adversarial" classifier trained to tell training rows from test rows reaches ROC-AUC
**0.67**. If both came from the same distribution, it would score 0.5. The test set has:
* 3.5× more high-value payments (10.8% vs 3.1% above 1,500). They come from old accounts, mostly AU/JP/MY
  customers in daytime, which looks like a wealthy-customer segment.
* 1.7× more new-device transactions (15% vs 9%), mostly on *established* accounts. That looks like
  ordinary device changes rather than the new-account fraud pattern seen in training.
* About 3× more velocity bursts (4.0% vs 1.4%) and older accounts overall (median 1,158 vs 790 days).

This shift is the most important fact for model selection: **a model that wins on ordinary
cross-validation is not necessarily the one that wins on the test set** (see §4).

**2.6 Missing values look random.** About 0.5–0.8% of each column is missing in training and about 2%
in test. Fraud rates among rows with missing values sit near the base rate, so we treat missingness
as random noise. The trees route NaN natively, and we do **not** use "is missing" flags as fraud
signals: they would only memorise noise.

## 3. Features

All features are computed per row, with no target encoding, so a row's label can never leak into its
own features (`features.py`).

| Feature | Rationale |
|---|---|
| `log_amount`, `log_spend_24h`, `log_age` | Money and age are heavy-tailed. The log scale makes ratios comparable. |
| `txn_1h`, `txn_24h`, `new_device`, `hour` | Raw behavioural signals |
| `amount_vs_avg_24h` | Is this payment unusually large *for this customer today*? (log of amount over average 24h ticket) |
| `amount_vs_merchant` | Is it unusually large *for this merchant type*? 2,000 at a luxury store is normal, at a grocery it is not. |
| `share_txn_last_1h` | Share of the day's transactions that happened in the last hour: a sudden burst vs steady use |
| `velocity_burst`, `young_account`, `night`, `high_risk_merchant`, `foreign` | Domain risk flags, with thresholds read off the data (§2.1) |
| `risk_signal_count` | Number of co-occurring signals. It captures the compounding effect directly (§2.2) and is the model's most important feature. |
| `merchant_category`, `country`, `transaction_channel` | Categoricals. CatBoost uses ordered target statistics; XGBoost uses native categorical splits. |

## 4. Model choice: what we tried and why we picked this one

### 4.1 Validation designed for a shifted test set
Ordinary cross-validation only measures performance on data that looks like the training set. Because
the test set does not (§2.5), we scored every model three ways (`validation.py`):

1. **Repeated stratified 5-fold CV** (3 repeats, 15 fits): in-distribution performance with variance averaged out.
2. **Shift-weighted CV:** the same out-of-fold predictions, but each training row is weighted by
   *p(test | x) / p(train | x)* from the adversarial classifier. This is importance weighting: the
   metric then describes a population that looks like the test set.
3. **Test-like hold-out:** train only on the 70% of rows that look *least* like the test set, and
   validate on the 30% that look *most* like it. This is the hardest check, because it rewards models that
   *extrapolate* sensibly into regions the test set over-represents.

### 4.2 Results (`python compare_models.py` → `reports/model_comparison.md`)

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

### 4.3 What the comparison taught us
* **Better features help every model.** The same XGBoost goes from 0.205 to 0.219 CV PR-AUC, and from
  0.304 to 0.352 on the test-like hold-out, just from the new features.
* **Logistic regression wins in-distribution but breaks under shift.** Spline logistic regression is
  among the best on ordinary CV (0.221) but collapses on the test-like hold-out (0.129). An additive
  model treats "large amount" as risk on its own, and its splines keep extrapolating that risk past the
  training range. In the hold-out, 58% of its top-3% alerts were payments above 1,500 from established
  accounts (median age 796 days), and only 21% of those alerts were fraud. The monotone XGBoost's top-3%
  alerts were new-device, young-account and velocity-burst rows, and 38% of them were fraud. Trees learn
  the "amount × other signals" interaction (§2.3) and extrapolate flat, which is safe. The EBM (a GAM
  with pairwise interactions) holds up much better than logistic regression, but still trails the trees.
* **Class weighting (`scale_pos_weight`) hurts.** Up-weighting the 353 frauds by about 55× makes the trees
  chase individual noisy positives. Ranking gets worse everywhere (test-like hold-out PR-AUC 0.352 → 0.202),
  and the probabilities lose calibration. We keep the natural class balance and handle imbalance where it
  belongs: in the threshold and in imbalance-aware metrics. For the same reason we did not use
  SMOTE or undersampling.
* **CatBoost and monotone XGBoost are the two strongest single models, with different strengths.**
  CatBoost has the best F1 in CV and on the test-like hold-out. Its ordered boosting and ordered target
  statistics for categoricals were designed to stop small, noisy datasets from overfitting their own
  targets. Monotone XGBoost has slightly higher PR-AUC. Neither dominates, which is exactly when blending pays off.
* **Monotone XGBoost adds complementary errors.** We force the score to be non-decreasing in amount,
  velocity, new device, night, merchant risk and signal count, and non-increasing in account age.
  These are rules a fraud analyst would sign off on, and they stop the trees learning implausible
  wiggles in sparse regions. Averaging it with CatBoost beats either model alone on PR-AUC in all three schemes.
* **Shallow, strongly regularised trees (depth 3, L2 penalty, row and column subsampling).** We tried
  depths 1–6. Deeper trees memorise the few hundred positives and lose ground, especially under shift.
* **Why not a neural network?** With 20k rows, 10 raw columns and 353 positives, gradient-boosted trees
  are the established state of the art for tabular data. A neural network would need far more data and
  would be harder to explain.

## 5. Choosing the decision threshold

**What the 0/1 column is judged on matters more than anything else in this section.** A 0/1 column has
only one point on its precision-recall curve, so its "PR-AUC" is
`recall × precision + (1 − recall) × fraud rate`. That rewards a *short, precise* list of alerts. F1 peaks
a little lower, and recall alone rewards flagging everything.

**Plug-in rule (default, `predict.py`).** If the probabilities are calibrated, flagging the top *k* rows
of the file being scored gives an expected number of caught frauds equal to the sum of their
probabilities. The expected total number of frauds is the sum over all rows. Expected precision, recall,
F1 and 0/1-column PR-AUC therefore follow for every *k* without any labels, and we flag the *k* that
maximises the expected 0/1-column PR-AUC. Two advantages over a fixed threshold:
* It **adapts to the risk mix of the file being scored**. The test set is riskier than the training set,
  and the rule sees that directly from the predicted probabilities.
* It **needs no extra tuning data**, only calibration, which we verified (`reports/calibration.md`).

On the test set it picks threshold 0.271 and flags 151 rows (1.3%), with expected precision 0.53 and
recall 0.30. We checked the rule out-of-fold by applying it to test-sized 4,000-row chunks of the
training predictions, exactly as `predict.py` applies it to the test set (`reports/threshold_tradeoff.md`):

| how the 0/1 column is cut                     |   flagged |   precision |   recall |     F1 |   PR-AUC of 0/1 column |   mean threshold |
|:----------------------------------------------|----------:|------------:|---------:|-------:|-----------------------:|-----------------:|
| plug-in: max expected binary PR-AUC (default) |    0.0067 |      0.5124 |   0.1936 | 0.2796 |                 0.1154 |           0.2833 |
| plug-in: max expected F1                      |    0.0118 |      0.3709 |   0.2484 | 0.2969 |                 0.1078 |           0.1586 |
| fixed threshold 0.095                         |    0.0198 |      0.2683 |   0.3022 | 0.2836 |                 0.0953 |           0.095  |
| fixed threshold 0.16                          |    0.0115 |      0.3777 |   0.2474 | 0.2982 |                 0.1088 |           0.16   |
| fixed threshold 0.25                          |    0.0072 |      0.4889 |   0.2011 | 0.2842 |                 0.1147 |           0.25   |
| fixed threshold 0.3                           |    0.006  |      0.5455 |   0.186  | 0.2769 |                 0.1179 |           0.3    |
| fixed threshold 0.4                           |    0.0047 |      0.608  |   0.1615 | 0.2544 |                 0.1163 |           0.4    |

The plug-in rule matches the best fixed thresholds (0.25–0.30) without being told them. It beats our
first submission's 0.095 cut by about 0.02 (+21%) on 0/1-column PR-AUC, at the cost of recall (0.19 vs
0.30 in CV; expected 0.30 on the riskier test set). To favour F1 or recall instead:
`python predict.py --rule f1` or `python predict.py --threshold 0.095`. `rethreshold.py` applies the same
rule to any existing submission file.

## 6. Assumptions of the model (and how we checked them)

| Assumption | Why it is needed | How we checked / mitigated it |
|---|---|---|
| **Transactions are independent.** There is no customer ID, so each row is scored on its own. | Needed for row-level CV and for tree models. | The velocity columns (`transactions_last_1h/24h`, `spend_last_24h`) already summarise each customer's recent history, so the most important sequential signal is present. |
| **The fraud mechanism P(fraud given features) is the same in train and test.** Only the *mix* of customers changes (covariate shift). | Any supervised model needs this. The labelled data is our only evidence of what fraud looks like. | We measured the shift (adversarial AUC 0.67) and **picked the model that holds up best when the shift is simulated** (shift-weighted CV, test-like hold-out). If fraudsters change tactics (concept drift), the model needs retraining on fresh labels. |
| **Labels are mostly correct, with random noise.** | Explains the PR-AUC ceiling. | 14% of frauds have no risk signals. We use strong regularisation, shallow trees and seed averaging so the model does not memorise them. |
| **Missing values are missing at random.** | Lets the trees treat NaN as "unknown" without inventing a meaning. | Fraud rates among missing rows sit near the base rate. We deliberately do not use missing-indicator features. |
| **Monotone effects:** more velocity, larger amounts, new devices, night-time, riskier merchants and more signals never *reduce* risk, and older accounts never *increase* it. | Encoded as XGBoost monotone constraints | Matches the empirical shape functions of an interpretable GAM (EBM) fitted to the data, and standard fraud domain knowledge. |
| **Features are available at decision time** (no leakage). | The model must work in real time. | Every feature comes from the transaction itself or the customer's *preceding* 1h/24h activity. There is no target encoding, and categorical encoders are fit inside each CV fold. |
| **The probabilities are calibrated** (no resampling or class weights) | The plug-in threshold rule reads expected frauds straight off the probabilities | Checked with a reliability table (`reports/calibration.md`): OOF predicted and observed fraud rates agree in every score decile, e.g. 9.0% vs 8.5% in the top decile. |
| **The leaderboard scores the submitted 0/1 labels** | Drives the threshold choice | Strongly supported by the leaderboard result (§7). If it turns out to read the probabilities instead, the threshold does not affect the score at all and nothing is lost. `rethreshold.py` gives a one-submission test. |

**Assumptions specific to the algorithms:**
* **Gradient-boosted trees** make no assumption about linearity, feature scale or distribution, and handle
  interactions and missing values natively. Their main limitation is piecewise-constant extrapolation:
  beyond the training range a tree predicts flat. Here that is an advantage, because the test set's
  extra-large amounts do not get extrapolated into ever-higher risk.
* **CatBoost** assumes rows are exchangeable: it builds its ordered statistics on random permutations of
  the training rows. That holds here, since rows are independent and the ids carry no time signal (the
  fraud rate is flat across id deciles).

## 7. What the leaderboard taught us

Our first submission (threshold 0.095) scored **0.1844**, below the original baseline's **0.1942**,
although every local test said it ranked frauds better. Instead of guessing, we tested explanations
against that gap (`python leaderboard_check.py` → `reports/leaderboard_check.md`):

| Explanation tested | Result |
|---|---|
| Covariate shift we had not modelled: re-weight validation rows to match the test set's exact mix of fraud-relevant segments (foreign × burst × young × new device × big amount × online) | Our model still ahead by +0.035 (± 0.013). **Rejected.** |
| The test set has 4× more missing values: inject test-level missingness into validation folds | Every model moves by less than 0.005. **Rejected.** |
| The leaderboard's "PR-AUC" is computed on the 0/1 column | Local gap −0.0102 vs leaderboard gap −0.0098. **Fits.** PR-AUC of the probabilities predicts the opposite sign (+0.021, about 2.9 paired standard errors from what was observed). |

Our first submission flagged 326 rows (2.7%) against the baseline's 226. The extra alerts had lower
precision, which costs more in a 0/1-column PR-AUC than the extra recall earns. This is a scoring
artefact, not a ranking problem: compared at the *same* number of alerts, our model has the highest
precision of the three at every alert budget we tried (0.4%–1.3% of rows).

**How much of a leaderboard move is noise?** With about 260 frauds in 12,000 test rows, one submission's
score has a bootstrap standard error of about 0.03. The *difference* between two similar submissions is
much less noisy (paired standard error about 0.003–0.01), but steps of +0.001 to +0.003 are still within
noise. We reconstructed the team's three v2 steps (dropping `share_of_24h_spend`, merchant-relative amount,
absolute amount deltas plus a country × merchant risk encoding). Locally, after each cumulative step the out-of-fold
PR-AUC sits +0.004, +0.003 and +0.000 above the baseline, all within about 1.5 standard errors. They are sensible features, but the
+0.006 leaderboard gain is not distinguishable from noise. The threshold effect is 3–4× larger.

**Takeaway for the submission:** decide the 0/1 cut for the metric it is judged on, and treat small
leaderboard differences as noise unless they are reproduced locally.

## 8. Limitations and next steps
* **The hidden-test score cannot be verified locally.** Our conclusions rest on out-of-fold evidence plus
  the leaderboard results we have. If the leaderboard turns out to score the probabilities, the threshold
  change is neutral and the remaining gap would point to concept drift in the test labels.
* With a customer or card ID we would add per-customer history features (deviation from that customer's
  usual amount, merchants and hours) and graph features linking devices to accounts. These are the biggest
  real-world gains in fraud detection.
* In production: monitor score and feature drift with the same adversarial-validation tool, retrain on
  fresh labels, and use the score in tiers (auto-approve, step-up verification, block) rather than as a
  single yes/no.

## 9. Reproducing

```bash
pip install -r requirements.txt
# put Track_2_Training_Dataset.csv and Track_2_Testing_Dataset.csv in data/
python analysis.py          # data findings      -> reports/data_analysis.md
python compare_models.py    # model comparison   -> reports/model_comparison.md   (~15 min)
python train.py             # final model        -> model/, reports/metrics.json, reports/threshold_tradeoff.md
python predict.py           # submission file    -> predictions.csv  (plug-in threshold; --rule f1 / --threshold T)
python leaderboard_check.py # metric diagnosis   -> reports/leaderboard_check.md   (~5 min)
python rethreshold.py their_submission.csv out.csv   # re-cut any submission's 0/1 column
```
