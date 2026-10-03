# Track 2 Fraud Detection: Approach, Model Choice and Assumptions

## TL;DR

* **Final model:** an equal-weight blend of **CatBoost** (depth 3, averaged over 3 seeds) and a
  **monotone-constrained XGBoost** (depth 3), trained on a small set of behavioural features.
* **Decision threshold:** 0.095 on the blended fraud probability. This is the lowest threshold whose
  out-of-fold F1 is within 5% of the best F1, so it gives up very little F1 for noticeably more recall.
* **Why this design:** the training data is small and noisy (353 frauds, 14% of which look exactly like
  normal transactions), and **the test set comes from a different distribution than the training set**.
  We therefore chose the model that held up best when we *simulated* that shift, not the one that won
  ordinary cross-validation.

| Validation scheme (see §4) | Metric | Original XGBoost | **Final model** | Change |
|---|---|---|---|---|
| Repeated 5-fold CV | PR-AUC | 0.205 | **0.226** | +10% |
| Shift-weighted CV (re-weighted to look like test) | PR-AUC | 0.321 | **0.356** | +11% |
| Test-like hold-out (train on least test-like 70%, validate on most test-like 30%) | PR-AUC | 0.304 | **0.364** | +20% |
| Test-like hold-out, at the chosen threshold | F1 / Recall | BASELINE_HOLDOUT_F1 / BASELINE_HOLDOUT_RECALL | **0.437 / 0.473** | |
| Shift-weighted CV, at the chosen threshold | F1 / Recall | BASELINE_SHIFT_F1 / BASELINE_SHIFT_RECALL | **0.372 / 0.449** | |

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
send high scores to step-up verification (OTP or call-back) rather than block them outright, so a false
alarm costs little friction while a missed fraud costs money. That asymmetry is why we pick the
**recall-leaning end** of the F1 plateau (§5) rather than the exact F1 maximum.

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

COMPARISON_TABLE

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
  wiggles in sparse regions. Averaging it with CatBoost beats either model alone on all three schemes.
* **Shallow, strongly regularised trees (depth 3, L2 penalty, row and column subsampling).** We tried
  depths 1–6. Deeper trees memorise the few hundred positives and lose ground, especially under shift.
* **Why not a neural network?** With 20k rows, 10 raw columns and 353 positives, gradient-boosted trees
  are the established state of the art for tabular data. A neural network would need far more data and
  would be harder to explain.

## 5. Choosing the decision threshold

The probability is turned into a 0/1 decision with a threshold chosen from **out-of-fold** predictions,
never from the training fit itself, so the choice is not optimistic.

* The F1-vs-threshold curve is **flat near its peak**: in CV, F1 varies by only ~0.015 between
  thresholds 0.09 and 0.20. With 353 frauds those points are statistically indistinguishable.
* Within that plateau we take the **lowest** threshold whose F1 is within 5% of the maximum: **0.095**.
  The F1 maximum itself is at 0.16. This trades almost no F1 for extra recall, consistent with the cost
  asymmetry in §1. On the test-like hold-out, 0.095 is also very close to the F1 peak.
* A probability threshold, unlike a fixed top-k%, **adapts to the test set's risk mix**. The test set
  contains more high-risk rows, so more of it gets flagged: 2.7% of test rows vs 2.0% of training rows.

Operating points (`reports/threshold_tradeoff.md`):

THRESHOLD_TABLE

Use `python predict.py --threshold T` to move along this curve.

## 6. Assumptions of the model (and how we checked them)

| Assumption | Why it is needed | How we checked / mitigated it |
|---|---|---|
| **Transactions are independent.** There is no customer ID, so each row is scored on its own. | Needed for row-level CV and for tree models. | The velocity columns (`transactions_last_1h/24h`, `spend_last_24h`) already summarise each customer's recent history, so the most important sequential signal is present. |
| **The fraud mechanism P(fraud given features) is the same in train and test.** Only the *mix* of customers changes (covariate shift). | Any supervised model needs this. The labelled data is our only evidence of what fraud looks like. | We measured the shift (adversarial AUC 0.67) and **picked the model that holds up best when the shift is simulated** (shift-weighted CV, test-like hold-out). If fraudsters change tactics (concept drift), the model needs retraining on fresh labels. |
| **Labels are mostly correct, with random noise.** | Explains the PR-AUC ceiling. | 14% of frauds have no risk signals. We use strong regularisation, shallow trees and seed averaging so the model does not memorise them. |
| **Missing values are missing at random.** | Lets the trees treat NaN as "unknown" without inventing a meaning. | Fraud rates among missing rows sit near the base rate. We deliberately do not use missing-indicator features. |
| **Monotone effects:** more velocity, larger amounts, new devices, night-time, riskier merchants and more signals never *reduce* risk, and older accounts never *increase* it. | Encoded as XGBoost monotone constraints | Matches the empirical shape functions of an interpretable GAM (EBM) fitted to the data, and standard fraud domain knowledge. |
| **Features are available at decision time** (no leakage). | The model must work in real time. | Every feature comes from the transaction itself or the customer's *preceding* 1h/24h activity. There is no target encoding, and categorical encoders are fit inside each CV fold. |
| **The probabilities are calibrated** (no resampling or class weights) | Lets one probability threshold carry over to a test set with a different risk mix | Checked with a reliability table: OOF predicted vs observed fraud rates agree within each of 20 bins. |
| **Missed fraud costs more than a false alarm** | Justifies the recall-leaning threshold | Configurable: `predict.py --threshold`. |

**Assumptions specific to the algorithms:**
* **Gradient-boosted trees** make no assumption about linearity, feature scale or distribution, and handle
  interactions and missing values natively. Their main limitation is piecewise-constant extrapolation:
  beyond the training range a tree predicts flat. Here that is an advantage, because the test set's
  extra-large amounts do not get extrapolated into ever-higher risk.
* **CatBoost** assumes rows are exchangeable: it builds its ordered statistics on random permutations of
  the training rows. That holds here, since rows are independent and the ids carry no time signal (the
  fraud rate is flat across id deciles).

## 7. Limitations and next steps
* **The hidden-test score cannot be verified locally.** Our improvements are measured on the training
  data under three validation schemes, including two that mimic the test shift. If the test labels follow a
  *different* fraud mechanism (concept drift), no training-only method can fully anticipate it.
* With a customer or card ID we would add per-customer history features (deviation from that customer's
  usual amount, merchants and hours) and graph features linking devices to accounts. These are the biggest
  real-world gains in fraud detection.
* In production: monitor score and feature drift with the same adversarial-validation tool, retrain on
  fresh labels, and use the score in tiers (auto-approve, step-up verification, block) rather than as a
  single yes/no.

## 8. Reproducing

```bash
pip install -r requirements.txt
# put Track_2_Training_Dataset.csv and Track_2_Testing_Dataset.csv in data/
python analysis.py          # data findings      -> reports/data_analysis.md
python compare_models.py    # model comparison   -> reports/model_comparison.md   (~15 min)
python train.py             # final model        -> model/, reports/metrics.json, reports/threshold_tradeoff.md
python predict.py           # submission file    -> predictions.csv
```
