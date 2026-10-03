"""Builds notebooks/fraud_detection.ipynb from cells defined here.

Run `python train.py` first (the notebook reads its outputs), then:
    python notebooks/build_notebook.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/fraud_detection.ipynb
"""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = []

cells += [md("""# Track 2 — Transaction fraud detection

**Goal.** Score each transaction with a fraud probability (judged by **PR-AUC**) and a hard fraud / not-fraud decision (judged by **F1** and **Recall**).

**Answer in one paragraph.** The training data has 353 frauds in 20,000 transactions (1.77%). We benchmarked nine models with 5-fold × 3-repeat stratified cross-validation; every family (linear, spline GAM, random forest, LightGBM, XGBoost, CatBoost, EBM) lands at PR-AUC ≈ 0.21–0.23, ~12× the 0.018 no-skill baseline, so the limit is signal in the data, not model capacity. We chose an **Explainable Boosting Machine (EBM)**: it ties for the best PR-AUC, produces calibrated probabilities, is fully interpretable (every prediction is a sum of per-feature contributions), and degrades gracefully under the train→test drift we found. The decision threshold is chosen to maximise **F<sub>√3</sub>**, which is algebraically the harmonic mean of F1 and Recall — the two threshold-dependent metrics in the rubric.

The full write-up is in [`reports/MODEL_REPORT.md`](../reports/MODEL_REPORT.md); this notebook holds the evidence."""),
code("""import sys, json, pickle, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np, pandas as pd
from IPython.display import Image, display
from scipy.stats import chi2_contingency
from sklearn.metrics import average_precision_score

import fraud_pipeline as fp
import plots

pd.set_option("display.precision", 4)
pd.set_option("display.width", 160)
train, test = fp.load_data()
y = train[fp.TARGET].to_numpy()
FIG = fp.FIG_DIR
OUT = fp.OUTPUT_DIR
metrics = json.load(open(OUT / "metrics.json"))
print(f"train {train.shape}  test {test.shape}")
print(f"fraud: {y.sum()} of {len(y)} = {y.mean():.3%}  (≈ 1 in {round(1 / y.mean())})")"""),
]

cells += [md("""## 1. Understanding the problem

* **Unit of prediction:** one card / account transaction, described by its amount, hour, merchant category, country, channel, the account's age, whether the device is new, and short-window velocity (transactions and spend in the last 1 h / 24 h).
* **Severe class imbalance (1 : 56).** A model that predicts "not fraud" for everyone is 98.2% accurate and useless, so accuracy and ROC-AUC are poor guides. **PR-AUC** (average precision) measures how well frauds are ranked above legitimate transactions *without* rewarding the huge pool of easy true negatives; its no-skill value equals the prevalence (0.018).
* **F1 and Recall need a threshold.** The model outputs a probability; the business decision (block / review) is `p ≥ t`. Recall = share of frauds caught; precision = share of alerts that are real fraud. In fraud operations a missed fraud (FN) usually costs far more than a manual review (FP), so we deliberately lean towards recall — but not so far that precision (and hence F1) collapses.
* **Test set is unlabelled and comes from a later period** (IDs 20001–32000 follow the training IDs 1–20000). We therefore estimate performance with cross-validation and check explicitly how different the test distribution is."""),
code("""display(train.head())
summary = pd.DataFrame({
    "dtype": train.dtypes.astype(str),
    "missing_train_%": train.isna().mean() * 100,
    "missing_test_%": test.isna().mean().reindex(train.columns) * 100,
    "n_unique": train.nunique(),
})
summary"""),
]

cells += [md("""## 2. What drives fraud? (exploratory analysis)

Fraud rate by binned feature value. The horizontal line is the overall 1.77% rate."""),
code("""plots.fig_risk_drivers(train, FIG / "01_risk_drivers.png")
display(Image(FIG / "01_risk_drivers.png"))"""),
md("""**Reading the chart**

| Signal | Pattern | Fraud intuition |
|---|---|---|
| Velocity (`transactions_last_24h`, `transactions_last_1h`) | flat until ~10/day or ≥3/hour, then 13–25% fraud | card-testing / account drain bursts |
| `new_device` | 7.5% vs 1.2% | account takeover from an unfamiliar device |
| `account_age` | 8–180 days ≈ 4.5%, > 1 year ≈ 1% | mule / freshly opened accounts |
| `transaction_amount` | rises from ~$500, peaks at $2–3k | cash-out, but very large tickets are rarer |
| Merchant / channel | cash transfer, luxury, electronics; bank transfer & e-commerce | easily resold goods, card-not-present |
| Hour | higher 23:00–04:00 | victims asleep, fewer real customers |
| Country | non-SG ≈ 2–2.5× SG | cross-border risk |

The effects are **non-linear and step-like** (thresholds on velocity, a hump in account age), which rules out a plain linear model on raw values and motivates models that learn shape functions."""),
code("""# The interaction we expected from domain knowledge: new device on a young account
pd.crosstab(train.new_device.map({0: "known device", 1: "new device"}),
            np.where(train.account_age < 30, "account < 30 days", "account ≥ 30 days"),
            values=train.fraud, aggfunc="mean").style.format("{:.1%}")"""),
]

cells += [md("""### Data quality

* Every non-ID column has a little missingness (0.5–0.8% in train, 2–2.4% in test). Missing rows are not materially more fraudulent (χ² tests below), consistent with *missing completely at random*. The EBM puts missing values in their own bin, so no imputation is needed.
* `account_age` is capped at 4000 days and `transaction_amount` at 9000 (spikes at the maximum).
* `spend_last_24h` is below the current amount in ~20% of rows, so it excludes the current transaction — the velocity features look backwards only (no leakage)."""),
code("""rows = []
for c in fp.RAW_FEATURES:
    m = train[c].isna()
    if m.sum():
        chi2, pval, *_ = chi2_contingency(pd.crosstab(m, train.fraud))
        rows.append({"feature": c, "n_missing": int(m.sum()), "fraud_rate_if_missing": train.fraud[m].mean(),
                     "fraud_rate_if_present": train.fraud[~m].mean(), "chi2_p_value": pval})
pd.DataFrame(rows)"""),
]

cells += [md("""## 3. Is the test set like the training set? (adversarial validation)

We train a classifier to distinguish training rows from test rows. AUC = 0.5 would mean identical distributions."""),
code("""adv_auc, adv_imp, iw = fp.adversarial_validation(train, test)
plots.fig_drift(train, test, adv_auc, adv_imp, FIG / "02_drift.png")
display(Image(FIG / "02_drift.png"))
print(f"adversarial AUC = {adv_auc:.3f}")"""),
code("""def profile(d):
    return pd.Series({
        "rows": len(d), "new_device": d.new_device.mean(), "median_account_age": d.account_age.median(),
        "account_age<=180d": (d.account_age <= 180).mean(), "24h_txns>=10": (d.transactions_last_24h >= 10).mean(),
        "night (23-04h)": d.transaction_hour.isin([23, 0, 1, 2, 3, 4]).mean(), "country=SG": (d.country == "SG").mean(),
    })
big_tr, big_te = train.transaction_amount > 1000, test.transaction_amount > 1000
pd.DataFrame({"train: fraud": profile(train[train.fraud == 1]), "train: legit": profile(train[train.fraud == 0]),
              "train: amount>1000": profile(train[big_tr]), "test: amount>1000": profile(test[big_te]),
              "test: amount<=1000": profile(test[~big_te])}).T.style.format("{:.3f}")"""),
md("""**Interpretation.** Within the training set the distribution is stable across IDs, but the test set is a step change (AUC 0.67):

1. **A new block of large transactions** (amount > $1k: 13% of test vs 5% of train; $2k–6k: 9% vs 2%) coming from *old* accounts (median ~6 years), on *known* devices, in *daytime*, mostly outside SG. Apart from the amount, these look like low-risk premium customers.
2. **Among ordinary-sized transactions, more risk signals**: new devices 17% vs 9%, high 24h velocity doubled.

Implications for modelling:
* A model that over-trusts `transaction_amount` would flood its alerts with the new large-ticket segment. The EBM's amount effect is **bounded and flat above ~$1k** (see shape functions below), and the segment is judged mainly on its other (benign) attributes.
* Our working assumption is **covariate shift**: P(x) changes but P(fraud | x) does not. Under that assumption a calibrated model's probabilities stay valid, the probability threshold transfers, and we can forecast test metrics (Section 7).
* As a robustness check every model is also scored with **importance-weighted PR-AUC** — training rows re-weighted by w(x) = p(test|x)/p(train|x) so the validation set "looks like" the test set (column `pr_auc_test_like` below)."""),
]

cells += [md("""## 4. Model selection

Nine candidate models, each evaluated on the **same** 15 folds (5-fold stratified × 3 repeats; stratification keeps ~70 frauds in every validation fold). With only 353 positives a single split would give PR-AUC ± 0.04 noise, so repetition matters. `threshold`, `precision_at_t`, `recall_at_t` and `f1_at_t` use each model's own F<sub>√3</sub>-optimal threshold."""),
code("""cv = pd.read_csv(OUT / "cv_results.csv")
cols = ["model", "pr_auc_mean", "pr_auc_std", "pr_auc_test_like", "roc_auc_mean", "best_f1",
        "threshold", "precision_at_t", "recall_at_t", "f1_at_t", "brier", "note"]
cv[cols].style.format({c: "{:.4f}" for c in cols if c not in ("model", "note")}).background_gradient(
    subset=["pr_auc_mean", "pr_auc_test_like"], cmap="Blues")"""),
code("""display(Image(FIG / "03_model_comparison.png"))"""),
md("""**What the benchmark tells us**

* **All model families are within one standard deviation of each other** (PR-AUC 0.20–0.23 with fold SD ≈ 0.04). Boosted trees, which can model any interaction, do *not* beat additive models. That is evidence the fraud signal is essentially **additive in the log-odds with non-linear per-feature shapes** — exactly the structure a GAM / EBM assumes.
* **Re-weighting for imbalance does not help.** Class-balanced logistic regression and LightGBM with `scale_pos_weight` score the same or worse on PR-AUC and produce badly mis-calibrated probabilities (higher Brier score). PR-AUC is a ranking metric, and re-weighting doesn't improve the ranking; it only shifts probabilities. We handle imbalance where it matters — at the **decision threshold** — and keep calibrated probabilities. (SMOTE was not used for the same reason, and because synthesising points in a 1:56 problem with heavy noise risks fabricating fraud patterns.)
* **Why the EBM:** it ties for the top PR-AUC and the top importance-weighted (test-like) PR-AUC, is well-calibrated, needs no imputation or encoding, and is a *glass box*: we can show the judges — or a fraud analyst — exactly why each transaction was flagged.

### Assumption check: is "additive + pairwise interactions" enough?
EBM with no interactions (a pure GAM) vs the default EBM (main effects + automatically detected pairwise interactions). LightGBM/XGBoost/CatBoost in the table above stand in for "unrestricted higher-order interactions"."""),
code("""splits = fp.cv_splits(y)
spec_gam = fp.ModelSpec("EBM, main effects only (pure GAM)", lambda: fp.make_ebm(interactions=0))
res_gam = fp.cross_validate(spec_gam, train, splits)
row = fp.summarise_cv(res_gam, y, weights=iw)
ebm_row = cv.loc[cv.model == fp.FINAL_MODEL_NAME].iloc[0]
gbm_best = cv.loc[cv.model.isin(["LightGBM", "XGBoost", "CatBoost"])].sort_values("pr_auc_mean").iloc[-1]
pd.DataFrame([
    {"model": row["model"], "PR-AUC": row["pr_auc_mean"], "sd": row["pr_auc_std"]},
    {"model": "EBM, main + pairwise (chosen)", "PR-AUC": ebm_row.pr_auc_mean, "sd": ebm_row.pr_auc_std},
    {"model": f"best unrestricted GBM ({gbm_best.model})", "PR-AUC": gbm_best.pr_auc_mean, "sd": gbm_best.pr_auc_std},
])"""),
md("""Pairwise interactions add a little; unrestricted interactions add nothing. The additivity assumption is supported by the data."""),
]

cells += [md("""## 5. Inside the chosen model

An EBM is a **Generalised Additive Model with pairwise interactions (GA²M)**:

$$\\text{logit}\\,P(\\text{fraud}\\mid x) = \\beta_0 + \\sum_j f_j(x_j) + \\sum_{(i,j)} f_{ij}(x_i, x_j)$$

Each $f_j$ is learned by cyclic gradient boosting of shallow trees on **one feature at a time** (with a small learning rate and many rounds, so feature order doesn't matter), then bagged over 14 outer bags. The result is a lookup table per feature: the prediction is literally the sum of the table entries, so the model is exactly interpretable."""),
code("""with open(OUT / "ebm_model.pkl", "rb") as f:
    bundle = pickle.load(f)
ebm, THRESHOLD = bundle["model"], bundle["threshold"]
display(Image(FIG / "07_ebm_importance.png"))
display(Image(FIG / "08_ebm_shapes.png"))"""),
md("""**How to read the shape functions** (y-axis = change in log-odds; +0.69 ≈ doubles the odds):

* `new_device` adds ≈ +1.3 log-odds (~3.6× the odds) — the single strongest binary signal.
* `transactions_last_24h` rises steadily beyond ~7 per day; `transactions_last_1h` jumps at ≥3 per hour.
* `account_age` is highest for accounts under ~6 months and falls steadily after.
* `transaction_amount` rises from ~$100 to ~$1k and then **plateaus**. In the raw data, fraud rates *drop* above $3k; the EBM attributes that drop to the other features of those transactions (old accounts, daytime, known devices), not to the amount itself. That is the conditional vs marginal distinction, and it is why the model doesn't over-react to the test set's new large-ticket segment.
* Merchant category (cash transfer, luxury, electronics), channel (e-commerce, bank transfer) and country (ID, PH, GB, VN, JP) behave as the EDA suggested.
* Interactions are small: the largest pair has under a tenth of the importance of the top main effect.

### Explaining an individual decision"""),
code("""display(Image(FIG / "09_local_explanation.png"))"""),
]

cells += [md("""## 6. Choosing the decision threshold

The rubric scores **F1 and Recall** at our chosen threshold. Their harmonic mean simplifies to an F-beta score:

$$\\text{HM}(F_1, R) = \\frac{2 F_1 R}{F_1 + R} = \\frac{4PR}{3P + R} = F_{\\beta}\\Big|_{\\beta=\\sqrt{3}}$$

So we pick the threshold that maximises **F<sub>√3</sub>** on out-of-fold predictions (averaged over the three CV repeats to smooth noise). β = √3 ≈ 1.73 means recall is weighted ~1.7× precision, which also matches the cost asymmetry of fraud.

A useful sanity check: for a calibrated model the F<sub>β</sub>-optimal threshold equals F<sub>β</sub>* / (1 + β²) (Lipton et al., 2014). With F<sub>√3</sub>* ≈ 0.28 that predicts t ≈ 0.07, close to the empirical optimum."""),
code("""table = pd.read_csv(OUT / "threshold_table.csv")
display(Image(FIG / "05_threshold_tradeoff.png"))
pts = [0.03, 0.05, 0.055, 0.07, THRESHOLD, 0.10, 0.12, 0.16, 0.20, 0.30, 0.50]
op = table.set_index(table.threshold.round(3)).loc[[round(p, 3) for p in pts],
        ["precision", "recall", "f1", "f2", "f_beta", "flag_rate"]]
op.style.format("{:.3f}").highlight_max(subset=["f1", "f_beta"], color="#cde2fb")"""),
code("""best_fb = table.f_beta.max()
print(f"chosen threshold          t = {THRESHOLD:.3f}")
print(f"theory: F_beta*/(1+beta^2)  = {best_fb / 4:.3f}")
f1row = table.loc[table.f1.idxmax()]
at = table.loc[table.threshold == THRESHOLD].iloc[0]
print(f"at chosen t:   precision {at.precision:.3f}  recall {at.recall:.3f}  F1 {at.f1:.3f}")
print(f"F1-optimal t = {f1row.threshold:.3f}: precision {f1row.precision:.3f}  recall {f1row.recall:.3f}  F1 {f1row.f1:.3f}")
print(f"=> we give up {f1row.f1 - at.f1:.3f} F1 to gain {at.recall - f1row.recall:.3f} recall")"""),
md("""The F1 curve is almost flat between t ≈ 0.08 and 0.30, while recall falls steeply. Moving from the F1-optimum to our threshold costs very little F1 and buys a large gain in recall. The default 0.5 threshold would catch only ~10% of fraud."""),
code("""display(Image(FIG / "04_pr_curve.png"))"""),
]

cells += [md("""## 7. Calibration and a forecast for the test set

The threshold rule (and the theory above) assumes the probabilities are **calibrated**: among transactions scored 5%, about 5% should be fraud."""),
code("""display(Image(FIG / "06_calibration.png"))
oof = pd.read_csv(OUT / "oof_predictions.csv")
print(f"mean OOF probability {oof.oof_probability.mean():.4f} vs observed fraud rate {y.mean():.4f}")"""),
md("""Calibration is close to the diagonal across the deciles. Under the covariate-shift assumption we can therefore forecast test-set metrics from the predicted probabilities alone (E[TP] = Σ p over flagged rows, etc.):"""),
code("""sub = pd.read_csv(OUT / "submission.csv")
p_test = sub.fraud_probability.to_numpy()
fc = pd.DataFrame([fp.plug_in_metrics(p_test, t) for t in [0.06, 0.07, THRESHOLD, 0.10, 0.12, 0.15]])
print(f"model-implied test fraud rate: {p_test.mean():.2%} (train {y.mean():.3%})")
fc.style.format("{:.3f}")"""),
md("""The model expects slightly more fraud in the test period (≈2.2%) and better separability there (the drift brought more clear-cut velocity / new-device cases). These are forecasts, valid only if P(fraud | x) is unchanged. If fraudsters changed tactics (concept drift), no model trained on these labels can know it; production monitoring would be needed."""),
]

cells += [md("""## 8. Assumptions of the model and how we checked them

| # | Assumption | Why it matters | Evidence / mitigation |
|---|---|---|---|
| 1 | **Additivity**: log-odds = sum of per-feature shapes + a few pairwise interactions | the EBM cannot represent 3-way interactions | unrestricted GBMs (LightGBM, XGBoost, CatBoost) do not beat it (§4) |
| 2 | **Independent observations** | CV assumes rows are exchangeable | no customer/card ID is given, so we can't group folds; the velocity features summarise each account's history; the train set shows no ID trend |
| 3 | **Covariate shift only**: P(fraud \\| x) is stable between train and test | needed for probabilities and threshold to transfer | adversarial AUC 0.67 shows P(x) shifts; we checked importance-weighted PR-AUC; P(y \\| x) cannot be verified without test labels |
| 4 | **Calibrated probabilities** | the threshold is a probability cut-off | reliability curve on the diagonal; mean OOF p = observed rate |
| 5 | **Missing values are uninformative / MCAR** and behave the same in test | the EBM gives missing its own bin | χ² tests: missing rows not significantly more fraudulent; test has 4× more missing, which these bins absorb |
| 6 | **Piecewise-constant shapes, flat extrapolation** | values beyond the training range get the edge bin's score | desirable here: the test's large amounts are not extrapolated into extreme risk |
| 7 | **Labels are correct and complete** | label noise caps achievable PR-AUC | the plateau at ≈0.22 across all models suggests irreducible noise; chargeback lag could mean some frauds are labelled 0 |
| 8 | **Features are available at decision time** | otherwise there is leakage | velocity / spend windows exclude the current transaction (spend < amount in ~20% of rows) |"""),
]

cells += [md("""## 9. Submission file"""),
code("""print(sub.shape)
print(sub.fraud_prediction.value_counts().rename({0: "legit", 1: "fraud"}))
sub.sort_values("fraud_probability", ascending=False).head(10)"""),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
out = Path(__file__).resolve().parent / "fraud_detection.ipynb"
nbf.write(nb, out)
print(f"wrote {out}")
