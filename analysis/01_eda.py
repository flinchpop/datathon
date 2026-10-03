"""Exploratory analysis: fraud rates per column value, missing data, and train/test shift.

Writes:
  results/missing_values.csv        missing count per column and fraud rate when missing
  results/fraud_rate_by_value.csv   fraud rate for each category / numeric bin
  results/train_vs_test_summary.csv mean / median / 90th percentile, train vs test
  results/train_vs_test_shift.csv   legit vs fraud vs test share for key indicators, and the
                                    test fraud rate each would imply if only fraud prevalence changed
  results/id_drift.csv              fraud rate and feature levels across training id blocks
"""
import numpy as np
import pandas as pd

from common import CAT, NUM, RESULTS, test, train

BINS = {
    "transaction_amount": [0, 50, 100, 250, 500, 1000, 2500, np.inf],
    "spend_last_24h": [-1, 100, 250, 500, 1000, 2500, np.inf],
    "account_age": [0, 30, 90, 180, 365, 730, 1825, np.inf],
}

# Missing values.
miss = pd.DataFrame({
    "missing_train": train.isna().sum(),
    "missing_test": test.isna().sum().reindex(train.columns),
    "fraud_rate_when_missing_%": {c: train.loc[train[c].isna(), "fraud"].mean() * 100 for c in train.columns},
}).drop(index=["id", "fraud"]).round(2)
miss.to_csv(RESULTS / "missing_values.csv", index_label="column")

# Fraud rate per value.
rows = []
for c in NUM + CAT:
    g = pd.cut(train[c], BINS[c]).astype(str) if c in BINS else train[c]
    g = pd.Series(g, index=train.index).where(train[c].notna(), "missing").astype(str)
    t = train.groupby(g)["fraud"].agg(["size", "mean"])
    for v, r in t.iterrows():
        rows.append({"column": c, "value": v, "rows": int(r["size"]), "fraud_rate_%": round(r["mean"] * 100, 2)})
pd.DataFrame(rows).to_csv(RESULTS / "fraud_rate_by_value.csv", index=False)

# Train vs test summary of numeric columns.
summ = []
for c in NUM:
    for name, d in (("train", train), ("test", test)):
        summ.append({"column": c, "set": name, "mean": d[c].mean(), "median": d[c].median(),
                     "p90": d[c].quantile(0.9), "max": d[c].max()})
pd.DataFrame(summ).round(2).to_csv(RESULTS / "train_vs_test_summary.csv", index=False)

# Shift test: if the test set only had a different fraud rate, every indicator would
# imply the same rate. They disagree wildly (even negative), so legit customers changed.
INDICATORS = {
    "new_device = 1": lambda d: d.new_device == 1,
    "amount > 640": lambda d: d.transaction_amount > 640,
    "amount > 2000": lambda d: d.transaction_amount > 2000,
    "account_age < 342": lambda d: d.account_age < 342,
    "account_age > 2681": lambda d: d.account_age > 2681,
    "transactions_last_1h >= 4": lambda d: d.transactions_last_1h >= 4,
    "transactions_last_24h >= 11": lambda d: d.transactions_last_24h >= 11,
    "country = SG": lambda d: d.country == "SG",
    "country = AU": lambda d: d.country == "AU",
    "country = JP": lambda d: d.country == "JP",
    "channel = ecommerce": lambda d: d.transaction_channel == "ecommerce",
    "merchant = luxury": lambda d: d.merchant_category == "luxury",
    "merchant = travel": lambda d: d.merchant_category == "travel",
    "hour in 23-3": lambda d: d.transaction_hour.isin([23, 0, 1, 2, 3]),
}
shift = []
for k, f in INDICATORS.items():
    a, b, t = f(train[train.fraud == 0]).mean(), f(train[train.fraud == 1]).mean(), f(test).mean()
    shift.append({"indicator": k, "share_legit_train": a, "share_fraud_train": b, "share_test": t,
                  "implied_test_fraud_rate_%": (t - a) / (b - a) * 100})
pd.DataFrame(shift).round(3).to_csv(RESULTS / "train_vs_test_shift.csv", index=False)

# Drift within the training set by id (ids look sequential; test ids follow train ids).
n = train.id.str[2:].astype(int)
blk = pd.cut(n, 8)
drift = train.groupby(blk, observed=True).agg(
    fraud_rate=("fraud", "mean"), amount_median=("transaction_amount", "median"),
    amount_p90=("transaction_amount", lambda s: s.quantile(0.9)), new_device_rate=("new_device", "mean"),
    account_age_median=("account_age", "median"), share_SG=("country", lambda s: (s == "SG").mean()))
drift.index = drift.index.astype(str)
drift.round(3).to_csv(RESULTS / "id_drift.csv", index_label="id_block")

print(miss.to_string(), "\n")
print(pd.DataFrame(shift).round(3).to_string(index=False))
print("\nWrote results to", RESULTS)
