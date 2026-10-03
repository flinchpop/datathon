"""Reproduce the data findings behind the modelling choices.

Usage: python analysis.py [train.csv] [test.csv]   -> reports/data_analysis.md
"""
import os
import sys

import numpy as np
import pandas as pd

from features import (BURST_TXN_1H, BURST_TXN_24H, HIGH_RISK_MERCHANTS, HOME_COUNTRY, NIGHT_HOURS,
                      YOUNG_ACCOUNT_DAYS)

train = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "data/Track_2_Training_Dataset.csv")
test = pd.read_csv(sys.argv[2] if len(sys.argv) > 2 else "data/Track_2_Testing_Dataset.csv")


def signals(d):
    return pd.DataFrame({
        "new_device": d.new_device == 1,
        f"young_account (<{YOUNG_ACCOUNT_DAYS}d)": d.account_age < YOUNG_ACCOUNT_DAYS,
        f"velocity_burst (1h>={BURST_TXN_1H} or 24h>={BURST_TXN_24H})":
            (d.transactions_last_1h >= BURST_TXN_1H) | (d.transactions_last_24h >= BURST_TXN_24H),
        "night (23:00-03:59)": d.transaction_hour.isin(NIGHT_HOURS),
        "amount > 500": d.transaction_amount > 500,
        "high_risk_merchant": d.merchant_category.isin(HIGH_RISK_MERCHANTS),
        "foreign (not SG)": d.country.notna() & (d.country != HOME_COUNTRY),
    })


def md(df, floatfmt=".4f"):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else format(v, floatfmt))
    return df.to_markdown()


out = []
y = train.fraud
out.append(f"# Data analysis\n\nTraining rows: {len(train)}, frauds: {y.sum()} ({y.mean():.2%}). "
           f"Test rows: {len(test)}.\n")

s_tr, s_te = signals(train), signals(test)
rows = []
for c in s_tr:
    m = s_tr[c]
    rows.append([c, y[m].mean(), y[~m].mean(), y[m].mean() / y[~m].mean(), m.mean(), s_te[c].mean()])
out.append("## 1. Single risk signals\n\nFraud rate with/without each signal, and how common the "
           "signal is in train vs test.\n")
out.append(md(pd.DataFrame(rows, columns=["signal", "fraud rate if present", "fraud rate if absent",
                                          "lift", "share of train", "share of test"]).set_index("signal")))

core = s_tr.columns[:6]
n_tr, n_te = s_tr[core].sum(axis=1), s_te[core].sum(axis=1)
t = train.groupby(n_tr).fraud.agg(["mean", "sum", "count"])
t.columns = ["fraud rate", "frauds", "rows"]
t["share of train"] = n_tr.value_counts(normalize=True).sort_index()
t["share of test"] = n_te.value_counts(normalize=True).sort_index()
out.append("\n\n## 2. Risk compounds\n\nFraud rate by the number of the first six signals present. "
           f"Risk climbs steeply as signals stack up. The test set has fewer signal-free rows "
           f"({(n_te == 0).mean():.0%} vs {(n_tr == 0).mean():.0%}) and {(n_te >= 4).mean() / (n_tr >= 4).mean():.1f}x "
           "the share of 4+ signal rows.\n")
out.append(md(t))

out.append(f"\n\n**Irreducible noise:** {int(y[n_tr == 0].sum())} of {y.sum()} frauds "
           f"({y[n_tr == 0].sum() / y.sum():.0%}) show none of the six signals. They are indistinguishable "
           "from ordinary transactions, which caps achievable PR-AUC/recall for any model.\n")

rows = []
for c in ["transaction_amount", "account_age", "spend_last_24h", "transactions_last_24h", "transactions_last_1h"]:
    rows.append([c, train[c].median(), test[c].median(), train[c].quantile(0.9), test[c].quantile(0.9)])
rows.append(["new_device (mean)", train.new_device.mean(), test.new_device.mean(), np.nan, np.nan])
out.append("\n## 3. Train -> test distribution shift\n\n")
out.append(md(pd.DataFrame(rows, columns=["feature", "train median", "test median", "train p90",
                                          "test p90"]).set_index("feature"), ".2f"))
hi_tr, hi_te = train[train.transaction_amount > 1500], test[test.transaction_amount > 1500]
out.append(f"\n\nHigh-value rows (> 1500): {len(hi_tr) / len(train):.1%} of train vs "
           f"{len(hi_te) / len(test):.1%} of test. In test they come from old accounts (median age "
           f"{hi_te.account_age.median():.0f}d vs {hi_tr.account_age.median():.0f}d in train), rarely a new "
           f"device ({hi_te.new_device.mean():.1%}), mostly daytime. In train, high-value payments from calm, "
           "established accounts are close to the base fraud rate - amount is only risky *together with* other "
           "signals, which is why tree models (that learn interactions) beat additive ones here.\n")

calm_old = (train.new_device == 0) & (train.transactions_last_1h <= 2) & \
           (train.transactions_last_24h <= 8) & (train.account_age > 365)
rows = []
for lo, hi in [(0, 200), (200, 500), (500, 1000), (1000, 2000), (2000, 1e9)]:
    a = train.transaction_amount.between(lo, hi, inclusive="left")
    rows.append([f"[{lo:g}, {hi:g})", y[a & calm_old].mean(), (a & calm_old).sum(),
                 y[a & ~calm_old].mean(), (a & ~calm_old).sum()])
out.append("\nFraud rate by amount, for calm established accounts vs everyone else:\n\n")
out.append(md(pd.DataFrame(rows, columns=["amount", "calm & old: fraud rate", "rows",
                                          "others: fraud rate", "rows"]).set_index("amount")))

rows = []
for c in train.columns.drop(["id", "fraud"]):
    m = train[c].isna()
    if m.any():
        rows.append([c, m.sum(), int(y[m].sum()), y[m].mean(), test[c].isna().mean()])
out.append("\n\n## 4. Missing values\n\nMissing values look random (fraud rate among missing rows is "
           "close to the 1.8% base rate), so we let the trees route NaNs natively and do not use "
           "'is missing' flags as fraud signals.\n\n")
out.append(md(pd.DataFrame(rows, columns=["column", "missing rows", "frauds", "fraud rate",
                                          "test missing share"]).set_index("column")))

os.makedirs("reports", exist_ok=True)
with open("reports/data_analysis.md", "w") as f:
    f.write("\n".join(out) + "\n")
print("\n".join(out))
