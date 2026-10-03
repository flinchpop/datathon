"""Explain the saved CatBoost model: how much weight each column and category gets.

Uses SHAP values (each feature's contribution to a transaction's fraud log-odds),
averaged over the seed models in model_catboost/.

  1. Column weight: share of the model's total |contribution| per column (sums to 100%).
  2. Value effect: for each category (and for bins of the numeric columns), the
     average contribution, shown as an odds multiplier (x2.0 = doubles the fraud odds
     relative to an average transaction, x0.5 = halves them).

Usage: python explain_catboost.py [path/to/Track_2_Training_Dataset.csv]
Writes outputs/catboost/column_weights.csv and outputs/catboost/value_effects.csv.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

import train_catboost as tc

DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/Track_2_Training_Dataset.csv"
# Bin edges for numeric columns; a bin covers (previous edge, edge].
NUMERIC_BINS = {
    "transaction_amount": [0, 50, 100, 250, 500, 1000, 2500, np.inf],
    "transaction_hour": [-1, 4, 8, 12, 16, 20, 23],
    "transactions_last_24h": [0, 3, 6, 9, 12, 15, np.inf],
    "spend_last_24h": [-1, 100, 250, 500, 1000, 2500, np.inf],
    "account_age": [0, 30, 90, 180, 365, 730, 1825, np.inf],
    "new_device": [-1, 0, 1],
    "transactions_last_1h": [-1, 0, 1, 2, 3, 4, 6, np.inf],
}


def bin_labels(edges):
    """Readable labels for integer-ish bins, e.g. (0, 50] -> "1-50", (2500, inf] -> ">2500"."""
    labels = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        lo_txt = int(lo) + 1
        if hi == np.inf:
            labels.append(f">{int(lo)}")
        elif lo_txt == int(hi):
            labels.append(str(int(hi)))
        else:
            labels.append(f"{lo_txt}-{int(hi)}")
    return labels


def shap_values(X):
    with open(os.path.join(tc.MODEL_DIR, "metadata.json")) as f:
        meta = json.load(f)
    pool = Pool(X, cat_features=tc.CATEGORICAL)
    vals = []
    for name in meta["model_files"]:
        m = CatBoostClassifier()
        m.load_model(os.path.join(tc.MODEL_DIR, name))
        vals.append(m.get_feature_importance(pool, type="ShapValues")[:, :-1])  # drop bias column
    return pd.DataFrame(np.mean(vals, axis=0), columns=X.columns, index=X.index)


def main():
    df = pd.read_csv(DATA_PATH)
    X = tc.prepare_features(df)
    shap = shap_values(X)

    weight = shap.abs().mean()
    col = (weight / weight.sum() * 100).sort_values(ascending=False).round(1).rename("weight_%")
    print("Column weights (share of total influence on the fraud score):")
    print(col.to_string(), "\n")

    rows = []
    for c in tc.FEATURES:
        if c in tc.CATEGORICAL:
            groups = X[c]
        else:
            edges = NUMERIC_BINS[c]
            groups = pd.cut(df[c], edges, labels=bin_labels(edges))
            groups = groups.cat.add_categories("missing").fillna("missing")
        for level, idx in shap[c].groupby(groups, observed=True).groups.items():
            rows.append({
                "column": c,
                "value": level,
                "rows": len(idx),
                "train_fraud_rate_%": round(df.loc[idx, tc.TARGET].mean() * 100, 2),
                "odds_multiplier": round(float(np.exp(shap.loc[idx, c].mean())), 2),
            })
    effects = pd.DataFrame(rows)
    for c in tc.FEATURES:
        t = effects[effects.column == c].drop(columns="column")
        if c in tc.CATEGORICAL:
            t = t.sort_values("odds_multiplier", ascending=False)
        print(f"{c}:\n{t.to_string(index=False)}\n")

    out_dir = os.path.join("outputs", "catboost")
    os.makedirs(out_dir, exist_ok=True)
    col.to_csv(os.path.join(out_dir, "column_weights.csv"), header=True, index_label="column")
    effects.to_csv(os.path.join(out_dir, "value_effects.csv"), index=False)
    print(f"Wrote column_weights.csv and value_effects.csv to {out_dir}/")


if __name__ == "__main__":
    main()
