"""Explain the saved XGBoost model (model/): how much weight each column gets.

Uses SHAP contributions: for every transaction, XGBoost splits its fraud score
(log-odds) exactly into one contribution per feature plus a baseline.

  1. Feature weight: each feature's share of the total |contribution| (sums to 100%),
     on the training and on the test data, next to XGBoost's own "gain" importance.
  2. Column weight: the same, but engineered features are added back onto the column
     they come from (e.g. merchant_category + merchant_category_risk +
     is_high_risk_merchant). Ratio features mix columns, so they stay separate.
  3. Value effects: for each category (and bins of numeric columns), the average
     combined contribution as an odds multiplier relative to an average transaction
     (x2.0 doubles the fraud odds, x0.5 halves them).

Usage: python explain_xgboost.py
Writes to outputs/xgboost/: feature_weights.csv, column_weights.csv, value_effects.csv,
test_features.csv, test_shap_values.csv
"""
import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb

import train_xgboost as tx

TRAIN_PATH = "data/Track_2_Training_Dataset.csv"
TEST_PATH = "data/Track_2_Testing_Dataset.csv"
RATIOS = ["avg_spend_per_txn_24h", "spike_ratio", "share_of_24h_spend", "urgency_ratio"]
# Engineered features that are derived from a single original column.
DERIVED = {
    "merchant_category": ["merchant_category_risk", "is_high_risk_merchant"],
    "country": ["country_risk"],
    "transaction_channel": ["transaction_channel_risk"],
    "account_age": ["is_new_account"],
}
NUMERIC_BINS = {
    "transaction_amount": [0, 50, 100, 250, 500, 1000, 2500, np.inf],
    "transaction_hour": [-1, 4, 8, 12, 16, 20, 23],
    "transactions_last_24h": [0, 3, 6, 9, 12, 15, np.inf],
    "spend_last_24h": [-1, 100, 250, 500, 1000, 2500, np.inf],
    "account_age": [0, 30, 90, 180, 365, 730, 1825, np.inf],
    "new_device": [-1, 0, 1],
    "transactions_last_1h": [-1, 0, 1, 2, 3, 4, 6, np.inf],
    "spike_ratio": [0, 0.5, 1, 2, 5, 10, np.inf],
    "urgency_ratio": [-0.01, 0, 0.25, 0.5, 1],
}


def feature_group(feature):
    if feature in RATIOS:
        return "ratio (engineered)"
    if feature.endswith("_risk"):
        return "risk score (engineered)"
    if feature.startswith("is_"):
        return "flag (engineered)"
    if feature.endswith("_missing"):
        return "missing flag (engineered)"
    return "original column"


def source_column(feature):
    """Original column a feature is credited to (ratios stay on their own)."""
    if feature.endswith("_missing"):
        return feature[: -len("_missing")]
    for col, derived in DERIVED.items():
        if feature in derived:
            return col
    return feature


def bin_labels(edges, continuous=False):
    """Labels for (lo, hi] bins: whole-number columns get "1-50" style, ratios "0.5-1"."""
    labels = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == np.inf:
            labels.append(f">{lo:g}")
        elif continuous:
            labels.append("0" if hi == 0 else f"{max(lo, 0):g}-{hi:g}")
        elif int(lo) + 1 == int(hi):
            labels.append(str(int(hi)))
        else:
            labels.append(f"{int(lo) + 1}-{int(hi)}")
    return labels


def shap_frame(model, X):
    contribs = model.get_booster().predict(xgb.DMatrix(X, enable_categorical=True), pred_contribs=True)
    return pd.DataFrame(contribs[:, :-1], columns=X.columns, index=X.index), contribs[:, -1]


def share(frame):
    w = frame.abs().mean()
    return w / w.sum() * 100


def main():
    with open(os.path.join(tx.MODEL_DIR, "metadata.json")) as f:
        meta = json.load(f)
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(tx.MODEL_DIR, "xgb_fraud.json"))
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    # Encode the training rows exactly as the final model saw them (out-of-fold risk scores).
    X_train = tx.build_train(train, train[tx.TARGET])[0][meta["features"]]
    X_test = tx.prepare_features(test, meta["encoders"])[meta["features"]]
    shap_train, _ = shap_frame(model, X_train)
    shap_test, bias = shap_frame(model, X_test)

    # 1. Per feature.
    gain = pd.Series(model.get_booster().get_score(importance_type="gain")).reindex(X_train.columns).fillna(0)
    features = pd.DataFrame({
        "group": [feature_group(c) for c in X_train.columns],
        "weight_train_%": share(shap_train),
        "weight_test_%": share(shap_test),
        "gain_share_%": gain / gain.sum() * 100,
    }).sort_values("weight_train_%", ascending=False).round(2)

    # 2. Per original column (engineered single-column features added back).
    sources = [source_column(c) for c in X_train.columns]
    grouped_train = shap_train.T.groupby(sources).sum().T
    grouped_test = shap_test.T.groupby(sources).sum().T
    columns = pd.DataFrame({
        "includes": pd.Series(X_train.columns, index=X_train.columns).groupby(sources).agg(", ".join),
        "weight_train_%": share(grouped_train),
        "weight_test_%": share(grouped_test),
    }).sort_values("weight_train_%", ascending=False).round(2)

    # 3. Value effects on the combined contribution of each original column.
    rows = []
    for col in tx.CATEGORICAL + tx.RAW_NUMERIC + ["spike_ratio", "urgency_ratio"]:
        if col in tx.CATEGORICAL:
            groups = train[col].fillna("missing")
        else:
            values = X_train[col] if col in RATIOS else train[col]
            edges = NUMERIC_BINS[col]
            groups = pd.cut(values, edges, labels=bin_labels(edges, continuous=col in RATIOS))
            groups = groups.cat.add_categories("missing").fillna("missing")
        for level, idx in grouped_train[col].groupby(groups, observed=True).groups.items():
            rows.append({
                "column": col,
                "value": level,
                "rows": len(idx),
                "train_fraud_rate_%": round(train.loc[idx, tx.TARGET].mean() * 100, 2),
                "odds_multiplier": round(float(np.exp(grouped_train.loc[idx, col].mean())), 2),
            })
    effects = pd.DataFrame(rows)

    os.makedirs(tx.OUTPUT_DIR, exist_ok=True)
    features.to_csv(os.path.join(tx.OUTPUT_DIR, "feature_weights.csv"), index_label="feature")
    columns.to_csv(os.path.join(tx.OUTPUT_DIR, "column_weights.csv"), index_label="column")
    effects.to_csv(os.path.join(tx.OUTPUT_DIR, "value_effects.csv"), index=False)
    X_test.assign(id=test["id"].to_numpy())[["id"] + list(X_test.columns)].to_csv(
        os.path.join(tx.OUTPUT_DIR, "test_features.csv"), index=False)
    out = shap_test.round(5)
    out.insert(0, "id", test["id"].to_numpy())
    out["baseline"] = np.round(bias, 5)
    out["prediction"] = model.predict_proba(X_test)[:, 1]
    out.to_csv(os.path.join(tx.OUTPUT_DIR, "test_shap_values.csv"), index=False)

    print("Weight per original column (engineered single-column features added back):")
    print(columns.to_string(), "\n")
    print("Weight per model feature:")
    print(features.to_string(), "\n")
    for col in effects.column.unique():
        t = effects[effects.column == col].drop(columns="column")
        if col in tx.CATEGORICAL:
            t = t.sort_values("odds_multiplier", ascending=False)
        print(f"{col}:\n{t.to_string(index=False)}\n")
    print(f"Wrote feature_weights.csv, column_weights.csv, value_effects.csv, test_features.csv, "
          f"test_shap_values.csv to {tx.OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
