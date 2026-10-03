"""Feature engineering shared by training, validation and prediction.

Every feature here is a fixed, label-free transform of a single row (plus the
training-set category levels and per-merchant typical amounts, which are learned
from training features only and stored with the model). There is no target
encoding, so nothing can leak a row's own label into its features.
"""
import numpy as np
import pandas as pd

TARGET = "fraud"
CATEGORICAL = ["merchant_category", "country", "transaction_channel"]
NUMERIC = ["transaction_amount", "transaction_hour", "transactions_last_24h", "spend_last_24h",
           "account_age", "new_device", "transactions_last_1h"]

# Domain thresholds, read off the training data (see analysis.py):
HIGH_RISK_MERCHANTS = ["luxury", "cash_transfer", "electronics"]  # 4-5% fraud vs ~1% elsewhere
NIGHT_HOURS = [0, 1, 2, 3, 23]                                      # ~2.5-3% fraud vs ~1.4% by day
YOUNG_ACCOUNT_DAYS = 120                                            # ~4% fraud below, ~1.2% above
BURST_TXN_1H, BURST_TXN_24H = 5, 16                                 # velocity attack: ~23% fraud
HOME_COUNTRY = "SG"                                                 # 70% of training traffic

# Expected sign of each feature's effect on fraud risk, enforced in the XGBoost model.
# A larger amount, more velocity, a younger account or a new device can never lower the score.
MONOTONE = {"log_amount": 1, "log_spend_24h": 1, "txn_24h": 1, "txn_1h": 1, "log_age": -1,
            "new_device": 1, "night": 1, "high_risk_merchant": 1, "young_account": 1,
            "velocity_burst": 1, "risk_signal_count": 1}


def fit_feature_state(df):
    """Learn the (label-free) state the features need from the training frame."""
    return {
        "categories": {c: sorted(df[c].dropna().unique().tolist()) for c in CATEGORICAL},
        "merchant_median_amount": df.groupby("merchant_category")["transaction_amount"].median().to_dict(),
    }


def make_features(df, state):
    """Raw transactions -> model features. Missing inputs stay NaN (both models handle them)."""
    amt, spend = df["transaction_amount"], df["spend_last_24h"]
    t1, t24, age = df["transactions_last_1h"], df["transactions_last_24h"], df["account_age"]
    hour, merchant, country = df["transaction_hour"], df["merchant_category"], df["country"]

    X = pd.DataFrame(index=df.index)
    # Raw signals (log scale for the heavy-tailed money/age columns).
    X["log_amount"] = np.log1p(amt)
    X["log_spend_24h"] = np.log1p(spend)
    X["txn_24h"] = t24
    X["txn_1h"] = t1
    X["log_age"] = np.log1p(age)
    X["new_device"] = df["new_device"]
    X["hour"] = hour

    # Behavioural ratios: is this payment unusual *for this customer / this merchant*?
    avg_24h = spend / t24.replace(0, np.nan)
    X["log_avg_spend_24h"] = np.log1p(avg_24h)
    X["amount_vs_avg_24h"] = np.log1p(amt) - np.log1p(avg_24h)
    merchant_typical = merchant.map(state["merchant_median_amount"]).astype(float)
    X["amount_vs_merchant"] = np.log1p(amt) - np.log1p(merchant_typical)
    X["share_txn_last_1h"] = t1 / t24.replace(0, np.nan)

    # Rule-of-thumb risk flags (NaN when the input is missing, rather than a fake 0).
    X["night"] = hour.isin(NIGHT_HOURS).astype(float)
    X["high_risk_merchant"] = merchant.isin(HIGH_RISK_MERCHANTS).astype(float).where(merchant.notna())
    X["foreign"] = (country != HOME_COUNTRY).astype(float).where(country.notna())
    X["young_account"] = (age < YOUNG_ACCOUNT_DAYS).astype(float).where(age.notna())
    X["velocity_burst"] = ((t1 >= BURST_TXN_1H) | (t24 >= BURST_TXN_24H)).astype(float)
    # Fraud risk compounds: the rate climbs from 0.7% (no signals) to >50% (4+ signals).
    X["risk_signal_count"] = (df["new_device"].fillna(0) + X["young_account"].fillna(0)
                              + X["velocity_burst"] + X["night"] + (amt > 500).astype(float)
                              + X["high_risk_merchant"].fillna(0))

    for c in CATEGORICAL:
        X[c] = pd.Categorical(df[c], categories=state["categories"][c])
    return X.replace([np.inf, -np.inf], np.nan)
