"""Feature engineering for Track 2 fraud detection.

Design principles
-----------------
* Every transaction row is independent (no customer id), so all features are
  per-row transformations of the 10 raw columns.
* Prefer *relative* features (ratios to the account's own 24h baseline, ratios to
  the merchant-category median) because the hidden test set is covariate-shifted
  (higher amounts / older accounts).  Relative features travel better.
* Keep a few *absolute* features too (log amount, dollar deltas) so the model is
  not blind to scale -- a 5x spike on a $2 coffee is not a 5x spike on a $1000 bill.
* Target encodings are computed strictly out-of-fold (see cv.py) to avoid leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RAW_NUM = [
    "transaction_amount", "transaction_hour", "transactions_last_24h",
    "spend_last_24h", "account_age", "new_device", "transactions_last_1h",
]
RAW_CAT = ["merchant_category", "country", "transaction_channel"]
HI_RISK_MERCHANTS = {"luxury", "cash_transfer", "electronics"}


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise engineered features that need no label information."""
    d = df.copy()
    amt = d["transaction_amount"]
    n24 = d["transactions_last_24h"]
    s24 = d["spend_last_24h"]
    n1 = d["transactions_last_1h"]
    age = d["account_age"]

    # --- scale
    d["log_amount"] = np.log1p(amt)
    d["log_spend_24h"] = np.log1p(s24)
    d["log_account_age"] = np.log1p(age)

    # --- account's own baseline over the last 24h
    avg_ticket = s24 / n24.replace(0, np.nan)
    d["avg_ticket_24h"] = avg_ticket
    d["spike_ratio"] = amt / (avg_ticket + 1.0)            # +1 avoids blow-up on tiny baselines
    d["log_spike_ratio"] = np.log1p(amt) - np.log1p(avg_ticket)
    d["amount_diff_avg_spend"] = amt - avg_ticket            # absolute dollar delta
    d["share_of_24h_spend"] = amt / (s24 + amt + 1.0)

    # --- velocity
    d["velocity_1h_share"] = n1 / n24.replace(0, np.nan)
    d["txn_24h_sq"] = n24 ** 2
    d["high_velocity_24h"] = (n24 >= 10).astype(float)
    d["high_velocity_1h"] = (n1 >= 3).astype(float)
    d["spend_per_hour_24h"] = s24 / 24.0

    # --- account maturity x device
    d["young_account"] = (age < 90).astype(float)
    d["new_device_young_account"] = d["new_device"] * d["young_account"]
    d["new_device_x_log_amount"] = d["new_device"] * d["log_amount"]
    d["age_per_txn"] = age / (n24 + 1.0)

    # --- time of day
    h = d["transaction_hour"]
    d["is_night"] = h.isin([23, 0, 1, 2, 3]).astype(float)
    d["hour_sin"] = np.sin(2 * np.pi * h / 24.0)
    d["hour_cos"] = np.cos(2 * np.pi * h / 24.0)
    d["night_new_device"] = d["is_night"] * d["new_device"]

    # --- categorical interactions (as strings, encoded later)
    d["hi_risk_merchant"] = d["merchant_category"].isin(HI_RISK_MERCHANTS).astype(float)
    d["hi_risk_merchant_new_device"] = d["hi_risk_merchant"] * d["new_device"]
    d["is_foreign"] = (d["country"] != "SG").astype(float)
    d["foreign_hi_risk"] = d["is_foreign"] * d["hi_risk_merchant"]
    d["country_merchant"] = d["country"].fillna("NA") + "_" + d["merchant_category"].fillna("NA")
    d["merchant_channel"] = d["merchant_category"].fillna("NA") + "_" + d["transaction_channel"].fillna("NA")
    d["country_channel"] = d["country"].fillna("NA") + "_" + d["transaction_channel"].fillna("NA")

    # --- missingness
    d["n_missing"] = df[RAW_NUM + RAW_CAT].isnull().sum(axis=1).astype(float)
    return d


def add_merchant_relative_features(d: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Amount relative to the merchant-category spend distribution.

    `ref` is the frame used to compute the per-category statistics.  Amounts are
    *not* labels, so computing the medians on the full training set (or even on
    train+test) is leakage-free; we still expose `ref` so callers can decide.
    """
    d = d.copy()
    g = ref.groupby("merchant_category")["transaction_amount"]
    med = g.median()
    logmean = np.log1p(ref["transaction_amount"]).groupby(ref["merchant_category"]).mean()
    logstd = np.log1p(ref["transaction_amount"]).groupby(ref["merchant_category"]).std()
    global_med = ref["transaction_amount"].median()
    m = d["merchant_category"].map(med).fillna(global_med)
    d["amount_ratio_to_merchant"] = d["transaction_amount"] / (m + 1.0)
    d["amount_diff_merchant_median"] = d["transaction_amount"] - m
    lm = d["merchant_category"].map(logmean).fillna(np.log1p(ref["transaction_amount"]).mean())
    ls = d["merchant_category"].map(logstd).fillna(np.log1p(ref["transaction_amount"]).std())
    d["amount_z_in_merchant"] = (np.log1p(d["transaction_amount"]) - lm) / (ls + 1e-6)
    return d


class TargetEncoder:
    """m-estimate (smoothed mean) target encoder.

    fit() on a training frame, transform() on anything.  For *training* rows use
    fit_transform_oof() so that each row's encoding never sees its own label.
    """

    def __init__(self, cols, m: float = 20.0, n_splits: int = 5, seed: int = 0):
        self.cols = list(cols)
        self.m = m
        self.n_splits = n_splits
        self.seed = seed
        self.maps_: dict[str, pd.Series] = {}
        self.prior_: float = 0.0

    def _fit_maps(self, X: pd.DataFrame, y: np.ndarray):
        prior = float(np.mean(y))
        maps = {}
        for c in self.cols:
            s = pd.DataFrame({"k": X[c].fillna("NA").astype(str).values, "y": y})
            agg = s.groupby("k")["y"].agg(["sum", "count"])
            maps[c] = (agg["sum"] + self.m * prior) / (agg["count"] + self.m)
        return maps, prior

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.maps_, self.prior_ = self._fit_maps(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for c in self.cols:
            out[f"te_{c}"] = X[c].fillna("NA").astype(str).map(self.maps_[c]).fillna(self.prior_).astype(float)
        return out

    def fit_transform_oof(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        from sklearn.model_selection import StratifiedKFold
        self.fit(X, y)  # maps used later for the validation / test rows
        out = pd.DataFrame(index=X.index, columns=[f"te_{c}" for c in self.cols], dtype=float)
        skf = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.seed)
        for tr_idx, va_idx in skf.split(X, y):
            maps, prior = self._fit_maps(X.iloc[tr_idx], y[tr_idx])
            for c in self.cols:
                out.iloc[va_idx, out.columns.get_loc(f"te_{c}")] = (
                    X.iloc[va_idx][c].fillna("NA").astype(str).map(maps[c]).fillna(prior).astype(float).values
                )
        return out


# ---------------------------------------------------------------------------
# Feature sets -- named so experiments can be reproduced exactly.
# ---------------------------------------------------------------------------
TE_COLS_BASE = ["merchant_category", "country", "transaction_channel"]
TE_COLS_INTER = ["country_merchant", "merchant_channel", "country_channel"]

FEATURE_SETS = {
    # raw columns only
    "raw": RAW_NUM,
    # raw + simple relative / velocity features
    "basic": RAW_NUM + [
        "log_amount", "avg_ticket_24h", "spike_ratio", "amount_diff_avg_spend",
        "velocity_1h_share", "young_account", "new_device_young_account", "is_night",
        "hi_risk_merchant", "hi_risk_merchant_new_device", "is_foreign", "n_missing",
    ],
    # basic + merchant-relative amounts
    "merchant": RAW_NUM + [
        "log_amount", "avg_ticket_24h", "spike_ratio", "amount_diff_avg_spend",
        "velocity_1h_share", "young_account", "new_device_young_account", "is_night",
        "hi_risk_merchant", "hi_risk_merchant_new_device", "is_foreign", "n_missing",
        "amount_ratio_to_merchant", "amount_diff_merchant_median", "amount_z_in_merchant",
    ],
    # everything numeric (lets the model choose; used with strong regularisation)
    "full": RAW_NUM + [
        "log_amount", "log_spend_24h", "log_account_age", "avg_ticket_24h", "spike_ratio",
        "log_spike_ratio", "amount_diff_avg_spend", "share_of_24h_spend", "velocity_1h_share",
        "txn_24h_sq", "high_velocity_24h", "high_velocity_1h", "spend_per_hour_24h",
        "young_account", "new_device_young_account", "new_device_x_log_amount", "age_per_txn",
        "is_night", "hour_sin", "hour_cos", "night_new_device", "hi_risk_merchant",
        "hi_risk_merchant_new_device", "is_foreign", "foreign_hi_risk", "n_missing",
        "amount_ratio_to_merchant", "amount_diff_merchant_median", "amount_z_in_merchant",
    ],
}
