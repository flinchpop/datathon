"""Self-contained fraud-detection ensemble for pickling (model.pkl).

    import pickle, pandas as pd
    model = pickle.load(open("model.pkl", "rb"))
    X = pd.read_csv("Track_2_Testing_Dataset.csv")      # id column optional
    scores = model.predict_proba(X)[:, 1]                # for PR-AUC
    labels = model.predict(X)                             # 0/1 at the chosen operating point (score >= 0.5)

Design
------
* Members: CatBoost x2 (native categoricals), XGBoost depth 2, LightGBM 4 leaves, logistic regression;
  each averaged over several seeds.  Identical to run_final.py / REPORT.md.
* Blending: each member's probability is mapped through its empirical CDF on a stored reference set
  (the test set at build time), then weight-averaged.  On the reference set this equals rank averaging;
  for any other batch it is a fixed, batch-independent monotone transform.
* Output score: blended value rescaled by a strictly increasing piecewise-linear map so that
  score >= 0.5  <=>  flagged (top `flag_share` of the reference set).  Ranks are preserved.
* Portability: fitted boosters are stored as native byte blobs (CatBoost .cbm, XGBoost UBJ, LightGBM text)
  and rebuilt lazily, so the pickle does not depend on those libraries' own pickle formats.  Learned
  statistics are plain dicts / numpy arrays (no pandas objects).  The class source is embedded in the
  pickle via __reduce__, so loading needs only: numpy, pandas, scikit-learn, xgboost, lightgbm, catboost.
"""
from __future__ import annotations

import io
import os
import tempfile

import numpy as np
import pandas as pd

RAW_NUM = ["transaction_amount", "transaction_hour", "transactions_last_24h", "spend_last_24h",
           "account_age", "new_device", "transactions_last_1h"]
RAW_CAT = ["merchant_category", "country", "transaction_channel"]
RAW_ALL = ["transaction_amount", "transaction_hour", "merchant_category", "country", "transaction_channel",
           "transactions_last_24h", "spend_last_24h", "account_age", "new_device", "transactions_last_1h"]
HI_RISK_MERCHANTS = {"luxury", "cash_transfer", "electronics"}

FEATURE_SETS = {
    "merchant": RAW_NUM + [
        "log_amount", "avg_ticket_24h", "spike_ratio", "amount_diff_avg_spend", "velocity_1h_share",
        "young_account", "new_device_young_account", "is_night", "hi_risk_merchant",
        "hi_risk_merchant_new_device", "is_foreign", "n_missing", "amount_ratio_to_merchant",
        "amount_diff_merchant_median", "amount_z_in_merchant"],
    "full": RAW_NUM + [
        "log_amount", "log_spend_24h", "log_account_age", "avg_ticket_24h", "spike_ratio", "log_spike_ratio",
        "amount_diff_avg_spend", "share_of_24h_spend", "velocity_1h_share", "txn_24h_sq", "high_velocity_24h",
        "high_velocity_1h", "spend_per_hour_24h", "young_account", "new_device_young_account",
        "new_device_x_log_amount", "age_per_txn", "is_night", "hour_sin", "hour_cos", "night_new_device",
        "hi_risk_merchant", "hi_risk_merchant_new_device", "is_foreign", "foreign_hi_risk", "n_missing",
        "amount_ratio_to_merchant", "amount_diff_merchant_median", "amount_z_in_merchant"],
}


# --------------------------------------------------------------------------------------
# feature engineering (mirrors src/features.py; kept here so the pickle is self-contained)
# --------------------------------------------------------------------------------------
def _coerce_input(X) -> pd.DataFrame:
    """Accept a DataFrame (any column order, optional id/fraud) or an ndarray in CSV column order."""
    if isinstance(X, pd.DataFrame):
        df = X.copy()
    else:
        arr = np.asarray(X, dtype=object)
        if arr.ndim != 2:
            raise ValueError("expected a 2-D array or DataFrame")
        if arr.shape[1] == len(RAW_ALL) + 1:          # id + 10 features
            df = pd.DataFrame(arr[:, 1:], columns=RAW_ALL)
        elif arr.shape[1] == len(RAW_ALL):
            df = pd.DataFrame(arr, columns=RAW_ALL)
        else:
            raise ValueError(f"expected {len(RAW_ALL)} or {len(RAW_ALL)+1} columns, got {arr.shape[1]}")
    missing = [c for c in RAW_ALL if c not in df.columns]
    if missing:
        raise ValueError(f"missing input columns: {missing}")
    df = df[RAW_ALL].copy()
    for c in RAW_NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    for c in RAW_CAT:
        df[c] = df[c].astype(object).where(df[c].notna(), np.nan)
    return df.reset_index(drop=True)


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    amt, n24, s24, n1, age = d["transaction_amount"], d["transactions_last_24h"], d["spend_last_24h"], \
        d["transactions_last_1h"], d["account_age"]
    d["log_amount"] = np.log1p(amt)
    d["log_spend_24h"] = np.log1p(s24)
    d["log_account_age"] = np.log1p(age)
    avg_ticket = s24 / n24.replace(0, np.nan)
    d["avg_ticket_24h"] = avg_ticket
    d["spike_ratio"] = amt / (avg_ticket + 1.0)
    d["log_spike_ratio"] = np.log1p(amt) - np.log1p(avg_ticket)
    d["amount_diff_avg_spend"] = amt - avg_ticket
    d["share_of_24h_spend"] = amt / (s24 + amt + 1.0)
    d["velocity_1h_share"] = n1 / n24.replace(0, np.nan)
    d["txn_24h_sq"] = n24 ** 2
    d["high_velocity_24h"] = (n24 >= 10).astype(float)
    d["high_velocity_1h"] = (n1 >= 3).astype(float)
    d["spend_per_hour_24h"] = s24 / 24.0
    d["young_account"] = (age < 90).astype(float)
    d["new_device_young_account"] = d["new_device"] * d["young_account"]
    d["new_device_x_log_amount"] = d["new_device"] * d["log_amount"]
    d["age_per_txn"] = age / (n24 + 1.0)
    h = d["transaction_hour"]
    d["is_night"] = h.isin([23, 0, 1, 2, 3]).astype(float)
    d["hour_sin"] = np.sin(2 * np.pi * h / 24.0)
    d["hour_cos"] = np.cos(2 * np.pi * h / 24.0)
    d["night_new_device"] = d["is_night"] * d["new_device"]
    d["hi_risk_merchant"] = d["merchant_category"].isin(HI_RISK_MERCHANTS).astype(float)
    d["hi_risk_merchant_new_device"] = d["hi_risk_merchant"] * d["new_device"]
    d["is_foreign"] = (d["country"] != "SG").astype(float)
    d["foreign_hi_risk"] = d["is_foreign"] * d["hi_risk_merchant"]
    d["n_missing"] = df[RAW_NUM + RAW_CAT].isnull().sum(axis=1).astype(float)
    return d


def fit_merchant_stats(ref: pd.DataFrame) -> dict:
    la = np.log1p(ref["transaction_amount"])
    g = ref.groupby("merchant_category")
    return {
        "median": g["transaction_amount"].median().to_dict(),
        "logmean": la.groupby(ref["merchant_category"]).mean().to_dict(),
        "logstd": la.groupby(ref["merchant_category"]).std().to_dict(),
        "global_median": float(ref["transaction_amount"].median()),
        "global_logmean": float(la.mean()), "global_logstd": float(la.std()),
    }


def add_merchant_relative_features(d: pd.DataFrame, st: dict) -> pd.DataFrame:
    d = d.copy()
    mc = d["merchant_category"]
    m = mc.map(st["median"]).astype(float).fillna(st["global_median"])
    d["amount_ratio_to_merchant"] = d["transaction_amount"] / (m + 1.0)
    d["amount_diff_merchant_median"] = d["transaction_amount"] - m
    lm = mc.map(st["logmean"]).astype(float).fillna(st["global_logmean"])
    ls = mc.map(st["logstd"]).astype(float).fillna(st["global_logstd"])
    d["amount_z_in_merchant"] = (np.log1p(d["transaction_amount"]) - lm) / (ls + 1e-6)
    return d


def _te_maps(keys: np.ndarray, y: np.ndarray, m: float) -> tuple[dict, float]:
    prior = float(np.mean(y))
    s = pd.DataFrame({"k": keys, "y": y}).groupby("k")["y"].agg(["sum", "count"])
    return ((s["sum"] + m * prior) / (s["count"] + m)).to_dict(), prior


def _te_keys(df: pd.DataFrame, c: str) -> np.ndarray:
    return df[c].fillna("NA").astype(str).values


def target_encode_train_oof(df: pd.DataFrame, y: np.ndarray, cols, m: float, seed: int):
    """OOF encodings for the training rows + full maps for scoring new rows."""
    from sklearn.model_selection import StratifiedKFold
    out = pd.DataFrame(index=df.index)
    maps = {}
    for c in cols:
        maps[c], prior = _te_maps(_te_keys(df, c), y, m)
        col = np.zeros(len(df))
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(df, y):
            mp, pr = _te_maps(_te_keys(df.iloc[tr], c), y[tr], m)
            col[va] = pd.Series(_te_keys(df.iloc[va], c)).map(mp).fillna(pr).values
        out[f"te_{c}"] = col
    return out, {"maps": maps, "prior": prior}


def target_encode_apply(df: pd.DataFrame, enc: dict, cols) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in cols:
        out[f"te_{c}"] = pd.Series(_te_keys(df, c)).map(enc["maps"][c]).fillna(enc["prior"]).values
    return out


# --------------------------------------------------------------------------------------
# member definitions
# --------------------------------------------------------------------------------------
MEMBERS = {
    # name: (kind, feature_set, uses_te, uses_native_cats, params, blend weight)
    "cat_d3_merch":    ("cat",  "merchant", False, True,  dict(iterations=600, depth=3, learning_rate=0.03, l2_leaf_reg=10), 1.00),
    "cat_d3_full":     ("cat",  "full",     False, True,  dict(iterations=600, depth=3, learning_rate=0.03, l2_leaf_reg=10), 1.00),
    "xgb_d2_full_te":  ("xgb",  "full",     True,  False, dict(n_estimators=400, max_depth=2, learning_rate=0.05, subsample=0.8,
                                                              colsample_bytree=0.8, min_child_weight=5, reg_lambda=5.0), 0.75),
    "lgbm_d2_full_te": ("lgbm", "full",     True,  False, dict(n_estimators=400, learning_rate=0.03, num_leaves=4, max_depth=2,
                                                              min_child_samples=40, subsample=0.8, subsample_freq=1,
                                                              colsample_bytree=0.8, reg_lambda=5.0), 0.75),
    "lr_full":         ("lr",   "full",     False, False, dict(C=0.1), 0.50),
}
TE_M = 20.0


class FraudEnsemble:
    """sklearn-style classifier: fit(train_df) / predict_proba(X) / predict(X)."""

    classes_ = np.array([0, 1])
    _estimator_type = "classifier"

    def __init__(self, seeds=(1000, 1001, 1002, 1003, 1004), flag_share=0.015, threads=4, source=None):
        self.seeds = list(seeds)
        self.flag_share = float(flag_share)
        self.threads = int(threads)
        self._source = source          # module source, embedded in the pickle (see __reduce__)
        self._cache = {}               # rebuilt boosters, never pickled

    # ----------------------------------------------------------------- feature matrices
    def _features(self, raw: pd.DataFrame, member: str, seed: int, train_y=None):
        kind, fs, use_te, use_cats, _, _ = MEMBERS[member]
        d = add_merchant_relative_features(add_basic_features(raw), self.merchant_stats_)
        X = d[FEATURE_SETS[fs]].astype(float)
        if use_te:
            if train_y is not None:
                te, enc = target_encode_train_oof(d, train_y, RAW_CAT, TE_M, seed)
                self.encoders_[(member, seed)] = enc
            else:
                te = target_encode_apply(d, self.encoders_[(member, seed)], RAW_CAT)
            X = pd.concat([X, te], axis=1)
        if use_cats:
            for c in RAW_CAT:
                X[c] = d[c].fillna("NA").astype(str).astype(object).values
        return X

    # ----------------------------------------------------------------- fitting
    def fit(self, train: pd.DataFrame, reference: pd.DataFrame | None = None):
        import xgboost as xgb, lightgbm as lgb
        from catboost import CatBoostClassifier, Pool
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression

        y = train["fraud"].values.astype(int)
        raw = _coerce_input(train)
        self.merchant_stats_ = fit_merchant_stats(raw)
        self.encoders_, self.blobs_, self.feature_names_ = {}, {}, {}
        for name, (kind, fs, use_te, use_cats, params, w) in MEMBERS.items():
            for s in self.seeds:
                X = self._features(raw, name, s, train_y=y)
                self.feature_names_[(name, s)] = list(X.columns)
                if kind == "cat":
                    m = CatBoostClassifier(**params, random_seed=s, verbose=0, thread_count=self.threads,
                                           allow_writing_files=False)
                    m.fit(Pool(X, y, cat_features=RAW_CAT))
                    with tempfile.TemporaryDirectory() as td:
                        p = os.path.join(td, "m.cbm"); m.save_model(p, format="cbm")
                        self.blobs_[(name, s)] = open(p, "rb").read()
                elif kind == "xgb":
                    m = xgb.XGBClassifier(**params, random_state=s, n_jobs=self.threads, verbosity=0)
                    m.fit(X, y)
                    self.blobs_[(name, s)] = bytes(m.get_booster().save_raw("ubj"))
                elif kind == "lgbm":
                    m = lgb.LGBMClassifier(**params, random_state=s, n_jobs=self.threads, verbose=-1)
                    m.fit(X, y)
                    self.blobs_[(name, s)] = m.booster_.model_to_string()
                elif kind == "lr":
                    imp = SimpleImputer(strategy="median").fit(X)
                    Xi = imp.transform(X)
                    sc = StandardScaler().fit(Xi)
                    lr = LogisticRegression(C=params["C"], max_iter=5000, class_weight="balanced").fit(sc.transform(Xi), y)
                    self.blobs_[(name, s)] = dict(median=imp.statistics_.astype(float), mean=sc.mean_.astype(float),
                                                  scale=sc.scale_.astype(float), coef=lr.coef_[0].astype(float),
                                                  intercept=float(lr.intercept_[0]))
        # reference distribution for the CDF blend + operating point
        ref = _coerce_input(reference) if reference is not None else raw
        member_probs = self._member_probs(ref)
        self.reference_ = {n: np.sort(p) for n, p in member_probs.items()}
        blend = self._blend(member_probs)
        self.threshold_ = float(np.quantile(blend, 1 - self.flag_share))
        self.blend_min_, self.blend_max_ = float(blend.min()), float(blend.max())
        return self

    # ----------------------------------------------------------------- scoring
    def _booster(self, key):
        if key in self._cache:
            return self._cache[key]
        kind = MEMBERS[key[0]][0]; blob = self.blobs_[key]
        if kind == "cat":
            from catboost import CatBoostClassifier
            b = CatBoostClassifier(); b.load_model(blob=blob)
        elif kind == "xgb":
            import xgboost as xgb
            b = xgb.Booster(); b.load_model(bytearray(blob))
        elif kind == "lgbm":
            import lightgbm as lgb
            b = lgb.Booster(model_str=blob)
        else:
            b = blob
        self._cache[key] = b
        return b

    def _member_probs(self, raw: pd.DataFrame) -> dict:
        out = {}
        for name, (kind, *_rest) in MEMBERS.items():
            ps = []
            for s in self.seeds:
                X = self._features(raw, name, s)
                X = X[self.feature_names_[(name, s)]]
                b = self._booster((name, s))
                if kind == "cat":
                    from catboost import Pool
                    p = b.predict_proba(Pool(X, cat_features=RAW_CAT))[:, 1]
                elif kind == "xgb":
                    import xgboost as xgb
                    p = b.predict(xgb.DMatrix(X.values.astype(float), feature_names=list(X.columns)))
                elif kind == "lgbm":
                    p = b.predict(X.values.astype(float))
                else:
                    Xv = X.values.astype(float)
                    Xv = np.where(np.isnan(Xv), b["median"], Xv)
                    z = ((Xv - b["mean"]) / b["scale"]) @ b["coef"] + b["intercept"]
                    p = 1.0 / (1.0 + np.exp(-z))
                ps.append(np.asarray(p, dtype=float))
            out[name] = np.mean(ps, axis=0)
        return out

    def _blend(self, member_probs: dict) -> np.ndarray:
        n = len(next(iter(member_probs.values())))
        acc, wsum = np.zeros(n), 0.0
        for name, p in member_probs.items():
            ref = self.reference_[name]; w = MEMBERS[name][5]
            # average-rank ECDF (equals scipy rankdata/len on the reference set itself)
            r = (np.searchsorted(ref, p, "left") + 1 + np.searchsorted(ref, p, "right")) / 2.0 / len(ref)
            acc += w * r; wsum += w
        return acc / wsum

    def _rescale(self, blend: np.ndarray) -> np.ndarray:
        thr, lo, hi = self.threshold_, self.blend_min_, self.blend_max_
        out = np.where(blend < thr, 0.5 * (blend - lo) / max(thr - lo, 1e-12),
                       0.5 + 0.5 * (blend - thr) / max(hi - thr, 1e-12))
        return np.clip(out, 0.0, 1.0)

    def decision_function(self, X) -> np.ndarray:
        """Score in [0,1]; >= 0.5 means flagged. Monotone in the rank-averaged ensemble."""
        raw = _coerce_input(X)
        return self._rescale(self._blend(self._member_probs(raw)))

    def predict_proba(self, X) -> np.ndarray:
        s = self.decision_function(X)
        return np.column_stack([1.0 - s, s])

    def predict(self, X) -> np.ndarray:
        return (self.decision_function(X) >= 0.5).astype(int)

    def predict_calibrated_proba(self, X) -> np.ndarray:
        """Mean of the tree members' probabilities: an interpretable fraud probability (training base rate 1.77%)."""
        mp = self._member_probs(_coerce_input(X))
        return np.mean([mp[n] for n in mp if MEMBERS[n][0] != "lr"], axis=0)

    # ----------------------------------------------------------------- pickling
    def __getstate__(self):
        st = dict(self.__dict__); st["_cache"] = {}
        return st

    def __reduce__(self):
        """Self-contained pickle: the class source is embedded and executed at load time, so unpickling needs
        no project module on the path.  (Plain `pickle.load` suffices; eval/exec are builtins.)"""
        if not self._source:
            raise RuntimeError("FraudEnsemble needs its module source (pass source=... at construction)")
        expr = ("(lambda ns: (exec(ns['__src__'], ns), ns['FraudEnsemble'].__new__(ns['FraudEnsemble']))[1])"
                "({'__src__': %r, '__name__': 'fraud_model_embedded'})") % (self._source,)
        return (eval, (expr,), self.__getstate__())

    def __setstate__(self, state):
        self.__dict__.update(state); self._cache = {}


def module_source() -> str:
    try:
        return open(__file__, encoding="utf-8").read()
    except NameError:  # executed from an embedded string
        return None
