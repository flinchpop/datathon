"""Shared helpers for the analysis scripts (data loading, CV, model builders).

Run every analysis script from the repo root, e.g. `python analysis/01_eda.py`.
Results are written to analysis/results/.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "analysis" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

import train_xgboost as tx  # noqa: E402  (the submitted model's feature pipeline)

CAT = ["merchant_category", "country", "transaction_channel"]
NUM = ["transaction_amount", "transaction_hour", "transactions_last_24h", "spend_last_24h",
       "account_age", "new_device", "transactions_last_1h"]

train = pd.read_csv(ROOT / "data" / "Track_2_Training_Dataset.csv")
test = pd.read_csv(ROOT / "data" / "Track_2_Testing_Dataset.csv")
y = train["fraud"].to_numpy()
LEVELS = {c: sorted(train[c].dropna().unique()) for c in CAT}


def raw_features(df):
    """The 10 original columns, categoricals encoded with the training levels."""
    X = df[NUM + CAT].copy()
    for c in CAT:
        X[c] = pd.Categorical(X[c], categories=LEVELS[c])
    return X


def weighted_average_precision(y_true, score, w):
    """PR-AUC (average precision) where each row counts with weight w."""
    order = np.argsort(-score)
    y_true, w = y_true[order], w[order]
    tp = np.cumsum(w * y_true)
    fp = np.cumsum(w * (1 - y_true))
    return float(np.sum(tp / (tp + fp) * w * y_true) / np.sum(w * y_true))


def load_adversarial_weights():
    """Test-likeness weights from 02_adversarial_validation.py (clipped at 10, mean 1)."""
    path = RESULTS / "adversarial_weights.csv"
    if not path.exists():
        sys.exit("Run analysis/02_adversarial_validation.py first.")
    w = pd.read_csv(path).set_index("id").loc[train["id"], "weight"].to_numpy()
    w = np.clip(w, 0, 10)
    return w / w.mean()


def out_of_fold(fit_predict, seed):
    """5-fold stratified OOF predictions; fit_predict(train_df, y_train, valid_df, seed)."""
    oof = np.zeros(len(y))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(train, y):
        oof[b] = fit_predict(train.iloc[a], y[a], train.iloc[b], seed)
    return oof


def scores(oof, w=None):
    out = {"pr_auc": average_precision_score(y, oof), "roc_auc": roc_auc_score(y, oof)}
    if w is not None:
        out["test_weighted_pr_auc"] = weighted_average_precision(y, oof, w)
    return out


# ---- model builders: each returns fit_predict(train_df, y_train, valid_df, seed) ----

def xgb_raw(depth=2, n=300, lr=0.03, spw=1.0, mcw=5, lam=5, sample_weight=None):
    def fp(a, ya, b, seed):
        m = xgb.XGBClassifier(n_estimators=n, learning_rate=lr, max_depth=depth, subsample=0.8,
                              colsample_bytree=0.8, min_child_weight=mcw, reg_lambda=lam,
                              scale_pos_weight=spw, enable_categorical=True, tree_method="hist",
                              random_state=seed, n_jobs=os.cpu_count())
        sw = None if sample_weight is None else sample_weight[a.index]
        m.fit(raw_features(a), ya, sample_weight=sw)
        return m.predict_proba(raw_features(b))[:, 1]
    return fp


def xgb_submitted(drop=()):
    """The submitted XGBoost pipeline (train_xgboost.py), optionally without some features."""
    drop = list(drop)

    def fp(a, ya, b, seed):
        Xa, enc = tx.build_train(a, pd.Series(ya, index=a.index))
        m = tx.make_model()
        m.set_params(random_state=seed)
        m.fit(Xa.drop(columns=drop), ya)
        return m.predict_proba(tx.prepare_features(b, enc).drop(columns=drop))[:, 1]
    return fp


def lgbm(n=400, lr=0.02, leaves=4):
    import lightgbm as lgb

    def fp(a, ya, b, seed):
        m = lgb.LGBMClassifier(n_estimators=n, learning_rate=lr, num_leaves=leaves, min_child_samples=40,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5,
                               random_state=seed, verbose=-1, n_jobs=os.cpu_count())
        m.fit(raw_features(a), ya)
        return m.predict_proba(raw_features(b))[:, 1]
    return fp


def catboost(depth=4, n=800, lr=0.03):
    from catboost import CatBoostClassifier

    def prep(df):
        X = df[NUM + CAT].copy()
        for c in CAT:
            X[c] = X[c].fillna("NA")
        return X

    def fp(a, ya, b, seed):
        m = CatBoostClassifier(iterations=n, learning_rate=lr, depth=depth, l2_leaf_reg=5, random_seed=seed,
                               verbose=0, cat_features=CAT, thread_count=-1, allow_writing_files=False)
        m.fit(prep(a), ya)
        return m.predict_proba(prep(b))[:, 1]
    return fp


def spline_logistic(C=0.3, knots=6):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

    def prep(df):
        X = df[NUM + CAT].copy()
        X["log_amt"] = np.log1p(X.transaction_amount)
        X["log_spend"] = np.log1p(X.spend_last_24h)
        X["log_age"] = np.log1p(X.account_age)
        for c in NUM + CAT:
            X[c + "_na"] = X[c].isna().astype(int)
        return X

    spl = ["log_amt", "log_spend", "log_age", "transactions_last_24h", "transactions_last_1h", "transaction_hour"]

    def fp(a, ya, b, seed):
        ct = ColumnTransformer([
            ("spl", make_pipeline(SimpleImputer(strategy="median"),
                                  SplineTransformer(n_knots=knots, extrapolation="linear")), spl),
            ("bin", SimpleImputer(strategy="most_frequent"), ["new_device"]),
            ("na", "passthrough", [c + "_na" for c in NUM + CAT]),
            ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="NA"),
                                  OneHotEncoder(handle_unknown="ignore")), CAT)])
        m = make_pipeline(ct, StandardScaler(with_mean=False), LogisticRegression(C=C, max_iter=3000))
        m.fit(prep(a), ya)
        return m.predict_proba(prep(b))[:, 1]
    return fp
