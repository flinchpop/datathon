"""Model factories used by experiments and the final pipeline."""
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb, lightgbm as lgb
from catboost import CatBoostClassifier, Pool


class CatNative:
    """CatBoost that treats every object/string column as a categorical feature.

    CatBoost's ordered target statistics give a leakage-safe target encoding for
    high-cardinality interactions (country x merchant) without hand-rolled OOF code.
    """
    def __init__(self, seed=0, threads=2, **params):
        self.params = dict(iterations=600, depth=3, learning_rate=0.03, l2_leaf_reg=10)
        self.params.update(params)
        self.seed, self.threads = seed, threads

    def _cats(self, X):
        return [c for c in X.columns if X[c].dtype == object or str(X[c].dtype) in ("string", "str")]

    def fit(self, X, y, sample_weight=None):
        self.cats_ = self._cats(X)
        self.model_ = CatBoostClassifier(**self.params, random_seed=self.seed, verbose=0,
                                         thread_count=self.threads, allow_writing_files=False)
        self.model_.fit(Pool(X, y, cat_features=self.cats_, weight=sample_weight))
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(Pool(X, cat_features=self.cats_))

    @property
    def feature_importances_(self):
        return self.model_.get_feature_importance()


def make_lr(seed=0, C=0.1):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))


def make_xgb(seed=0, threads=2, **kw):
    p = dict(n_estimators=400, max_depth=2, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
             min_child_weight=5, reg_lambda=5.0)
    p.update(kw)
    return xgb.XGBClassifier(**p, random_state=seed, n_jobs=threads, verbosity=0)


def make_lgbm(seed=0, threads=2, **kw):
    p = dict(n_estimators=400, learning_rate=0.03, num_leaves=4, max_depth=2, min_child_samples=40,
             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0)
    p.update(kw)
    return lgb.LGBMClassifier(**p, random_state=seed, n_jobs=threads, verbose=-1)


def make_cat(seed=0, threads=2, **kw):
    return CatNative(seed=seed, threads=threads, **kw)
