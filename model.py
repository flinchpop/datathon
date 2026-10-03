"""The fraud model: an equal-weight blend of CatBoost and monotone-constrained XGBoost.

Why two gradient-boosted tree models:
  * CatBoost had the best F1 of any single model, in CV and on the test-like
    hold-out (see compare_models.py). Its ordered boosting and ordered target
    statistics for the categorical columns are designed for small, noisy data like
    ours (353 frauds).
  * XGBoost with monotone constraints encodes domain knowledge (more velocity, a
    larger amount, a newer account or a new device never *lowers* risk), which
    keeps it sensible in sparse regions the test set over-represents.
  * The two make different errors, so averaging their probabilities ranks better
    than either alone in every validation scheme we ran.
Both are shallow (depth 3) and strongly regularised: the signal is weak and
noisy, and deeper trees just memorise the few hundred positives.
"""
import json
import os

import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

from features import CATEGORICAL, MONOTONE, fit_feature_state, make_features

CATBOOST_PARAMS = dict(iterations=500, learning_rate=0.03, depth=3, l2_leaf_reg=10)
CATBOOST_SEEDS = (0, 1, 2)  # average a few seeds to reduce variance on a small dataset
XGB_PARAMS = dict(n_estimators=300, learning_rate=0.03, max_depth=3, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=5, reg_lambda=5)


def _for_catboost(X):
    X = X.copy()
    for c in CATEGORICAL:
        X[c] = X[c].astype(object).fillna("NA").astype(str)
    return X


class FraudEnsemble:
    def __init__(self, n_jobs=-1):
        self.n_jobs = n_jobs
        self.state = None
        self.cat_models, self.xgb_model = [], None

    def fit(self, df, y):
        y = np.asarray(y)
        self.state = fit_feature_state(df)
        X = make_features(df, self.state)
        self.cat_models = []
        for seed in CATBOOST_SEEDS:
            m = CatBoostClassifier(verbose=0, random_seed=seed, cat_features=CATEGORICAL,
                                   thread_count=self.n_jobs, allow_writing_files=False,
                                   **CATBOOST_PARAMS)
            m.fit(_for_catboost(X), y)
            self.cat_models.append(m)
        self.xgb_model = xgb.XGBClassifier(
            **XGB_PARAMS, enable_categorical=True, tree_method="hist", random_state=0,
            n_jobs=self.n_jobs, monotone_constraints=tuple(MONOTONE.get(c, 0) for c in X.columns))
        self.xgb_model.fit(X, y)
        return self

    def predict_components(self, df):
        X = make_features(df, self.state)
        p_cat = np.mean([m.predict_proba(_for_catboost(X))[:, 1] for m in self.cat_models], axis=0)
        p_xgb = self.xgb_model.predict_proba(X)[:, 1]
        return p_cat, p_xgb

    def predict_proba(self, df):
        p_cat, p_xgb = self.predict_components(df)
        return (p_cat + p_xgb) / 2

    def feature_importance(self, df):
        """Mean |SHAP| contribution (log-odds) per feature, averaged over both models."""
        X = make_features(df, self.state)
        pool = Pool(_for_catboost(X), cat_features=CATEGORICAL)
        cat_shap = np.mean([np.abs(m.get_feature_importance(data=pool, type="ShapValues")[:, :-1]).mean(0)
                            for m in self.cat_models], axis=0)
        dm = xgb.DMatrix(X, enable_categorical=True)
        xgb_shap = np.abs(self.xgb_model.get_booster().predict(dm, pred_contribs=True)[:, :-1]).mean(0)
        return dict(zip(X.columns, (cat_shap + xgb_shap) / 2))

    def save(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        for i, m in enumerate(self.cat_models):
            m.save_model(os.path.join(model_dir, f"catboost_{i}.cbm"))
        self.xgb_model.save_model(os.path.join(model_dir, "xgboost.json"))
        with open(os.path.join(model_dir, "feature_state.json"), "w") as f:
            json.dump(self.state, f, indent=2)

    @classmethod
    def load(cls, model_dir):
        self = cls()
        with open(os.path.join(model_dir, "feature_state.json")) as f:
            self.state = json.load(f)
        self.cat_models = []
        for i in range(len(CATBOOST_SEEDS)):
            m = CatBoostClassifier()
            m.load_model(os.path.join(model_dir, f"catboost_{i}.cbm"))
            self.cat_models.append(m)
        self.xgb_model = xgb.XGBClassifier()
        self.xgb_model.load_model(os.path.join(model_dir, "xgboost.json"))
        return self

