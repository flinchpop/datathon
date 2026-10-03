"""Compare candidate models under the three validation schemes in validation.py.

Usage: python compare_models.py [train.csv] [test.csv]
Writes reports/model_comparison.md. Takes ~15 minutes (EBM is the slow one and is
skipped if `interpret` is not installed).
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from features import CATEGORICAL, MONOTONE, NUMERIC, TARGET, fit_feature_state, make_features
from model import CATBOOST_PARAMS, CATBOOST_SEEDS, XGB_PARAMS, FraudEnsemble, _for_catboost
from validation import adversarial_weights, cross_validate, format_results

train_path = sys.argv[1] if len(sys.argv) > 1 else "data/Track_2_Training_Dataset.csv"
test_path = sys.argv[2] if len(sys.argv) > 2 else "data/Track_2_Testing_Dataset.csv"
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
y = train[TARGET].to_numpy()

_spec = importlib.util.spec_from_file_location("baseline", os.path.join("baseline", "train_xgboost.py"))
baseline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baseline)


def features_for(a, b):
    state = fit_feature_state(train.iloc[a])
    return make_features(train.iloc[a], state), make_features(train.iloc[b], state)


def original_baseline(a, b):
    X_tr, enc = baseline.build_train(train.iloc[a], train[TARGET].iloc[a])
    m = baseline.make_model()
    m.fit(X_tr, y[a])
    return m.predict_proba(baseline.prepare_features(train.iloc[b], enc))[:, 1]


def xgb_plain(a, b):
    Xa, Xb = features_for(a, b)
    m = xgb.XGBClassifier(**{**XGB_PARAMS, "max_depth": 2}, enable_categorical=True,
                          tree_method="hist", random_state=0, n_jobs=-1)
    return m.fit(Xa, y[a]).predict_proba(Xb)[:, 1]


def xgb_weighted(a, b):
    Xa, Xb = features_for(a, b)
    m = xgb.XGBClassifier(**{**XGB_PARAMS, "max_depth": 2}, enable_categorical=True,
                          tree_method="hist", random_state=0, n_jobs=-1,
                          scale_pos_weight=(y[a] == 0).sum() / y[a].sum())
    return m.fit(Xa, y[a]).predict_proba(Xb)[:, 1]


def xgb_monotone(a, b):
    Xa, Xb = features_for(a, b)
    m = xgb.XGBClassifier(**XGB_PARAMS, enable_categorical=True, tree_method="hist", random_state=0,
                          n_jobs=-1, monotone_constraints=tuple(MONOTONE.get(c, 0) for c in Xa.columns))
    return m.fit(Xa, y[a]).predict_proba(Xb)[:, 1]


def catboost_only(a, b):
    Xa, Xb = features_for(a, b)
    ps = [CatBoostClassifier(verbose=0, random_seed=s, cat_features=CATEGORICAL, allow_writing_files=False,
                             **CATBOOST_PARAMS).fit(_for_catboost(Xa), y[a]).predict_proba(_for_catboost(Xb))[:, 1]
          for s in CATBOOST_SEEDS]
    return np.mean(ps, axis=0)


def logistic_splines(a, b):
    Xa, Xb = features_for(a, b)
    num = [c for c in Xa.columns if c not in CATEGORICAL]
    for X in (Xa, Xb):
        for c in CATEGORICAL:
            X[c] = X[c].astype(object)
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), SplineTransformer(n_knots=5, degree=2),
                              StandardScaler()), num),
        ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="NA"),
                              OneHotEncoder(handle_unknown="ignore")), CATEGORICAL)])
    m = make_pipeline(pre, LogisticRegression(C=0.03, max_iter=2000))
    return m.fit(Xa, y[a]).predict_proba(Xb)[:, 1]


def ebm(a, b):
    from interpret.glassbox import ExplainableBoostingClassifier
    m = ExplainableBoostingClassifier(interactions=10, random_state=0)
    cols = NUMERIC + CATEGORICAL
    return m.fit(train.iloc[a][cols], y[a]).predict_proba(train.iloc[b][cols])[:, 1]


def ensemble(a, b):
    return FraudEnsemble().fit(train.iloc[a], y[a]).predict_proba(train.iloc[b])


CANDIDATES = [
    ("Original baseline (XGBoost d2)", original_baseline),
    ("XGBoost d2, new features", xgb_plain),
    ("XGBoost d2 + scale_pos_weight", xgb_weighted),
    ("Logistic regression (splines)", logistic_splines),
    ("EBM (GAM + 10 interactions)", ebm),
    ("XGBoost d3 monotone", xgb_monotone),
    ("CatBoost d3 (3 seeds)", catboost_only),
    ("FINAL: CatBoost + monotone XGBoost", ensemble),
]


def main():
    weights, adv_auc = adversarial_weights(train, test)
    print(f"Adversarial train-vs-test ROC-AUC {adv_auc:.3f}")
    rows = []
    for name, fp in CANDIDATES:
        if fp is ebm and importlib.util.find_spec("interpret") is None:
            print(f"Skipping {name}: pip install interpret-core")
            continue
        r, *_ = cross_validate(fp, train, y, weights)
        print(format_results(name, r), flush=True)
        rows.append([name, r["cv"]["pr_auc"], r["cv"]["roc_auc"], r["cv"]["best_f1"],
                     r["shift_weighted_cv"]["pr_auc"], r["shift_weighted_cv"]["best_f1"],
                     r["adversarial_holdout"]["pr_auc"], r["adversarial_holdout"]["best_f1"]])
    df = pd.DataFrame(rows, columns=["Model", "CV PR-AUC", "CV ROC-AUC", "CV best F1", "Shift-wtd PR-AUC",
                                     "Shift-wtd best F1", "Test-like hold-out PR-AUC",
                                     "Test-like hold-out best F1"])
    os.makedirs("reports", exist_ok=True)
    with open("reports/model_comparison.md", "w") as f:
        f.write("# Model comparison\n\n")
        f.write(f"Adversarial train-vs-test ROC-AUC: {adv_auc:.3f}. CV = 3x repeated stratified 5-fold; "
                "shift-weighted = same OOF predictions weighted by p(test|x)/p(train|x); test-like hold-out = "
                "train on the 70% least test-like rows, validate on the 30% most test-like.\n\n")
        f.write(df.to_markdown(index=False, floatfmt=".4f"))
        f.write("\n")
    print("\nWrote reports/model_comparison.md")


if __name__ == "__main__":
    main()
