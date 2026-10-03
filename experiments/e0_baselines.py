"""E0: model x feature-set baselines (no target encoding yet)."""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb, lightgbm as lgb
from catboost import CatBoostClassifier
from src.cv import run_cv

df = pd.read_csv("data/train.csv")
POS_W = (df.fraud == 0).sum() / (df.fraud == 1).sum()  # ~55.6

def lr(seed):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"))

def xgb_shallow(seed, depth=2, n=400, lr_=0.05):
    return xgb.XGBClassifier(n_estimators=n, max_depth=depth, learning_rate=lr_, subsample=0.8,
                             colsample_bytree=0.8, min_child_weight=5, reg_lambda=5.0,
                             eval_metric="aucpr", random_state=seed, n_jobs=4, verbosity=0)

def lgbm(seed):
    return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=4, max_depth=2,
                              min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              reg_lambda=5.0, random_state=seed, n_jobs=4, verbose=-1)

def cat(seed):
    return CatBoostClassifier(iterations=600, depth=3, learning_rate=0.03, l2_leaf_reg=10,
                              random_seed=seed, verbose=0, thread_count=4, allow_writing_files=False)

results = []
for fs in ["raw", "basic", "merchant", "full"]:
    for name, mk in [("LogReg", lr), ("XGB_d2", xgb_shallow), ("LGBM_d2", lgbm), ("CatBoost_d3", cat)]:
        print(f"[{fs:8s}] {name}")
        r = run_cv(df, mk, feature_set=fs, n_repeats=4)
        results.append(dict(feature_set=fs, model=name, pr_auc=r["pooled_pr_auc_mean"], pr_auc_std=r["pooled_pr_auc_std"],
                            fold_pr_auc=r["fold_pr_auc_mean"], roc=r["roc_auc"], f1=r["best_f1"], recall=r["recall_at_best_f1"]))
res = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
print("\n=== SUMMARY (sorted by pooled PR-AUC) ===")
print(res.round(4).to_string(index=False))
res.to_csv("outputs/e0_baselines.csv", index=False)
