"""E1: categorical handling -- OOF target encodings vs CatBoost native categoricals."""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb, lightgbm as lgb
from catboost import CatBoostClassifier
from src.cv import run_cv
from src.features import TE_COLS_BASE, TE_COLS_INTER

df = pd.read_csv("data/train.csv")
NREP = 6
def lr(seed):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"))
def xgb_d2(seed):
    return xgb.XGBClassifier(n_estimators=400, max_depth=2, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                             min_child_weight=5, reg_lambda=5.0, random_state=seed, n_jobs=2, verbosity=0)
def lgbm(seed):
    return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=4, max_depth=2, min_child_samples=40,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, random_state=seed, n_jobs=2, verbose=-1)
def cat(seed):
    return CatBoostClassifier(iterations=600, depth=3, learning_rate=0.03, l2_leaf_reg=10, random_seed=seed, verbose=0, thread_count=2, allow_writing_files=False)

rows = []
def log(tag, fs, te, r):
    rows.append(dict(tag=tag, feature_set=fs, te=str(te), pr_auc=r["pooled_pr_auc_mean"], std=r["pooled_pr_auc_std"],
                     fold=r["fold_pr_auc_mean"], roc=r["roc_auc"], f1=r["best_f1"], recall=r["recall_at_best_f1"]))

for fs in ["merchant", "full"]:
    for te_name, te in [("none", None), ("base", TE_COLS_BASE), ("base+inter", TE_COLS_BASE + TE_COLS_INTER), ("inter_only", TE_COLS_INTER)]:
        for name, mk in [("LogReg", lr), ("XGB_d2", xgb_d2), ("LGBM_d2", lgbm), ("CatBoost_d3", cat)]:
            print(f"[{fs}] te={te_name} {name}", flush=True)
            r = run_cv(df, mk, feature_set=fs, te_cols=te, te_m=20.0, n_repeats=NREP)
            log(name, fs, te_name, r)

# TE smoothing strength
for m in [5.0, 50.0, 100.0]:
    print(f"[merchant] te=base+inter m={m} XGB_d2", flush=True)
    r = run_cv(df, xgb_d2, feature_set="merchant", te_cols=TE_COLS_BASE + TE_COLS_INTER, te_m=m, n_repeats=NREP)
    log(f"XGB_d2_m{m}", "merchant", "base+inter", r)

res = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
print("\n=== E1 SUMMARY ===\n", res.round(4).to_string(index=False))
res.to_csv("outputs/e1_encodings.csv", index=False)
