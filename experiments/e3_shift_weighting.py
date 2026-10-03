"""E3: covariate-shift handling.

(a) Importance weighting: w(x) = p(test|x)/p(train|x) from an adversarial classifier,
    used as sample_weight when fitting.  Evaluated two ways: plain CV PR-AUC and
    *importance-weighted* CV PR-AUC (an estimate of test-set PR-AUC under covariate shift).
(b) merchant_ref = train+val pooled medians vs train-only.
"""
import sys, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, pandas as pd
import xgboost as xgb, lightgbm as lgb
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import average_precision_score
from catboost import CatBoostClassifier
from src.cv import run_cv
from src.features import TE_COLS_BASE, TE_COLS_INTER, RAW_NUM, RAW_CAT

df = pd.read_csv("data/train.csv"); te_df = pd.read_csv("data/test.csv")
NREP = 6
# --- adversarial weights on train rows (OOF so they are honest)
X = pd.concat([df.drop(columns="fraud"), te_df], ignore_index=True).drop(columns="id")
for c in RAW_CAT: X[c] = X[c].astype("category")
yy = np.r_[np.zeros(len(df)), np.ones(len(te_df))]
adv = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15, verbose=-1, n_jobs=2)
p = cross_val_predict(adv, X, yy, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1][: len(df)]
prior = len(te_df) / len(df)
w_raw = (p / (1 - p)) / prior
w = np.clip(w_raw, 0.2, 5.0); w = w / w.mean()
df["_w"] = w
print("importance weights: mean %.3f, min %.3f, max %.3f, p90 %.3f" % (w.mean(), w.min(), w.max(), np.quantile(w, .9)))
print("weighted fraud rate (estimate of test base rate under covariate shift): %.4f" % np.average(df.fraud, weights=w))

def xgb_d2(seed):
    return xgb.XGBClassifier(n_estimators=400, max_depth=2, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                             min_child_weight=5, reg_lambda=5.0, random_state=seed, n_jobs=2, verbosity=0)
def cat(seed):
    return CatBoostClassifier(iterations=600, depth=3, learning_rate=0.03, l2_leaf_reg=10, random_seed=seed, verbose=0, thread_count=2, allow_writing_files=False)

def weighted_ap(y, s, w):
    # importance-weighted average precision
    o = np.argsort(-s); y = y[o]; w = w[o]
    tp = np.cumsum(w * y); fp = np.cumsum(w * (1 - y)); prec = tp / (tp + fp); rec = tp / tp[-1]
    return float(np.sum(np.diff(np.r_[0, rec]) * prec))

rows = []
TE = TE_COLS_BASE + TE_COLS_INTER
for name, mk in [("XGB_d2", xgb_d2), ("CatBoost_d3", cat)]:
    for fs in ["merchant", "full"]:
        for wt in ["none", "importance", "importance_sqrt"]:
            fn = None
            if wt == "importance": fn = lambda Xtr, ytr: df.loc[Xtr.index, "_w"].values
            if wt == "importance_sqrt": fn = lambda Xtr, ytr: np.sqrt(df.loc[Xtr.index, "_w"].values)
            print(f"[{fs}] {name} weight={wt}", flush=True)
            r = run_cv(df, mk, feature_set=fs, te_cols=TE, n_repeats=NREP, sample_weight_fn=fn, return_oof=True)
            wap = np.mean([weighted_ap(df.fraud.values, r["oof"][i], w) for i in range(NREP)])
            print(f"    importance-weighted PR-AUC (test-like): {wap:.4f}")
            rows.append(dict(model=name, feature_set=fs, weighting=wt, pr_auc=r["pooled_pr_auc_mean"], weighted_pr_auc=wap, f1=r["best_f1"], recall=r["recall_at_best_f1"]))
# merchant_ref pooled
for name, mk in [("XGB_d2", xgb_d2), ("CatBoost_d3", cat)]:
    print(f"[merchant] {name} merchant_ref=pooled", flush=True)
    r = run_cv(df, mk, feature_set="merchant", te_cols=TE, n_repeats=NREP, merchant_ref="pooled", return_oof=True)
    wap = np.mean([weighted_ap(df.fraud.values, r["oof"][i], w) for i in range(NREP)])
    rows.append(dict(model=name, feature_set="merchant", weighting="none|merchant_ref=pooled", pr_auc=r["pooled_pr_auc_mean"], weighted_pr_auc=wap, f1=r["best_f1"], recall=r["recall_at_best_f1"]))
res = pd.DataFrame(rows)
print("\n=== E3 SUMMARY ===\n", res.round(4).to_string(index=False))
res.to_csv("outputs/e3_shift.csv", index=False)
