"""E2: CatBoost with native categorical features (ordered target statistics) + depth / regularisation sweep."""
import sys, warnings, time; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from catboost import CatBoostClassifier, Pool
from src.features import add_basic_features, add_merchant_relative_features, FEATURE_SETS
from src.cv import metrics

df = pd.read_csv("data/train.csv"); y = df.fraud.values
NREP = 6
CAT_SETS = {
    "raw3": ["merchant_category", "country", "transaction_channel"],
    "raw3+inter": ["merchant_category", "country", "transaction_channel", "country_merchant", "merchant_channel", "country_channel"],
}

def run(fs, cat_set, params, n_repeats=NREP, seed=42):
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=seed)
    oof = np.zeros((n_repeats, len(df))); t0 = time.time()
    cats = CAT_SETS[cat_set]
    for k, (tr_idx, va_idx) in enumerate(rskf.split(df, y)):
        tr = add_basic_features(df.iloc[tr_idx]); va = add_basic_features(df.iloc[va_idx])
        tr = add_merchant_relative_features(tr, tr); va = add_merchant_relative_features(va, tr)
        cols = FEATURE_SETS[fs] + cats
        Xtr = tr[cols].copy(); Xva = va[cols].copy()
        for c in cats: Xtr[c] = Xtr[c].fillna("NA").astype(str).astype(object); Xva[c] = Xva[c].fillna("NA").astype(str).astype(object)
        m = CatBoostClassifier(**params, random_seed=seed + k, verbose=0, thread_count=2, allow_writing_files=False)
        m.fit(Pool(Xtr, y[tr_idx], cat_features=cats))
        oof[k // 5, va_idx] = m.predict_proba(Pool(Xva, cat_features=cats))[:, 1]
    pooled = pd.DataFrame([metrics(y, oof[r]) for r in range(n_repeats)])
    out = dict(pr_auc=pooled.pr_auc.mean(), std=pooled.pr_auc.std(), roc=pooled.roc_auc.mean(), f1=pooled.best_f1.mean(), recall=pooled.recall_at_best_f1.mean(), secs=time.time() - t0)
    print(f"  PR-AUC {out['pr_auc']:.4f}±{out['std']:.4f} ROC {out['roc']:.4f} F1 {out['f1']:.4f} R {out['recall']:.3f} ({out['secs']:.0f}s)", flush=True)
    return out

rows = []
base = dict(iterations=600, depth=3, learning_rate=0.03, l2_leaf_reg=10)
for fs in ["merchant", "full"]:
    for cs in CAT_SETS:
        print(f"[{fs}] cats={cs} base", flush=True)
        r = run(fs, cs, base); rows.append(dict(feature_set=fs, cats=cs, params=str(base), **r))

# depth / lr / l2 sweep on the better combo (full + raw3+inter assumed; re-evaluated anyway)
for depth in [2, 3, 4, 5, 6]:
    for lr, iters in [(0.03, 600), (0.015, 1200)]:
        for l2 in [3, 10, 30]:
            p = dict(iterations=iters, depth=depth, learning_rate=lr, l2_leaf_reg=l2)
            print(f"[merchant] cats=raw3+inter {p}", flush=True)
            r = run("merchant", "raw3+inter", p, n_repeats=4); rows.append(dict(feature_set="merchant", cats="raw3+inter", params=str(p), **r))

res = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
print("\n=== E2 SUMMARY ===\n", res.round(4).to_string(index=False))
res.to_csv("outputs/e2_catboost_native.csv", index=False)
