"""E4: blending diverse members on identical CV splits + seed bagging.

All members are evaluated with run_cv(seed=42) so their OOF matrices line up row-for-row
and repeat-for-repeat; blends are then scored exactly like a single model would be.
"""
import sys, warnings, itertools, json; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, pandas as pd
from scipy.stats import rankdata
from src.cv import run_cv, metrics
from src.models import make_lr, make_xgb, make_lgbm, make_cat
from src.features import TE_COLS_BASE, TE_COLS_INTER

df = pd.read_csv("data/train.csv"); y = df.fraud.values
NREP = int(sys.argv[1]) if len(sys.argv) > 1 else 6
TE = TE_COLS_BASE + TE_COLS_INTER
CATS = ["merchant_category", "country", "transaction_channel", "country_merchant", "merchant_channel", "country_channel"]

members = {
    "lr_full_te":      dict(make_model=lambda s: make_lr(s, C=0.1), feature_set="full", te_cols=TE),
    "xgb_d2_merch_te": dict(make_model=lambda s: make_xgb(s), feature_set="merchant", te_cols=TE),
    "lgbm_d2_merch_te":dict(make_model=lambda s: make_lgbm(s), feature_set="merchant", te_cols=TE),
    "cat_d3_merch":    dict(make_model=lambda s: make_cat(s), feature_set="merchant", cat_cols=CATS),
    "cat_d3_full":     dict(make_model=lambda s: make_cat(s), feature_set="full", cat_cols=CATS),
    "cat_d4_merch":    dict(make_model=lambda s: make_cat(s, depth=4, l2_leaf_reg=30), feature_set="merchant", cat_cols=CATS),
}
oofs = {}
for name, cfg in members.items():
    print(name, flush=True)
    r = run_cv(df, n_repeats=NREP, return_oof=True, **cfg)
    oofs[name] = r["oof"]
np.save("outputs/e4_oofs.npy", oofs, allow_pickle=True)

def score(oof):
    pooled = pd.DataFrame([metrics(y, oof[r]) for r in range(NREP)])
    return pooled.pr_auc.mean(), pooled.pr_auc.std(), pooled.best_f1.mean(), pooled.recall_at_best_f1.mean()

def rank_blend(names, weights=None):
    weights = weights or [1.0] * len(names)
    out = np.zeros_like(oofs[names[0]])
    for n, w in zip(names, weights):
        for r in range(NREP):
            out[r] += w * rankdata(oofs[n][r]) / len(y)
    return out / sum(weights)

def prob_blend(names, weights=None):
    weights = weights or [1.0] * len(names)
    return sum(w * oofs[n] for n, w in zip(names, weights)) / sum(weights)

rows = []
for n in members: 
    pr, sd, f1, rec = score(oofs[n]); rows.append(dict(blend=n, kind="single", pr_auc=pr, std=sd, f1=f1, recall=rec))
names = list(members)
for k in [2, 3, 4]:
    for combo in itertools.combinations(names, k):
        for kind, fn in [("rank", rank_blend), ("prob", prob_blend)]:
            pr, sd, f1, rec = score(fn(list(combo))); rows.append(dict(blend="+".join(combo), kind=kind, pr_auc=pr, std=sd, f1=f1, recall=rec))
pr, sd, f1, rec = score(rank_blend(names)); rows.append(dict(blend="ALL", kind="rank", pr_auc=pr, std=sd, f1=f1, recall=rec))
res = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
print("\n=== E4 SUMMARY (top 25) ===\n", res.head(25).round(4).to_string(index=False))
print("\n=== singles ===\n", res[res.kind == "single"].round(4).to_string(index=False))
res.to_csv("outputs/e4_ensemble.csv", index=False)

# --- paired comparison: best blend vs best single, per repeat
best_single = res[res.kind == "single"].iloc[0].blend
best_blend_row = res[res.kind != "single"].iloc[0]
bb = best_blend_row.blend.split("+") if best_blend_row.blend != "ALL" else names
fn = rank_blend if best_blend_row.kind == "rank" else prob_blend
a = np.array([metrics(y, oofs[best_single][r])["pr_auc"] for r in range(NREP)])
b = np.array([metrics(y, fn(bb)[r])["pr_auc"] for r in range(NREP)])
print(f"\npaired per-repeat PR-AUC: {best_single}={a.round(4)}  vs  blend={b.round(4)}  diff mean={np.mean(b-a):+.4f} wins={np.sum(b>a)}/{NREP}")
