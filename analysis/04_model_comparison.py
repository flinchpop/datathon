"""Compare model families and settings with repeated 5-fold CV (seeds 0, 1, 2).

Reports mean PR-AUC, ROC-AUC and test-weighted PR-AUC (see 02_adversarial_validation.py),
then tries blends (rank- and probability-averages) of the saved OOF predictions.

Writes: results/model_comparison.csv, results/blend_comparison.csv,
        results/oof_predictions_by_model.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from common import (RESULTS, catboost, lgbm, load_adversarial_weights, out_of_fold, scores,
                    spline_logistic, train, xgb_raw, xgb_submitted, y)

W = load_adversarial_weights()
SEEDS = (0, 1, 2)
MODELS = {
    "xgb_submitted (depth2, engineered features)": xgb_submitted(),
    "xgb_raw depth2 300t lr0.03": xgb_raw(),
    "xgb_raw depth1 600t lr0.05": xgb_raw(depth=1, n=600, lr=0.05),
    "xgb_raw depth3 600t lr0.02": xgb_raw(depth=3, n=600, lr=0.02),
    "xgb_raw depth2 test-likeness weighted": xgb_raw(sample_weight=pd.Series(np.clip(W, 0, 10))),
    "lightgbm 4 leaves": lgbm(),
    "spline logistic regression": spline_logistic(),
    "catboost depth4 800it lr0.03": catboost(),
    "catboost depth3 800it lr0.05": catboost(3, 800, 0.05),
    "catboost depth5 800it lr0.03": catboost(5, 800, 0.03),
    "catboost depth6 800it lr0.03": catboost(6, 800, 0.03),
    "catboost depth4 1500it lr0.02": catboost(4, 1500, 0.02),
    "catboost depth6 1500it lr0.015": catboost(6, 1500, 0.015),
}

rows, oofs = [], {}
for name, fp in MODELS.items():
    runs = []
    for s in SEEDS:
        oof = out_of_fold(fp, s)
        oofs[(name, s)] = oof
        runs.append(scores(oof, W))
    r = pd.DataFrame(runs)
    row = {"model": name, **{f"{k}_mean": r[k].mean() for k in r}, **{f"{k}_std": r[k].std(ddof=0) for k in r}}
    rows.append(row)
    print(f"{name:45s} PR-AUC {row['pr_auc_mean']:.4f}±{row['pr_auc_std']:.4f}  "
          f"test-weighted {row['test_weighted_pr_auc_mean']:.4f}  ROC {row['roc_auc_mean']:.4f}", flush=True)
pd.DataFrame(rows).round(4).to_csv(RESULTS / "model_comparison.csv", index=False)

oof_df = pd.DataFrame({"id": train["id"], "fraud": y})
for (name, s), v in oofs.items():
    oof_df[f"{name} | seed{s}"] = v
oof_df.to_csv(RESULTS / "oof_predictions_by_model.csv", index=False)

BLENDS = [
    ("xgb_submitted (depth2, engineered features)", "catboost depth4 800it lr0.03"),
    ("xgb_raw depth3 600t lr0.02", "catboost depth4 800it lr0.03"),
    ("lightgbm 4 leaves", "catboost depth4 800it lr0.03"),
    ("xgb_raw depth3 600t lr0.02", "lightgbm 4 leaves", "catboost depth4 800it lr0.03"),
    ("catboost depth3 800it lr0.05", "catboost depth4 800it lr0.03"),
    ("spline logistic regression", "catboost depth4 800it lr0.03"),
    ("xgb_raw depth3 600t lr0.02", "lightgbm 4 leaves", "spline logistic regression",
     "catboost depth4 800it lr0.03"),
]
brows = []
for keys in BLENDS:
    for method in ("rank average", "probability average"):
        runs = []
        for s in SEEDS:
            parts = [rankdata(oofs[(k, s)]) if method == "rank average" else oofs[(k, s)] for k in keys]
            runs.append(scores(np.mean(parts, axis=0), W))
        r = pd.DataFrame(runs).mean()
        brows.append({"blend": " + ".join(keys), "method": method, **r.to_dict()})
pd.DataFrame(brows).round(4).to_csv(RESULTS / "blend_comparison.csv", index=False)
print(pd.DataFrame(brows).round(4).to_string(index=False))
