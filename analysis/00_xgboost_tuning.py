"""First XGBoost tuning step (raw columns): tree depth, rounds and fraud up-weighting.

The very first model (depth 5, scale_pos_weight ~56, early stopping) overfit; this grid
showed shallow, regularised trees without class weighting work best, which is where the
submitted settings (depth 2, 300 trees, lr 0.03) come from.

Writes: results/xgboost_tuning.csv
"""
import pandas as pd

from common import RESULTS, out_of_fold, scores, xgb_raw, y

spw_full = (y == 0).sum() / (y == 1).sum()
rows = []
for depth in (2, 3, 4, 5):
    for n, lr in ((300, 0.03), (600, 0.02)):
        for spw in (1.0, spw_full ** 0.5, spw_full):
            runs = pd.DataFrame([scores(out_of_fold(xgb_raw(depth, n, lr, spw), s)) for s in (0, 1)])
            rows.append({"max_depth": depth, "n_estimators": n, "learning_rate": lr,
                         "scale_pos_weight": round(spw, 1), "pr_auc": runs.pr_auc.mean(),
                         "roc_auc": runs.roc_auc.mean()})
            print(rows[-1], flush=True)
pd.DataFrame(rows).round(4).sort_values("pr_auc", ascending=False).to_csv(
    RESULTS / "xgboost_tuning.csv", index=False)
