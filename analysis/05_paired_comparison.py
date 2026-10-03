"""Head-to-head comparison on 10 different 5-fold splits (seeds 100-109).

All three models see exactly the same folds in each repeat, so the per-repeat
differences are a fair, paired comparison.

Writes: results/paired_comparison_by_split.csv, results/paired_comparison_summary.csv
"""
import pandas as pd

from common import RESULTS, catboost, load_adversarial_weights, out_of_fold, scores, xgb_raw, xgb_submitted

W = load_adversarial_weights()
MODELS = {"xgb_submitted": xgb_submitted(), "xgb_raw": xgb_raw(), "catboost_depth4": catboost()}
BASE = "xgb_submitted"

rows = []
for rep in range(10):
    for name, fp in MODELS.items():
        rows.append({"split": rep, "model": name, **scores(out_of_fold(fp, 100 + rep), W)})
    print(rep, {r["model"]: round(r["pr_auc"], 4) for r in rows[-3:]}, flush=True)
by_split = pd.DataFrame(rows)
by_split.round(4).to_csv(RESULTS / "paired_comparison_by_split.csv", index=False)

pr = by_split.pivot(index="split", columns="model", values="pr_auc")
wpr = by_split.pivot(index="split", columns="model", values="test_weighted_pr_auc")
summary = pd.DataFrame({
    "cv_pr_auc_mean": pr.mean(),
    "diff_vs_submitted_xgb": pr.sub(pr[BASE], axis=0).mean(),
    "splits_better_than_submitted_xgb": pr.gt(pr[BASE], axis=0).sum(),
    "test_weighted_pr_auc_mean": wpr.mean(),
}).round(4)
summary.to_csv(RESULTS / "paired_comparison_summary.csv", index_label="model")
print(summary.to_string())
