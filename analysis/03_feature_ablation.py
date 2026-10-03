"""Which engineered feature groups help the submitted XGBoost model?

Runs the submitted pipeline (train_xgboost.py) with whole feature groups removed,
5-fold CV repeated over 5 seeds. Risk scores are fit inside each fold.

Writes: results/feature_ablation_xgboost.csv
"""
import pandas as pd

from common import CAT, NUM, RESULTS, load_adversarial_weights, out_of_fold, scores, xgb_submitted

W = load_adversarial_weights()
MISSING = [f"{c}_missing" for c in NUM + CAT]
RATIOS = ["avg_spend_per_txn_24h", "spike_ratio", "share_of_24h_spend", "urgency_ratio"]
FLAGS = ["is_high_risk_merchant", "is_new_account"]
RISK = [f"{c}_risk" for c in CAT]
SETS = {
    "all features (submitted model)": [],
    "raw columns only": MISSING + RATIOS + FLAGS + RISK,
    "without missing flags": MISSING,
    "without ratios": RATIOS,
    "without flags": FLAGS,
    "without risk scores": RISK,
}

rows = []
for name, drop in SETS.items():
    runs = pd.DataFrame([scores(out_of_fold(xgb_submitted(drop), s), W) for s in range(1, 6)])
    rows.append({"feature_set": name, **{f"{k}_mean": runs[k].mean() for k in runs},
                 **{f"{k}_std": runs[k].std(ddof=0) for k in runs}})
    print(f"{name:32s} PR-AUC {runs.pr_auc.mean():.4f}±{runs.pr_auc.std(ddof=0):.4f}  "
          f"ROC {runs.roc_auc.mean():.4f}", flush=True)
pd.DataFrame(rows).round(4).to_csv(RESULTS / "feature_ablation_xgboost.csv", index=False)
