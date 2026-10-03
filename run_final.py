"""Train the final ensemble on all labelled data, score the hidden test set and write submissions.

Usage:  python3 run_final.py [--seeds 5] [--cv-repeats 6]

Outputs (in outputs/):
  submission.csv              id, fraud            -> fraud probability-like score in [0,1] (rank-blended)
  submission_binary.csv       id, fraud            -> 0/1 label at the F1-optimal CV threshold
  submission_full.csv         id, fraud_score, fraud_prob, fraud_label, plus each member's probability
  cv_results.csv              per-member and blended CV metrics
  feature_importance.csv      averaged gain importances per member
"""
import sys, json, argparse, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, ".")
import numpy as np, pandas as pd
from src.pipeline import run_pipeline
from src.models import make_lr, make_xgb, make_lgbm, make_cat
from src.features import TE_COLS_BASE

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--cv-repeats", type=int, default=6)
ap.add_argument("--threads", type=int, default=4)
args = ap.parse_args()

train = pd.read_csv("data/train.csv"); test = pd.read_csv("data/test.csv")
CATS = ["merchant_category", "country", "transaction_channel"]
T = args.threads

# ---- ensemble definition (chosen from experiments/e0..e4; see REPORT.md) ----
MEMBERS = {
    "cat_d3_merch":    dict(make_model=lambda s: make_cat(s, threads=T), feature_set="merchant", cat_cols=CATS),
    "cat_d3_full":     dict(make_model=lambda s: make_cat(s, threads=T), feature_set="full", cat_cols=CATS),
    "xgb_d2_full_te":  dict(make_model=lambda s: make_xgb(s, threads=T), feature_set="full", te_cols=TE_COLS_BASE),
    "lgbm_d2_full_te": dict(make_model=lambda s: make_lgbm(s, threads=T), feature_set="full", te_cols=TE_COLS_BASE),
    "lr_full":         dict(make_model=lambda s: make_lr(s, C=0.1), feature_set="full"),
}
WEIGHTS = {"cat_d3_merch": 1.0, "cat_d3_full": 1.0, "xgb_d2_full_te": 0.75, "lgbm_d2_full_te": 0.75, "lr_full": 0.5}

seeds = [1000 + i for i in range(args.seeds)]
res = run_pipeline(train, test, MEMBERS, WEIGHTS, seeds, n_repeats_cv=args.cv_repeats, out_dir="outputs")

# ---- write outputs ----
score = res["test_rank_score"]; prob = res["test_prob"]; thr = res["threshold"]
label = (score >= thr).astype(int)
pd.DataFrame({"id": test.id, "fraud": score}).to_csv("outputs/submission.csv", index=False)
pd.DataFrame({"id": test.id, "fraud": label}).to_csv("outputs/submission_binary.csv", index=False)
full = pd.DataFrame({"id": test.id, "fraud_score": score, "fraud_prob": prob, "fraud_label": label})
for n, s in res["test_member_scores"].items(): full[f"p_{n}"] = s
full.to_csv("outputs/submission_full.csv", index=False)
res["cv_table"].to_csv("outputs/cv_results.csv", index=False)
imp = pd.concat({k: v for k, v in res["importances"].items() if v is not None}, axis=1)
imp.to_csv("outputs/feature_importance.csv")
summary = dict(threshold_rank_score=thr, cv_flag_rate=res["flag_rate"], test_flag_rate=float(label.mean()),
               test_mean_prob=float(prob.mean()), n_test=int(len(test)), n_flagged=int(label.sum()), seeds=seeds,
               members=list(MEMBERS), weights=WEIGHTS, secs=res["secs"])
json.dump(summary, open("outputs/run_summary.json", "w"), indent=2)
print("\n=== RUN SUMMARY ===\n", json.dumps(summary, indent=2))
print("\nTop features per member:")
for n, v in res["importances"].items():
    if v is not None: print(f"  {n}: " + ", ".join(f"{i}={x:.3f}" for i, x in (v / v.sum()).head(8).items()))

# ---- final upload files (monotone rescaling so that 0.5 <=> chosen operating point) ----
import subprocess
subprocess.run([sys.executable, "make_submissions.py"], check=True)
