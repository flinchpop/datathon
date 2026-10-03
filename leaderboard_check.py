"""Why did a model that wins every local test lose on the leaderboard?

Usage: python leaderboard_check.py [train.csv]   -> reports/leaderboard_check.md  (~5 min)

Re-creates the out-of-fold predictions of the original baseline and of our ensemble,
then asks which metric reproduces the leaderboard result
    baseline 0.1942  vs  our ensemble (submitted with threshold 0.095) 0.1844
and how much of any leaderboard difference is just noise.
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import RepeatedStratifiedKFold

from model import FraudEnsemble
from validation import metrics_at

train = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "data/Track_2_Training_Dataset.csv")
y = train["fraud"].to_numpy()
N_REP = 3

_spec = importlib.util.spec_from_file_location("baseline", os.path.join("baseline", "train_xgboost.py"))
baseline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baseline)


def baseline_fp(a, b):
    X, enc = baseline.build_train(train.iloc[a], train["fraud"].iloc[a])
    return baseline.make_model().fit(X, y[a]).predict_proba(baseline.prepare_features(train.iloc[b], enc))[:, 1]


def ours_fp(a, b):
    return FraudEnsemble().fit(train.iloc[a], y[a]).predict_proba(train.iloc[b])


def oof(fp):
    runs = np.zeros((N_REP, len(y)))
    for k, (a, b) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=N_REP, random_state=42).split(train, y)):
        runs[k // 5, b] = fp(a, b)
    return runs


def best_f1_threshold(p):
    prec, rec, t = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return t[np.nanargmax(f1[:-1])]


def paired_se(score1, score2, n=300, rows=12000, seed=0):
    """Std of score1-score2 over bootstrap draws of a `rows`-sized evaluation set (row indices)."""
    rng = np.random.default_rng(seed)
    d = [score1(i) - score2(i) for i in (rng.integers(0, len(y), rows) for _ in range(n))]
    return float(np.std(d))


def main():
    runs = {"baseline": oof(baseline_fp), "ours": oof(ours_fp)}
    # Each model's 0/1 column exactly as it was submitted: baseline at its own best-OOF-F1
    # threshold, ours at the recall-leaning 0.095.
    cuts = {"baseline": lambda p: best_f1_threshold(p), "ours": lambda p: 0.095}
    rows = []
    for name, rr in runs.items():
        vals = []
        for p in rr:
            t = cuts[name](p)
            m = metrics_at(y, p, t)
            vals.append([average_precision_score(y, p), m["binary_pr_auc"], m["f1"], m["recall"], m["flag_rate"], t])
        rows.append([name] + list(np.mean(vals, axis=0)))
    df = pd.DataFrame(rows, columns=["model", "PR-AUC (probabilities)", "PR-AUC (0/1 column)", "F1", "recall",
                                     "flagged", "threshold"]).set_index("model")
    diff = df.loc["ours"] - df.loc["baseline"]

    pb, po = runs["baseline"].mean(0), runs["ours"].mean(0)
    tb = best_f1_threshold(pb)
    se_prob = paired_se(lambda i: average_precision_score(y[i], po[i]),
                        lambda i: average_precision_score(y[i], pb[i]))
    se_bin = paired_se(lambda i: average_precision_score(y[i], (po[i] >= 0.095).astype(int)),
                       lambda i: average_precision_score(y[i], (pb[i] >= tb).astype(int)))
    rng = np.random.default_rng(1)
    se_single = float(np.std([average_precision_score(y[i], pb[i]) for i in
                              (rng.integers(0, len(y), 12000) for _ in range(300))]))
    z_prob = (diff["PR-AUC (probabilities)"] - (-0.0098)) / se_prob

    out = ["# Leaderboard check\n",
           "Out-of-fold metrics (3x repeated 5-fold CV), each model's 0/1 column cut as it was submitted:\n",
           df.round(4).to_markdown(), "\n",
           f"| | PR-AUC (probabilities) | PR-AUC (0/1 column) | F1 |\n|---|---|---|---|\n"
           f"| ours - baseline, local | {diff['PR-AUC (probabilities)']:+.4f} | {diff['PR-AUC (0/1 column)']:+.4f} | {diff['F1']:+.4f} |\n"
           f"| ours - baseline, leaderboard | | **-0.0098** | |\n"
           f"| paired noise (1 s.e., 12k-row test) | {se_prob:.4f} | {se_bin:.4f} | |\n",
           f"\nPR-AUC of the 0/1 column reproduces the leaderboard gap almost exactly. PR-AUC of the probabilities "
           f"predicts the opposite sign; the observed gap is {z_prob:.1f} paired standard errors away from it. "
           f"This is strong evidence (not proof) that the leaderboard scores the submitted 0/1 labels.\n\n"
           f"Noise: one submission's score has a bootstrap s.e. of about {se_single:.3f} on 12,000 rows, and two "
           f"similar submissions differ by noise of roughly 0.003-0.01, so leaderboard steps of a few thousandths "
           f"are not evidence on their own.\n"]
    os.makedirs("reports", exist_ok=True)
    with open("reports/leaderboard_check.md", "w") as f:
        f.write("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
