"""Train the fraud ensemble, validate it three ways, check the threshold rule, save to model/.

Usage: python train.py [train.csv] [test.csv]
  train.csv defaults to data/Track_2_Training_Dataset.csv
  test.csv  defaults to data/Track_2_Testing_Dataset.csv (features only; used to
            measure the train->test shift for validation, never for fitting)
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from features import TARGET
from model import FraudEnsemble
from validation import (adversarial_weights, cross_validate, format_results, metrics_at,
                        plugin_threshold)

MODEL_DIR = "model"
REPORT_DIR = "reports"


def main():
    train_path = sys.argv[1] if len(sys.argv) > 1 else "data/Track_2_Training_Dataset.csv"
    test_path = sys.argv[2] if len(sys.argv) > 2 else "data/Track_2_Testing_Dataset.csv"
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    y = train[TARGET].to_numpy()
    print(f"Training rows {len(train)}, frauds {y.sum()} ({y.mean():.2%}); test rows {len(test)}")

    weights, adv_auc = adversarial_weights(train, test)
    print(f"Train-vs-test adversarial ROC-AUC {adv_auc:.3f} (0.5 = same distribution)")

    def fit_predict(a, b):
        return FraudEnsemble().fit(train.iloc[a], y[a]).predict_proba(train.iloc[b])

    results, oof_runs, holdout, hold_idx = cross_validate(fit_predict, train, y, weights)
    print(format_results("CatBoost + monotone XGBoost", results))

    # The 0/1 column is cut at predict time with a plug-in rule (validation.plugin_threshold):
    # the threshold that maximises the *expected* PR-AUC of the 0/1 column on the file being
    # scored. Check here, out-of-fold, that the rule picks good thresholds: score test-sized
    # (4,000-row) chunks of the OOF predictions exactly as predict.py would score the test set.
    def chunk_eval(rule=None, fixed=None):
        rows = []
        for r, p in enumerate(oof_runs):
            for _, idx in StratifiedKFold(5, shuffle=True, random_state=r).split(p, y):
                t = fixed if fixed is not None else plugin_threshold(p[idx], rule)[0]
                m = metrics_at(y[idx], p[idx], t)
                rows.append([m["flag_rate"], m["precision"], m["recall"], m["f1"], m["binary_pr_auc"], t])
        return np.mean(rows, axis=0)

    rules = [("plug-in: max expected binary PR-AUC (default)", "binary_pr_auc", None),
             ("plug-in: max expected F1", "f1", None)] + \
            [(f"fixed threshold {t}", None, t) for t in (0.095, 0.16, 0.25, 0.30, 0.40)]
    tradeoff = pd.DataFrame([chunk_eval(r, t) for _, r, t in rules], index=[n for n, _, _ in rules],
                            columns=["flagged", "precision", "recall", "F1", "PR-AUC of 0/1 column",
                                     "mean threshold"]).rename_axis("how the 0/1 column is cut")

    # Fallback fixed threshold (predict.py --rule saved): the plug-in choice on all OOF rows.
    threshold = plugin_threshold(np.mean(oof_runs, axis=0), "binary_pr_auc")[0]
    at_thr = {
        "cv": {k: float(np.mean([metrics_at(y, p, threshold)[k] for p in oof_runs]))
               for k in ("precision", "recall", "f1", "flag_rate", "binary_pr_auc")},
        "shift_weighted_cv": {k: float(np.mean([metrics_at(y, p, threshold, weights)[k] for p in oof_runs]))
                              for k in ("precision", "recall", "f1", "flag_rate", "binary_pr_auc")},
        "adversarial_holdout": {k: float(v) for k, v in metrics_at(y[hold_idx], holdout, threshold).items()},
    }
    print(f"\nSaved fallback threshold {threshold:.3f} (plug-in on all OOF predictions)")
    for k, v in at_thr.items():
        print(f"  {k:20s} precision {v['precision']:.3f}  recall {v['recall']:.3f}  F1 {v['f1']:.3f}  "
              f"PR-AUC(0/1) {v['binary_pr_auc']:.3f}  flagged {v['flag_rate']:.2%}")

    # Reliability check: predicted vs observed fraud rate in 10 equal-size OOF score bins.
    oof_mean = np.mean(oof_runs, axis=0)
    calib = (pd.DataFrame({"predicted": oof_mean, "observed": y})
             .groupby(pd.qcut(oof_mean, 10, labels=False)).agg(["mean", "size"]))
    calib = pd.DataFrame({"mean predicted": calib[("predicted", "mean")],
                          "observed fraud rate": calib[("observed", "mean")],
                          "rows": calib[("observed", "size")]}).rename_axis("score decile")

    final = FraudEnsemble().fit(train, y)
    final.save(MODEL_DIR)
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump({"threshold": threshold, "threshold_rule": "plug-in binary PR-AUC on OOF"}, f, indent=2)

    imp = pd.Series(final.feature_importance(train)).sort_values(ascending=False)
    print("\nFeature importance (mean |SHAP|, log-odds, avg of both models):")
    print(imp.round(4).to_string())

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, "metrics.json"), "w") as f:
        json.dump({"adversarial_auc": adv_auc, "threshold": threshold, "threshold_free": results,
                   "at_threshold": at_thr, "feature_importance": imp.round(5).to_dict()}, f, indent=2)
    with open(os.path.join(REPORT_DIR, "threshold_tradeoff.md"), "w") as f:
        f.write("# How the 0/1 column is cut (out-of-fold, test-sized 4,000-row chunks)\n\n"
                + tradeoff.round(4).to_markdown() + "\n")
    with open(os.path.join(REPORT_DIR, "calibration.md"), "w") as f:
        f.write("# Calibration (out-of-fold, averaged over CV repeats)\n\n" + calib.round(4).to_markdown() + "\n")
    print("\n" + tradeoff.round(3).to_string())
    print("\n" + calib.round(4).to_string())
    print(f"\nSaved model to {MODEL_DIR}/ and reports to {REPORT_DIR}/")


if __name__ == "__main__":
    main()
