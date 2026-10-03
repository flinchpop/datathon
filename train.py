"""Train the fraud ensemble, validate it three ways, choose the threshold, save to model/.

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

from features import TARGET
from model import FraudEnsemble
from validation import (adversarial_weights, choose_threshold, cross_validate, format_results,
                        metrics_at)

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

    threshold, f1_peak_threshold, best_f1 = choose_threshold(y, oof_runs)

    def operating_point(t):
        return {
            "cv": {k: float(np.mean([metrics_at(y, p, t)[k] for p in oof_runs]))
                   for k in ("precision", "recall", "f1", "flag_rate")},
            "shift_weighted_cv": {k: float(np.mean([metrics_at(y, p, t, weights)[k] for p in oof_runs]))
                                  for k in ("precision", "recall", "f1", "flag_rate")},
            "adversarial_holdout": {k: float(v) for k, v in metrics_at(y[hold_idx], holdout, t).items()},
        }

    at_thr = operating_point(threshold)
    print(f"\nThreshold {threshold:.3f} (F1 peaks at {f1_peak_threshold:.3f} with OOF F1 {best_f1:.4f}; "
          f"we take the lowest threshold within 5% of that peak to favour recall)")
    for k, v in at_thr.items():
        print(f"  {k:20s} precision {v['precision']:.3f}  recall {v['recall']:.3f}  F1 {v['f1']:.3f}  "
              f"flagged {v['flag_rate']:.2%}")

    # Precision/recall at a range of thresholds, so the operating point can be moved knowingly.
    rows = []
    for t in sorted({0.04, 0.06, 0.08, threshold, 0.12, f1_peak_threshold, 0.25}):
        op = operating_point(t)
        label = f"{t:.3f}" + (" (chosen)" if t == threshold else " (F1 peak)" if t == f1_peak_threshold else "")
        rows.append([label] + [op[s][k] for s in ("cv", "shift_weighted_cv", "adversarial_holdout")
                               for k in ("flag_rate", "precision", "recall", "f1")])
    cols = [f"{s} {k}" for s in ("CV", "shift-wtd", "test-like") for k in ("flagged", "precision", "recall", "F1")]
    tradeoff = pd.DataFrame(rows, columns=["threshold"] + cols).set_index("threshold")

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
        json.dump({"threshold": threshold, "f1_peak_threshold": f1_peak_threshold}, f, indent=2)

    imp = pd.Series(final.feature_importance(train)).sort_values(ascending=False)
    print("\nFeature importance (mean |SHAP|, log-odds, avg of both models):")
    print(imp.round(4).to_string())

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, "metrics.json"), "w") as f:
        json.dump({"adversarial_auc": adv_auc, "threshold": threshold,
                   "f1_peak_threshold": f1_peak_threshold, "threshold_free": results,
                   "at_threshold": at_thr, "feature_importance": imp.round(5).to_dict()}, f, indent=2)
    with open(os.path.join(REPORT_DIR, "threshold_tradeoff.md"), "w") as f:
        f.write("# Threshold trade-off (out-of-fold)\n\n" + tradeoff.to_markdown(floatfmt=".3f") + "\n")
    with open(os.path.join(REPORT_DIR, "calibration.md"), "w") as f:
        f.write("# Calibration (out-of-fold, averaged over CV repeats)\n\n" + calib.round(4).to_markdown() + "\n")
    print("\n" + tradeoff.round(3).to_string())
    print("\n" + calib.round(4).to_string())
    print(f"\nSaved model to {MODEL_DIR}/ and reports to {REPORT_DIR}/")


if __name__ == "__main__":
    main()
