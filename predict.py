"""Score transactions with the model saved by train.py.

Usage: python predict.py [input.csv] [output.csv] [--rule RULE | --threshold T]
  input.csv  defaults to data/Track_2_Testing_Dataset.csv
  output.csv defaults to predictions.csv
  --rule     how the 0/1 `fraud` column is cut (default: binary_pr_auc)
               binary_pr_auc  plug-in threshold maximising expected PR-AUC of the 0/1 column
               f1             plug-in threshold maximising expected F1
               saved          the fixed out-of-fold threshold stored in model/metadata.json
  --threshold a fixed probability threshold (overrides --rule)

Writes id, fraud_probability and fraud (0/1). If the input has a `fraud` column,
it also reports how well the predictions match it.
"""
import argparse
import json
import os

import pandas as pd
from sklearn.metrics import (average_precision_score, classification_report, confusion_matrix,
                             roc_auc_score)

from features import CATEGORICAL, TARGET
from model import FraudEnsemble
from train import MODEL_DIR
from validation import plugin_threshold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?", default="data/Track_2_Testing_Dataset.csv")
    ap.add_argument("output", nargs="?", default="predictions.csv")
    ap.add_argument("--rule", choices=["binary_pr_auc", "f1", "saved"], default="binary_pr_auc")
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    model = FraudEnsemble.load(MODEL_DIR)
    df = pd.read_csv(args.input)
    for col in CATEGORICAL:
        unseen = set(df[col].dropna()) - set(model.state["categories"][col])
        if unseen:
            print(f"Warning: {col} has values not seen in training: {sorted(unseen)}")
    proba = model.predict_proba(df)

    if args.threshold is not None:
        threshold, how = args.threshold, "fixed (--threshold)"
    elif args.rule == "saved":
        with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
            threshold, how = json.load(f)["threshold"], "saved out-of-fold threshold"
    else:
        threshold, exp = plugin_threshold(proba, args.rule)
        how = (f"plug-in, maximising expected {args.rule} (expected: {exp['expected_frauds']:.0f} frauds in file, "
               f"precision {exp['precision']:.2f}, recall {exp['recall']:.2f}, {args.rule} {exp[args.rule]:.3f})")

    out = pd.DataFrame({"id": df["id"], "fraud_probability": proba,
                        "fraud": (proba >= threshold).astype(int)})
    out.to_csv(args.output, index=False)
    print(f"Scored {len(out)} rows -> {args.output}  ({out['fraud'].sum()} flagged = {out['fraud'].mean():.2%}, "
          f"threshold {threshold:.3f}: {how})")

    if TARGET in df.columns:
        y = df[TARGET]
        print(f"\nEvaluation against true labels (n={len(y)}, frauds={int(y.sum())})")
        print(f"ROC-AUC {roc_auc_score(y, proba):.4f}  PR-AUC (probabilities) {average_precision_score(y, proba):.4f}  "
              f"PR-AUC (0/1 column) {average_precision_score(y, out['fraud']):.4f}")
        print("Confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(y, out["fraud"]))
        print(classification_report(y, out["fraud"], digits=4))


if __name__ == "__main__":
    main()
