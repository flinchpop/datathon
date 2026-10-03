"""Score transactions with the model saved by train.py.

Usage: python predict.py [input.csv] [output.csv] [--threshold T]
  input.csv  defaults to data/Track_2_Testing_Dataset.csv
  output.csv defaults to predictions.csv
  --threshold overrides the saved decision threshold (e.g. to trade precision for recall)

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?", default="data/Track_2_Testing_Dataset.csv")
    ap.add_argument("output", nargs="?", default="predictions.csv")
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
        meta = json.load(f)
    threshold = meta["threshold"] if args.threshold is None else args.threshold
    model = FraudEnsemble.load(MODEL_DIR)

    df = pd.read_csv(args.input)
    for col in CATEGORICAL:
        unseen = set(df[col].dropna()) - set(model.state["categories"][col])
        if unseen:
            print(f"Warning: {col} has values not seen in training: {sorted(unseen)}")

    proba = model.predict_proba(df)
    out = pd.DataFrame({"id": df["id"], "fraud_probability": proba,
                        "fraud": (proba >= threshold).astype(int)})
    out.to_csv(args.output, index=False)
    print(f"Scored {len(out)} rows -> {args.output}  "
          f"({out['fraud'].sum()} flagged as fraud = {out['fraud'].mean():.2%}, threshold {threshold:.3f})")

    if TARGET in df.columns:
        y = df[TARGET]
        print(f"\nEvaluation against true labels (n={len(y)}, frauds={int(y.sum())})")
        print(f"ROC-AUC {roc_auc_score(y, proba):.4f}  PR-AUC {average_precision_score(y, proba):.4f}")
        print("Confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(y, out["fraud"]))
        print(classification_report(y, out["fraud"], digits=4))


if __name__ == "__main__":
    main()
