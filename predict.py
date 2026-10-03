"""Score new transactions with the model saved by train_xgboost.py.

Usage: python predict.py path/to/new_data.csv [output.csv]

The input needs the same columns as the training data (the `fraud` column is
optional). Writes id, fraud_probability and fraud (0/1). If the input has a
`fraud` column, it also reports how well the predictions match it.
"""
import json
import os
import sys

import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from train_xgboost import MODEL_DIR, TARGET, prepare_features


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    in_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "predictions.csv"

    with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
        meta = json.load(f)
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(MODEL_DIR, "xgb_fraud.json"))

    df = pd.read_csv(in_path)
    X = prepare_features(df, meta["categories"])[meta["features"]]
    for col, levels in meta["categories"].items():
        unseen = set(df[col].dropna()) - set(levels)
        if unseen:
            print(f"Warning: {col} has values not seen in training (treated as missing): {sorted(unseen)}")

    proba = model.predict_proba(X)[:, 1]
    out = pd.DataFrame({"id": df["id"], "fraud_probability": proba,
                        "fraud": (proba >= meta["threshold"]).astype(int)})
    out.to_csv(out_path, index=False)
    print(f"Scored {len(out)} rows -> {out_path}  "
          f"({out['fraud'].sum()} flagged as fraud at threshold {meta['threshold']:.3f})")

    if TARGET in df.columns:
        y = df[TARGET]
        print(f"\nEvaluation against true labels (n={len(y)}, frauds={int(y.sum())})")
        print(f"ROC-AUC {roc_auc_score(y, proba):.4f}  PR-AUC {average_precision_score(y, proba):.4f}")
        print("Confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(y, out["fraud"]))
        print(classification_report(y, out["fraud"], digits=4))


if __name__ == "__main__":
    main()
