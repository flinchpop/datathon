"""Score new transactions with a saved model.

Usage: python predict.py [input.csv] [output.csv] [--model catboost|xgboost]
  input.csv  defaults to data/Track_2_Testing_Dataset.csv
  output.csv defaults to predictions.csv
  --model    catboost (default, from train_catboost.py) or xgboost (from train_xgboost.py)

The input needs the same columns as the training data (the `fraud` column is
optional). Writes the submission format: id, prediction (fraud probability).
If the input has a `fraud` column, it also reports ROC-AUC and PR-AUC.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

TARGET = "fraud"


def predict_catboost(df):
    from catboost import CatBoostClassifier

    import train_catboost as tc

    with open(os.path.join(tc.MODEL_DIR, "metadata.json")) as f:
        meta = json.load(f)
    X = tc.prepare_features(df)[meta["features"]]
    probas = []
    for name in meta["model_files"]:
        m = CatBoostClassifier()
        m.load_model(os.path.join(tc.MODEL_DIR, name))
        probas.append(m.predict_proba(X)[:, 1])
    return np.mean(probas, axis=0)


def predict_xgboost(df):
    import xgboost as xgb

    import train_xgboost as tx

    with open(os.path.join(tx.MODEL_DIR, "metadata.json")) as f:
        meta = json.load(f)
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(tx.MODEL_DIR, "xgb_fraud.json"))
    X = tx.prepare_features(df, meta["encoders"])[meta["features"]]
    return model.predict_proba(X)[:, 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", nargs="?", default="data/Track_2_Testing_Dataset.csv")
    parser.add_argument("output", nargs="?", default="predictions.csv")
    parser.add_argument("--model", choices=["catboost", "xgboost"], default="catboost")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    proba = predict_catboost(df) if args.model == "catboost" else predict_xgboost(df)
    pd.DataFrame({"id": df["id"], "prediction": proba}).to_csv(args.output, index=False)
    print(f"Scored {len(df)} rows with {args.model} -> {args.output}")

    if TARGET in df.columns:
        y = df[TARGET]
        print(f"Evaluation against true labels (n={len(y)}, frauds={int(y.sum())}): "
              f"ROC-AUC {roc_auc_score(y, proba):.4f}  PR-AUC {average_precision_score(y, proba):.4f}")


if __name__ == "__main__":
    main()
