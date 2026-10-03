"""XGBoost on the 10 original columns only (no engineered features).

Same model settings as train_xgboost.py. In repeated CV it scored PR-AUC 0.205
vs 0.197 for the engineered-feature version (analysis/results/model_comparison.csv).

Reports 5-fold CV, then fits on all rows and writes the test submission.

Usage: python train_xgboost_raw.py
Writes: outputs/submission_xgboost_raw.csv, outputs/xgboost_raw/oof_predictions.csv,
        outputs/xgboost_raw/feature_importance_gain.csv
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import train_xgboost as tx

TRAIN_PATH = "data/Track_2_Training_Dataset.csv"
TEST_PATH = "data/Track_2_Testing_Dataset.csv"
OUTPUT_DIR = os.path.join("outputs", "xgboost_raw")
FEATURES = tx.RAW_NUMERIC + tx.CATEGORICAL


def prepare(df, levels):
    X = df[FEATURES].copy()
    for col in tx.CATEGORICAL:
        X[col] = pd.Categorical(X[col], categories=levels[col])
    return X


def main():
    train, test = pd.read_csv(TRAIN_PATH), pd.read_csv(TEST_PATH)
    y = train[tx.TARGET]
    levels = {c: sorted(train[c].dropna().unique()) for c in tx.CATEGORICAL}
    X, X_test = prepare(train, levels), prepare(test, levels)

    oof = np.zeros(len(y))
    for fold, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=tx.SEED).split(X, y), 1):
        m = tx.make_model()
        m.fit(X.iloc[a], y.iloc[a])
        oof[b] = m.predict_proba(X.iloc[b])[:, 1]
        print(f"Fold {fold}: ROC-AUC {roc_auc_score(y.iloc[b], oof[b]):.4f}  "
              f"PR-AUC {average_precision_score(y.iloc[b], oof[b]):.4f}")
    print(f"CV (OOF)  ROC-AUC {roc_auc_score(y, oof):.4f}  PR-AUC {average_precision_score(y, oof):.4f}")

    final = tx.make_model()
    final.fit(X, y)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pd.DataFrame({"id": test["id"], "prediction": final.predict_proba(X_test)[:, 1]}).to_csv(
        os.path.join("outputs", "submission_xgboost_raw.csv"), index=False)
    pd.DataFrame({"id": train["id"], "fraud": y, "oof_prediction": oof}).to_csv(
        os.path.join(OUTPUT_DIR, "oof_predictions.csv"), index=False)
    gain = pd.Series(final.get_booster().get_score(importance_type="gain")).reindex(FEATURES).fillna(0)
    (gain / gain.sum() * 100).sort_values(ascending=False).round(2).rename("gain_share_%").to_csv(
        os.path.join(OUTPUT_DIR, "feature_importance_gain.csv"), index_label="feature")
    print("Wrote outputs/submission_xgboost_raw.csv")


if __name__ == "__main__":
    main()
