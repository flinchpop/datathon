"""Train a CatBoost fraud classifier on the Track 2 training dataset.

CatBoost beat XGBoost, LightGBM and a spline logistic regression in repeated
5-fold CV (PR-AUC 0.218 vs 0.205 for the XGBoost model), and none of the
engineered features or model blends improved on it, so it uses the raw columns.

Evaluates with 5-fold CV, then fits the final model (an average of several
seeds) on all rows and saves it to model_catboost/ for use by predict.py.

Usage: python train_catboost.py [path/to/Track_2_Training_Dataset.csv]
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/Track_2_Training_Dataset.csv"
MODEL_DIR = "model_catboost"
TARGET = "fraud"
CATEGORICAL = ["merchant_category", "country", "transaction_channel"]
NUMERIC = ["transaction_amount", "transaction_hour", "transactions_last_24h", "spend_last_24h",
           "account_age", "new_device", "transactions_last_1h"]
FEATURES = NUMERIC + CATEGORICAL
SEED = 42
FINAL_SEEDS = 5


def prepare_features(df):
    """Select model columns; CatBoost needs missing categories as a string."""
    X = df[FEATURES].copy()
    for col in CATEGORICAL:
        X[col] = X[col].fillna("NA").astype(str)
    return X


def make_model(seed=SEED):
    # Depth 4 / 800 rounds / lr 0.03 won a small search over depth 3-6 and
    # 800-1500 rounds; deeper or longer models overfit the ~350 frauds.
    return CatBoostClassifier(
        iterations=800,
        learning_rate=0.03,
        depth=4,
        l2_leaf_reg=5,
        cat_features=CATEGORICAL,
        random_seed=seed,
        verbose=0,
        thread_count=-1,
    )


def main():
    df = pd.read_csv(DATA_PATH)
    X, y = prepare_features(df), df[TARGET]
    print(f"Rows: {len(df)}  fraud rate: {y.mean():.2%}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        m = make_model()
        m.fit(X.iloc[tr], y.iloc[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        print(f"Fold {fold}: ROC-AUC {roc_auc_score(y.iloc[va], oof[va]):.4f}  "
              f"PR-AUC {average_precision_score(y.iloc[va], oof[va]):.4f}")
    print(f"CV (OOF)  ROC-AUC {roc_auc_score(y, oof):.4f}  PR-AUC {average_precision_score(y, oof):.4f}")

    # Final model: several seeds on all rows; predict.py averages their outputs.
    os.makedirs(MODEL_DIR, exist_ok=True)
    files = []
    for i in range(FINAL_SEEDS):
        m = make_model(SEED + i)
        m.fit(X, y)
        name = f"catboost_seed{i}.cbm"
        m.save_model(os.path.join(MODEL_DIR, name))
        files.append(name)
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump({"model_type": "catboost", "model_files": files, "features": FEATURES}, f, indent=2)
    print(f"\nSaved final model ({FINAL_SEEDS} seeds, all {len(X)} rows) to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
