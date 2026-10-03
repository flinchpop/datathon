"""Train an XGBoost fraud classifier on the Track 2 training dataset.

Evaluates the model (5-fold CV + hold-out), then fits a final model on all rows
and saves it to model/ for use by predict.py.

Usage: python train_xgboost.py [path/to/Track_2_Training_Dataset.csv]
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/Track_2_Training_Dataset.csv"
TARGET = "fraud"
CATEGORICAL = ["merchant_category", "country", "transaction_channel"]
SEED = 42


MODEL_DIR = "model"


def prepare_features(df, categories):
    """Turn a raw transactions frame into model features.

    `categories` fixes the category levels so new data is encoded exactly like
    the training data (unseen values become missing).
    """
    X = df.drop(columns=["id", TARGET], errors="ignore")
    for col in CATEGORICAL:
        X[col] = pd.Categorical(X[col], categories=categories[col])
    # Simple behavioural ratios; XGBoost handles the NaNs these may produce.
    X["amount_vs_avg_24h"] = X["transaction_amount"] / (
        X["spend_last_24h"] / X["transactions_last_24h"].replace(0, np.nan)
    )
    X["share_of_24h_spend"] = X["transaction_amount"] / X["spend_last_24h"].replace(0, np.nan)
    X = X.replace([np.inf, -np.inf], np.nan)
    return X


def load(path):
    df = pd.read_csv(path)
    categories = {col: sorted(df[col].dropna().unique().tolist()) for col in CATEGORICAL}
    return prepare_features(df, categories), df[TARGET], categories


def make_model():
    # Shallow, regularised trees with a fixed number of rounds: a small grid search
    # (depth 2-5, with/without scale_pos_weight) found the signal is weak and deeper
    # or class-weighted models overfit the ~350 positives.
    return xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=2,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=5,
        enable_categorical=True,
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
    )


def best_f1_threshold(y_true, proba):
    p, r, t = precision_recall_curve(y_true, proba)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    i = np.nanargmax(f1[:-1])
    return t[i], f1[i]


def main():
    X, y, categories = load(DATA_PATH)
    print(f"Rows: {len(X)}  fraud rate: {y.mean():.2%}  missing cells: {X.isna().sum().sum()}")

    # 5-fold stratified CV on the full data.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        m = make_model()
        m.fit(X.iloc[tr], y.iloc[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        print(f"Fold {fold}: ROC-AUC {roc_auc_score(y.iloc[va], oof[va]):.4f}  "
              f"PR-AUC {average_precision_score(y.iloc[va], oof[va]):.4f}")
    print(f"CV (OOF)  ROC-AUC {roc_auc_score(y, oof):.4f}  PR-AUC {average_precision_score(y, oof):.4f}")

    # Hold-out evaluation: decision threshold chosen on a validation slice of the
    # training part, then reported on the untouched test set.
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    X_fit, X_val, y_fit, y_val = train_test_split(X_tr, y_tr, test_size=0.2, stratify=y_tr, random_state=SEED)
    model = make_model()
    model.fit(X_fit, y_fit)
    thr, val_f1 = best_f1_threshold(y_val, model.predict_proba(X_val)[:, 1])

    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= thr).astype(int)
    print(f"\nHold-out test (n={len(y_te)}, frauds={y_te.sum()})")
    print(f"ROC-AUC {roc_auc_score(y_te, proba):.4f}  PR-AUC {average_precision_score(y_te, proba):.4f}")
    print(f"Threshold {thr:.3f} (val F1 {val_f1:.3f})  test F1 {f1_score(y_te, pred):.4f}")
    print("Confusion matrix [[TN FP] [FN TP]]:\n", confusion_matrix(y_te, pred))
    print(classification_report(y_te, pred, digits=4))

    imp = pd.Series(model.get_booster().get_score(importance_type="gain")).sort_values(ascending=False)
    print("Feature importance (gain):")
    print((imp / imp.sum()).round(4).to_string())

    # Final model: fit on every row. The threshold comes from the CV out-of-fold
    # predictions, which cover all 20k rows and are steadier than one small split.
    final_thr, oof_f1 = best_f1_threshold(y, oof)
    final = make_model()
    final.fit(X, y)
    os.makedirs(MODEL_DIR, exist_ok=True)
    final.save_model(os.path.join(MODEL_DIR, "xgb_fraud.json"))
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump({"threshold": float(final_thr), "categories": categories,
                   "features": list(X.columns)}, f, indent=2)
    print(f"\nSaved final model (all {len(X)} rows) to {MODEL_DIR}/  "
          f"threshold {final_thr:.3f} (OOF F1 {oof_f1:.3f})")


if __name__ == "__main__":
    main()
