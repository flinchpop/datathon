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
RAW_NUMERIC = ["transaction_amount", "transaction_hour", "transactions_last_24h", "spend_last_24h",
               "account_age", "new_device", "transactions_last_1h"]
HIGH_RISK_MERCHANTS = ["luxury", "cash_transfer", "electronics"]
NEW_ACCOUNT_DAYS = 30
# Smoothing for risk scores: a category's rate is blended with the overall rate as if
# it had RISK_SMOOTHING extra rows at the overall rate, so small categories aren't noisy.
RISK_SMOOTHING = 50


def fit_encoders(df, y):
    """Learn category levels and smoothed fraud-rate ("risk") scores from training rows."""
    prior = float(y.mean())
    categories, risk = {}, {}
    for col in CATEGORICAL:
        categories[col] = sorted(df[col].dropna().unique().tolist())
        stats = y.groupby(df[col]).agg(["sum", "count"])
        rate = (stats["sum"] + RISK_SMOOTHING * prior) / (stats["count"] + RISK_SMOOTHING)
        risk[col] = {str(k): float(v) for k, v in rate.items()}
    return {"categories": categories, "risk": risk, "prior": prior}


def prepare_features(df, encoders, risk_override=None):
    """Turn a raw transactions frame into model features.

    `encoders` (from fit_encoders) fixes category levels and risk scores so new data is
    encoded exactly like the training data; unseen categories become missing / get the
    overall fraud rate. `risk_override` supplies precomputed risk columns (used for
    training rows, see out_of_fold_risk).
    """
    X = df.drop(columns=["id", TARGET], errors="ignore").copy()

    # Missing-value flags: record *that* a value was missing before anything else.
    for col in RAW_NUMERIC + CATEGORICAL:
        X[f"{col}_missing"] = X[col].isna().astype(int)

    # Velocity and spike ratios.
    txns_24h = X["transactions_last_24h"].replace(0, np.nan)
    X["avg_spend_per_txn_24h"] = X["spend_last_24h"] / txns_24h
    X["spike_ratio"] = X["transaction_amount"] / X["avg_spend_per_txn_24h"].replace(0, np.nan)
    X["share_of_24h_spend"] = X["transaction_amount"] / X["spend_last_24h"].replace(0, np.nan)
    X["urgency_ratio"] = X["transactions_last_1h"] / txns_24h

    # Rule-of-thumb flags (missing inputs stay missing rather than becoming 0).
    X["is_high_risk_merchant"] = X["merchant_category"].isin(HIGH_RISK_MERCHANTS).astype(float)
    X.loc[X["merchant_category"].isna(), "is_high_risk_merchant"] = np.nan
    X["is_new_account"] = (X["account_age"] < NEW_ACCOUNT_DAYS).astype(float)
    X.loc[X["account_age"].isna(), "is_new_account"] = np.nan

    # Risk (target) encoding: historical fraud rate of each category.
    for col in CATEGORICAL:
        if risk_override is not None:
            X[f"{col}_risk"] = risk_override[col].to_numpy()
        else:
            X[f"{col}_risk"] = X[col].map(encoders["risk"][col]).astype(float).fillna(encoders["prior"])

    for col in CATEGORICAL:
        X[col] = pd.Categorical(X[col], categories=encoders["categories"][col])
    return X.replace([np.inf, -np.inf], np.nan)


def out_of_fold_risk(df, y, n_splits=5):
    """Risk scores for training rows computed without each row's own label.

    Scoring a row with a rate that includes its own label leaks the answer and makes
    the model over-trust the risk columns, so each row is scored by encoders fit on the
    other folds.
    """
    risk = pd.DataFrame(index=df.index, columns=CATEGORICAL, dtype=float)
    for tr, va in StratifiedKFold(n_splits, shuffle=True, random_state=SEED).split(df, y):
        enc = fit_encoders(df.iloc[tr], y.iloc[tr])
        for col in CATEGORICAL:
            risk.iloc[va, risk.columns.get_loc(col)] = (
                df[col].iloc[va].map(enc["risk"][col]).astype(float).fillna(enc["prior"]).to_numpy()
            )
    return risk


def build_train(df, y):
    """Fit encoders on (df, y) and return leak-free training features plus the encoders."""
    enc = fit_encoders(df, y)
    return prepare_features(df, enc, risk_override=out_of_fold_risk(df, y)), enc


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
    df = pd.read_csv(DATA_PATH)
    y = df[TARGET]
    print(f"Rows: {len(df)}  fraud rate: {y.mean():.2%}  missing cells: {df.isna().sum().sum()}")

    # 5-fold stratified CV. Risk scores are fit inside each fold so the validation
    # rows' labels never influence their own features.
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for fold, (tr, va) in enumerate(skf.split(df, y), 1):
        X_tr, enc = build_train(df.iloc[tr], y.iloc[tr])
        m = make_model()
        m.fit(X_tr, y.iloc[tr])
        oof[va] = m.predict_proba(prepare_features(df.iloc[va], enc))[:, 1]
        print(f"Fold {fold}: ROC-AUC {roc_auc_score(y.iloc[va], oof[va]):.4f}  "
              f"PR-AUC {average_precision_score(y.iloc[va], oof[va]):.4f}")
    print(f"CV (OOF)  ROC-AUC {roc_auc_score(y, oof):.4f}  PR-AUC {average_precision_score(y, oof):.4f}")

    # Hold-out evaluation: decision threshold chosen on a validation slice of the
    # training part, then reported on the untouched test set.
    df_tr, df_te = train_test_split(df, test_size=0.2, stratify=y, random_state=SEED)
    df_fit, df_val = train_test_split(df_tr, test_size=0.2, stratify=df_tr[TARGET], random_state=SEED)
    X_fit, enc = build_train(df_fit, df_fit[TARGET])
    model = make_model()
    model.fit(X_fit, df_fit[TARGET])
    y_val, y_te = df_val[TARGET], df_te[TARGET]
    thr, val_f1 = best_f1_threshold(y_val, model.predict_proba(prepare_features(df_val, enc))[:, 1])

    proba = model.predict_proba(prepare_features(df_te, enc))[:, 1]
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
    X, enc = build_train(df, y)
    final = make_model()
    final.fit(X, y)
    os.makedirs(MODEL_DIR, exist_ok=True)
    final.save_model(os.path.join(MODEL_DIR, "xgb_fraud.json"))
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w") as f:
        json.dump({"threshold": float(final_thr), "encoders": enc,
                   "features": list(X.columns)}, f, indent=2)
    print(f"\nSaved final model (all {len(X)} rows) to {MODEL_DIR}/  "
          f"threshold {final_thr:.3f} (OOF F1 {oof_f1:.3f})")


if __name__ == "__main__":
    main()
