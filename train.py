"""Train the fraud model and predict on new data.

Usage:
    python train.py                                   # cross-validate, pick, fit, save model.joblib
    python train.py predict data/test.csv submission.csv   # id, probability, %, 0/1 label

Scoring is 60% PR-AUC plus 40% unknown metrics, so the pipeline aims for:
  * the best ranking (PR-AUC, also ROC-AUC): the model is chosen by out-of-fold PR-AUC;
  * honest probabilities (log loss, Brier): the chosen model's scores are Platt-calibrated
    on out-of-fold predictions; this is monotonic, so PR-AUC/ROC-AUC are unchanged;
  * sensible hard labels (F1, precision, recall, MCC): the 0/1 cutoff maximises
    out-of-fold F1.
Imputation (preprocess.clean) is re-learned inside every fold, so no validation rows leak
into the fill values.
"""
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
                             f1_score, log_loss, matthews_corrcoef, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from preprocess import IMPUTED_COLS, clean, fit_stats

TRAIN_CSV = "data/Track_2_Training_Dataset.csv"
MODEL_PATH = "model.joblib"
N_SPLITS, N_REPEATS = 5, 3

CATEGORICAL = ["merchant_category", "country", "transaction_channel"]
FLAGS = [f"{c}_was_missing" for c in IMPUTED_COLS]
BINARY = ["new_device", "is_young_account", "new_device_young_account", "is_high_risk_merchant",
          "is_night", "zero_spend_with_txns", "account_age_capped"]
# Linear models get log-scaled versions of the heavy-tailed columns.
LINEAR_NUMERIC = ["log_amount", "log_spend_24h", "log_amount_to_avg_ratio", "log_account_age",
                  "transactions_last_24h", "transactions_last_1h", "txn_share_last_1h",
                  "hour_sin", "hour_cos"]
# Tree models are scale-free, so they get the raw columns plus the ratios.
TREE_NUMERIC = ["transaction_amount", "spend_last_24h", "avg_spend_per_txn_24h", "account_age",
                "amount_to_avg_ratio", "transactions_last_24h", "transactions_last_1h",
                "txn_share_last_1h", "transaction_hour", "hour_sin", "hour_cos"]


def _encoder(numeric):
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", StandardScaler(), numeric + BINARY + FLAGS),
    ], sparse_threshold=0)


def logreg(class_weight=None, C=0.1):
    return make_pipeline(_encoder(LINEAR_NUMERIC),
                         LogisticRegression(C=C, class_weight=class_weight, max_iter=5000))


def boosting():
    return make_pipeline(_encoder(TREE_NUMERIC), HistGradientBoostingClassifier(
        learning_rate=0.03, max_iter=400, max_leaf_nodes=8, min_samples_leaf=40,
        l2_regularization=5.0, random_state=0))


CANDIDATES = {
    "logistic regression": lambda: [logreg()],
    "logistic regression (balanced weights)": lambda: [logreg("balanced")],
    "gradient boosting": lambda: [boosting()],
    "blend: logistic regression + boosting": lambda: [logreg(), boosting()],
}


def to_logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def score(models, X):
    """Blend = average of the members' log-odds."""
    return np.mean([to_logit(m.predict_proba(X)[:, 1]) for m in models], axis=0)


def prepare(raw_train, raw_other):
    stats = fit_stats(raw_train)
    return clean(raw_train, stats, verbose=False), clean(raw_other, stats, verbose=False), stats


def cross_validate(raw):
    y = raw["fraud"].to_numpy()
    oof = {name: np.zeros((N_REPEATS, len(raw))) for name in CANDIDATES}
    for rep in range(N_REPEATS):
        print(f"  repeat {rep + 1}/{N_REPEATS} ...", flush=True)
        for tr, va in StratifiedKFold(N_SPLITS, shuffle=True, random_state=rep).split(raw, y):
            train, valid, _ = prepare(raw.iloc[tr], raw.iloc[va])
            for name, make in CANDIDATES.items():
                models = [m.fit(train, y[tr]) for m in make()]
                oof[name][rep, va] = score(models, valid)
    return y, oof


def best_f1_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(thr[np.argmax(f1[:-1])])


def report(y, p, threshold):
    pred = (p >= threshold).astype(int)
    return {
        "PR-AUC": average_precision_score(y, p), "ROC-AUC": roc_auc_score(y, p),
        "log loss": log_loss(y, p), "Brier": brier_score_loss(y, p),
        "F1": f1_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred), "MCC": matthews_corrcoef(y, pred),
        "balanced acc": balanced_accuracy_score(y, pred),
    }


def train():
    raw = pd.read_csv(TRAIN_CSV)
    print(f"Loaded {len(raw)} rows, fraud rate {raw['fraud'].mean():.3%} "
          f"(PR-AUC of random guessing = {raw['fraud'].mean():.3f})")
    print(f"\n{N_REPEATS}x repeated {N_SPLITS}-fold CV, imputation re-learned in every fold:")
    y, oof = cross_validate(raw)

    print("\nOut-of-fold PR-AUC (mean ± std over repeats) / ROC-AUC:")
    pr = {n: [average_precision_score(y, s) for s in oof[n]] for n in CANDIDATES}
    for n in sorted(CANDIDATES, key=lambda n: -np.mean(pr[n])):
        roc = np.mean([roc_auc_score(y, s) for s in oof[n]])
        print(f"  {n:42s} {np.mean(pr[n]):.4f} ± {np.std(pr[n]):.4f}   {roc:.4f}")
    best = max(CANDIDATES, key=lambda n: np.mean(pr[n]))
    print(f"\nChosen model: {best}")

    # Platt calibration on the out-of-fold scores of the chosen model.
    s = oof[best].ravel()
    calibrator = LogisticRegression(C=1e6).fit(s.reshape(-1, 1), np.tile(y, N_REPEATS))
    p = calibrator.predict_proba(s.reshape(-1, 1))[:, 1]
    threshold = best_f1_threshold(np.tile(y, N_REPEATS), p)
    uncal = 1 / (1 + np.exp(-s))
    print(f"Calibrated probabilities: log loss {log_loss(np.tile(y, N_REPEATS), uncal):.4f} "
          f"-> {log_loss(np.tile(y, N_REPEATS), p):.4f}")
    print(f"Decision threshold (max out-of-fold F1): {threshold:.4f}")

    print("\nExpected test performance (out-of-fold, mean over repeats):")
    rows = [report(y, calibrator.predict_proba(r.reshape(-1, 1))[:, 1], threshold)
            for r in oof[best]]
    for k in rows[0]:
        print(f"  {k:13s} {np.mean([r[k] for r in rows]):.4f}")

    stats = fit_stats(raw)
    full = clean(raw, stats, verbose=False)
    models = [m.fit(full, y) for m in CANDIDATES[best]()]
    joblib.dump({"name": best, "models": models, "calibrator": calibrator, "stats": stats,
                 "threshold": threshold}, MODEL_PATH)
    print(f"\nFitted {best} on all {len(raw)} rows and saved it to {MODEL_PATH}")


def predict(src, dst):
    bundle = joblib.load(MODEL_PATH)
    raw = pd.read_csv(src)
    data = clean(raw, bundle["stats"], verbose=False)
    p = bundle["calibrator"].predict_proba(score(bundle["models"], data).reshape(-1, 1))[:, 1]
    out = pd.DataFrame({"id": raw["id"], "fraud_probability": p,
                        "fraud_probability_pct": (100 * p).round(2),
                        "fraud": (p >= bundle["threshold"]).astype(int)})
    out.to_csv(dst, index=False)
    print(f"Predicted {len(out)} rows with {bundle['name']}: {out['fraud'].sum()} flagged as fraud "
          f"(threshold {bundle['threshold']:.4f}); saved to {dst}")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "predict":
        predict(*sys.argv[2:4])
    else:
        train()
