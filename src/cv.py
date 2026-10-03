"""Repeated stratified cross-validation harness.

With only 353 positives, a single 5-fold split gives PR-AUC estimates with a
standard error of ~0.02 -- larger than most of the gains we are hunting for.
We therefore use RepeatedStratifiedKFold (default 5 folds x 6 repeats = 30 fits)
and report mean +/- std, plus a paired comparison against a baseline when asked.

All label-dependent preprocessing (target encoding) is done *inside* each fold:
fit on the training part, nested-OOF for the training rows, plain transform for
the validation rows.  Merchant-relative amount statistics use amounts only (no
labels) and are fitted on the training fold.
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve

from .features import (
    add_basic_features, add_merchant_relative_features, TargetEncoder,
    FEATURE_SETS, TE_COLS_BASE, TE_COLS_INTER,
)


def best_f1(y_true, score):
    """Max F1 over all thresholds; returns (f1, recall, precision, threshold)."""
    p, r, t = precision_recall_curve(y_true, score)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    i = int(np.nanargmax(f1[:-1]))  # last point has no threshold
    return float(f1[i]), float(r[i]), float(p[i]), float(t[i])


def metrics(y_true, score) -> dict:
    f1, rec, prec, thr = best_f1(y_true, score)
    return {
        "pr_auc": average_precision_score(y_true, score),
        "roc_auc": roc_auc_score(y_true, score),
        "best_f1": f1, "recall_at_best_f1": rec, "precision_at_best_f1": prec, "thr": thr,
    }


def build_fold_features(tr_raw, va_raw, feature_set: str, te_cols=None, te_m: float = 20.0,
                        seed: int = 0, merchant_ref: str = "train", cat_cols=None):
    """Return (X_tr, X_va, feature_names) for one fold."""
    y_tr = tr_raw["fraud"].values
    tr = add_basic_features(tr_raw)
    va = add_basic_features(va_raw)
    ref = tr if merchant_ref == "train" else pd.concat([tr, va])
    tr = add_merchant_relative_features(tr, ref)
    va = add_merchant_relative_features(va, ref)
    cols = list(FEATURE_SETS[feature_set])
    X_tr = tr[cols].astype(float)
    X_va = va[cols].astype(float)
    if te_cols:
        te = TargetEncoder(te_cols, m=te_m, seed=seed)
        X_tr = pd.concat([X_tr, te.fit_transform_oof(tr, y_tr)], axis=1)
        X_va = pd.concat([X_va, te.transform(va)], axis=1)
    if cat_cols:  # raw string categoricals for models with native categorical support (CatBoost)
        for c in cat_cols:
            X_tr[c] = tr[c].fillna("NA").astype(str).astype(object).values
            X_va[c] = va[c].fillna("NA").astype(str).astype(object).values
    return X_tr, X_va, list(X_tr.columns)


def run_cv(df: pd.DataFrame, make_model, feature_set="basic", te_cols=None, te_m=20.0,
           n_splits=5, n_repeats=6, seed=42, fit_kwargs=None, sample_weight_fn=None,
           merchant_ref="train", verbose=True, return_oof=False, cat_cols=None):
    """Run repeated stratified CV.

    make_model(seed) -> estimator with fit/predict_proba.
    sample_weight_fn(X_tr, y_tr) -> weights (optional, for cost-sensitive / shift weighting).
    Returns dict with per-fold metrics, mean/std, and (optionally) OOF scores per repeat.
    """
    y = df["fraud"].values
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    rows, oof = [], np.zeros((n_repeats, len(df)))
    t0 = time.time()
    for k, (tr_idx, va_idx) in enumerate(rskf.split(df, y)):
        rep = k // n_splits
        X_tr, X_va, names = build_fold_features(df.iloc[tr_idx], df.iloc[va_idx], feature_set,
                                                te_cols, te_m, seed=seed + k, merchant_ref=merchant_ref,
                                                cat_cols=cat_cols)
        y_tr, y_va = y[tr_idx], y[va_idx]
        model = make_model(seed + k)
        kw = dict(fit_kwargs or {})
        if sample_weight_fn is not None:
            kw["sample_weight"] = sample_weight_fn(X_tr, y_tr)
        model.fit(X_tr, y_tr, **kw)
        s = model.predict_proba(X_va)[:, 1]
        oof[rep, va_idx] = s
        m = metrics(y_va, s); m["fold"] = k; m["repeat"] = rep
        rows.append(m)
    res = pd.DataFrame(rows)
    # pooled per-repeat metrics (threshold-based metrics are more stable when pooled)
    pooled = pd.DataFrame([metrics(y, oof[r]) for r in range(n_repeats)])
    out = {
        "fold_pr_auc_mean": res.pr_auc.mean(), "fold_pr_auc_std": res.pr_auc.std(),
        "pooled_pr_auc_mean": pooled.pr_auc.mean(), "pooled_pr_auc_std": pooled.pr_auc.std(),
        "roc_auc": pooled.roc_auc.mean(),
        "best_f1": pooled.best_f1.mean(), "recall_at_best_f1": pooled.recall_at_best_f1.mean(),
        "precision_at_best_f1": pooled.precision_at_best_f1.mean(),
        "fold_results": res, "pooled_results": pooled, "features": names, "secs": time.time() - t0,
    }
    if return_oof:
        out["oof"] = oof
    if verbose:
        print(f"  PR-AUC fold {out['fold_pr_auc_mean']:.4f}±{out['fold_pr_auc_std']:.4f} | pooled {out['pooled_pr_auc_mean']:.4f}±{out['pooled_pr_auc_std']:.4f}"
              f" | ROC {out['roc_auc']:.4f} | bestF1 {out['best_f1']:.4f} (R={out['recall_at_best_f1']:.3f}, P={out['precision_at_best_f1']:.3f})"
              f" | {len(names)} feats | {out['secs']:.0f}s")
    return out
