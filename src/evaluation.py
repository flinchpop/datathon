"""Metrics and a validation scheme that mimics the test set.

The test set differs from training in two measurable ways (REPORT.md, s.2):
  * ~18% of rows miss at least one sensor (vs ~6% in training), often two;
  * its covariates sit more often in the tails (heat waves, packed rooms,
    unusual previous-hour usage): a classifier separates train from test
    with AUC ~0.68.
So besides ordinary K-fold CV we report a "test-like" score: validation rows
get the test set's missingness patterns injected and are re-weighted by the
density ratio p(test | x) / p(train | x).
"""
from __future__ import annotations

from typing import Callable

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .features import SENSORS, TARGET, missing_pattern


def metrics(y, p, w=None) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    w = w / w.sum()
    err = y - p
    ybar = np.sum(w * y)
    return {
        "RMSE": float(np.sqrt(np.sum(w * err**2))),
        "MAE": float(np.sum(w * np.abs(err))),
        "R2": float(1 - np.sum(w * err**2) / np.sum(w * (y - ybar) ** 2)),
    }


def test_pattern_distribution(test: pd.DataFrame) -> pd.Series:
    return missing_pattern(test).value_counts(normalize=True)


def inject_missing(df: pd.DataFrame, pattern_probs: pd.Series, seed: int) -> pd.DataFrame:
    """Give each row a missingness pattern drawn from the test distribution.

    Rows that already miss something keep their own pattern; the draw is
    rescaled so the overall share of incomplete rows matches the test set.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    complete = out[SENSORS].notna().all(axis=1).to_numpy()
    pats = list(pattern_probs.index)
    probs = pattern_probs.to_numpy().copy()
    i_full = pats.index(())
    target_incomplete = 1 - probs[i_full]
    have_incomplete = 1 - complete.mean()
    p_drop = (target_incomplete - have_incomplete) / complete.mean()
    probs[i_full] = 0
    probs = probs / probs.sum()
    idx = np.flatnonzero(complete)
    hit = idx[rng.random(len(idx)) < p_drop]
    choice = rng.choice(len(pats), size=len(hit), p=probs)
    for row, c in zip(hit, choice):
        for col in pats[c]:
            out.iat[row, out.columns.get_loc(col)] = np.nan
    return out


def shift_weights(train: pd.DataFrame, test: pd.DataFrame, clip_q: float = 0.99,
                  seed: int = 0) -> np.ndarray:
    """Density-ratio weights w(x) = p(test|x)/p(train|x) for training rows.

    Uses an out-of-fold LightGBM classifier on the covariates only (the
    missingness itself is handled by injection, so NaNs are filled with
    building/hour medians first to stop the classifier keying on them).
    """
    cols = ["hour", "dow", "month"] + SENSORS
    both = pd.concat([train.assign(_t=0), test.assign(_t=1)], ignore_index=True)
    for s in SENSORS:
        med = both.groupby(["building_id", "hour"])[s].transform("median")
        both[s] = both[s].fillna(med).fillna(both[s].median())
    X = both[cols].copy()
    X["building_id"] = pd.Categorical(both["building_id"])
    t = both["_t"].to_numpy()
    p = np.zeros(len(both))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, t):
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                 min_child_samples=40, verbose=-1, random_state=seed)
        clf.fit(X.iloc[a], t[a])
        p[b] = clf.predict_proba(X.iloc[b])[:, 1]
    p_tr = np.clip(p[t == 0], 1e-3, 1 - 1e-3)
    w = p_tr / (1 - p_tr) * (t == 0).sum() / (t == 1).sum()
    w = np.minimum(w, np.quantile(w, clip_q))
    return w / w.mean()


def cross_validate(factory: Callable[[], object], train: pd.DataFrame,
                   pattern_probs: pd.Series, n_splits: int = 5, n_repeats: int = 3,
                   seed: int = 0, train_weights=None) -> dict:
    """Out-of-fold predictions on clean and on test-like (injected) rows."""
    y = train[TARGET].to_numpy()
    clean = np.zeros((n_repeats, len(y)))
    injected = np.zeros((n_repeats, len(y)))
    strata = pd.qcut(y, 10, labels=False)
    for r in range(n_repeats):
        dirty = inject_missing(train, pattern_probs, seed=1000 + r)
        skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed + r)
        for a, b in skf.split(train, strata):
            w = None if train_weights is None else train_weights[a]
            model = factory().fit(train.iloc[a], y[a], w)
            clean[r, b] = model.predict(train.iloc[b])
            injected[r, b] = model.predict(dirty.iloc[b])
    return {"clean": clean, "injected": injected, "y": y}


def summarise(cv: dict, weights: np.ndarray) -> dict:
    """Average the metrics over repeats for the three evaluation views."""
    y = cv["y"]
    views = {
        "cv": [metrics(y, p) for p in cv["clean"]],
        "cv_missing": [metrics(y, p) for p in cv["injected"]],
        "test_like": [metrics(y, p, weights) for p in cv["injected"]],
    }
    out = {}
    for v, rows in views.items():
        for k in ("RMSE", "MAE", "R2"):
            out[f"{v}_{k}"] = float(np.mean([r[k] for r in rows]))
    return out
