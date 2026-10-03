"""Validation tools: three complementary ways to estimate hidden-test performance.

The hidden test set is *not* drawn from the same distribution as the training set
(an adversarial classifier separates them with ROC-AUC ~0.67: test rows have
older accounts, bigger amounts, twice as many new devices and velocity bursts).
Ordinary cross-validation only measures in-distribution performance, so every
model is scored three ways:

1. Repeated stratified 5-fold CV (3 repeats): in-distribution performance.
2. Shift-weighted CV: the same out-of-fold predictions, but each training row is
   weighted by p(test|x) / p(train|x) from the adversarial classifier, so the
   metrics describe a population that looks like the test set.
3. Adversarial hold-out: train on the 70% of rows that look *least* like the test
   set and validate on the 30% that look *most* like it. This tests whether a
   model extrapolates to the regions the test set over-represents.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

from features import CATEGORICAL, NUMERIC

SEED = 42


def adversarial_weights(train, test, seed=0):
    """Importance weights p(test|x)/p(train|x) for every training row, out-of-fold."""
    both = pd.concat([train[NUMERIC + CATEGORICAL], test[NUMERIC + CATEGORICAL]], ignore_index=True)
    for c in CATEGORICAL:
        both[c] = both[c].astype("category")
    is_test = np.r_[np.zeros(len(train)), np.ones(len(test))]
    p = np.zeros(len(both))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(both, is_test):
        m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15,
                               min_child_samples=50, verbose=-1, random_state=seed)
        m.fit(both.iloc[a], is_test[a])
        p[b] = m.predict_proba(both.iloc[b])[:, 1]
    auc = roc_auc_score(is_test, p)
    p = np.clip(p[: len(train)], 0.02, 0.98)
    w = np.clip((p / (1 - p)) / np.mean(p / (1 - p)), 0, 10)
    return w / w.mean(), auc


def adversarial_split(weights, holdout_frac=0.3):
    """Indices (train_part, test_like_part) splitting rows by how test-like they are."""
    order = np.argsort(weights, kind="stable")
    cut = int(len(weights) * (1 - holdout_frac))
    return np.sort(order[:cut]), np.sort(order[cut:])


def pr_points(y, p, w=None):
    precision, recall, thr = precision_recall_curve(y, p, sample_weight=w)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return precision[:-1], recall[:-1], f1[:-1], thr


def metrics_at(y, p, threshold, w=None):
    w = np.ones(len(y)) if w is None else w
    pred = p >= threshold
    tp = np.sum(w * (pred & (y == 1)))
    fp = np.sum(w * (pred & (y == 0)))
    fn = np.sum(w * (~pred & (y == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "flag_rate": np.sum(w * pred) / np.sum(w)}


def summary(y, p, w=None):
    """Threshold-free metrics plus the best achievable F1."""
    _, _, f1, _ = pr_points(y, p, w)
    return {"pr_auc": average_precision_score(y, p, sample_weight=w),
            "roc_auc": roc_auc_score(y, p, sample_weight=w),
            "best_f1": float(np.max(f1))}


def choose_threshold(y, oof_runs, tolerance=0.05):
    """Decision threshold from out-of-fold predictions.

    The F1-vs-threshold curve is flat near its peak (with ~350 frauds the top few
    thresholds are statistically indistinguishable), so we take the *lowest*
    threshold whose F1 is within `tolerance` of the best. That buys extra recall
    (missed fraud is costlier than a false alarm, and recall is graded) for a
    negligible F1 cost.
    """
    grid = np.round(np.arange(0.02, 0.50, 0.005), 4)
    f1 = np.array([np.mean([metrics_at(y, p, t)["f1"] for p in oof_runs]) for t in grid])
    best = f1.max()
    t = grid[np.argmax(f1 >= (1 - tolerance) * best)]
    return float(t), float(grid[np.argmax(f1)]), float(best)


def cross_validate(fit_predict, train, y, weights, n_repeats=3):
    """Run repeated CV + adversarial hold-out for `fit_predict(train_idx, valid_idx) -> proba`.

    Returns (results dict, list of OOF prediction arrays, hold-out predictions, hold-out idx).
    """
    oof_runs = []
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=SEED)
    oof = np.zeros(len(y))
    for k, (a, b) in enumerate(rskf.split(train, y)):
        oof[b] = fit_predict(a, b)
        if k % 5 == 4:
            oof_runs.append(oof.copy())
    a, b = adversarial_split(weights)
    holdout = fit_predict(a, b)

    def avg(rows):
        return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}

    results = {
        "cv": avg([summary(y, p) for p in oof_runs]),
        "shift_weighted_cv": avg([summary(y, p, weights) for p in oof_runs]),
        "adversarial_holdout": summary(y[b], holdout),
    }
    return results, oof_runs, holdout, b


def format_results(name, r):
    return (f"{name:34s} | CV PR-AUC {r['cv']['pr_auc']:.4f}  ROC {r['cv']['roc_auc']:.4f}  "
            f"bestF1 {r['cv']['best_f1']:.4f} | shift-wtd PR-AUC {r['shift_weighted_cv']['pr_auc']:.4f}  "
            f"bestF1 {r['shift_weighted_cv']['best_f1']:.4f} | test-like hold-out PR-AUC "
            f"{r['adversarial_holdout']['pr_auc']:.4f}  bestF1 {r['adversarial_holdout']['best_f1']:.4f}")
