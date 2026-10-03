"""Final training pipeline: fit the ensemble on the full training set and score the test set.

Steps
-----
1. Build engineered features for train and test (merchant-relative stats fitted on train).
2. Target encodings: OOF for train rows (so the model never sees a row's own label),
   full-train maps for test rows.  CatBoost members use raw string categoricals
   instead (CatBoost's ordered target statistics do the same job internally).
3. Fit each ensemble member with several seeds, average seeds within a member,
   then rank-average across members with the configured weights.
4. Choose the decision threshold from repeated-CV OOF predictions (F1-optimal), so
   the same threshold can be applied to the test scores.
"""
from __future__ import annotations
import json, time
import numpy as np, pandas as pd
from scipy.stats import rankdata

from .features import add_basic_features, add_merchant_relative_features, TargetEncoder, FEATURE_SETS
from .cv import run_cv, metrics, best_f1


def build_train_test(train: pd.DataFrame, test: pd.DataFrame, feature_set: str, te_cols=None, te_m=20.0,
                     cat_cols=None, seed=0):
    y = train["fraud"].values
    tr = add_basic_features(train); te = add_basic_features(test)
    tr = add_merchant_relative_features(tr, tr); te = add_merchant_relative_features(te, tr)
    cols = list(FEATURE_SETS[feature_set])
    X_tr = tr[cols].astype(float); X_te = te[cols].astype(float)
    if te_cols:
        enc = TargetEncoder(te_cols, m=te_m, seed=seed)
        X_tr = pd.concat([X_tr, enc.fit_transform_oof(tr, y)], axis=1)
        X_te = pd.concat([X_te, enc.transform(te)], axis=1)
    if cat_cols:
        for c in cat_cols:
            X_tr[c] = tr[c].fillna("NA").astype(str).astype(object).values
            X_te[c] = te[c].fillna("NA").astype(str).astype(object).values
    return X_tr, X_te, y


def fit_predict_member(train, test, cfg, seeds, verbose=True):
    """Average predictions of one member over several seeds; returns (test_scores, importances)."""
    preds, imps = [], []
    for s in seeds:
        X_tr, X_te, y = build_train_test(train, test, cfg["feature_set"], cfg.get("te_cols"), cfg.get("te_m", 20.0),
                                         cfg.get("cat_cols"), seed=s)
        m = cfg["make_model"](s)
        m.fit(X_tr, y)
        preds.append(m.predict_proba(X_te)[:, 1])
        imp = getattr(m, "feature_importances_", None)
        if imp is not None:
            imps.append(pd.Series(np.asarray(imp, dtype=float), index=X_tr.columns))
    p = np.mean(preds, axis=0)
    imp = pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False) if imps else None
    return p, imp


def rank_average(score_list, weights):
    n = len(score_list[0])
    out = np.zeros(n)
    for s, w in zip(score_list, weights):
        out += w * rankdata(s) / n
    return out / float(np.sum(weights))


def run_pipeline(train, test, members: dict, weights: dict, seeds, n_repeats_cv=6, out_dir="outputs", verbose=True):
    t0 = time.time()
    # ---- 1. CV for each member on identical splits -> OOF for blending + threshold selection
    oofs, cv_rows = {}, []
    for name, cfg in members.items():
        if verbose: print(f"[cv] {name}", flush=True)
        r = run_cv(train, cfg["make_model"], feature_set=cfg["feature_set"], te_cols=cfg.get("te_cols"),
                   te_m=cfg.get("te_m", 20.0), cat_cols=cfg.get("cat_cols"), n_repeats=n_repeats_cv, return_oof=True, verbose=verbose)
        oofs[name] = r["oof"]
        cv_rows.append(dict(member=name, pr_auc=r["pooled_pr_auc_mean"], pr_auc_std=r["pooled_pr_auc_std"], roc_auc=r["roc_auc"],
                            best_f1=r["best_f1"], recall_at_best_f1=r["recall_at_best_f1"]))
    y = train["fraud"].values
    names = list(members)
    w = [weights[n] for n in names]
    blend_oof = np.stack([rank_average([oofs[n][r] for n in names], w) for r in range(n_repeats_cv)])
    pooled = pd.DataFrame([metrics(y, blend_oof[r]) for r in range(n_repeats_cv)])
    cv_rows.append(dict(member="BLEND", pr_auc=pooled.pr_auc.mean(), pr_auc_std=pooled.pr_auc.std(), roc_auc=pooled.roc_auc.mean(),
                        best_f1=pooled.best_f1.mean(), recall_at_best_f1=pooled.recall_at_best_f1.mean()))
    cv_table = pd.DataFrame(cv_rows)
    if verbose: print("\n", cv_table.round(4).to_string(index=False), flush=True)

    # ---- threshold from OOF blend: the blend is a rank score in (0,1]; pick the F1-optimal quantile
    thr_rows = [best_f1(y, blend_oof[r]) for r in range(n_repeats_cv)]
    thr = float(np.median([t[3] for t in thr_rows]))                  # rank-score threshold
    flag_rate = float(np.mean([(blend_oof[r] >= thr).mean() for r in range(n_repeats_cv)]))  # share flagged
    if verbose: print(f"\nF1-optimal rank threshold {thr:.4f} -> flags {flag_rate*100:.2f}% of transactions "
                      f"(CV F1 {np.mean([t[0] for t in thr_rows]):.4f}, recall {np.mean([t[1] for t in thr_rows]):.3f}, precision {np.mean([t[2] for t in thr_rows]):.3f})")

    # ---- 2. fit on full train, predict test
    test_scores, importances = {}, {}
    for name, cfg in members.items():
        if verbose: print(f"[fit] {name} x {len(seeds)} seeds", flush=True)
        test_scores[name], importances[name] = fit_predict_member(train, test, cfg, seeds, verbose)
    blend_test = rank_average([test_scores[n] for n in names], w)
    # calibrated-probability companion: average of member probabilities (for human-readable risk)
    prob_test = np.mean([test_scores[n] for n in names], axis=0)

    result = dict(cv_table=cv_table, blend_oof=blend_oof, oofs=oofs, threshold=thr, flag_rate=flag_rate,
                  test_rank_score=blend_test, test_prob=prob_test, test_member_scores=test_scores,
                  importances=importances, secs=time.time() - t0)
    return result
