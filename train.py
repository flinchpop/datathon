"""End-to-end training pipeline for the Track 2 fraud-detection task.

    python train.py                  # full run: benchmark all models, fit EBM, predict test
    python train.py --skip-benchmark # only cross-validate + fit the final EBM

Outputs
    outputs/submission.csv        id, fraud_probability, fraud_prediction (test set)
    outputs/cv_results.csv        repeated-CV metrics for every benchmark model
    outputs/threshold_table.csv   precision / recall / F1 / F-beta at each threshold
    outputs/oof_predictions.csv   out-of-fold EBM probabilities on the training set
    outputs/metrics.json          headline numbers (threshold, CV metrics, drift, test forecast)
    outputs/ebm_model.pkl         the fitted final model
    reports/figures/*.png         figures used in the report
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import plots  # noqa: E402
from fraud_pipeline import (  # noqa: E402
    DEFAULT_BETA, FIG_DIR, FINAL_MODEL_NAME, ID, OUTPUT_DIR, RAW_FEATURES, TARGET,
    adversarial_validation, choose_threshold, cross_validate, cv_splits, load_data,
    make_ebm, model_zoo, plug_in_metrics, summarise_cv,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-benchmark", action="store_true", help="only run the final EBM")
    ap.add_argument("--beta", type=float, default=DEFAULT_BETA,
                    help="F-beta used to pick the threshold (default sqrt(3) = equal weight on F1 and Recall)")
    args = ap.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    train, test = load_data()
    y = train[TARGET].to_numpy()
    prevalence = y.mean()
    print(f"train {train.shape}, test {test.shape}, fraud rate {prevalence:.4f} ({y.sum()} positives)")

    # ---- 1. Drift diagnostics ------------------------------------------------
    adv_auc, adv_imp, iw = adversarial_validation(train, test)
    print(f"adversarial validation AUC (train vs test): {adv_auc:.3f}")
    plots.fig_risk_drivers(train, FIG_DIR / "01_risk_drivers.png")
    plots.fig_drift(train, test, adv_auc, adv_imp, FIG_DIR / "02_drift.png")

    # ---- 2. Benchmark with repeated stratified CV ---------------------------
    splits = cv_splits(y)
    specs = model_zoo()
    if args.skip_benchmark:
        specs = [s for s in specs if s.name == FINAL_MODEL_NAME]
    results, rows = {}, []
    for spec in specs:
        t0 = time.time()
        res = cross_validate(spec, train, splits)
        results[spec.name] = res
        row = summarise_cv(res, y, args.beta, weights=iw)
        row["note"] = spec.note
        rows.append(row)
        print(f"  {spec.name:40s} PR-AUC {row['pr_auc_mean']:.4f} ± {row['pr_auc_std']:.4f} | "
              f"ROC {row['roc_auc_mean']:.4f} | F1@t {row['f1_at_t']:.3f} Rec@t {row['recall_at_t']:.3f} | "
              f"{time.time() - t0:.0f}s", flush=True)
    cv = pd.DataFrame(rows).sort_values("pr_auc_mean", ascending=False)
    if not args.skip_benchmark:
        cv.to_csv(OUTPUT_DIR / "cv_results.csv", index=False)
        plots.fig_model_comparison(cv, prevalence, FINAL_MODEL_NAME, FIG_DIR / "03_model_comparison.png")

    # ---- 3. Threshold selection on EBM out-of-fold predictions --------------
    ebm_cv = results[FINAL_MODEL_NAME]
    threshold, table = choose_threshold(y, ebm_cv.oof, args.beta)
    table.to_csv(OUTPUT_DIR / "threshold_table.csv", index=False)
    oof_mean = ebm_cv.oof.mean(axis=0)
    pd.DataFrame({ID: train[ID], TARGET: y, "oof_probability": oof_mean}).to_csv(
        OUTPUT_DIR / "oof_predictions.csv", index=False)
    at_t = table.loc[table["threshold"] == threshold].iloc[0]
    f1_best = table.loc[table["f1"].idxmax()]
    print(f"chosen threshold {threshold:.3f}: precision {at_t.precision:.3f} recall {at_t.recall:.3f} "
          f"F1 {at_t.f1:.3f} (F1-optimal t={f1_best.threshold:.3f} gives F1 {f1_best.f1:.3f}, recall {f1_best.recall:.3f})")
    plots.fig_pr_curve(y, oof_mean, threshold, FIG_DIR / "04_pr_curve.png")
    plots.fig_threshold(table, threshold, args.beta, FIG_DIR / "05_threshold_tradeoff.png")
    plots.fig_calibration(y, oof_mean, FIG_DIR / "06_calibration.png")

    # ---- 4. Final model on all training data --------------------------------
    ebm = make_ebm().fit(train[RAW_FEATURES], y)
    with open(OUTPUT_DIR / "ebm_model.pkl", "wb") as f:
        pickle.dump({"model": ebm, "threshold": threshold, "features": RAW_FEATURES, "beta": args.beta}, f)
    plots.fig_ebm_importance(ebm, FIG_DIR / "07_ebm_importance.png")
    plots.fig_ebm_shapes(ebm, FIG_DIR / "08_ebm_shapes.png")

    # ---- 5. Predict the test set --------------------------------------------
    p_test = ebm.predict_proba(test[RAW_FEATURES])[:, 1]
    sub = pd.DataFrame({ID: test[ID], "fraud_probability": p_test,
                        "fraud_prediction": (p_test >= threshold).astype(int)})
    sub.to_csv(OUTPUT_DIR / "submission.csv", index=False, float_format="%.8f")
    forecast = plug_in_metrics(p_test, threshold, args.beta)
    print(f"test: {sub.fraud_prediction.sum()} flagged ({sub.fraud_prediction.mean():.2%}); "
          f"model-implied fraud rate {p_test.mean():.4f}; plug-in recall {forecast['expected_recall']:.3f} "
          f"F1 {forecast['expected_f1']:.3f}")

    top = int(np.argmax(p_test))
    plots.fig_local_explanation(
        ebm, test[RAW_FEATURES].iloc[[top]],
        f"Why test transaction {test[ID].iloc[top]} was flagged (p = {p_test[top]:.2f})",
        FIG_DIR / "09_local_explanation.png")

    metrics = {
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "train_fraud_rate": float(prevalence), "train_positives": int(y.sum()),
        "adversarial_auc": float(adv_auc),
        "adversarial_importance": adv_imp.round(4).to_dict(),
        "final_model": FINAL_MODEL_NAME,
        "beta": args.beta,
        "threshold": threshold,
        "cv": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
               for k, v in cv.loc[cv.model == FINAL_MODEL_NAME].iloc[0].items()},
        "cv_at_threshold": {k: float(at_t[k]) for k in ["precision", "recall", "f1", "f2", "f_beta", "flag_rate"]},
        "cv_f1_optimal": {k: float(f1_best[k]) for k in ["threshold", "precision", "recall", "f1"]},
        "test_flagged": int(sub.fraud_prediction.sum()),
        "test_forecast_if_calibrated": {k: float(v) for k, v in forecast.items()},
    }
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("done.")


if __name__ == "__main__":
    main()
