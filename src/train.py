"""Fit the final model, report its validation scores, write the submission.

    python -m src.train            # -> outputs/submission.csv, outputs/final_metrics.json
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .evaluation import (cross_validate, inject_missing, metrics, shift_weights, summarise,
                         test_pattern_distribution)
from .features import ROOT, SENSORS, TARGET, load_test, load_train, missing_pattern
from .models import LinearPlusGBM, ReducedFeatureModel

OUT = ROOT / "outputs"


def final_model():
    """Structured linear model + boosted residual correction, one per
    missing-sensor pattern (see REPORT.md for why each piece is there)."""
    return ReducedFeatureModel(lambda sensors: LinearPlusGBM(sensors))


def main(n_repeats: int = 3) -> None:
    OUT.mkdir(exist_ok=True)
    train, test = load_train(), load_test()
    y = train[TARGET].to_numpy()

    # 1. Honest validation scores (5-fold x n_repeats).
    probs = test_pattern_distribution(test)
    weights = shift_weights(train, test)
    cv = cross_validate(final_model, train, probs, n_repeats=n_repeats)
    scores = summarise(cv, weights)
    oof = cv["clean"].mean(axis=0)
    # Error split by how many sensors were missing, on the test-like copy
    # (repeat 0 used seed 1000 inside cross_validate).
    dirty = inject_missing(train, probs, seed=1000)
    n_gaps = dirty[SENSORS].isna().sum(axis=1).to_numpy()
    scores["by_missing_sensors_test_like"] = {
        f"{k} missing": {**metrics(y[n_gaps == k], cv["injected"][0][n_gaps == k]),
                         "n": int((n_gaps == k).sum())}
        for k in sorted(set(n_gaps))
    }
    pd.DataFrame({"id": train["id"], "actual": y, "oof_pred": oof,
                  "oof_pred_test_like_gaps": cv["injected"][0],
                  "n_missing_test_like": n_gaps}).to_csv(OUT / "oof_predictions.csv", index=False)

    # 2. Fit on all 8,000 rows and predict the 3,000 test rows.
    model = final_model().fit(train, y)
    pred = model.predict(test)
    assert len(pred) == len(test) and np.isfinite(pred).all()
    pred = np.clip(pred, 0, None)  # energy cannot be negative
    sub = pd.DataFrame({"id": test["id"], TARGET: np.round(pred, 4)})
    sub.to_csv(OUT / "submission.csv", index=False)

    with open(OUT / "final_metrics.json", "w") as f:
        json.dump(scores, f, indent=2)
    print(json.dumps(scores, indent=2))
    print(f"submission: {len(sub)} rows, mean {pred.mean():.2f}, "
          f"{len(model.models_)} sub-models for {len(set(missing_pattern(test)))} gap patterns")


if __name__ == "__main__":
    main()
