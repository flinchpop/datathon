"""Compare candidate models under the three validation views.

    python -m src.run_cv            # writes outputs/cv_results.csv
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .evaluation import cross_validate, shift_weights, summarise, test_pattern_distribution
from .features import COOLING_BASE, ROOT, SENSORS, load_test, load_train
from .models import GBMModel, LinearPlusGBM, ReducedFeatureModel, StructuredLinearModel


class Persistence:
    """Naive baseline: this hour = previous hour (building-hour mean if unknown)."""

    def fit(self, df, y, sample_weight=None):
        d = df.assign(_y=y)
        self.cell_ = d.groupby(["building_id", "hour"])["_y"].mean()
        return self

    def predict(self, df):
        fallback = self.cell_.reindex(pd.MultiIndex.from_frame(df[["building_id", "hour"]])).to_numpy()
        return np.where(df["previous_usage"].notna(), df["previous_usage"], fallback)


class ContextImputed:
    """Fill missing sensors with the training building x hour median, then predict."""

    def __init__(self, model):
        self.model = model

    def _fill(self, df):
        df = df.copy()
        for s in SENSORS:
            key = pd.MultiIndex.from_frame(df[["building_id", "hour"]])
            df[s] = df[s].fillna(pd.Series(self.med_[s].reindex(key).to_numpy(), index=df.index))
        df["cooling_deg"] = np.clip(df["temperature"] - COOLING_BASE, 0, None)
        return df

    def fit(self, df, y, sample_weight=None):
        self.med_ = {s: df.groupby(["building_id", "hour"])[s].median() for s in SENSORS}
        self.model.fit(self._fill(df), y, sample_weight)
        return self

    def predict(self, df):
        return self.model.predict(self._fill(df))


def candidates() -> dict:
    lin = lambda s: StructuredLinearModel(s)
    gbm = lambda s: GBMModel(s)
    hyb = lambda s: LinearPlusGBM(s)
    return {
        "Persistence baseline": lambda: Persistence(),
        "Linear + median imputation": lambda: ContextImputed(StructuredLinearModel()),
        "Linear (reduced-feature)": lambda: ReducedFeatureModel(lin),
        "LightGBM (native NaN)": lambda: GBMModel(),
        "LightGBM (reduced-feature)": lambda: ReducedFeatureModel(gbm),
        "Linear+GBM hybrid (reduced-feature)": lambda: ReducedFeatureModel(hyb),
    }


def main(n_repeats: int = 3, only: list[str] | None = None) -> pd.DataFrame:
    train, test = load_train(), load_test()
    probs = test_pattern_distribution(test)
    weights = shift_weights(train, test)
    rows = []
    for name, factory in candidates().items():
        if only and name not in only:
            continue
        t0 = time.time()
        cv = cross_validate(factory, train, probs, n_repeats=n_repeats)
        rows.append({"model": name, **summarise(cv, weights), "seconds": round(time.time() - t0)})
        print(f"{name}: done in {rows[-1]['seconds']}s", flush=True)
    res = pd.DataFrame(rows).round(4)
    out = ROOT / "outputs" / "cv_results.csv"
    res.to_csv(out, index=False)
    print(res.to_string(index=False))
    return res


if __name__ == "__main__":
    main()
