"""Models: structured linear regression, LightGBM, their hybrid, and the
reduced-feature wrapper that handles missing sensor readings."""
from __future__ import annotations

from typing import Callable, Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import SENSORS, missing_pattern


# --------------------------------------------------------------------------
# 1. Structured linear model
# --------------------------------------------------------------------------
class StructuredLinearModel:
    """Least squares on an explicit, interpretable design matrix.

    energy = building offset
           + daily load profile of the building type (type x hour)
           + weekend shift of the building type (type x weekend)
           + month (academic-calendar seasonality)
           + type-specific temperature slope + cooling-degree hinge
           + humidity
           + type-specific occupancy slope
           + previous-hour usage (persistence)
           + noise

    Continuous inputs are centred on their training means so that the
    building/hour coefficients read as "usage under average conditions".
    """

    def __init__(self, sensors: Sequence[str] = SENSORS, alpha: float = 1.0):
        self.sensors = [s for s in SENSORS if s in sensors]
        self.alpha = alpha  # tiny ridge penalty, for numerical stability only

    def _levels(self, df: pd.DataFrame) -> None:
        self.buildings_ = sorted(df["building_id"].unique())
        self.types_ = sorted(df["building_type"].unique())
        self.months_ = sorted(df["month"].unique())
        self.centre_ = {s: df[s].mean() for s in self.sensors}

    def design(self, df: pd.DataFrame) -> pd.DataFrame:
        cols: dict[str, np.ndarray] = {"const": np.ones(len(df))}
        bld, typ = df["building_id"].to_numpy(), df["building_type"].to_numpy()
        hour, wk = df["hour"].to_numpy(), df["weekend"].to_numpy()
        for b in self.buildings_[1:]:
            cols[f"bld[{b}]"] = (bld == b).astype(float)
        for t in self.types_:
            is_t = (typ == t).astype(float)
            for h in range(1, 24):
                cols[f"{t}:hour{h}"] = is_t * (hour == h)
            cols[f"{t}:weekend"] = is_t * wk
        for m in self.months_[1:]:
            cols[f"month{m}"] = (df["month"].to_numpy() == m).astype(float)
        c = {s: df[s].to_numpy(dtype=float) - self.centre_[s] for s in self.sensors}
        if "temperature" in c:
            for t in self.types_:
                cols[f"{t}:temperature"] = (typ == t) * c["temperature"]
            cols["cooling_deg"] = df["cooling_deg"].to_numpy(dtype=float)
        if "humidity" in c:
            cols["humidity"] = c["humidity"]
        if "occupancy" in c:
            for t in self.types_:
                cols[f"{t}:occupancy"] = (typ == t) * c["occupancy"]
        if "previous_usage" in c:
            cols["previous_usage"] = c["previous_usage"]
        return pd.DataFrame(cols, index=df.index)

    def fit(self, df: pd.DataFrame, y, sample_weight=None):
        self._levels(df)
        X = self.design(df).to_numpy()
        y = np.asarray(y, dtype=float)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        Xw, yw = X * np.sqrt(w)[:, None], y * np.sqrt(w)
        pen = self.alpha * np.eye(X.shape[1])
        pen[0, 0] = 0.0  # never shrink the intercept
        self.coef_ = np.linalg.solve(Xw.T @ Xw + pen, Xw.T @ yw)
        self.columns_ = list(self.design(df.iloc[:1]).columns)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.design(df).to_numpy() @ self.coef_


# --------------------------------------------------------------------------
# 2. Gradient-boosted trees
# --------------------------------------------------------------------------
GBM_PARAMS = dict(
    objective="regression", learning_rate=0.02, n_estimators=1500,
    num_leaves=15, min_child_samples=20, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, reg_lambda=1.0, verbose=-1,
)


def gbm_frame(df: pd.DataFrame, sensors: Sequence[str]) -> pd.DataFrame:
    X = df[["hour", "dow", "weekend", "month"]].copy()
    X["building_id"] = pd.Categorical(df["building_id"])
    X["building_type"] = pd.Categorical(df["building_type"])
    for s in SENSORS:
        if s in sensors:
            X[s] = df[s]
    if "temperature" in sensors:
        X["cooling_deg"] = df["cooling_deg"]
    return X


class GBMModel:
    """LightGBM on the raw inputs; learns interactions without a formula."""

    def __init__(self, sensors: Sequence[str] = SENSORS, seed: int = 0, **params):
        self.sensors = [s for s in SENSORS if s in sensors]
        self.params = {**GBM_PARAMS, **params, "random_state": seed}

    def fit(self, df, y, sample_weight=None):
        self.model_ = lgb.LGBMRegressor(**self.params)
        self.model_.fit(gbm_frame(df, self.sensors), np.asarray(y, float),
                        sample_weight=sample_weight)
        return self

    def predict(self, df):
        return self.model_.predict(gbm_frame(df, self.sensors))


# --------------------------------------------------------------------------
# 3. Hybrid: linear structure + boosted trees on what the line misses
# --------------------------------------------------------------------------
HYBRID_GBM_PARAMS = dict(
    learning_rate=0.01, n_estimators=900, num_leaves=7, min_child_samples=80,
    reg_lambda=5.0,
)


class LinearPlusGBM:
    """Fit the structured linear model, then boost trees on its residuals.

    The linear part carries the main effects and extrapolates sensibly into
    the sparse tails; the shallow, heavily-regularised trees only add the
    non-linear corrections the formula cannot express. Training residuals are
    cross-fitted so the trees learn from honest (out-of-sample) errors.
    """

    def __init__(self, sensors: Sequence[str] = SENSORS, seed: int = 0,
                 n_inner: int = 5, **gbm_params):
        self.sensors = [s for s in SENSORS if s in sensors]
        self.seed, self.n_inner = seed, n_inner
        self.gbm_params = {**HYBRID_GBM_PARAMS, **gbm_params}

    def fit(self, df, y, sample_weight=None):
        y = np.asarray(y, float)
        oof = np.empty(len(y))
        folds = np.random.default_rng(self.seed).integers(0, self.n_inner, len(y))
        for k in range(self.n_inner):
            tr, va = folds != k, folds == k
            w = None if sample_weight is None else np.asarray(sample_weight)[tr]
            m = StructuredLinearModel(self.sensors).fit(df[tr], y[tr], w)
            oof[va] = m.predict(df[va])
        self.linear_ = StructuredLinearModel(self.sensors).fit(df, y, sample_weight)
        self.gbm_ = GBMModel(self.sensors, self.seed, **self.gbm_params)
        self.gbm_.fit(df, y - oof, sample_weight)
        return self

    def predict(self, df):
        return self.linear_.predict(df) + self.gbm_.predict(df)


# --------------------------------------------------------------------------
# 4. Missing data: one model per pattern of available sensors
# --------------------------------------------------------------------------
class ReducedFeatureModel:
    """Predict each row with a model trained only on the sensors it has.

    For a row with, say, occupancy missing we fit (once, then cache) the base
    model on all training rows using every input except occupancy. This gives
    the best estimate of E[usage | what we actually observed] without
    inventing values, and it is exact for the linear model's conditional
    mean (unlike plugging in an imputed number, which ignores the extra
    uncertainty). Training rows are used whenever they have the sensors that
    sub-model needs, so every sub-model sees ~8,000 rows.
    """

    def __init__(self, factory: Callable[[Sequence[str]], object]):
        self.factory = factory

    def fit(self, df, y, sample_weight=None):
        self.df_, self.y_ = df, np.asarray(y, float)
        self.w_ = None if sample_weight is None else np.asarray(sample_weight, float)
        self.models_: dict[tuple, object] = {}
        return self

    def _model(self, missing: tuple):
        if missing not in self.models_:
            used = [s for s in SENSORS if s not in missing]
            ok = self.df_[used].notna().all(axis=1).to_numpy()
            w = None if self.w_ is None else self.w_[ok]
            self.models_[missing] = self.factory(used).fit(self.df_[ok], self.y_[ok], w)
        return self.models_[missing]

    def predict(self, df):
        out = np.empty(len(df))
        pats = missing_pattern(df).tolist()
        for p in dict.fromkeys(pats):
            rows = np.array([q == p for q in pats])
            out[rows] = self._model(p).predict(df[rows])
        return out


class Blend:
    """Weighted average of already-defined models."""

    def __init__(self, models: Sequence, weights: Sequence[float]):
        self.models, self.weights = list(models), np.asarray(weights, float)

    def fit(self, df, y, sample_weight=None):
        for m in self.models:
            m.fit(df, y, sample_weight)
        return self

    def predict(self, df):
        return sum(w * m.predict(df) for m, w in zip(self.models, self.weights))
