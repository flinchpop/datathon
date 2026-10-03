"""Shared building blocks for the Track 2 fraud-detection pipeline.

Everything that both `train.py` and the analysis notebook need lives here:
data loading, feature engineering for the benchmark models, the model zoo,
repeated stratified cross-validation, threshold selection, drift diagnostics
and the plotting style.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = ROOT / "reports" / "figures"

TARGET = "fraud"
ID = "id"
CAT = ["merchant_category", "country", "transaction_channel"]
NUM = [
    "transaction_amount",
    "transaction_hour",
    "transactions_last_24h",
    "spend_last_24h",
    "account_age",
    "new_device",
    "transactions_last_1h",
]
RAW_FEATURES = NUM + CAT

SEED = 42
N_SPLITS = 5
N_REPEATS = 3

# The rubric scores F1 *and* Recall. The harmonic mean of F1 and Recall is
# algebraically identical to F-beta with beta = sqrt(3):
#   HM(F1, R) = 4PR / (3P + R) = F_{sqrt 3}
# so we pick the decision threshold that maximises F_{sqrt 3} on out-of-fold
# predictions. This weights recall ~1.7x precision, in line with the fraud
# cost asymmetry (a missed fraud costs more than a false alarm).
DEFAULT_BETA = float(np.sqrt(3))


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_data(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(data_dir / "Track_2_Training_Dataset.csv")
    test = pd.read_csv(data_dir / "Track_2_Testing_Dataset.csv")
    return train, test


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Domain features used by the linear / spline benchmark models.

    Tree models and the EBM learn these shapes from the raw columns, so they are
    only needed where the model is linear in its inputs.
    """
    d = df.copy()
    d["log_amount"] = np.log1p(d["transaction_amount"])
    d["log_spend24"] = np.log1p(d["spend_last_24h"])
    d["log_age"] = np.log1p(d["account_age"])
    # How unusual is this amount versus the customer's recent average ticket?
    avg_prior = d["spend_last_24h"] / d["transactions_last_24h"]
    d["amt_to_avg"] = np.log1p(d["transaction_amount"] / (avg_prior + 1))
    # Hour is cyclical: 23:00 is next to 00:00.
    d["hour_sin"] = np.sin(2 * np.pi * d["transaction_hour"] / 24)
    d["hour_cos"] = np.cos(2 * np.pi * d["transaction_hour"] / 24)
    d["is_night"] = d["transaction_hour"].isin([23, 0, 1, 2, 3, 4]).astype(int)
    young = (d["account_age"] <= 180).astype(float).where(d["account_age"].notna())
    d["newdev_x_young"] = d["new_device"] * young
    return d


SPLINE_FEATURES = [
    "log_amount", "log_age", "transactions_last_24h", "transactions_last_1h",
    "log_spend24", "amt_to_avg", "hour_sin", "hour_cos",
]
FLAG_FEATURES = ["new_device", "is_night", "newdev_x_young"]
LINEAR_RAW_NUM = NUM


# --------------------------------------------------------------------------- #
# Model zoo
# --------------------------------------------------------------------------- #
def _ohe():
    return make_pipeline(
        SimpleImputer(strategy="constant", fill_value="NA"),
        OneHotEncoder(handle_unknown="ignore"),
    )


def make_logreg(class_weight=None):
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), LINEAR_RAW_NUM),
        ("cat", _ohe(), CAT),
    ])
    return make_pipeline(pre, LogisticRegression(C=1.0, class_weight=class_weight, max_iter=5000))


def make_spline_logreg():
    pre = ColumnTransformer([
        ("spl", make_pipeline(
            SimpleImputer(strategy="median"),
            SplineTransformer(n_knots=5, degree=3, knots="quantile", extrapolation="constant"),
            StandardScaler(),
        ), SPLINE_FEATURES),
        ("flag", SimpleImputer(strategy="most_frequent"), FLAG_FEATURES),
        ("cat", _ohe(), CAT),
    ])
    return make_pipeline(pre, LogisticRegression(C=0.1, max_iter=5000))


def make_random_forest():
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median", add_indicator=True), NUM),
        ("cat", _ohe(), CAT),
    ])
    rf = RandomForestClassifier(
        n_estimators=500, min_samples_leaf=3, max_features="sqrt",
        class_weight="balanced_subsample", n_jobs=-1, random_state=SEED,
    )
    return make_pipeline(pre, rf)


def make_lightgbm(scale_pos_weight=1.0):
    import lightgbm as lgb
    return lgb.LGBMClassifier(
        n_estimators=600, learning_rate=0.02, max_depth=2, num_leaves=4,
        min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        reg_lambda=5, scale_pos_weight=scale_pos_weight, verbose=-1, random_state=SEED,
    )


def make_xgboost():
    import xgboost as xgb
    return xgb.XGBClassifier(
        n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.8,
        colsample_bytree=0.7, min_child_weight=3, reg_lambda=5,
        enable_categorical=True, tree_method="hist", n_jobs=-1, random_state=SEED,
    )


class CatBoostWrapper:
    """CatBoost needs string categoricals with no NaN; wrap that up."""

    def __init__(self, **params):
        import catboost as cb
        self.model = cb.CatBoostClassifier(
            iterations=800, learning_rate=0.03, depth=5, random_seed=SEED,
            thread_count=-1, verbose=False, **params,
        )

    @staticmethod
    def _prep(X):
        X = X.copy()
        for c in CAT:
            X[c] = X[c].astype(object).where(X[c].notna(), "NA").astype(str)
        return X

    def fit(self, X, y):
        self.model.fit(self._prep(X), y, cat_features=CAT)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(self._prep(X))


def make_ebm(**params):
    from interpret.glassbox import ExplainableBoostingClassifier
    return ExplainableBoostingClassifier(random_state=SEED, n_jobs=-1, **params)


@dataclass
class ModelSpec:
    name: str
    factory: callable
    features: str = "raw"          # "raw" or "engineered"
    native_categories: bool = False  # LightGBM / XGBoost want pandas category dtype
    note: str = ""


def model_zoo() -> list[ModelSpec]:
    return [
        ModelSpec("Logistic regression", make_logreg, note="Linear baseline"),
        ModelSpec("Logistic regression (class-balanced)", lambda: make_logreg("balanced"), note="Re-weighting for imbalance"),
        ModelSpec("Spline logistic GAM", make_spline_logreg, "engineered", note="Hand-built GAM + domain features"),
        ModelSpec("Random forest", make_random_forest, note="Bagged deep trees"),
        ModelSpec("LightGBM", make_lightgbm, native_categories=True, note="Gradient boosting"),
        ModelSpec("LightGBM (scale_pos_weight=5)", lambda: make_lightgbm(5.0), native_categories=True, note="Boosting + re-weighting"),
        ModelSpec("XGBoost", make_xgboost, native_categories=True, note="Gradient boosting"),
        ModelSpec("CatBoost", CatBoostWrapper, note="Ordered boosting, native categoricals"),
        ModelSpec("Explainable Boosting Machine", make_ebm, note="Glass-box GA2M (chosen)"),
    ]


FINAL_MODEL_NAME = "Explainable Boosting Machine"


def prepare_X(spec: ModelSpec, df: pd.DataFrame, categories: dict | None = None) -> pd.DataFrame:
    if spec.features == "engineered":
        d = add_features(df)
        X = d[SPLINE_FEATURES + FLAG_FEATURES + CAT].copy()
    else:
        X = df[RAW_FEATURES].copy()
    if spec.native_categories:
        for c in CAT:
            cats = None if categories is None else categories[c]
            X[c] = pd.Categorical(X[c].astype(object), categories=cats)
    return X


def category_levels(df: pd.DataFrame) -> dict:
    return {c: sorted(df[c].dropna().unique().tolist()) for c in CAT}


# --------------------------------------------------------------------------- #
# Metrics & thresholds
# --------------------------------------------------------------------------- #
def fbeta(precision, recall, beta):
    b2 = beta ** 2
    return (1 + b2) * precision * recall / np.maximum(b2 * precision + recall, 1e-12)


def metrics_at_threshold(y, p, t, beta=DEFAULT_BETA) -> dict:
    pred = p >= t
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "threshold": t,
        "precision": precision,
        "recall": recall,
        "f1": fbeta(precision, recall, 1.0),
        "f2": fbeta(precision, recall, 2.0),
        "f_beta": fbeta(precision, recall, beta),
        "flag_rate": pred.mean(),
        "tp": tp, "fp": fp, "fn": fn,
    }


THRESHOLD_GRID = np.round(np.arange(0.005, 0.996, 0.005), 3)


def threshold_table(y, oof_repeats: np.ndarray, beta=DEFAULT_BETA, grid=THRESHOLD_GRID) -> pd.DataFrame:
    """Metrics at each threshold, averaged across CV repeats (smooths the curve)."""
    rows = []
    for t in grid:
        per_rep = [metrics_at_threshold(y, oof_repeats[r], t, beta) for r in range(len(oof_repeats))]
        rows.append(pd.DataFrame(per_rep).mean(numeric_only=True))
    return pd.DataFrame(rows).reset_index(drop=True)


def choose_threshold(y, oof_repeats, beta=DEFAULT_BETA) -> tuple[float, pd.DataFrame]:
    table = threshold_table(y, oof_repeats, beta)
    best = table.loc[table["f_beta"].idxmax()]
    return float(best["threshold"]), table


def plug_in_metrics(p_unlabelled: np.ndarray, t: float, beta=DEFAULT_BETA) -> dict:
    """Expected precision / recall / F on an unlabelled set, assuming calibration.

    With calibrated probabilities p_i, E[TP] = sum p_i over flagged rows,
    E[FP] = sum (1 - p_i) over flagged rows and E[FN] = sum p_i over the rest.
    """
    flag = p_unlabelled >= t
    etp = p_unlabelled[flag].sum()
    efp = (1 - p_unlabelled[flag]).sum()
    efn = p_unlabelled[~flag].sum()
    precision = etp / max(etp + efp, 1e-12)
    recall = etp / max(etp + efn, 1e-12)
    return {
        "threshold": t,
        "expected_prevalence": p_unlabelled.mean(),
        "flag_rate": flag.mean(),
        "expected_precision": precision,
        "expected_recall": recall,
        "expected_f1": fbeta(precision, recall, 1.0),
        "expected_f_beta": fbeta(precision, recall, beta),
    }


# --------------------------------------------------------------------------- #
# Cross-validation
# --------------------------------------------------------------------------- #
@dataclass
class CVResult:
    name: str
    oof: np.ndarray                       # shape (n_repeats, n_samples)
    fold_ap: list = field(default_factory=list)
    fold_auc: list = field(default_factory=list)


def cv_splits(y, n_splits=N_SPLITS, n_repeats=N_REPEATS, seed=SEED):
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    return list(cv.split(np.zeros(len(y)), y))


def cross_validate(spec: ModelSpec, train: pd.DataFrame, splits, n_repeats=N_REPEATS) -> CVResult:
    y = train[TARGET].to_numpy()
    X = prepare_X(spec, train, category_levels(train))
    n_splits = len(splits) // n_repeats
    res = CVResult(spec.name, np.zeros((n_repeats, len(y))))
    for i, (tr_idx, va_idx) in enumerate(splits):
        model = spec.factory()
        model.fit(X.iloc[tr_idx], y[tr_idx])
        p = model.predict_proba(X.iloc[va_idx])[:, 1]
        res.oof[i // n_splits, va_idx] = p
        res.fold_ap.append(average_precision_score(y[va_idx], p))
        res.fold_auc.append(roc_auc_score(y[va_idx], p))
    return res


def summarise_cv(res: CVResult, y, beta=DEFAULT_BETA, weights=None) -> dict:
    t, table = choose_threshold(y, res.oof, beta)
    at_t = table.loc[table["threshold"] == t].iloc[0]
    best_f1 = []
    for r in range(len(res.oof)):
        pr, rc, _ = precision_recall_curve(y, res.oof[r])
        best_f1.append(np.max(fbeta(pr[:-1], rc[:-1], 1.0)))
    out = {
        "model": res.name,
        "pr_auc_mean": np.mean(res.fold_ap),
        "pr_auc_std": np.std(res.fold_ap),
        "roc_auc_mean": np.mean(res.fold_auc),
        "best_f1": np.mean(best_f1),
        "threshold": t,
        "precision_at_t": at_t["precision"],
        "recall_at_t": at_t["recall"],
        "f1_at_t": at_t["f1"],
        "brier": np.mean([brier_score_loss(y, res.oof[r]) for r in range(len(res.oof))]),
    }
    if weights is not None:
        out["pr_auc_test_like"] = np.mean(
            [average_precision_score(y, res.oof[r], sample_weight=weights) for r in range(len(res.oof))]
        )
    return out


# --------------------------------------------------------------------------- #
# Drift diagnostics
# --------------------------------------------------------------------------- #
def adversarial_validation(train: pd.DataFrame, test: pd.DataFrame, seed=SEED):
    """Train a classifier to tell train rows from test rows.

    Returns (AUC, per-feature importance, density-ratio weights for train rows).
    AUC ~0.5 means no drift; the weights w(x) = p(test|x)/p(train|x) re-weight
    training rows to look like the test set (importance-weighted validation).
    """
    import lightgbm as lgb
    from sklearn.model_selection import cross_val_predict

    both = pd.concat([train[RAW_FEATURES], test[RAW_FEATURES]], ignore_index=True)
    for c in CAT:
        both[c] = pd.Categorical(both[c].astype(object))
    is_test = np.r_[np.zeros(len(train)), np.ones(len(test))]
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15,
                             min_child_samples=50, verbose=-1, random_state=seed)
    p = cross_val_predict(clf, both, is_test, method="predict_proba",
                          cv=StratifiedKFold(5, shuffle=True, random_state=seed))[:, 1]
    auc = roc_auc_score(is_test, p)
    clf.fit(both, is_test)
    imp = pd.Series(clf.booster_.feature_importance("gain"), index=RAW_FEATURES).sort_values(ascending=False)
    imp = imp / imp.sum()
    p_tr = p[: len(train)]
    w = p_tr / (1 - p_tr) * (len(train) / len(test))
    w = np.clip(w, 0.05, np.quantile(w, 0.99))
    w = w / w.mean()
    return auc, imp, w


# --------------------------------------------------------------------------- #
# Plot style (reference palette from the dataviz guidance)
# --------------------------------------------------------------------------- #
PALETTE = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "violet": "#4a3aa7",
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "deemph": "#c3c2b7",
}


def set_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": PALETTE["surface"],
        "axes.facecolor": PALETTE["surface"],
        "savefig.facecolor": PALETTE["surface"],
        "axes.edgecolor": PALETTE["axis"],
        "axes.labelcolor": PALETTE["ink2"],
        "axes.titlecolor": PALETTE["ink"],
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.8,
        "xtick.color": PALETTE["muted"],
        "ytick.color": PALETTE["muted"],
        "xtick.labelcolor": PALETTE["ink2"],
        "ytick.labelcolor": PALETTE["ink2"],
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
        "font.family": "sans-serif",
        "font.size": 9,
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })
