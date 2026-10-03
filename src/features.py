"""Data loading and feature engineering shared by every model."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "Track_1_Training_Dataset.csv"
TEST_PATH = ROOT / "data" / "Track_1_Testing_Dataset.csv"

TARGET = "energy_usage"
# Measured covariates that can be missing (calendar/building columns never are).
SENSORS = ["temperature", "humidity", "occupancy", "previous_usage"]
DOW = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
       "Friday": 4, "Saturday": 5, "Sunday": 6}
# Above this outdoor temperature the chillers start working disproportionately
# hard; chosen by cross-validation (see REPORT.md, section 4).
COOLING_BASE = 30.0


def load(path: Path) -> pd.DataFrame:
    return prepare(pd.read_csv(path))


def load_train() -> pd.DataFrame:
    return load(TRAIN_PATH)


def load_test() -> pd.DataFrame:
    return load(TEST_PATH)


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar and physically-motivated features. Never fills NaNs."""
    df = df.copy()
    df["dow"] = df["day_of_week"].map(DOW).astype(int)
    df["weekend"] = (df["dow"] >= 5).astype(int)
    # Cooling degrees: zero until the outdoor temperature passes the base,
    # then grows linearly (hinge), the standard degree-day load model.
    df["cooling_deg"] = np.clip(df["temperature"] - COOLING_BASE, 0, None)
    return df


def missing_pattern(df: pd.DataFrame) -> pd.Series:
    """Tuple of missing sensor columns for each row, e.g. ('occupancy',)."""
    miss = df[SENSORS].isna().to_numpy()
    return pd.Series([tuple(c for c, m in zip(SENSORS, row) if m) for row in miss],
                     index=df.index)
