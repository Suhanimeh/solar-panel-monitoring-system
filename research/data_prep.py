"""Data loading, quality checks, feature engineering and chronological splitting.

Each function supports one step of the research question: "can operating
measurements reveal unusual operating conditions?"  Good detection starts
with trustworthy data, so loading validates the schema and the quality report
makes data problems visible instead of silently absorbing them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "timestamp_utc", "voltage_v", "current_a", "power_w",
    "temp_c", "humidity_pct", "light_pct",
]
NUMERIC_COLUMNS = REQUIRED_COLUMNS[1:]

# Inputs available to every learned detector.
RAW_FEATURES = ["voltage_v", "current_a", "power_w", "temp_c", "humidity_pct", "light_pct"]
# Raw inputs + two engineered features (see add_features).
ENGINEERED_FEATURES = RAW_FEATURES + ["current_per_light", "current_roll_std"]

LABEL_COLUMNS = ["is_anomaly", "fault_type", "event_id"]

ROLL_WINDOW = 5           # samples (= 10 minutes at the 2-minute logging interval)
MIN_LIGHT_FOR_RATIO = 5.0  # avoids dividing by ~0 at dawn/dusk
SENTINEL = -999.0          # value the firmware writes when the DHT22 fails


def load_readings(path) -> pd.DataFrame:
    """Load a CSV produced by data_logger/logger.py (or research/simulate.py).

    Raises FileNotFoundError / ValueError with a specific message when the file
    is missing, lacks required columns, or has unparseable timestamps.
    Non-numeric measurement values become NaN (and are then counted by
    quality_report) rather than crashing the load.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    if df["timestamp_utc"].isna().any():
        raise ValueError(f"{path.name} contains unparseable timestamps")
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("timestamp_utc").reset_index(drop=True)


def quality_report(df: pd.DataFrame, ranges: Dict[str, Tuple[float, float]]) -> dict:
    """Summarise data quality: missing values, duplicates, sentinels, ranges,
    sampling regularity, and internal consistency (power ~= voltage x current)."""
    ts = df["timestamp_utc"]
    gaps = ts.diff().dt.total_seconds().dropna()
    median_gap = float(gaps.median()) if len(gaps) else float("nan")
    out_of_range = {}
    for col, (lo, hi) in ranges.items():
        if col in df.columns:
            out_of_range[col] = int(((df[col] < lo) | (df[col] > hi)).sum())
    cols = [c for c in NUMERIC_COLUMNS if c in df.columns]
    power_gap = (df["power_w"] - df["voltage_v"] * df["current_a"]).abs()
    return {
        "n_rows": int(len(df)),
        "time_span": [str(ts.min()), str(ts.max())],
        "missing_by_column": {c: int(df[c].isna().sum()) for c in cols},
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_timestamps": int(ts.duplicated().sum()),
        "sentinel_values": {c: int((df[c] == SENTINEL).sum()) for c in cols},
        "out_of_range": out_of_range,
        "median_interval_s": median_gap,
        "n_gaps_over_2x_median": int((gaps > 2 * median_gap).sum()) if len(gaps) else 0,
        "max_abs_power_vs_VxI_w": round(float(power_gap.max()), 4),
        "status_counts": df["status"].value_counts().to_dict() if "status" in df.columns else None,
    }


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features. Both are *causal*: a row's value uses only that
    row and earlier rows of the same day, so no future information leaks in.

    current_per_light : current / max(light, 5). Normalises current by available
        sunlight so a "low" current at low light is not mistaken for a problem.
    current_roll_std  : std of the last 5 current readings (same day). A frozen
        sensor has ~0 variation even though real panels fluctuate.
    """
    out = df.copy()
    out["current_per_light"] = out["current_a"] / np.maximum(out["light_pct"], MIN_LIGHT_FOR_RATIO)
    day = out["timestamp_utc"].dt.date
    out["current_roll_std"] = (
        out.groupby(day)["current_a"]
        .transform(lambda s: s.rolling(ROLL_WINDOW, min_periods=ROLL_WINDOW).std())
    )
    return out


def split_by_day(df: pd.DataFrame, n_train_days: int, n_val_days: int
                 ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split on whole calendar days: train < validation < test.

    Whole days (not random rows) are used because consecutive samples are highly
    autocorrelated; a random split would put near-duplicates of test rows in the
    training set and give over-optimistic results.
    """
    if n_train_days < 1 or n_val_days < 1:
        raise ValueError("n_train_days and n_val_days must be >= 1")
    days = sorted(df["timestamp_utc"].dt.date.unique())
    if len(days) <= n_train_days + n_val_days:
        raise ValueError(
            f"Need more than {n_train_days + n_val_days} days to leave a test set; got {len(days)}"
        )
    day = df["timestamp_utc"].dt.date
    train_days = set(days[:n_train_days])
    val_days = set(days[n_train_days:n_train_days + n_val_days])
    train = df[day.isin(train_days)].reset_index(drop=True)
    val = df[day.isin(val_days)].reset_index(drop=True)
    test = df[~day.isin(train_days | val_days)].reset_index(drop=True)
    return train, val, test


def correlation_table(df: pd.DataFrame, cols: Iterable[str] = tuple(RAW_FEATURES)) -> pd.DataFrame:
    return df[list(cols)].corr().round(3)
