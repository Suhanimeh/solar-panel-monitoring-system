"""Synthetic multi-day dataset with labelled, injected faults.

*** THIS DATA IS SIMULATED. It is NOT hardware data. ***

Why it exists: the repository's sample file is one fault-free simulated day with
no labels, so no detector can be evaluated on it. To measure how well a detector
finds unusual behaviour we must know where the unusual behaviour is. Injecting
known deviations into simulated data is the standard way to get that ground
truth, at the cost that results describe *my injected patterns*, not real
panel failures.

Calibration: the relationships between light and voltage / current /
temperature / humidity were obtained by least-squares on
data/sample/solar_readings_sample.csv (coefficients in COEF). Day-to-day
variation (cloud events, overcast days, ambient offsets) is added so that
"normal" is not one identical curve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SAMPLES_PER_DAY = 360     # 06:00-18:00 at 2-minute intervals, as in the sample file
INTERVAL_S = 120
PEAK_LIGHT = 96.0

# (intercept, slope on light_pct, residual std) fitted on the repo sample file.
COEF = {
    "voltage_v": (11.034, 0.0840, 0.201),
    "current_a": (0.012, 0.0366, 0.093),
    "temp_c": (19.965, 0.1275, 0.573),
    "humidity_pct": (60.040, -0.2123, 1.723),
}

FAULT_TYPES = ["current_drop", "stuck_current", "voltage_spike", "range_violation"]


def generate_clean_days(n_days: int, seed: int, start_date: str = "2026-06-01") -> pd.DataFrame:
    """Generate fault-free days. Columns mirror the logger's output plus
    data_source='SIMULATED' and empty label columns."""
    if n_days < 1:
        raise ValueError("n_days must be >= 1")
    rng = np.random.default_rng(seed)
    idx = np.arange(SAMPLES_PER_DAY)
    hours = idx * INTERVAL_S / 3600.0
    clear_sky = PEAK_LIGHT * np.sin(np.pi * hours / 12.0)
    frames = []
    for d in range(n_days):
        scale = rng.uniform(0.85, 1.0)
        if rng.random() < 0.2:                       # overcast day
            scale *= rng.uniform(0.35, 0.6)
        factor = np.ones(SAMPLES_PER_DAY)
        for _ in range(rng.poisson(2)):              # passing clouds
            width = int(rng.integers(5, 30))
            start = int(rng.integers(0, SAMPLES_PER_DAY - width))
            depth = rng.uniform(0.3, 0.8)
            factor[start:start + width] *= 1 - depth * np.hanning(width + 2)[1:-1]
        light = np.clip(clear_sky * scale * factor + rng.normal(0, 1.2, SAMPLES_PER_DAY), 0, 100)

        def lin(name, offset=0.0):
            a, b, sd = COEF[name]
            return a + b * light + offset + rng.normal(0, sd, SAMPLES_PER_DAY)

        voltage = np.round(lin("voltage_v"), 2)
        current = np.round(np.clip(lin("current_a"), 0, None), 2)
        temp = np.round(lin("temp_c", rng.normal(0, 1.5)), 1)
        humidity = np.round(np.clip(lin("humidity_pct", rng.normal(0, 3.0)), 0, 100), 1)
        ts = (pd.Timestamp(start_date, tz="UTC") + pd.Timedelta(days=d, hours=6)
              + pd.to_timedelta(idx * INTERVAL_S, unit="s"))
        frames.append(pd.DataFrame({
            "timestamp_utc": ts,
            "timestamp_ms": (d * 86400 + idx * INTERVAL_S) * 1000,
            "voltage_v": voltage,
            "current_a": current,
            "power_w": np.round(voltage * current, 2),
            "temp_c": temp,
            "humidity_pct": humidity,
            "light_pct": np.round(light, 1),
            "status": "OK",
        }))
    df = pd.concat(frames, ignore_index=True)
    df["data_source"] = "SIMULATED"
    df["is_anomaly"] = 0
    df["fault_type"] = "none"
    df["event_id"] = 0
    return df


def inject_faults(df: pd.DataFrame, fault_days: list, seed: int) -> pd.DataFrame:
    """Inject one episode of each fault type into every day listed in
    `fault_days` (0-based day indices). Episodes sit in separate slots of the
    day, so they never overlap. Returns a labelled copy.

    current_drop     : current x 0.55-0.80 for 20-48 min (partial shading / soiling-like).
    stuck_current    : current frozen at the last good value for 16-32 min (frozen sensor).
    voltage_spike    : voltage +3-5 V for 1-3 samples, kept inside the valid range.
    range_violation  : voltage 25-30 V for 1-2 samples. The existing logger would
                       *reject* such rows; they are included so the range-check
                       baseline gets a fair chance.
    """
    rng = np.random.default_rng(seed + 10_000)
    out = df.copy().reset_index(drop=True)
    day_idx = (out["timestamp_utc"].dt.normalize() - out["timestamp_utc"].dt.normalize().min()).dt.days
    next_event = 1
    for d in fault_days:
        rows = np.flatnonzero(day_idx.to_numpy() == d)
        if len(rows) != SAMPLES_PER_DAY:
            raise ValueError(f"Day {d} does not have {SAMPLES_PER_DAY} rows")
        slots = np.array_split(np.arange(20, 340), len(FAULT_TYPES))
        order = rng.permutation(len(FAULT_TYPES))
        for slot, t in zip(slots, order):
            ftype = FAULT_TYPES[t]
            dur = {"current_drop": rng.integers(10, 25), "stuck_current": rng.integers(8, 17),
                   "voltage_spike": rng.integers(1, 4), "range_violation": rng.integers(1, 3)}[ftype]
            s = int(rng.integers(slot[0] + 1, slot[-1] - dur))
            sel = rows[s:s + dur]
            if ftype == "current_drop":
                out.loc[sel, "current_a"] = np.round(out.loc[sel, "current_a"] * rng.uniform(0.55, 0.80), 2)
            elif ftype == "stuck_current":
                out.loc[sel, "current_a"] = out.loc[rows[s - 1], "current_a"]
            elif ftype == "voltage_spike":
                out.loc[sel, "voltage_v"] = np.round(np.minimum(out.loc[sel, "voltage_v"] + rng.uniform(3, 5), 23.5), 2)
            else:
                out.loc[sel, "voltage_v"] = np.round(rng.uniform(25, 30, len(sel)), 2)
            out.loc[sel, "power_w"] = np.round(out.loc[sel, "voltage_v"] * out.loc[sel, "current_a"], 2)
            out.loc[sel, ["is_anomaly", "fault_type", "event_id"]] = [1, ftype, next_event]
            next_event += 1
    return out


def make_dataset(seed: int, n_days: int = 14, n_train_days: int = 8, n_val_days: int = 2) -> pd.DataFrame:
    """Clean training/validation days; faults only in the remaining (test) days."""
    df = generate_clean_days(n_days, seed)
    return inject_faults(df, list(range(n_train_days + n_val_days, n_days)), seed)
