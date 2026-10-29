"""
System performance analysis for the IoT Solar Panel Monitoring System.

Loads one or more CSV logs produced by data_logger/logger.py, computes daily
energy yield, peak power, uptime/data-quality metrics, and produces plots
summarizing panel performance over the logging period.

Usage:
    python analyze_performance.py --input ../data/sample/solar_readings_sample.csv --outdir ./output
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load_readings(paths):
    frames = [pd.read_csv(p, parse_dates=["timestamp_utc"]) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    return df


def compute_summary(df: pd.DataFrame) -> dict:
    duration_hours = (
        (df["timestamp_utc"].max() - df["timestamp_utc"].min()).total_seconds() / 3600
    )
    # Trapezoidal integration of power over time gives energy in Wh
    df = df.copy()
    df["t_hours"] = (
        df["timestamp_utc"] - df["timestamp_utc"].iloc[0]
    ).dt.total_seconds() / 3600
    energy_wh = pd.Series(
        data=(df["power_w"].values[1:] + df["power_w"].values[:-1]) / 2
        * (df["t_hours"].values[1:] - df["t_hours"].values[:-1])
    ).sum()

    return {
        "num_readings": len(df),
        "duration_hours": round(duration_hours, 2),
        "peak_power_w": round(df["power_w"].max(), 2),
        "avg_power_w": round(df["power_w"].mean(), 2),
        "estimated_energy_wh": round(energy_wh, 2),
        "avg_voltage_v": round(df["voltage_v"].mean(), 2),
        "avg_current_a": round(df["current_a"].mean(), 2),
        "avg_temp_c": round(df["temp_c"].mean(), 2),
        "avg_humidity_pct": round(df["humidity_pct"].mean(), 2),
        "avg_light_pct": round(df["light_pct"].mean(), 2),
    }


def plot_performance(df: pd.DataFrame, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(df["timestamp_utc"], df["power_w"], color="#d97706")
    axes[0].set_ylabel("Power (W)")
    axes[0].set_title("Panel Power Output Over Time")

    axes[1].plot(df["timestamp_utc"], df["voltage_v"], label="Voltage (V)", color="#2563eb")
    axes[1].plot(df["timestamp_utc"], df["current_a"], label="Current (A)", color="#16a34a")
    axes[1].set_ylabel("Voltage / Current")
    axes[1].legend(loc="upper right")
    axes[1].set_title("Voltage and Current Over Time")

    axes[2].plot(df["timestamp_utc"], df["temp_c"], label="Temp (C)", color="#dc2626")
    axes[2].plot(df["timestamp_utc"], df["light_pct"], label="Light (%)", color="#7c3aed")
    axes[2].set_ylabel("Temp / Light")
    axes[2].set_xlabel("Time")
    axes[2].legend(loc="upper right")
    axes[2].set_title("Ambient Temperature and Light Intensity Over Time")

    fig.tight_layout()
    fig_path = outdir / "performance_summary.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return fig_path


def main():
    parser = argparse.ArgumentParser(description="Analyze solar panel monitoring logs")
    parser.add_argument("--input", nargs="+", required=True, help="One or more CSV log files")
    parser.add_argument("--outdir", type=Path, default=Path("./output"), help="Directory for plots/reports")
    args = parser.parse_args()

    df = load_readings(args.input)
    summary = compute_summary(df)

    print("\n--- System Performance Summary ---")
    for key, value in summary.items():
        print(f"{key:22s}: {value}")

    fig_path = plot_performance(df, args.outdir)
    print(f"\nSaved performance chart to: {fig_path}")


if __name__ == "__main__":
    main()
