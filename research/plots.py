"""Figures for the report. Matplotlib only; Agg backend so it runs headless."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from .evaluation import confusion_counts

FAULT_COLORS = {"current_drop": "#d97706", "stuck_current": "#7c3aed",
                "voltage_spike": "#dc2626", "range_violation": "#0f766e"}


def fig_eda(sample: pd.DataFrame, sim_train: pd.DataFrame, corr: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    hrs = lambda d: d["timestamp_utc"].dt.hour + d["timestamp_utc"].dt.minute / 60
    for day, g in sim_train.groupby(sim_train["timestamp_utc"].dt.date):
        ax[0, 0].plot(hrs(g), g["light_pct"], color="grey", alpha=0.35, lw=0.8)
    ax[0, 0].plot(hrs(sample), sample["light_pct"], color="#d97706", lw=1.5, label="Repo sample day (simulated)")
    ax[0, 0].plot([], [], color="grey", label="Simulated training days")
    ax[0, 0].set(title="Light intensity by hour of day", xlabel="Hour (UTC)", ylabel="Light (%)")
    ax[0, 0].legend()
    ax[0, 1].scatter(sim_train["light_pct"], sim_train["power_w"], s=4, alpha=0.3, color="grey", label="Simulated training")
    ax[0, 1].scatter(sample["light_pct"], sample["power_w"], s=6, color="#d97706", label="Repo sample")
    ax[0, 1].set(title="Power vs light (normal operation)", xlabel="Light (%)", ylabel="Power (W)")
    ax[0, 1].legend()
    im = ax[1, 0].imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax[1, 0].set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax[1, 0].set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax[1, 0].text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    ax[1, 0].set_title("Correlations (simulated training data)")
    fig.colorbar(im, ax=ax[1, 0], fraction=0.046)
    ax[1, 1].scatter(sim_train["light_pct"], sim_train["current_a"], s=4, alpha=0.3, color="grey")
    ax[1, 1].set(title="Current vs light (the relationship the residual model learns)",
                 xlabel="Light (%)", ylabel="Current (A)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_timeline(test: pd.DataFrame, det, day_index: int, path: Path):
    days = sorted(test["timestamp_utc"].dt.date.unique())
    g = test[test["timestamp_utc"].dt.date == days[day_index]].reset_index(drop=True)
    h = g["timestamp_utc"].dt.hour + g["timestamp_utc"].dt.minute / 60
    score, pred = det.score(g), det.predict(g)
    fig, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    for a, col in zip(ax[:2], ["current_a", "voltage_v"]):
        a.plot(h, g[col], color="#374151", lw=1)
        for t, c in FAULT_COLORS.items():
            m = (g["fault_type"] == t).to_numpy()
            a.scatter(h[m], g.loc[m, col], s=22, color=c, label=f"true: {t}", zorder=3)
        f = pred == 1
        a.scatter(h[f], g.loc[f, col], s=70, facecolors="none", edgecolors="black", label="flagged", zorder=4)
        a.set_ylabel(col)
    ax[0].legend(ncol=3, fontsize=8)
    ax[0].set_title(f"{det.name}: test day {day_index + 1} (SIMULATED data, injected faults)")
    ax[2].plot(h, score, color="#2563eb", lw=1)
    ax[2].axhline(det.threshold_, color="red", ls="--", label="alert threshold (set on clean validation data)")
    ax[2].set(ylabel="anomaly score", xlabel="Hour (UTC)"); ax[2].legend()
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_confusion(test: pd.DataFrame, detectors, path: Path):
    fig, axes = plt.subplots(1, len(detectors), figsize=(4 * len(detectors), 4))
    for a, d in zip(np.atleast_1d(axes), detectors):
        c = confusion_counts(test["is_anomaly"], d.predict(test))
        m = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
        a.imshow(m, cmap="Blues")
        for i in range(2):
            for j in range(2):
                a.text(j, i, m[i, j], ha="center", va="center", fontsize=14,
                       color="white" if m[i, j] > m.max() / 2 else "black")
        a.set_xticks([0, 1], ["normal", "flagged"]); a.set_yticks([0, 1], ["normal", "anomaly"])
        a.set_xlabel("predicted"); a.set_ylabel("actual")
        a.set_title(d.name, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def fig_pr_and_types(test: pd.DataFrame, detectors, per_type: pd.DataFrame, path: Path):
    """Left: precision-recall curves for score-based methods. Right: recall by fault type."""
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    y = test["is_anomaly"].to_numpy()
    for d in detectors:
        s = d.score(test)
        if len(np.unique(s)) > 2:
            p, r, _ = precision_recall_curve(y, s)
            ax[0].plot(r, p, label=d.name)
    ax[0].axhline(y.mean(), color="grey", ls=":", label=f"no-skill (prevalence {y.mean():.2f})")
    ax[0].set(title="Precision-recall (seed 42, test set)", xlabel="Recall", ylabel="Precision", ylim=(0, 1.02))
    ax[0].legend(fontsize=8)
    per_type.plot.bar(ax=ax[1], rot=20)
    ax[1].set(title="Recall by fault type (mean over seeds)", ylabel="Recall", ylim=(0, 1.05))
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
