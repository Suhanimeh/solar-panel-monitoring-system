"""Run the full anomaly-detection benchmark and write results + figures.

From the repository root:
    python -m research.run_experiment                # defaults: 10 seeds
    python -m research.run_experiment --seeds 3 --outdir /tmp/results

Nothing here is hand-edited: every number in research/results/ is produced by
this script.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import sklearn

from . import plots
from .data_prep import (add_features, correlation_table, load_readings,
                        quality_report, split_by_day)
from .detectors import VALID_RANGES, build_detectors
from .evaluation import evaluate_detector
from .simulate import generate_clean_days, make_dataset

REPO = Path(__file__).resolve().parent.parent
METRIC_COLS = ["precision", "recall", "f1", "roc_auc", "pr_auc", "event_recall", "false_alarms_per_day"]
TYPE_COLS = ["recall_current_drop", "recall_stuck_current", "recall_voltage_spike", "recall_range_violation"]


def prepare(seed, args):
    df = make_dataset(seed, args.n_days, args.train_days, args.val_days)
    train, val, test = split_by_day(df, args.train_days, args.val_days)
    return [add_features(x) for x in (train, val, test)]


def run_seed(seed, args):
    train, val, test = prepare(seed, args)
    detectors = build_detectors(random_state=seed)
    rows = []
    for d in detectors:
        d.fit(train).calibrate(val, args.target_fpr)
        rows.append({"seed": seed, **evaluate_detector(d, test)})
    return rows, detectors, (train, val, test)


def fpr_sensitivity(seed, args):
    train, val, test = prepare(seed, args)
    rows = []
    for d in build_detectors(random_state=seed)[1:]:  # baseline has no tunable threshold
        d.fit(train)
        for fpr in [0.001, 0.005, 0.01, 0.02, 0.05]:
            d.calibrate(val, fpr)
            r = evaluate_detector(d, test)
            rows.append({"seed": seed, "method": d.name, "target_fpr": fpr, "precision": r["precision"],
                         "recall": r["recall"], "f1": r["f1"], "event_recall": r["event_recall"],
                         "false_alarms_per_day": r["false_alarms_per_day"]})
    return rows


def mean_sd(df, cols):
    g = df.groupby("method", sort=False)[cols]
    m, s = g.mean(), g.std()
    out = pd.DataFrame(index=m.index)
    for c in cols:
        out[c] = [("n/a" if np.isnan(a) else f"{a:.3f} ± {b:.3f}") for a, b in zip(m[c], s[c])]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=10, help="number of simulation seeds (default 10)")
    ap.add_argument("--base-seed", type=int, default=42)
    ap.add_argument("--n-days", type=int, default=14)
    ap.add_argument("--train-days", type=int, default=8)
    ap.add_argument("--val-days", type=int, default=2)
    ap.add_argument("--target-fpr", type=float, default=0.01,
                    help="false-alarm rate allowed on clean validation data (default 0.01)")
    ap.add_argument("--sample-csv", type=Path, default=REPO / "data/sample/solar_readings_sample.csv")
    ap.add_argument("--outdir", type=Path, default=REPO / "research/results")
    args = ap.parse_args()
    if args.seeds < 2:
        ap.error("--seeds must be >= 2 so a standard deviation can be computed")
    args.outdir.mkdir(parents=True, exist_ok=True)

    # 1. Data quality + EDA on the repository's own sample file
    if args.sample_csv.exists():
        sample = load_readings(args.sample_csv)
        sample_source = str(args.sample_csv)
    elif args.sample_csv == REPO / "data/sample/solar_readings_sample.csv":
        sample = generate_clean_days(1, seed=args.base_seed)
        sample_source = "generated in memory by research.simulate.generate_clean_days (SIMULATED)"
        print("Optional sample CSV not found; using one generated simulated day instead.")
    else:
        ap.error(f"Sample CSV not found: {args.sample_csv}")
    report = quality_report(sample, VALID_RANGES)
    (args.outdir / "data_quality_repo_sample.json").write_text(json.dumps(report, indent=2))
    print("Repo sample data quality:", json.dumps(report, indent=2))

    # 2. Benchmark over several simulation seeds
    seeds = [args.base_seed + i for i in range(args.seeds)]
    all_rows = []
    for s in seeds:
        rows, detectors, splits = run_seed(s, args)
        all_rows += rows
        if s == seeds[0]:
            main_detectors, (train, val, test) = detectors, splits
    res = pd.DataFrame(all_rows)
    res.to_csv(args.outdir / "per_seed_results.csv", index=False)

    table = mean_sd(res, METRIC_COLS)
    table.to_csv(args.outdir / "benchmark_summary.csv")
    (args.outdir / "benchmark_summary.md").write_text(table.to_markdown())
    per_type = res.groupby("method", sort=False)[TYPE_COLS].mean()
    per_type.columns = [c.replace("recall_", "") for c in per_type.columns]
    per_type.round(3).to_csv(args.outdir / "recall_by_fault_type.csv")
    sens = pd.DataFrame([r for s in seeds for r in fpr_sensitivity(s, args)])
    sens_mean = sens.drop(columns="seed").groupby(["method", "target_fpr"], sort=False).mean().round(3)
    sens_mean.to_csv(args.outdir / "fpr_sensitivity.csv")

    # 3. Sanity check: how often do the (seed-0) detectors flag the repo's clean sample day?
    sample_f = add_features(sample)
    sanity = {d.name: float(d.predict(sample_f).mean()) for d in main_detectors}
    (args.outdir / "sample_day_flag_rate.json").write_text(json.dumps(sanity, indent=2))

    # 4. Figures (first seed)
    plots.fig_eda(sample, train, correlation_table(train), args.outdir / "fig1_eda.png")
    plots.fig_timeline(test, main_detectors[1], 0, args.outdir / "fig2_timeline_residual_model.png")
    plots.fig_confusion(test, main_detectors, args.outdir / "fig3_confusion_matrices.png")
    plots.fig_pr_and_types(test, main_detectors, per_type.T, args.outdir / "fig4_pr_and_fault_types.png")

    meta = {
        "args": {k: str(v) for k, v in vars(args).items()},
        "seeds": seeds,
        "n_rows": {"train": len(train), "validation": len(val), "test": len(test)},
        "test_prevalence_seed0": float(test["is_anomaly"].mean()),
        "test_events_seed0": int(test["event_id"].max() - test.loc[test.event_id > 0, "event_id"].min() + 1),
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                     "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__},
        "sample_source": sample_source,
        "data_note": "All benchmark training, validation, and test data is SIMULATED with injected faults. No private hardware data is included in this repository.",
    }
    (args.outdir / "run_metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"\n=== Benchmark: mean ± sd over {len(seeds)} simulation seeds (SIMULATED data) ===")
    print(table.to_string())
    print("\nRecall by fault type:\n", per_type.round(3).to_string())
    print("\nFlag rate on repo's clean sample day:", sanity)
    print(f"\nResults written to {args.outdir}")


if __name__ == "__main__":
    main()
