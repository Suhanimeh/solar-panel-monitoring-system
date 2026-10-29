# Project Plan and Presentation Outline

Effort figures are planning estimates for a student working part-time, not a record of time spent.

| # | Milestone | Deliverable | Est. effort |
|---|---|---|---|
| 1 | Repository audit and research question | Audit table (methodology.md §A); question and terminology | 4–6 h |
| 2 | Dataset preparation and EDA | `data_prep.py`, `simulate.py`, quality report, fig1 | 8–10 h |
| 3 | Baseline implementation | Range-threshold detector, evaluation functions, tests | 4–6 h |
| 4 | Training, testing, benchmarking | Detectors, `run_experiment.py`, results tables | 8–10 h |
| 5 | Interpretation and error analysis | Per-fault recall, Isolation Forest probe, sensitivity analysis | 6–8 h |
| 6 | Documentation and presentation | README, report, slides | 8–12 h |

**Total ≈ 38–52 h (about 3–4 weeks at 12–15 h/week).** Dependencies: 2 → 3 → 4 → 5 → 6. Main risk: spending time on extra models rather than on interpretation.

## Presentation outline (8 slides, ~8 minutes)

1. **Problem.** Range checks can't see context-dependent problems (1.1 A is fine at dawn, suspicious at noon).
2. **Scope and terminology.** Monitoring vs anomaly detection vs classification vs prediction; what I do and don't do.
3. **Data.** Existing system; why the sample can't be evaluated; simulated data with injected labelled faults (state clearly).
4. **Approach.** Chronological split; three methods; threshold rule; metrics.
5. **Results.** Benchmark table; confusion matrices.
6. **Error analysis.** Recall by fault type; Isolation Forest cannot extrapolate; stuck sensors detected late (fig2).
7. **Limitations.** Simulated; partly circular; false-alarm burden; flags ≠ faults.
8. **Next steps.** Real logs, physics-based model, rolling-variance rule, persistence alerts, public dataset.
