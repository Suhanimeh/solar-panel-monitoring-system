# Research Methodology and Interview Notes

## A. Repository audit (what was actually there)

Audited from the uploaded repository archive (not the live GitHub page).

| Area | Finding |
|---|---|
| Firmware | Reads voltage (A0), current (ACS712, A1), light (A2), DHT22 temp/humidity; computes power; classifies each row OK / SENSOR_FAULT / OUT_OF_RANGE. Calibration constants are placeholders. |
| Logger | Schema + range validation; rejects with reasons. Good basis for the baseline. `timestamp_utc` is host receive time, not sampling time. |
| Analysis | Descriptive only (peak/avg power, trapezoidal energy, plots). **No ML, no anomaly detection, no labels.** |
| Data | One simulated day, 360 rows, 2-min interval, all `OK`, no faults. No real hardware data is included. |
| Tests | 15 logger tests, which pass (run via a minimal stand-in for pytest). No tests for analysis. |
| Documentation mismatches to verify | The original README references `docs/architecture.md`, `docs/testing_procedures.md` and `.github/workflows/tests.yml`; none were in the archive. Check whether they exist in the GitHub repo, and remove the links if not. |
| Firmware comment mismatch | The `.ino` header mentions a ZMPT101B for voltage (an AC sensor), while the README and code use a resistive voltage divider. Worth correcting. |
| Data realism | Voltage, current, temperature and light correlate at 0.99–1.00, impossible on real hardware; the sample is a clean simulation. |

Original code was not modified.

## B. Why this research question

*Can a model of expected behaviour given conditions find unusual operating conditions that fixed range checks miss?*

- It follows from a genuine gap: the project already does range checking; the open question is what lies beyond it.
- It reflects a core predictive-maintenance idea: judge a measurement against what is expected under the current operating conditions.
- It is answerable with the available data. Fault *classification* needs labelled real faults; *failure prediction* needs run-to-failure histories. Neither exists, so neither is attempted.

Solar monitoring is **not** equivalent to industrial machinery maintenance. What transfers is the workflow: data validation, baseline, model comparison, honest evaluation.

## C. Why simulated data, and its consequences

The sample has no faults and no labels, so no detector can be scored against it. I generate multi-day data calibrated to the sample and **inject known faults** so ground truth exists. Consequences, to state upfront in any discussion:
- Results show detectability of *my injected patterns*, not real failures.
- A model similar in form to the data-generating process has an advantage by construction.
- Absolute numbers are not transferable; the *relative* behaviour and failure modes (e.g., Isolation Forest not extrapolating) are the informative parts.

## D. Data preparation decisions

| Decision | Reason |
|---|---|
| Validate schema on load | Fail early with a clear message |
| Quality report | Missing values, duplicates, sentinels, ranges, gaps, power ≈ V×I consistency |
| Whole-day chronological split | Adjacent samples are near-duplicates; random splits leak |
| Train and validation days are clean | Models learn "normal"; thresholds need clean data |
| Features are causal | Rolling std uses only past values (tested) |
| No scaling for Isolation Forest | Tree splits are scale-invariant |
| Train-median imputation | First samples of each day lack a 5-sample window; fill value comes from training only |

## E. Method choices and training/testing protocol

1. **Range thresholds**: the existing logic, so the comparison answers "what does ML add over what we have?"
2. **Expected-behaviour residual**: interpretable ("predicted 3.4 A, measured 2.3 A"), cheap, domain-informed.
3. **Isolation Forest**: standard unsupervised detector needing no labels. Run with raw and engineered features to test whether feature design helps.

Protocol: fit on clean train days → set threshold at the 99th percentile of validation scores (same rule for all methods) → evaluate once on test → repeat for 10 seeds, report mean ± sd.

Metrics: precision, recall, F1, confusion matrix (standard); event recall and false alarms/day (operational); ROC-AUC and PR-AUC (threshold-free; PR-AUC preferred for rare events). Accuracy is deliberately **not** reported: a detector that flags nothing would score ~91 % accuracy here.

## F. Results in one paragraph

Range checks: perfect precision, recall 0.05. Residual model: F1 0.79, event recall 0.87, ~3.3 false alarms/day. Isolation Forest: F1 0.03–0.06. See README Section 9 for the full table.

## G. Limitations

Simulated, injected faults; elevated prevalence (~9 %); daylight only; no seasonality; one fault episode of each type per test day; residual model advantage partly built-in; 1 % false-alarm budget still means ~3 alerts/day; a flag is not a diagnosis.

## H. Likely interview questions and defensible answers

**Why not just report accuracy?** About 91 % of test rows are normal, so "flag nothing" scores ~91 %. Precision/recall and event recall describe the actual job.

**Why a chronological split?** Deployment predicts the future from the past, and neighbouring samples are almost identical, so random splitting inflates results.

**How can you evaluate without real faults?** I can't, for real performance. Injection gives ground truth for *detectability of known patterns*. I label everything simulated and do not generalise.

**Isn't the residual model's win circular?** Partly, yes: it models the same kind of relationship the simulator used. I treat the margin over Isolation Forest as indicative, not as proof, and the real test is real data.

**Why did Isolation Forest fail?** Evidence: its score for 30 V was no higher than for 19.3 V (probe on seed 42), consistent with splits being drawn only from the training range. Engineered features didn't fix stuck-sensor detection even though the feature was exactly 0. Hypothesis: the same non-extrapolation. I did not run a further ablation to prove it.

**How did you pick the threshold?** 99th percentile of scores on clean validation data, identical for all methods; no labels or test data involved (there is a test for this). `fpr_sensitivity.csv` shows the trade-off.

**What would you do with more time/data?** Real multi-week logs, panel-temperature and irradiance inputs, a rolling-variance rule for stuck sensors, persistence-based alerting, maintenance-confirmed labels.

**Is this predictive maintenance?** It is the anomaly-detection step that often precedes it. I do not predict failures.

## I. Honest reproducibility statement

`python -m research.run_experiment` regenerates every number and figure; two runs gave identical output files. Tests: 23 new tests pass; the 15 original tests pass (run through a minimal stand-in for pytest because pytest could not be installed offline; run `pytest tests/ -v` yourself to confirm).
