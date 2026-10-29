# Mapping to the Research Assistant Job Description

| Job requirement | Evidence from this project | Strength |
|---|---|---|
| AI methodology selection | Baseline vs domain-informed model vs generic unsupervised method; reasons in methodology.md §E; task chosen to match what data supports | **Direct** |
| Data preparation | Schema validation, quality report, causal features, leakage-safe split, train-only imputation (`data_prep.py`, tests) | **Direct** |
| Model training and testing | Reproducible pipeline over 10 seeds, identical outputs on re-run (`run_experiment.py`) | **Direct** |
| Model benchmarking | Four methods compared with precision/recall/F1, PR-AUC, event recall, false alarms/day, per-fault recall | **Direct** (on simulated data) |
| Research analysis | Isolation Forest saturation probe, stuck-sensor lag, threshold sensitivity, circularity discussion | **Direct** |
| Documentation and reporting | README, methodology, report outline | **Direct** (report outline only; full report still to be written) |
| Project planning | Milestones with deliverables and effort estimates | **Partial**: a plan for a solo project, with no real timeline or stakeholders |
| Communication | Presentation outline and interview Q&A | **Partial**: prepared, not yet delivered or tested on an audience |
| Coordination with professors, coordinators and other RAs | **Not demonstrated** | Show this through real experience (course projects, team work, supervisor feedback) |

Limits: a solo academic project on simulated data does not show professional collaboration, working with real industrial data, or domain expertise in maintenance.
