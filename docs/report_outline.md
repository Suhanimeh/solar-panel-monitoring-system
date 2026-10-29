# Technical Report Outline

Target length: 6–8 pages. Numbers below are the actual outputs of the experiment (10 seeds, simulated data).

1. **Abstract** (150 words). Problem: fixed range checks miss context-dependent anomalies. Approach: compare range checks, an expected-behaviour regression model, and Isolation Forest on simulated multi-day solar data with injected, labelled faults, using a chronological split. Result: the regression model reached F1 0.79 and event recall 0.87 versus 0.09 and 0.25 for range checks; Isolation Forest scored F1 below 0.07. Caveat: simulated data; advantage partly by construction.
2. **Introduction.** Condition monitoring in context; why solar panels as an accessible testbed; explicit difference between monitoring, anomaly detection, fault classification, failure prediction.
3. **Problem Statement and Objectives.** Research question; four objectives (README §2).
4. **Background and Related Work.** Cover: anomaly-detection overview (Chandola, Banerjee & Kumar, 2009, *ACM Computing Surveys*, "Anomaly detection: A survey"); Isolation Forest (Liu, Ting & Zhou, 2008, *IEEE ICDM*); scikit-learn (Pedregosa et al., 2011, *JMLR*); PV-specific fault-detection literature (**to be found and read by you**; do not cite anything you have not read). Public dataset: Kaggle "Solar Power Generation Data".
5. **Methodology.** Data generation and calibration; features; split; detectors; threshold rule; metrics (docs/methodology.md D–E).
6. **Implementation.** Package structure, reproducibility (seeds, metadata), tests; mention the unchanged IoT stack.
7. **Results and Discussion.** Table, fig3 confusion matrices, fig4 PR curves and per-fault recall, fig2 timeline; failure-mode analysis (Isolation Forest saturation, stuck-sensor lag); false-alarm trade-off.
8. **Limitations.** Simulated; injected; circularity; prevalence; daylight only; flags ≠ faults.
9. **Conclusion and Future Work.** Real data, physically grounded model, rolling-variance rule, persistence alerting, public dataset.
10. **References.** Only sources you have read. Verified candidates: Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008), Isolation Forest, ICDM; Chandola, V., Banerjee, A., & Kumar, V. (2009), ACM Computing Surveys 41(3); Pedregosa, F. et al. (2011), JMLR 12. Check volume/page details before submitting.
