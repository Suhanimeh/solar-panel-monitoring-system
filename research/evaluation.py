"""Evaluation helpers. Implemented by hand (and cross-checked against
scikit-learn in the tests) so every number can be explained."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .detectors import BaseDetector


def _check(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same length")
    if not (np.isin(y_true, [0, 1]).all() and np.isin(y_pred, [0, 1]).all()):
        raise ValueError("labels and predictions must be 0/1")
    return y_true.astype(int), y_pred.astype(int)


def confusion_counts(y_true, y_pred) -> dict:
    y_true, y_pred = _check(y_true, y_pred)
    return {
        "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
        "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
    }


def precision_recall_f1(y_true, y_pred) -> dict:
    c = confusion_counts(y_true, y_pred)
    p = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) else 0.0
    r = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def event_recall(event_id, y_pred) -> float:
    """Fraction of fault *episodes* with at least one flagged sample.
    Operators act on episodes, not individual rows, so this is the more
    practical 'did we notice the problem?' measure."""
    event_id, y_pred = np.asarray(event_id), np.asarray(y_pred)
    ids = np.unique(event_id[event_id > 0])
    if len(ids) == 0:
        return float("nan")
    return float(np.mean([y_pred[event_id == i].any() for i in ids]))


def recall_by_fault_type(fault_type, y_pred) -> dict:
    fault_type, y_pred = np.asarray(fault_type), np.asarray(y_pred)
    return {t: float(y_pred[fault_type == t].mean()) for t in np.unique(fault_type) if t != "none"}


def evaluate_detector(det: BaseDetector, test_df: pd.DataFrame) -> dict:
    """Metrics for one detector on a labelled test set."""
    y = test_df["is_anomaly"].to_numpy()
    score, pred = det.score(test_df), det.predict(test_df)
    res = {"method": det.name, **confusion_counts(y, pred), **precision_recall_f1(y, pred)}
    res["event_recall"] = event_recall(test_df["event_id"], pred)
    n_days = test_df["timestamp_utc"].dt.date.nunique()
    res["false_alarms_per_day"] = res["fp"] / n_days
    # Rank-based metrics only make sense when the score is graded, not binary.
    graded = len(np.unique(score)) > 2
    res["roc_auc"] = float(roc_auc_score(y, score)) if graded else float("nan")
    res["pr_auc"] = float(average_precision_score(y, score)) if graded else float("nan")
    res["prevalence"] = float(y.mean())
    for t, v in recall_by_fault_type(test_df["fault_type"], pred).items():
        res[f"recall_{t}"] = v
    return res
