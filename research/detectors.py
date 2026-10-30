"""Three kinds of anomaly detector behind one small interface.

    fit(train_df)               learn what "normal" looks like (clean data only)
    calibrate(val_df, fpr)      pick an alert threshold on *clean validation* data
    score(df)                   higher = more unusual
    predict(df)                 1 = flagged, 0 = not flagged

All score-based detectors use the same threshold rule (the (1-fpr) quantile of
scores on clean validation data) so the comparison is fair and no labels are
needed to tune anything.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from .data_prep import ENGINEERED_FEATURES, RAW_FEATURES

# Reuse the *existing* logger's range limits as the single source of truth.
from data_logger.logger import VALID_RANGES


class BaseDetector:
    name = "base"
    threshold_: float | None = None

    def fit(self, train_df: pd.DataFrame) -> "BaseDetector":
        raise NotImplementedError

    def score(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def calibrate(self, val_df: pd.DataFrame, target_fpr: float = 0.01) -> "BaseDetector":
        if not 0 < target_fpr < 1:
            raise ValueError("target_fpr must be between 0 and 1")
        self.threshold_ = float(np.quantile(self.score(val_df), 1 - target_fpr))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        if self.threshold_ is None:
            raise RuntimeError("Call calibrate() before predict()")
        return (self.score(df) > self.threshold_).astype(int)


class RangeThresholdDetector(BaseDetector):
    """Baseline: flag a row if any reading is outside the fixed valid range.
    This is the validation the existing logger already performs, so it is the
    honest 'what do we already have?' reference. It needs no training."""
    name = "Range thresholds (existing logic)"

    def __init__(self, ranges: Dict[str, tuple] | None = None):
        self.ranges = ranges or VALID_RANGES
        self.threshold_ = 0.0

    def fit(self, train_df):
        return self

    def calibrate(self, val_df, target_fpr=0.01):
        return self  # nothing to calibrate: limits come from sensor specs

    def score(self, df):
        n_violations = np.zeros(len(df))
        for col, (lo, hi) in self.ranges.items():
            if col in df.columns:
                n_violations += ((df[col] < lo) | (df[col] > hi)).to_numpy()
        return n_violations


class ConditionalResidualDetector(BaseDetector):
    """'Expected behaviour' model: predict voltage and current from light and
    temperature using a quadratic regression trained on clean data, then flag
    rows whose measured value is far from the prediction.

    score = max over {voltage, current} of |residual| / robust sigma, where sigma
    is estimated from training residuals via the median absolute deviation."""
    name = "Expected-behaviour residual"
    predictors = ["light_pct", "temp_c"]
    targets = ["voltage_v", "current_a"]

    def fit(self, train_df):
        X = train_df[self.predictors]
        self.models_, self.sigma_ = {}, {}
        for t in self.targets:
            m = make_pipeline(PolynomialFeatures(2, include_bias=False), LinearRegression())
            m.fit(X, train_df[t])
            resid = train_df[t] - m.predict(X)
            self.models_[t] = m
            self.sigma_[t] = 1.4826 * float(np.median(np.abs(resid - np.median(resid))))
        return self

    def score(self, df):
        X = df[self.predictors]
        z = [np.abs(df[t] - self.models_[t].predict(X)) / self.sigma_[t] for t in self.targets]
        return np.max(np.vstack(z), axis=0)


class IsolationForestDetector(BaseDetector):
    """Isolation Forest (Liu, Ting & Zhou, 2008): unsupervised; isolates unusual
    points with fewer random splits. Trees are scale-invariant, so no feature
    scaling is applied. Missing feature values (e.g. the first samples of a day
    for rolling features) are filled with the *training* median."""

    def __init__(self, features: List[str], name: str, n_estimators: int = 200, random_state: int = 0):
        self.features, self.name = features, name
        self.model = IsolationForest(n_estimators=n_estimators, random_state=random_state)

    def fit(self, train_df):
        self.fill_ = train_df[self.features].median()
        self.model.fit(train_df[self.features].fillna(self.fill_))
        return self

    def score(self, df):
        return -self.model.score_samples(df[self.features].fillna(self.fill_))


def build_detectors(random_state: int = 0) -> List[BaseDetector]:
    return [
        RangeThresholdDetector(),
        ConditionalResidualDetector(),
        IsolationForestDetector(RAW_FEATURES, "Isolation Forest (raw features)", random_state=random_state),
        IsolationForestDetector(ENGINEERED_FEATURES, "Isolation Forest (+ engineered features)", random_state=random_state),
    ]
