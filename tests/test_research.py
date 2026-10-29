"""Tests for the research/ package. Written with unittest so they run with
either `python -m unittest discover tests` or `pytest tests/`."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.data_prep import (add_features, load_readings, quality_report,  # noqa: E402
                                split_by_day, RAW_FEATURES)
from research.detectors import (ConditionalResidualDetector, IsolationForestDetector,  # noqa: E402
                                RangeThresholdDetector, VALID_RANGES)
from research.evaluation import (confusion_counts, event_recall,  # noqa: E402
                                 precision_recall_f1, recall_by_fault_type)
from research.simulate import SAMPLES_PER_DAY, generate_clean_days, make_dataset  # noqa: E402


class TestLoadAndQuality(unittest.TestCase):
    def test_missing_column_raises(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.csv"
            pd.DataFrame({"timestamp_utc": ["2026-01-01T00:00:00Z"], "voltage_v": [1]}).to_csv(p, index=False)
            with self.assertRaisesRegex(ValueError, "missing required columns"):
                load_readings(p)

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_readings("does_not_exist.csv")

    def test_quality_report_counts_problems(self):
        df = pd.DataFrame({
            "timestamp_utc": pd.to_datetime(["2026-01-01 00:00", "2026-01-01 00:02", "2026-01-01 00:02", "2026-01-01 00:10"], utc=True),
            "voltage_v": [12.0, 99.0, 99.0, 12.0], "current_a": [1.0, 1.0, 1.0, 1.0],
            "power_w": [12.0, 99.0, 99.0, 12.0], "temp_c": [20.0, 20.0, 20.0, -999.0],
            "humidity_pct": [50.0, 50.0, 50.0, np.nan], "light_pct": [50.0] * 4,
        })
        r = quality_report(df, VALID_RANGES)
        self.assertEqual(r["duplicate_rows"], 1)
        self.assertEqual(r["duplicate_timestamps"], 1)
        self.assertEqual(r["out_of_range"]["voltage_v"], 2)
        self.assertEqual(r["sentinel_values"]["temp_c"], 1)
        self.assertEqual(r["missing_by_column"]["humidity_pct"], 1)
        self.assertEqual(r["n_gaps_over_2x_median"], 1)


class TestSplitAndFeatures(unittest.TestCase):
    def test_split_is_chronological_and_disjoint(self):
        df = generate_clean_days(6, seed=1)
        tr, va, te = split_by_day(df, 3, 1)
        self.assertLess(tr.timestamp_utc.max(), va.timestamp_utc.min())
        self.assertLess(va.timestamp_utc.max(), te.timestamp_utc.min())
        self.assertEqual(len(tr) + len(va) + len(te), len(df))

    def test_split_needs_a_test_set(self):
        with self.assertRaises(ValueError):
            split_by_day(generate_clean_days(3, seed=1), 2, 1)

    def test_features_are_causal(self):
        """Changing future readings must not change earlier feature values."""
        df = generate_clean_days(1, seed=2)
        base = add_features(df)
        altered = df.copy()
        altered.loc[200:, "current_a"] = 99.0
        again = add_features(altered)
        pd.testing.assert_series_equal(base.loc[:199, "current_roll_std"], again.loc[:199, "current_roll_std"])

    def test_current_per_light_no_division_blowup(self):
        df = generate_clean_days(1, seed=3)
        df.loc[0, "light_pct"] = 0.0
        self.assertTrue(np.isfinite(add_features(df)["current_per_light"]).all())


class TestSimulation(unittest.TestCase):
    def test_reproducible(self):
        pd.testing.assert_frame_equal(make_dataset(7), make_dataset(7))

    def test_labels_only_in_test_days(self):
        df = make_dataset(5)
        tr, va, te = split_by_day(df, 8, 2)
        self.assertEqual(tr.is_anomaly.sum() + va.is_anomaly.sum(), 0)
        self.assertEqual(te.event_id[te.event_id > 0].nunique(), 16)  # 4 faults x 4 days

    def test_faults_labelled_and_power_consistent(self):
        df = make_dataset(5)
        self.assertEqual(set(df.fault_type.unique()),
                         {"none", "current_drop", "stuck_current", "voltage_spike", "range_violation"})
        self.assertTrue(((df.fault_type != "none") == (df.is_anomaly == 1)).all())
        self.assertLess((df.power_w - df.voltage_v * df.current_a).abs().max(), 0.01)
        self.assertTrue((df.data_source == "SIMULATED").all())

    def test_only_range_violations_leave_valid_range(self):
        df = make_dataset(5)
        outside = df[(df.voltage_v > VALID_RANGES["voltage_v"][1])]
        self.assertTrue((outside.fault_type == "range_violation").all())

    def test_clean_day_size(self):
        self.assertEqual(len(generate_clean_days(2, 0)), 2 * SAMPLES_PER_DAY)


class TestEvaluation(unittest.TestCase):
    def test_metrics_match_sklearn(self):
        rng = np.random.default_rng(0)
        y, p = rng.integers(0, 2, 200), rng.integers(0, 2, 200)
        m = precision_recall_f1(y, p)
        self.assertAlmostEqual(m["precision"], precision_score(y, p))
        self.assertAlmostEqual(m["recall"], recall_score(y, p))
        self.assertAlmostEqual(m["f1"], f1_score(y, p))
        c = confusion_counts(y, p)
        self.assertEqual(sum(c.values()), 200)

    def test_zero_division_is_safe(self):
        m = precision_recall_f1([0, 0, 1], [0, 0, 0])
        self.assertEqual((m["precision"], m["recall"], m["f1"]), (0.0, 0.0, 0.0))

    def test_event_recall_hand_example(self):
        ev = [0, 1, 1, 0, 2, 2, 0, 3]
        pred = [0, 0, 1, 0, 0, 0, 0, 1]
        self.assertAlmostEqual(event_recall(ev, pred), 2 / 3)

    def test_recall_by_type(self):
        r = recall_by_fault_type(["none", "a", "a", "b"], [1, 1, 0, 0])
        self.assertEqual(r, {"a": 0.5, "b": 0.0})

    def test_bad_input_rejected(self):
        with self.assertRaises(ValueError):
            confusion_counts([0, 1], [0])
        with self.assertRaises(ValueError):
            confusion_counts([0, 2], [0, 1])


class TestDetectors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = make_dataset(11)
        tr, va, te = split_by_day(df, 8, 2)
        cls.tr, cls.va, cls.te = (add_features(x) for x in (tr, va, te))

    def test_range_detector_flags_out_of_range_only(self):
        d = RangeThresholdDetector()
        pred = d.predict(self.te)
        flagged_types = set(self.te.loc[pred == 1, "fault_type"])
        self.assertEqual(flagged_types, {"range_violation"})

    def test_calibration_hits_target_false_alarm_rate_on_validation(self):
        d = ConditionalResidualDetector().fit(self.tr).calibrate(self.va, 0.02)
        rate = d.predict(self.va).mean()
        # quantile interpolation can land one or two samples off the target
        self.assertAlmostEqual(rate, 0.02, delta=2 / len(self.va))
        self.assertGreater(rate, 0.0)

    def test_predict_before_calibrate_fails(self):
        with self.assertRaises(RuntimeError):
            ConditionalResidualDetector().fit(self.tr).predict(self.te)

    def test_residual_detector_catches_obvious_current_drop(self):
        d = ConditionalResidualDetector().fit(self.tr).calibrate(self.va, 0.01)
        row = self.va[self.va.light_pct > 80].iloc[[0]].copy()
        self.assertEqual(d.predict(row)[0], 0)
        row["current_a"] = row["current_a"] * 0.5
        self.assertEqual(d.predict(row)[0], 1)

    def test_isolation_forest_scores_and_is_seeded(self):
        a = IsolationForestDetector(RAW_FEATURES, "a", random_state=1).fit(self.tr)
        b = IsolationForestDetector(RAW_FEATURES, "b", random_state=1).fit(self.tr)
        s = a.score(self.te)
        self.assertEqual(s.shape, (len(self.te),))
        np.testing.assert_allclose(s, b.score(self.te))

    def test_calibration_does_not_use_test_data(self):
        """Threshold must be identical whatever the test labels/values are."""
        d1 = ConditionalResidualDetector().fit(self.tr).calibrate(self.va, 0.01)
        d2 = ConditionalResidualDetector().fit(self.tr).calibrate(self.va, 0.01)
        _ = d2.predict(self.te)
        self.assertEqual(d1.threshold_, d2.threshold_)


if __name__ == "__main__":
    unittest.main()
