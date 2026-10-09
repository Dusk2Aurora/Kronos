"""Pure synthetic diagnostics tests; no real study sources or holdout reads."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from research.frozen.experiment_03 import diagnostics as d


class DiagnosticsTests(unittest.TestCase):
    def fixture(self, start="2025-01-01", n=12):
        frame = pd.DataFrame({"decision_at": pd.date_range(start, periods=n, freq="8h", tz="UTC"),
                              "RV_raw": np.linspace(0., .002, n)})
        predictions = {"ewma": np.full(n, .0005), "R1": np.linspace(.0001, .001, n),
                       "R2": np.linspace(.0002, .002, n), "B2": np.full(n, .0009)}
        return frame, predictions

    def test_unique_fixed_bins_and_exact_boundary(self):
        cuts = d.quantile_cuts([1., 1., 1., 1.], [.25, .5, .75])
        self.assertEqual(cuts, [1.])
        np.testing.assert_array_equal(d.assign_bins([0., 1., 2.], cuts), [0, 1, 1])
        with self.assertRaises(ValueError):
            d.assign_bins([1.], [1., 1.])

    def test_sealed_training_bins_unchanged_by_future_perturbation(self):
        train, pred = self.fixture()
        with tempfile.TemporaryDirectory() as root:
            seal = d.seal_calibration(train, pred, Path(root) / "seal")
            original = copy.deepcopy(seal)
            future, futurepred = self.fixture("2026-03-01")
            first = d.diagnostic_tables(d.row_metrics(future, futurepred), seal)
            future.RV_raw *= 100
            futurepred["R2"] *= 100
            changed = d.diagnostic_tables(d.row_metrics(future, futurepred), seal)
            self.assertEqual(seal, original)
            self.assertNotEqual(first["period_losses"].mean_prediction_RV.tolist(),
                                changed["period_losses"].mean_prediction_RV.tolist())
            for table in (first["model_calibration"], changed["model_calibration"]):
                self.assertEqual(table[(table.period == "POOLED") & (table.family == "R2")
                                       & (table.bin_type == "prediction_decile")]["count"].sum(), len(future))
            seal["thresholds"]["train_effective_RV_q90"] *= 2
            with self.assertRaises(ValueError):
                d.diagnostic_tables(d.row_metrics(future, futurepred), seal)
            with self.assertRaises(FileExistsError):
                d.seal_calibration(train, pred, Path(root) / "seal")

    def test_same_epsilon_identity_including_floor_and_underprediction(self):
        frame, pred = self.fixture()
        rows = d.row_metrics(frame, pred)
        np.testing.assert_allclose(rows.additive_epsilon_logRV_MSE, rows.surprise_MSE,
                                   rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(rows.RV_effective, np.maximum(rows.RV_raw, 1e-12))
        # Strict underestimation comparator: exactly one half is not counted.
        observed = pd.DataFrame({"decision_at": pd.date_range("2026-03-01", periods=3, freq="8h", tz="UTC"),
                                 "RV_raw": [.004, .004, .004]})
        synthetic = {"ewma": np.array([.002, .001, .003])}
        with tempfile.TemporaryDirectory() as root:
            seal = d.seal_calibration(frame, {"ewma": pred["ewma"]}, Path(root) / "seal")
            table = d.diagnostic_tables(d.row_metrics(observed, synthetic), seal)["high_extreme_RV_underprediction"]
            pooled = table[table.period == "POOLED"]
            np.testing.assert_array_equal(pooled.underprediction_count, [1, 1])
            np.testing.assert_array_equal(pooled["count"], [3, 3])

    def test_calendar_weighted_rolling_not_daily_mean_average(self):
        frame = pd.DataFrame({"decision_at": ["2026-03-01T04:01Z", "2026-03-01T12:01Z",
                              "2026-03-01T20:01Z", "2026-03-03T04:01Z", "2026-03-08T04:01Z"],
                              "RV_raw": [.0001, .0002, .0003, .003, .004]})
        rows = d.row_metrics(frame, {"ewma": np.full(5, .0005)})
        rolling = d.calendar_rolling(rows)
        self.assertEqual(len(rolling), 8)
        self.assertEqual(rolling.iloc[1].daily_count, 0)
        self.assertEqual(rolling.iloc[2].rolling_7day_count, 4)
        self.assertAlmostEqual(rolling.iloc[2].rolling_7day_raw_QLIKE_mean, rows.raw_QLIKE.iloc[:4].mean())
        self.assertEqual(rolling.iloc[7].rolling_7day_count, 2)
        self.assertAlmostEqual(rolling.iloc[7].rolling_7day_raw_QLIKE_mean, rows.raw_QLIKE.iloc[3:].mean())

    def test_analyze_context_immutable_output_and_artifacts(self):
        train, pred = self.fixture()
        future, futurepred = self.fixture("2026-03-01")
        with tempfile.TemporaryDirectory() as root:
            seal = d.seal_calibration(train, pred, Path(root) / "seal")
            out = Path(root) / "synthetic_diagnostics"
            result = d.analyze(future, futurepred, seal, out, context="DEV_validation_engineering")
            self.assertFalse(result["manifest"]["thresholds_refit"])
            for filename in ("RV_timeline.png", "RV_timeline.svg", "manifest.json", "row_metrics_and_timeline.csv"):
                self.assertTrue((out / filename).is_file())
            with self.assertRaises(FileExistsError):
                d.analyze(future, futurepred, seal, out, context="DEV_validation_engineering")
            with self.assertRaises(ValueError):
                d.analyze(future, futurepred, seal, Path(root) / "wrong", context="authorized_holdout")


if __name__ == "__main__":
    unittest.main()
