"""Synthetic-only risk-03 engineering checks; no market/holdout inputs."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from research.frozen.experiment_03 import models as m


class B2ContractTests(unittest.TestCase):
    def test_exact_selection_tie_and_nonfinite(self):
        candidates = [dict(eligible=True, num_leaves=a, min_data_in_leaf=b,
                           validation_qlike=2.) for a, b in m.B2_GRID]
        self.assertIs(m.select_b2(candidates), candidates[1])
        candidates[0]["validation_qlike"] = np.nextafter(2., -np.inf)
        self.assertIs(m.select_b2(candidates), candidates[0])
        with self.assertRaises(ValueError):
            m.select_b2([dict(eligible=False, validation_qlike=1.)])

    def test_width_finite_and_grid_guards_before_dependency(self):
        for x in (np.ones((80, 32)), np.ones((80, 34)), np.full((80, 33), np.nan)):
            with self.assertRaises(ValueError):
                m.fit_b2(x, np.ones(80), np.ones((40, 33)), np.ones(40), 7, 30)
        with self.assertRaises(ValueError):
            m.fit_b2(np.ones((80, 33)), np.ones(80), np.ones((40, 33)), np.ones(40), 8, 30)
        with self.assertRaises(ValueError):
            m.fit_b2(np.ones((80, 33)), -np.ones(80), np.ones((40, 33)), np.ones(40), 7, 30)

    def test_absolute_clip_and_target_floor(self):
        p, stats = m._restore_log_prediction(np.array([-1000., 0., 1000.]), 1e-4)
        np.testing.assert_allclose(p, [1e-12, 1e-4, 1.], rtol=1e-14)
        self.assertEqual(stats["lower_clip_count"], 1)
        self.assertEqual(stats["upper_clip_count"], 1)
        np.testing.assert_array_equal(m._effective_y([0., 1e-13, 1e-3]), [1e-12, 1e-12, 1e-3])

    def test_locked_configuration(self):
        cfg = m._locked_b2()
        self.assertEqual(cfg["library_version"], "4.6.0")
        self.assertEqual(cfg["early_stopping_rounds"], 30)
        self.assertEqual(cfg["num_boost_round_max"], 300)
        self.assertEqual(cfg["params"]["objective"], "gamma")
        self.assertEqual(cfg["params"]["metric"], "None")

    @unittest.skipUnless((m.DEPENDENCY_PATH / "lightgbm/__init__.py").exists(), "isolated dependency not installed")
    def test_synthetic_gamma_link_reload_metric_and_determinism(self):
        rng = np.random.default_rng(177)
        x = rng.normal(size=(480, 33))
        y = 1e-4 * np.exp(.8 * x[:, 0]) * rng.gamma(5., .2, size=len(x))
        # Validation has a different target median: scale must remain train-only.
        y[360:] *= 1.3
        print(json.dumps({"event": "fit_started", "context": "synthetic",
                          "family": "B2", "num_leaves": 7, "min_data_in_leaf": 30,
                          "purpose": "gamma_link_reload_engineering_check"}))
        model = m.fit_b2(x[:360], y[:360], x[360:], y[360:], 7, 30)
        print(json.dumps({"event": "fit_completed", "context": "synthetic",
                          "family": "B2", "best_iteration": model["best_iteration"],
                          "training_iterations": model["training_iterations"]}))
        self.assertEqual(model["scale"], np.median(y[:360]))
        self.assertLessEqual(model["training_iterations"], 300)
        self.assertGreaterEqual(model["training_iterations"], model["best_iteration"])
        self.assertEqual(len(model["history"]["train"]["raw_QLIKE"]),
                         len(model["history"]["validation"]["raw_QLIKE"]))
        self.assertEqual(model["effective_params"]["objective"], "gamma")
        self.assertEqual(model["effective_params"]["num_threads"], "1")
        booster = m._lightgbm().Booster(model_str=model["model_text"])
        raw = booster.predict(x[360:], raw_score=True, num_threads=1)
        transformed = booster.predict(x[360:], num_threads=1)
        np.testing.assert_allclose(transformed, np.exp(raw), rtol=1e-13)
        p = m.predict_b2(model, x[360:])
        np.testing.assert_allclose(p, np.clip(transformed * model["scale"], 1e-12, 1.), rtol=1e-13)
        self.assertAlmostEqual(model["validation_qlike"], np.mean(m.qlike(y[360:], p)), places=12)
        self.assertAlmostEqual(model["validation_qlike"],
                               model["history"]["validation"]["raw_QLIKE"][model["best_iteration"] - 1], places=12)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic_b2.json"
            m.save_b2(model, path)
            reloaded = m.load_b2(path)
            np.testing.assert_array_equal(p, m.predict_b2(reloaded, x[360:]))
            self.assertEqual(model, reloaded)
            with self.assertRaises(FileExistsError):
                m.save_b2(model, path)
        print(json.dumps({"event": "fit_started", "context": "synthetic",
                          "family": "B2", "num_leaves": 7, "min_data_in_leaf": 30,
                          "purpose": "determinism_engineering_check"}))
        replay = m.fit_b2(x[:360], y[:360], x[360:], y[360:], 7, 30)
        print(json.dumps({"event": "fit_completed", "context": "synthetic",
                          "family": "B2", "best_iteration": replay["best_iteration"],
                          "training_iterations": replay["training_iterations"]}))
        self.assertEqual(model, replay)


if __name__ == "__main__":
    unittest.main()
