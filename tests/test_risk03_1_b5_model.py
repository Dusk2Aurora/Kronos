"""Small synthetic checks; no research data or experiment execution."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from research.frozen.experiment_03_1.b5_model import (
    B5TCN, configure, fit_b5, load_b5, predict_b5, raw_qlike, save_b5,
)


class B5ModelTests(unittest.TestCase):
    def setUp(self):
        configure(17, "cpu")
        rng = np.random.default_rng(17)
        self.x = rng.normal(size=(4, 256, 11)).astype(np.float32)
        self.ordinary = rng.normal(size=(4, 33)).astype(np.float32)

    def test_causal_hidden_and_batch_independence(self):
        model = B5TCN(4).eval()
        with torch.no_grad():
            before = model.hidden(torch.from_numpy(self.x))
            changed = self.x.copy()
            changed[:, 100:] += 100
            after = model.hidden(torch.from_numpy(changed))
            torch.testing.assert_close(before[:, :100], after[:, :100], rtol=0, atol=0)
            alone = model.hidden(torch.from_numpy(self.x[:1]))
            torch.testing.assert_close(before[:1], alone, rtol=1e-5, atol=1e-6)
            changed[1:] *= 10
            changed[0] = self.x[0]
            mixed = model.hidden(torch.from_numpy(changed))
            torch.testing.assert_close(before[:1], mixed[:1], rtol=0, atol=0)

    def test_float64_loss_equivalence(self):
        y = np.array([1e-8, 0.001, 0.02], dtype=np.float64)
        scale = np.median(y)
        z = np.array([-5, 0.5, 1], dtype=np.float64)
        original = raw_qlike(y, np.exp(z) * scale)
        scaled = np.mean(z + y / scale * np.exp(-z)) + np.log(scale)
        self.assertAlmostEqual(original, scaled, places=13)

    def test_fit_and_weights_only_reload(self):
        y = np.array([0.001, 0.002, 0.003, 0.004])
        payload = fit_b5(self.x, self.ordinary, y, self.x[:2], self.ordinary[:2], y[:2],
                         width=4, weight_decay=0.001, seed=17, device="cpu", max_epochs=2)
        self.assertEqual(len(payload["history"]), 2)
        self.assertTrue(payload["metadata"]["finite_checks_passed"])
        self.assertGreater(payload["resources"]["parameters"], 0)
        prediction = predict_b5(payload, self.x, self.ordinary)
        self.assertTrue(np.isfinite(prediction["prediction"]).all())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "b5.pt"
            save_b5(payload, path)
            loaded = load_b5(path)
            for index in (0, len(self.x) // 2, len(self.x) - 1):
                original = predict_b5(payload, self.x[index:index + 1], self.ordinary[index:index + 1])
                restored = predict_b5(loaded, self.x[index:index + 1], self.ordinary[index:index + 1])
                repeat = predict_b5(loaded, self.x[index:index + 1], self.ordinary[index:index + 1])
                np.testing.assert_array_equal(original["raw_score"], restored["raw_score"])
                np.testing.assert_array_equal(restored["prediction"], repeat["prediction"])
            np.testing.assert_array_equal(prediction["raw_score"],
                                          predict_b5(loaded, self.x, self.ordinary)["raw_score"])


if __name__ == "__main__":
    unittest.main()

