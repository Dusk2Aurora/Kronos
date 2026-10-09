"""Synthetic statistics contracts; no real research values, fits or state writes."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.stats import norm

from research.m2_ci import statistics as stats


class StatisticsContracts(unittest.TestCase):
    def test_degenerate_planning_has_no_numeric_mde_or_power(self):
        for se, alpha in [(1., 0.), (0., .5), (np.nan, .5)]:
            with self.subTest(se=se, alpha=alpha):
                rows = stats.planning_scenarios(se, .1, [3, 6], alpha)
                for row in rows:
                    self.assertEqual(row['planning_status'],
                                     'NOT_APPLICABLE_ALPHA_ZERO_OR_DEGENERATE')
                    for key in ('planning_SE_7day', 'MDE_80power_two_sided95',
                                'power_at_5pct_base_Regret_approx'):
                        self.assertTrue(np.isnan(row[key]))

    def test_planning_power_is_two_sided_and_scales_with_sample_size(self):
        # At zero effect a two-sided 5% test has nominal 5% power, not 2.5%.
        null = stats.planning_scenarios(.2, 0., [3], .5)[0]
        self.assertAlmostEqual(null['power_at_5pct_base_Regret_approx'], .05)
        rows = stats.planning_scenarios(.2, .1, [3, 12], .5)
        for row in rows:
            expected_n = row['months'] * 365.25 / 12 * 3
            expected_se = .2 * np.sqrt(1267 / expected_n)
            delta = .1 / expected_se
            expected_power = norm.cdf(delta - norm.ppf(.975)) + norm.cdf(-delta - norm.ppf(.975))
            self.assertEqual(row['planning_status'], 'CONDITIONAL_SCENARIO')
            self.assertAlmostEqual(row['planning_SE_7day'], expected_se)
            self.assertAlmostEqual(row['power_at_5pct_base_Regret_approx'], expected_power)
            self.assertAlmostEqual(row['MDE_80power_two_sided95'],
                                   (norm.ppf(.975) + norm.ppf(.8)) * expected_se)
        self.assertAlmostEqual(rows[1]['planning_SE_7day'], rows[0]['planning_SE_7day'] / 2)
        self.assertGreater(rows[1]['power_at_5pct_base_Regret_approx'],
                           rows[0]['power_at_5pct_base_Regret_approx'])

    def test_paired_raw_and_regret_gains_have_identical_statistics(self):
        target = np.array([0., 1e-5, .1, 2., 100.])
        first = stats.losses(target, [1e-8, 1e-4, .3, 1., 50.])
        second = stats.losses(target, [2e-8, 2e-5, .2, 3., 80.])
        raw = first['raw_QLIKE'] - second['raw_QLIKE']
        regret = first['QLIKE_Regret'] - second['QLIKE_Regret']
        np.testing.assert_allclose(raw, regret, rtol=1e-12, atol=1e-12)
        counts = np.array([[1, 1, 1, 1, 1], [2, 0, 1, 0, 2], [0, 2, 0, 2, 1]])
        raw_means = counts @ raw / counts.sum(axis=1)
        regret_means = counts @ regret / counts.sum(axis=1)
        np.testing.assert_allclose(raw_means, regret_means, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(np.quantile(raw_means, [.025, .975]),
                                   np.quantile(regret_means, [.025, .975]), atol=1e-12)

    def test_invalid_predictions_and_alignment_are_rejected(self):
        for prediction in ([0., 1.], [-1., 1.], [np.nan, 1.],
                           [np.inf, 1.], [1.], [[1., 1.]]):
            with self.subTest(prediction=prediction), self.assertRaises(ValueError):
                stats.losses([1., 2.], prediction)
        for target in ([np.nan, 2.], [np.inf, 2.]):
            with self.subTest(target=target), self.assertRaises(ValueError):
                stats.losses(target, [1., 1.])

    def test_four_decisions_and_zero_alpha_override(self):
        good = dict(gain=.1, alpha=.5, relative=.06, positive_folds=4,
                    validation_gain=.01, ci_lower=.01, ci_upper=.2,
                    cheap_screens=[False, False], specificity=True)
        cases = [({}, 'CANDIDATE_INCREMENT_NOT_CONFIRMED'),
                 ({'specificity': False, 'cheap_screens': [True, False]},
                  'NON_SPECIFIC_ENSEMBLE_GAIN'),
                 ({'ci_lower': -.01}, 'INCONCLUSIVE'),
                 ({'alpha': 0.}, 'NO_DEVELOPMENT_INCREMENT'),
                 ({'gain': 0.}, 'NO_DEVELOPMENT_INCREMENT'),
                 ({'ci_upper': 0.}, 'NO_DEVELOPMENT_INCREMENT'),
                 ({'positive_folds': 2, 'validation_gain': 0.}, 'NO_DEVELOPMENT_INCREMENT')]
        for change, expected in cases:
            with self.subTest(change=change):
                self.assertEqual(stats.classify(**(good | change)), expected)
        self.assertEqual(stats.classify(**(good | {'relative': .049})), 'INCONCLUSIVE')
        self.assertEqual(stats.classify(**(good | {'positive_folds': 3})), 'INCONCLUSIVE')
        self.assertEqual(stats.classify(**(good | {'validation_gain': 0.})), 'INCONCLUSIVE')

    def test_constant_rv_surprise_ranking_comes_from_same_ewma_denominator(self):
        event = np.array([True, True, False, False])
        ewma = np.array([1., 2., 4., 8.])
        constant = np.full(4, 4.)
        surprise_score = np.log((constant + stats.EPS) / (ewma + stats.EPS))
        inverse_ewma_score = -np.log(ewma + stats.EPS)
        self.assertEqual(stats.ranking(event, surprise_score),
                         stats.ranking(event, inverse_ewma_score))
        self.assertEqual(stats.ranking(event, surprise_score)['AUROC'], 1.)
        absolute = stats.ranking(event, constant)
        self.assertEqual(absolute['AUROC'], .5)
        self.assertEqual(absolute['AP'], float(event.mean()))
        one_class = stats.ranking(np.zeros(4), surprise_score)
        self.assertTrue(all(np.isnan(value) for value in one_class.values()))

    def test_weight_selection_loads_oof_only_without_validation(self):
        folds = [f'WF{i:02d}' for i in range(1, 6)]
        predictions = pd.DataFrame({'fold_id': np.resize(folds, 1267),
                                    'B5': np.full(1267, 2.), 'R2': np.ones(1267)})
        labels = pd.DataFrame({'RV_raw': np.ones(1267)})
        configuration = {'roles': {'folds': [{'id': f} for f in folds]},
            'fusion': {'comparators': ['R2'], 'alpha_candidates': [0., .5, 1.],
                       'selection': 'minimum OOF QLIKE', 'tie_break': 'smallest alpha'},
            'evidence_class': 'SYNTHETIC'}
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(stats, 'RUN', Path(temporary)), \
                patch.object(stats, 'config', return_value=configuration), \
                patch.object(stats, 'check_source_seal'), \
                patch.object(stats, 'load_tables', return_value=(predictions, labels)) as load, \
                patch.object(stats, 'sha', return_value='synthetic'), \
                patch.object(stats, 'save'), patch.object(stats, 'write') as write, \
                contextlib.redirect_stdout(io.StringIO()):
            stats.choose_weights()
        load.assert_called_once_with('oof_predictions.csv')
        selection = write.call_args.args[1]
        self.assertFalse(selection['original_VALID_opened_by_this_selection'])
        self.assertEqual(selection['selected']['R2']['alpha'], 1.)


if __name__ == '__main__':
    unittest.main()
