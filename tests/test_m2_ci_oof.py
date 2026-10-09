"""Critical synthetic causal contracts; no research data or scientific fits."""
import inspect
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from research.m2_ci import data, runner


class OOFContracts(unittest.TestCase):
    def test_strict_two_clock_purge(self):
        frame = pd.DataFrame({'decision_boundary_at':['2025-01-01T04:00:00Z']*3,
            'label_end':['2025-01-02T00:00:00Z','2025-01-01T23:00:00Z','2025-01-01T23:00:00Z'],
            'labelable_at':['2025-01-02T00:01:00Z','2025-01-02T00:00:00Z','2025-01-01T23:01:00Z']})
        np.testing.assert_array_equal(data.role_indices(frame,'2025-01-01T00:00:00Z','2025-01-02T00:00:00Z'),[2])

    def test_prediction_interface_has_no_targets(self):
        parameters = set(inspect.signature(runner.predict_only).parameters)
        self.assertEqual(parameters, {'features','indices','selected','scaler','device'})
        source = inspect.getsource(runner.predict_only)
        self.assertNotIn('RV_raw',source)
        self.assertNotIn('fit_',source)

    def test_outer_target_perturbation_cannot_change_fit_inputs(self):
        labels = pd.DataFrame({'RV_raw':[1.,2.,3.,4.,5.,6.]})
        fit, inner, outer = np.array([0,1]),np.array([2,3]),np.array([4,5])
        a = runner.fit_role_targets(labels,fit)
        b = runner.fit_role_targets(labels,inner)
        changed = labels.copy()
        changed.loc[outer,'RV_raw'] *= 1e9
        np.testing.assert_array_equal(a,runner.fit_role_targets(changed,fit))
        np.testing.assert_array_equal(b,runner.fit_role_targets(changed,inner))

    def test_r0_excluded_from_fitted_matrix(self):
        features = {'ordinary_raw':np.zeros((3,33)), 'hidden':np.zeros((3,512)),
                    'har':np.zeros((3,3)), 'r0':np.full((3,2),999)}
        self.assertEqual(runner.feature_matrix(features,'R2',np.array([0])).shape,(1,545))
        self.assertEqual(runner.feature_matrix(features,'R1',np.array([0])).shape,(1,33))
        self.assertFalse(np.any(runner.feature_matrix(features,'R2',np.array([0])) == 999))

    def test_old_output_paths_rejected_before_value_reads(self):
        with self.assertRaises(PermissionError):
            data.new_scope(data.ROOT / 'research/runs/FROZEN_RISK_03_v1')

    def test_registry_hash_mismatch_blocks_before_configuration_load(self):
        registry = {'configuration':{'path':'research/m2_ci/config.yaml','sha256':'wrong'},
                    'protocol_sha256':'wrong'}
        with patch.object(data,'read',return_value=registry), patch.object(data,'sha',return_value='actual'):
            with self.assertRaises(PermissionError):
                data.config()

    def test_real_fit_thread_and_cuda_contract(self):
        self.assertEqual(runner.os.environ['CUBLAS_WORKSPACE_CONFIG'],':4096:8')
        self.assertTrue(hasattr(runner.train_oof,'__wrapped__'))

    def test_prediction_csv_has_no_future_label_clocks(self):
        metadata = pd.DataFrame({k:['2025-01-01T04:00:00Z'] for k in
                    ['opportunity_id','decision_boundary_at','decision_at','entry_at','history_start_at','history_end_exclusive']})
        metadata['RV_raw'] = 999
        metadata['labelable_at'] = '2099-01-01T00:00:00Z'
        result = runner.prediction_frame(metadata,np.array([0]),{'B5':np.array([.001])},'WF01','2025-01-01T00:00:00Z')
        self.assertNotIn('RV_raw',result)
        self.assertNotIn('labelable_at',result)


if __name__ == '__main__':
    unittest.main()
