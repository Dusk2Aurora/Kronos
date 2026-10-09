"""Synthetic-only engineering tests; no research source or holdout is opened."""
import copy
import json
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score, average_precision_score
from research.frozen.experiment_03.statistics import (
    weighted_ranking, stationary_day_counts, equal_quarter_differences,
    classify_status, evaluate_quarters, load_config, variance_losses,
)


def test_ties_match_sklearn_expansion_and_pairwise_rank():
    y = np.array([0, 1, 0, 1, 1, 0], dtype=bool)
    score = np.array([1., 1., 2., 3., 3., 3.])
    counts = np.array([3, 2, 0, 4, 1, 2])
    actual = weighted_ranking(y, score, counts)
    yy, ss = np.repeat(y, counts), np.repeat(score, counts)
    assert actual['AUROC'] == pytest.approx(roc_auc_score(yy, ss))
    assert actual['AP'] == pytest.approx(average_precision_score(yy, ss))
    positive, negative = ss[yy], ss[~yy]
    independent = np.mean((positive[:, None] > negative) + .5 * (positive[:, None] == negative))
    assert actual['AUROC'] == pytest.approx(independent)


def test_full_calendar_shared_seed_and_empty_dates():
    counts = stationary_day_counts(20, 7, 40, 20268008)
    assert counts.shape == (40, 20)
    np.testing.assert_array_equal(counts.sum(axis=1), 20)
    np.testing.assert_array_equal(counts, stationary_day_counts(20, 7, 40, 20268008))
    assert not np.array_equal(counts, stationary_day_counts(20, 7, 40, 20268009))
    # Calendar days 1, 4 and 8 have no opportunities, but stay in the sampling frame.
    day_ids = np.array([0, 0, 2, 3, 5, 6, 7, 9])
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    score = np.array([0, 2, 0, 1, 2, 3, 1, 3])
    for row in counts:
        weights = row[day_ids]
        if weights[labels == 1].sum() and weights[labels == 0].sum():
            expanded_dates = np.repeat(np.arange(20), row)
            expanded_rows = np.concatenate([np.flatnonzero(day_ids == d) for d in expanded_dates])
            result = weighted_ranking(labels, score, weights)
            assert result['AUROC'] == pytest.approx(roc_auc_score(labels[expanded_rows], score[expanded_rows]))


def test_equal_quarters_is_not_pooled_or_sample_weighted():
    assert equal_quarter_differences([.10, -.02]) == pytest.approx(.04)
    assert equal_quarter_differences([.10, -.02]) != pytest.approx((.10 * 90 - .02 * 92) / 182)
    assert np.isnan(equal_quarter_differences([.10, np.nan]))


def criteria(mean=.04, lower=.01, upper=.08, quarters=(.03, .05)):
    return {'mean_delta': mean, 'ci_lower': lower, 'ci_upper': upper, 'quarter_deltas': quarters}


@pytest.mark.parametrize('r1,b2,options,expected', [
    (criteria(), criteria(), {}, 'CONFIRMED_RANKING_INCREMENT'),
    (criteria(), criteria(), {'risk_magnitude_red_flag': True}, 'RANKING_ONLY'),
    (criteria(), criteria(lower=-.01), {}, 'NONLINEAR_BASELINE_COMPETITIVE'),
    (criteria(upper=-.01, lower=-.1, mean=-.03), criteria(), {}, 'NOT_SUPPORTED'),
    (criteria(mean=.02), criteria(), {}, 'NOT_SUPPORTED'),
    (criteria(lower=-.01), criteria(), {}, 'INCONCLUSIVE'),
    (criteria(quarters=(-.01, .09)), criteria(), {}, 'INCONCLUSIVE'),
    (criteria(), criteria(), {'adequate_events': False}, 'INCONCLUSIVE'),
    (criteria(), criteria(), {'invalid_fraction': .010001}, 'INCONCLUSIVE'),
    (criteria(), criteria(), {'invalid_fraction': .01}, 'CONFIRMED_RANKING_INCREMENT'),
    (criteria(), criteria(), {'invalid_fraction': .5, 'risk_magnitude_red_flag': True, 'invalidated': True}, 'INVALIDATED'),
    (criteria(), criteria(), {'invalid_fraction': .5, 'risk_magnitude_red_flag': True}, 'INCONCLUSIVE'),
])
def test_registered_status_priority(r1, b2, options, expected):
    assert classify_status({'R2_minus_R1': r1, 'R2_minus_B2': b2}, **options) == expected


def synthetic_config():
    cfg = copy.deepcopy(load_config())
    cfg['roles']['test_quarters'] = [
        {'id': 'SYNTHETIC_A', 'start': '2000-01-01T00:00:00Z', 'end_exclusive': '2000-01-11T00:00:00Z'},
        {'id': 'SYNTHETIC_B', 'start': '2000-01-11T00:00:00Z', 'end_exclusive': '2000-01-31T00:00:00Z'},
    ]
    cfg['bootstrap']['iterations'] = 40
    return cfg


def synthetic_frame(oneclass=False):
    dates = pd.date_range('2000-01-01T04:01:00Z', periods=30, freq='D')
    raw = np.where(np.arange(30) % 2, .04, .01)
    if oneclass:
        raw[:] = .01
    frame = pd.DataFrame({'decision_at': dates, 'RV_raw': raw})
    predictions = {'ewma': np.full(30, .01), 'R1': np.full(30, .02),
                   'B2': np.where(np.arange(30) % 3, .03, .01), 'R2': raw.copy()}
    return frame, predictions


def test_one_class_nan_no_redraw_and_artifacts(tmp_path):
    frame, preds = synthetic_frame(oneclass=True)
    cfg = synthetic_config()
    cfg['bootstrap']['iterations'] = 5000
    summary = evaluate_quarters(frame, preds, tmp_path, cfg)
    assert summary['status'] == 'INCONCLUSIVE'
    for block in [7, 3, 14]:
        draws = pd.read_csv(tmp_path / f'draws_block{block}.csv')
        assert len(draws) == 5000
        assert draws['equal_quarter_R2_minus_R1'].isna().all()
        assert not draws['joint_valid'].any()
        assert summary['bootstrap'][str(block)]['joint_valid_draws'] == 0
        assert summary['bootstrap'][str(block)]['invalid_fractions']['equal_quarter_joint'] == 1
    saved = json.loads((tmp_path / 'summary.json').read_text())
    assert saved['bootstrap']['7']['comparisons']['R2_minus_R1']['ci_lower'] is None
    with pytest.raises(FileExistsError):
        evaluate_quarters(frame, preds, tmp_path, cfg)


def test_paired_counts_scores_joint_percentile_and_unequal_quarters(tmp_path):
    frame, preds = synthetic_frame()
    cfg = synthetic_config()
    summary = evaluate_quarters(frame, preds, tmp_path, cfg, training_thresholds={'high_RV': .03})
    table = pd.read_csv(tmp_path / 'draws_block7.csv')
    archive = np.load(tmp_path / 'multiplicities_block7.npz')
    assert archive['SYNTHETIC_A_counts'].shape == (40, 10)
    assert archive['SYNTHETIC_B_counts'].shape == (40, 20)
    assert int(archive['SYNTHETIC_B_seed']) == 20261008 + 7000 + 1
    for qi, name in enumerate(['SYNTHETIC_A', 'SYNTHETIC_B']):
        start, stop = (0, 10) if qi == 0 else (10, 30)
        raw = frame['RV_raw'].to_numpy()[start:stop]
        denominator = preds['ewma'][start:stop]
        event = np.log((raw + 1e-12) / (denominator + 1e-12)) > np.log(2)
        for d in range(40):
            w = archive[f'{name}_counts'][d]
            for model in preds:
                score = np.log((preds[model][start:stop] + 1e-12) / (denominator + 1e-12))
                expected = weighted_ranking(event, score, w)['AUROC']
                actual = table.loc[d, f'{name}_{model}_AUROC']
                assert actual == pytest.approx(expected, nan_ok=True)
    for key in ['R2_minus_R1', 'R2_minus_B2']:
        delta = (table[f'SYNTHETIC_A_{key}'] + table[f'SYNTHETIC_B_{key}']) / 2
        np.testing.assert_allclose(table[f'equal_quarter_{key}'], delta, equal_nan=True)
        valid = table['joint_valid']
        if valid.any():
            lower, upper = np.quantile(delta[valid], [.025, .975])
            result = summary['bootstrap']['7']['comparisons'][key]
            assert result['ci_lower'] == pytest.approx(lower)
            assert result['ci_upper'] == pytest.approx(upper)
    assert summary['high_RV']['sealed_threshold'] == .03
    assert summary['thresholds_refit'] is False


def test_surprise_uses_raw_additive_epsilon_and_strict_threshold(tmp_path):
    frame, preds = synthetic_frame()
    frame['RV_raw'] = .02
    result = evaluate_quarters(frame, preds, tmp_path, synthetic_config())
    # (.02 + eps)/(.01 + eps) is strictly below 2: no surprise event.
    assert all(q['positive'] == 0 for q in result['quarter_metrics'])


def test_invalid_input_and_purge(tmp_path):
    frame, preds = synthetic_frame()
    preds['R2'][0] = np.nan
    with pytest.raises(ValueError, match='prediction'):
        evaluate_quarters(frame, preds, tmp_path, synthetic_config())
    frame, preds = synthetic_frame()
    frame['labelable_at'] = frame['decision_at'] + pd.Timedelta(hours=6)
    frame.loc[9, 'labelable_at'] = pd.Timestamp('2000-01-11T00:00:00Z')
    with pytest.raises(ValueError, match='unpurged'):
        evaluate_quarters(frame, preds, tmp_path, synthetic_config())


def test_variance_regret_and_logmse_are_explicit():
    result = variance_losses([0., .01], [1e-12, .01])
    np.testing.assert_allclose(result['QLIKE_Regret'], 0)
    np.testing.assert_allclose(result['logRV_MSE'], 0)
    assert result['additive_epsilon_logRV_MSE'][0] > 0

