"""Same-coordinate quantization readability diagnostic, never portfolio bootstrap."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'baselines'))
from prepare_e00 import ROOT, require, sha, write_json
from summarize_e00r import development_labels, prediction_summary, matched_random, stationary_means, clean
from run_e00 import read_result
from mlp_analyze import daily, stratum_expected_weights
import numpy as np
import pandas as pd
import yaml

FOLDS = ('WF01', 'WF02', 'WF03', 'WF04')
FAMILIES = ('ridge_quant_continuous', 'ridge_quant_binary')
GATES = ('rank50', 'rank25', 'rank75', 'economic')
SEEDS = (17, 29, 43)
HISTORICAL = ('E00_20261008_phase4_v1', 'E00R_20261008_phase4_v1',
              'FROZEN_20261008_v1', 'FROZEN_MLP_20261008_v2')


def registered_old(run_id, relative, inputs):
    run = ROOT / 'research/runs' / run_id
    regpath = ROOT / 'research/registry' / f'{run_id}.json'
    registry = json.loads(regpath.read_text(encoding='utf-8'))
    require(registry['status'].startswith('completed'), 'Historical run not accepted: ' + run_id)
    hashes = {k.replace('\\', '/'): v for k, v in registry['artifacts']['final_hashes'].items()}
    require(sha(run / relative) == hashes.get(relative), 'Historical artifact changed: ' + str(run / relative))
    inputs[str(run / relative)] = sha(run / relative)
    inputs[str(regpath)] = sha(regpath)
    return run / relative


def validate_protocol(common):
    require(common['head']['type'] == 'Ridge' and common['head']['lambda_candidates'] == [.001, .01, .1, 1.], 'Ridge design differs')
    require(common['paired_uncertainty']['draws'] == 5000 and common['paired_uncertainty']['seed'] == 17
            and common['paired_uncertainty']['mean_block_days'] == 7
            and common['paired_uncertainty']['sensitivity_days'] == [3, 14], 'Unsupported bootstrap design')
    q = common['quantization_screen']
    require(q['median_relative_MSE_improvement_continuous_vs_binary_min'] == .01
            and q['positive_folds_min'] == 3 and q['equal_fold_MSE_block_lower_95_min'] == 0
            and q['mse_is_task_readability_not_general_information_loss'] is True, 'Unsupported quantization screen')
    screen = common['research_screen']
    required = {'prediction_median_relative_MSE_improvement_min': .01, 'matched_random_median_lift_bps_min': 1.,
                'positive_increment_folds_min': 3, 'information_equal_fold_primary_block_lower_95_bound_min': 0.,
                'economic_median_quarterly_increment_vs_ordinary_min_fraction': .01,
                'economic_positive_quarters_vs_cash_min': 3, 'economic_median_closed_trade_drawdown_change_max_fraction': 0.}
    require(all(screen[k] == v for k, v in required.items()), 'Unsupported information/economic screen')
    require(common['final_holdout_access'] == 'forbidden', 'Holdout must remain sealed')


def run(output):
    output = output.resolve()
    require(output.is_relative_to((ROOT / 'research/runs').resolve()) and output != (ROOT / 'research/runs').resolve(), 'Not a research run')
    registry = json.loads((ROOT / 'research/registry' / f'{output.name}.json').read_text(encoding='utf-8'))
    require(registry['status'] == 'engine_verified_pending_analysis', 'Official replay not accepted')
    require(not (output / 'analysis_report.json').exists() and not (output / 'provenance/analysis_source').exists(), 'Refuse analysis overwrite')
    inputs = {}
    for group in ('preparation_hashes', 'engine_hashes'):
        for relative, value in registry['artifacts'][group].items():
            path = (output / relative).resolve()
            require(path.is_relative_to(output) and sha(path) == value, 'Changed input: ' + relative)
            inputs[str(path)] = value
    common = yaml.safe_load((output / 'provenance/common_config.yaml').read_text(encoding='utf-8'))
    config = yaml.safe_load((output / 'provenance/experiment_config.yaml').read_text(encoding='utf-8'))
    require(sha(output / 'provenance/common_config.yaml') == registry['configuration']['sha256']
            and sha(output / 'provenance/experiment_config.yaml') == registry['parent_configuration']['sha256'], 'Protocol/config copy changed')
    validate_protocol(common)
    label_bundle = ROOT / config['labels_and_costs']['verified_label_bundle']
    require(sha(label_bundle / 'manifest.json') == registry['provenance']['label_manifest_sha256'], 'Verified label manifest changed')
    label_manifest = json.loads((label_bundle / 'manifest.json').read_text(encoding='utf-8'))
    for relative in ('labels_forbidden_as_features.csv', 'opportunities.csv', 'split_index.csv'):
        require(sha(label_bundle / relative) == label_manifest['artifact_sha256'][relative], 'Verified label artifact changed')
    labels, split, bundle = development_labels(config)
    for relative in ('split_index.csv', 'labels_forbidden_as_features.csv'):
        inputs[str(bundle / relative)] = sha(bundle / relative)
    source = output / 'provenance/analysis_source'
    source.mkdir()
    for path in (Path(__file__), ROOT / 'research/frozen/mlp_analyze.py', ROOT / 'research/baselines/summarize_e00r.py', ROOT / 'research/baselines/run_e00.py'):
        shutil.copy2(path, source / path.name)
    write_json(source / 'analysis_plan.json', {'protocol_sha256': registry['configuration']['sha256'],
               'primary_quantization_screen': common['quantization_screen'], 'information_comparators': ['ordinary19', 'matched_month_direction_counts'],
               'bootstrap_seed_derivation': 'base_seed17+fold_index*100; inherited unchanged from prior common workflow; four folds independently bootstrapped',
               'secondary_backbone_dimension_different': True, 'bootstrap_scope': 'independent_reference_utility_and_prediction_loss_not_compounded_portfolio'})
    new_predictions = pd.read_csv(output / 'development_test_predictions.csv')
    require(set(new_predictions.feature_set) == set(FAMILIES) and set(new_predictions.fold_id) == set(FOLDS), 'Unexpected prediction scope')
    base_predictions = pd.read_csv(output / 'baseline_reuse/development_test_predictions.csv')
    base_predictions = base_predictions.loc[base_predictions.feature_set.isin(['ridge19', 'ridge13'])]
    predictions = pd.concat([new_predictions, base_predictions], ignore_index=True)
    pm = prediction_summary(predictions, labels, split)
    require(len(pm) == 16, 'Expected two quant and two ordinary predictions per fold')
    pm.to_csv(output / 'prediction_metrics.csv', index=False)
    previous_predictions = pd.read_csv(registered_old('FROZEN_20261008_v1', 'prediction_metrics.csv', inputs))
    pp = []
    for fold in FOLDS:
        rows = pm.loc[pm.fold_id.eq(fold)].set_index('feature_variant')
        old = previous_predictions.loc[previous_predictions.fold_id.eq(fold)].set_index('feature_variant')
        u, q, ordinary = [float(rows.loc[name, 'MSE_bps_squared']) for name in (*FAMILIES, 'ridge19')]
        pp.append({'fold_id': fold, 'continuous_MSE_bps_squared': u, 'binary_MSE_bps_squared': q,
                   'ordinary19_MSE_bps_squared': ordinary, 'ordinary13_MSE_bps_squared': rows.loc['ridge13', 'MSE_bps_squared'],
                   'historical_pretrained531_MSE_bps_squared': old.loc['ridge_pretrained', 'MSE_bps_squared'],
                   'historical_random531_seed_mean_MSE_bps_squared': old.loc[[f'ridge_random_s{s}' for s in SEEDS], 'MSE_bps_squared'].mean(),
                   'relative_MSE_improvement_continuous_vs_binary': 1 - u / q,
                   'relative_MSE_improvement_continuous_vs_ordinary': 1 - u / ordinary,
                   'relative_MSE_improvement_binary_vs_ordinary': 1 - q / ordinary,
                   'historical_dimension_different_reference_only': True})
    pp = pd.DataFrame(pp)
    pp.to_csv(output / 'paired_prediction_metrics.csv', index=False)
    gates, reference, matched = {}, [], []
    origins = [(output, json.loads((output / 'signals_manifest.json').read_text(encoding='utf-8'))['signals']),
               (ROOT / common['variants']['E00R']['reuse_run'], json.loads((output / 'baseline_reuse/signals_manifest.json').read_text(encoding='utf-8'))['signals'])]
    expected = {(fold, f'{family}_{gate}') for fold in FOLDS for family in (*FAMILIES, 'ridge19', 'ridge13') for gate in GATES}
    for origin, signals in origins:
        for s in signals:
            fold, mode = s['fold_id'], s['mode']
            if origin != output and not mode.startswith(('ridge19_', 'ridge13_')):
                continue
            require((fold, mode) in expected and (fold, mode) not in gates, 'Unexpected or duplicate gate')
            ids = sorted(split.loc[split.fold_id.eq(fold) & split.role.eq('test'), 'opportunity_id'])
            path = (origin / s['path']).resolve()
            require(path.is_relative_to(origin.resolve()) and sha(path) == s['sha256'], 'Sidecar path/hash changed')
            rows = json.loads(path.read_text(encoding='utf-8'))['decisions']
            require([r['signal_at'] for r in rows] == ids and all(set(r) == {'signal_at', 'direction', 'exposure_fraction'} for r in rows), 'Gate IDs/schema differ')
            actions = np.asarray([r['exposure_fraction'] for r in rows], float)
            direction = labels.loc[ids, 'direction'].to_numpy()
            require(np.isin(actions, [0, 1]).all() and int(actions.sum()) == s['selected_count']
                    and np.array_equal(direction, [r['direction'] for r in rows]), 'Gate action/direction mismatch')
            gates[(fold, mode)] = (ids, actions)
            inputs[str(path)] = sha(path)
            entry = labels.loc[ids, 'entry_at'].tolist()
            weights = stratum_expected_weights(entry, direction, actions)
            for cost in ('base', 'stress'):
                y = labels.loc[ids, cost + '_profit_ratio'].to_numpy() * 10000
                reference.append({'fold_id': fold, 'mode': mode, 'cost': cost, 'opportunities': len(ids),
                    'selected_count': int(actions.sum()), 'participation': actions.mean(),
                    'selected_mean_net_bps': float(y[actions > 0].mean()) if actions.any() else None,
                    'gated_net_bps_per_eligible_opportunity': float(np.mean(y * actions)),
                    'metric_basis': 'independent_official_reference_net_labels_not_portfolio'})
                if mode.startswith(FAMILIES):
                    result = matched_random(y, actions > 0, entry, direction)
                    null_mean = float(y @ weights / actions.sum()) if actions.any() else None
                    matched.append({'fold_id': fold, 'mode': mode, 'cost': cost, **result, 'exact_null_mean_bps': null_mean,
                        'exact_selection_lift_bps': float(y @ actions / actions.sum() - null_mean) if actions.any() else None})
    require(set(gates) == expected, 'Missing quant/ordinary gates')
    reference, matched = pd.DataFrame(reference), pd.DataFrame(matched)
    reference.to_csv(output / 'reference_gate_metrics.csv', index=False)
    matched.to_csv(output / 'matched_random_summary.csv', index=False)
    boots, daily_rows, draws = [], [], {}
    fold_lookup = {f['id']: f for f in config['splits']['walk_forward_dates']}
    for fi, fold in enumerate(FOLDS):
        spec = fold_lookup[fold]
        ids = gates[(fold, FAMILIES[0] + '_rank50')][0]
        days = len(pd.date_range(spec['test'][0], spec['test'][1], freq='D', inclusive='left'))
        scores = {name: predictions.loc[predictions.fold_id.eq(fold) & predictions.feature_set.eq(name)].set_index('opportunity_id').loc[ids, 'score_bps'].to_numpy()
                  for name in (*FAMILIES, 'ridge19')}
        for cost in ('base', 'stress'):
            y = labels.loc[ids, cost + '_profit_ratio'].to_numpy() * 10000
            series = {}
            for family in FAMILIES:
                a = gates[(fold, family + '_rank50')][1]
                ordinary = gates[(fold, 'ridge19_rank50')][1]
                weights = stratum_expected_weights(labels.loc[ids, 'entry_at'].tolist(), labels.loc[ids, 'direction'].to_numpy(), a)
                e = gates[(fold, family + '_economic')][1]
                oe = gates[(fold, 'ridge19_economic')][1]
                for name, values in [('rank50_vs_ordinary', (a - ordinary) * y), ('rank50_vs_matched_counts', (a - weights) * y),
                                     ('economic_vs_cash', e * y), ('economic_vs_ordinary', (e - oe) * y)]:
                    series[family + '_' + name] = (values, 'independent_reference_utility_bps_per_calendar_day', 1.)
                if cost == 'base':
                    series[family + '_MSE_vs_ordinary'] = ((scores['ridge19'] - y) ** 2 - (scores[family] - y) ** 2,
                                                        'MSE_difference_bps_squared', days / len(ids))
            if cost == 'base':
                series['continuous_vs_binary_MSE'] = ((scores[FAMILIES[1]] - y) ** 2 - (scores[FAMILIES[0]] - y) ** 2,
                                                     'MSE_difference_bps_squared', days / len(ids))
            for comparison, (values, unit, scale) in series.items():
                calendar = daily(values, ids, spec)
                require(len(calendar) == days and np.isclose(calendar.sum(), np.sum(values)), 'Complete paired calendar lost observations')
                daily_rows.extend({'fold_id': fold, 'comparison': comparison, 'cost': cost, 'at': at.isoformat(),
                    'value': float(v), 'scale_from_daily_mean': scale, 'reported_unit': unit} for at, v in calendar.items())
                for block in (7, 3, 14):
                    sample = stationary_means(calendar.to_numpy(), block, draws=5000, seed=17 + fi * 100) * scale
                    draws.setdefault((comparison, cost, block, unit), []).append(sample)
                    boots.append({'fold_id': fold, 'comparison': comparison, 'cost': cost, 'mean_block_days': block,
                        'point_difference': calendar.mean() * scale, 'lower_95': np.quantile(sample, .025), 'upper_95': np.quantile(sample, .975),
                        'unit': unit, 'scope': 'reference_or_prediction_not_compounded_portfolio', 'calendar_days': days,
                        'effective_blocks_approx': days / block, 'draws': 5000, 'seed': 17 + fi * 100})
    for (comparison, cost, block, unit), parts in draws.items():
        require(len(parts) == 4, 'All four folds required')
        sample = np.mean(parts, axis=0)
        points = [r['point_difference'] for r in boots if (r['comparison'], r['cost'], r['mean_block_days']) == (comparison, cost, block)]
        boots.append({'fold_id': 'EQUAL_FOLD_MEAN', 'comparison': comparison, 'cost': cost, 'mean_block_days': block,
            'point_difference': np.mean(points), 'lower_95': np.quantile(sample, .025), 'upper_95': np.quantile(sample, .975),
            'unit': unit, 'scope': 'reference_or_prediction_not_compounded_portfolio', 'draws': 5000, 'seed': '17+fold_index*100'})
    bootstrap = pd.DataFrame(boots)
    bootstrap.to_csv(output / 'paired_block_bootstrap.csv', index=False)
    pd.DataFrame(daily_rows).to_csv(output / 'paired_daily_series.csv', index=False)
    metrics = pd.read_csv(output / 'metrics.csv')
    expected_metrics = {(f, family + '_' + gate, c) for f in FOLDS for family in FAMILIES for gate in GATES for c in ('base', 'stress')}
    require(len(metrics) == 64 and set(metrics[['fold_id', 'mode', 'cost']].itertuples(index=False, name=None)) == expected_metrics, 'New export scope differs')
    combined = [metrics.assign(source_run=output.name, comparison_role='same_coordinate_quantization_diagnostic')]
    for run_id in HISTORICAL:
        frame = pd.read_csv(registered_old(run_id, 'metrics.csv', inputs))
        require(set(frame.fold_id) == set(FOLDS) and not frame.duplicated(['fold_id', 'mode', 'cost']).any(), 'Historical metric scope differs')
        frame.loc[frame['mode'].eq('ordinary_features_gate'), 'mode'] = 'legacy_E00_logistic_gate'
        combined.append(frame.assign(source_run=run_id, comparison_role='ordinary19_primary_comparator' if run_id.startswith('E00R_') else 'secondary_dimension_or_head_reference'))
    combined = pd.concat(combined, ignore_index=True)
    require(len(combined) == 512 and not combined.duplicated(['fold_id', 'mode', 'cost']).any(), 'Combined official scope differs')
    combined.to_csv(output / 'all_portfolio_metrics.csv', index=False)
    paired, concentration = [], []
    for fold in FOLDS:
        for cost in ('base', 'stress'):
            rows = combined.loc[combined.fold_id.eq(fold) & combined.cost.eq(cost)].set_index('mode')
            for family in FAMILIES:
                for gate in ('rank50', 'economic'):
                    quant, ordinary = rows.loc[family + '_' + gate], rows.loc['ridge19_' + gate]
                    paired.append({'fold_id': fold, 'family': family, 'cost': cost, 'gate': gate,
                        'quant_return_pct': quant.net_return_pct, 'ordinary_return_pct': ordinary.net_return_pct,
                        'increment_vs_ordinary_pp': quant.net_return_pct - ordinary.net_return_pct,
                        'closed_drawdown_change_vs_ordinary_pp': quant.closed_trade_max_drawdown_pct - ordinary.closed_trade_max_drawdown_pct,
                        'quant_trades': quant.trades, 'ordinary_trades': ordinary.trades})
    paired = pd.DataFrame(paired)
    paired.to_csv(output / 'paired_portfolio_metrics.csv', index=False)
    for row in metrics.itertuples(index=False):
        require(sha(row.archive) == row.archive_sha256, 'Official archive changed')
        trades = read_result(Path(row.archive))['trades']
        profit = np.asarray([t['profit_abs'] for t in trades], float)
        positive, top = profit[profit > 0], max(1, int(np.ceil(len(profit) * .05)))
        turnover = sum(float(t['amount']) * (float(t['open_rate']) + float(t['close_rate'])) for t in trades)
        concentration.append({'fold_id': row.fold_id, 'mode': row.mode, 'cost': row.cost, 'trades': len(trades),
            'positive_profit_top5pct_trade_count_share': np.sort(positive)[-top:].sum() / positive.sum() if positive.sum() else None,
            'largest_profit_USDT': profit.max() if len(profit) else None, 'largest_loss_USDT': profit.min() if len(profit) else None,
            'gross_entry_exit_notional_USDT': turnover, 'turnover_over_initial_wallet': turnover / 10000})
    pd.DataFrame(concentration).to_csv(output / 'portfolio_concentration_turnover.csv', index=False)
    primary = bootstrap.loc[bootstrap.fold_id.eq('EQUAL_FOLD_MEAN') & bootstrap.cost.eq('base') & bootstrap.mean_block_days.eq(7)].set_index('comparison')
    q_checks = {'median_relative_MSE_continuous_vs_binary_ge_1pct': pp.relative_MSE_improvement_continuous_vs_binary.median() >= .01,
                'positive_MSE_improvement_folds_ge_3': int((pp.relative_MSE_improvement_continuous_vs_binary > 0).sum()) >= 3,
                'equal_fold_7day_MSE_difference_lower_95_positive': primary.loc['continuous_vs_binary_MSE', 'lower_95'] > 0}
    information, economic = {}, {}
    for family, short in zip(FAMILIES, ('continuous', 'binary')):
        lift = matched.loc[matched.cost.eq('base') & matched['mode'].eq(family + '_rank50')]
        comparisons = [family + '_rank50_vs_ordinary', family + '_rank50_vs_matched_counts']
        improvement = pp['relative_MSE_improvement_' + short + '_vs_ordinary']
        checks = {'median_relative_MSE_vs_ordinary_ge_1pct': improvement.median() >= .01,
                  'positive_prediction_improvement_folds_ge_3': int((improvement > 0).sum()) >= 3,
                  'median_matched_selection_lift_ge_1bps': lift.exact_selection_lift_bps.median() >= 1,
                  'positive_matched_lift_folds_ge_3': int((lift.exact_selection_lift_bps > 0).sum()) >= 3,
                  'paired_primary_95_lower_positive_all_comparators': bool((primary.loc[comparisons, 'lower_95'] > 0).all()),
                  'positive_paired_utility_folds_ge_3_all_comparators': all(int((bootstrap.loc[bootstrap.fold_id.isin(FOLDS) & bootstrap.cost.eq('base')
                      & bootstrap.mean_block_days.eq(7) & bootstrap.comparison.eq(c), 'point_difference'] > 0).sum()) >= 3 for c in comparisons)}
        information[family] = {'checks': checks, 'passed': all(checks.values())}
        economic[family] = {}
        for gate in ('rank50', 'economic'):
            b = paired.loc[paired.family.eq(family) & paired.gate.eq(gate) & paired.cost.eq('base')]
            s = paired.loc[paired.family.eq(family) & paired.gate.eq(gate) & paired.cost.eq('stress')]
            checks = {'positive_base_increment_folds_ge_3': int((b.increment_vs_ordinary_pp > 0).sum()) >= 3,
                      'median_base_increment_ge_1pp': b.increment_vs_ordinary_pp.median() >= 1,
                      'positive_base_quarters_vs_cash_ge_3': int((b.quant_return_pct > 0).sum()) >= 3,
                      'stress_median_return_positive': s.quant_return_pct.median() > 0,
                      'stress_median_increment_positive': s.increment_vs_ordinary_pp.median() > 0,
                      'median_closed_drawdown_not_worse': b.closed_drawdown_change_vs_ordinary_pp.median() <= 0}
            economic[family][gate] = {'checks': checks, 'passed': all(checks.values())}
    passed = all(q_checks.values()) or any(x['passed'] for x in information.values()) or any(v['passed'] for x in economic.values() for v in x.values())
    report = {'status': 'completed_exploratory_development_analysis', 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'experiment': output.name, 'quantization_screen': {'checks': q_checks, 'passed': all(q_checks.values())},
        'information_screens': information, 'economic_screens': economic,
        'conclusion': 'support_further_research' if passed else 'insufficient_evidence', 'automatic_promotion': False,
        'final_holdout_inspected': False, 'backbone_or_tokenizer_trained': False, 'protocol_sha256': registry['configuration']['sha256'],
        'new_official_exports': 64, 'new_Ridge_fits': 32, 'prediction_metrics_rows': 16, 'metrics_rows': len(combined),
        'economic_participation_counts': {family: {f: int(gates[(f, family + '_economic')][1].sum()) for f in FOLDS} for family in FAMILIES},
        'input_sha256': inputs, 'historical_random_backbone_is_hard_screen_comparator': False,
        'limitations': ['Task-specific same-coordinate continuous versus binary Ridge readability; not general information loss.',
            'Previously seen development folds and checkpoint cutoff unproven; no strict ex-ante OOS claim.',
            'Historical backbone512 plus ordinary19 differs in dimension and downstream path; descriptive reference only, no causal attribution.',
            'Stationary bootstrap covers independent reference utility and prediction loss, never compounded portfolio return.',
            'Complete UTC decision-date calendars retain zero-opportunity days; equal weighting of independently bootstrapped folds.',
            'Month x direction matched-count null is posthoc diagnostic, not executable policy or portfolio confidence interval.',
            'Sparse or wide-interval evidence stays insufficient; no threshold or capacity extension.',
            'Quarter wallets independently reset; closed-trade equity excludes unrealized intrahour exposure risk.',
            'Replication requires separately registered independent data; no automatic holdout, training or deployment permission.']}
    write_json(output / 'analysis_report.json', clean(report))
    print(json.dumps(clean(report), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    run(parser.parse_args().output_dir)
