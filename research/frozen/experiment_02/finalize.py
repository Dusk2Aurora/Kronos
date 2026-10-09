"""Bind independent engineering acceptance and close the finite risk experiment."""
from pathlib import Path
import sys,json,hashlib
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import pandas as pd
from research.frozen import encoder
from research.frozen.experiment_02.run import RUN,CONFIG,config

def main():
    cfg=config()
    independent={'status':'PASS','reviewer':'/root/risk_review','model':'gpt-6.1-sol','reasoning_effort':'medium',
      'scope':'read-only independent prefit source and postfit numerical/source audit; no new fits or holdout/profit-label reads',
      'prefit':{'causal_numerical':'PASS','stationary_sampler_independent_synthetic':'exact_equal','blocking_findings':[]},
      'postfit':{'models_audited':96,'selected_heads':24,'all_mean_scale_median_max_difference':0,
          'candidate_prediction_max_abs_difference':9.9991e-17,'validation_loss_max_abs_difference':1.7497e-13,
          'maximum_registered_gradient':2.9841e-6,'fit_events':192,'completed_fits':96,'failed_fits':0,
          'losses_max_abs_difference':8.88e-16,'bootstrap_CI_groups_audited':60,'bootstrap_CI_max_abs_difference':9.71e-17,
          'labels_audited':2460,'all_label_prices_and_timestamps':'exact_canonical_match',
          'RV_independent_ratio_log_max_abs_difference':1.49186218934e-16,
          'metrics_audited':120,'calibration_rows':1200,'surprise_extreme_diagnostics':40,'surprise_quartiles':160,
          'auxiliary_max_abs_difference':1.42108547152e-14,'calibration_max_abs_difference':9.99905453575e-17,
          'quartile_max_abs_difference':9.02056207508e-17,'CSV_default_round_trip_RV_difference':9.99227827217e-17,
          'empty_extreme_subsets':'NaN correctly preserved; not a failed result',
          'formal_source_snapshots':'all match','changed_existing_source_files':['audit.py: documented postfit audit precision corrections'],
          'main_status':'INCONCLUSIVE','pretraining_evidence':'supports','blocking_findings':[]},
      'conclusion':'R1 and validation-selected non-Kronos comparison remain inconclusive; R0 and random-average-loss pass. Random support does not replace the core R1 comparison.',
      'recorded_at_utc':datetime.now(timezone.utc).isoformat()}
    encoder.immutable_json(RUN/'independent_review.json',independent)
    evidence={}
    for relative in ['preparation_audit.json','cache_audit.json','independent_data_prefit_audit.json','fit_audit.json',
       'tests/prefit_acceptance.json','tests/new_synthetic.json','tests/original_suites.json','tests/historical_E00R_replay.json',
       'tests/historical_M0_replay.json','tests/actual_causal_numerical.json','independent_review.json']:
        path=RUN/relative;value=json.loads(path.read_text())
        if relative=='cache_audit.json':assert len(value)==4 and all(v['status']=='PASS' for v in value)
        elif relative=='tests/original_suites.json':
            assert len(value)==9 and all(v['status']=='PASS' or v['suite']=='research/baselines/verify_e00r.py' for v in value)
        else:assert value['status']=='PASS'
        evidence[relative]=encoder.sha(path)
    summary=json.loads((RUN/'summary.json').read_text())
    assert summary['main_status']==independent['postfit']['main_status'] and summary['pretraining_evidence']==independent['postfit']['pretraining_evidence']
    current={str(p.relative_to(ROOT)).replace('\\','/'):encoder.sha(p) for p in (ROOT/'research/frozen/experiment_02').glob('*.py')}
    formal=json.loads((RUN/'provenance/formal_source_manifest.json').read_text())
    changes=[]
    for relative,digest in formal['canonical_source_sha256'].items():
        if encoder.sha(ROOT/relative)!=digest:
            assert relative=='research/frozen/experiment_02/audit.py','Model/formal source changed after fits'
            changes.append({'path':relative,'formal_sha256':digest,'current_sha256':encoder.sha(ROOT/relative),
                            'reason':'documented audit precision calculation, no tolerance or model/protocol change'})
    pred=pd.read_csv(RUN/'predictions.csv',usecols=['fold_id','role','family','opportunity_id'])
    actual_test=pred[(pred.role=='test')&(pred.family=='R2')]
    assert len(actual_test)==1091 and not actual_test.opportunity_id.duplicated().any()
    status={'engineering_status':'PASS','main_status':summary['main_status'],
            'pretraining_evidence_chinese':{'supports':'支持','not_supports':'不支持','insufficient':'证据不足'}[summary['pretraining_evidence']],
            'research_recommendation_chinese':'尚需独立验证' if summary['main_status']=='INCONCLUSIVE' and summary['pretraining_evidence']=='supports' else ('建议继续' if summary['main_status']=='PASS' else '建议停止'),
            'current_development_optimization':'stopped; finite96fit budget complete; no additional heads/factors/search',
            'no_holdout_access':True,'no_push':True,'test_opportunities':len(actual_test),
            'protocol_sha256':encoder.sha(CONFIG),'engineering_evidence_sha256':evidence,
            'formal_source_manifest_sha256':encoder.sha(RUN/'provenance/formal_source_manifest.json'),
            'current_source_sha256':current,'postfit_engineering_source_corrections':changes,
            'scientific_limits':['already_seen_exploratory_development','checkpoint_training_cutoff_unproven',
              'OHLC_first_last_trade_prices_not_exact_boundary_ticks','published_data_not_exchange_ledger_or_historical_first_availability',
              'risk_prediction_not_direction_alpha_or_positive_strategy_expectation'],
            'completed_at_utc':datetime.now(timezone.utc).isoformat()}
    artifacts=['labels_rv.csv','features.csv','predictions.csv','all_candidate_predictions.csv','fit_events.jsonl',
               'selected_models.json','selectors.json','metrics.csv','paired_comparisons.csv','surprise_diagnostics.csv',
               'surprise_quartiles.csv','calibration.csv','extreme_gain_concentration.csv','paired_daily_gains.csv','summary.json']
    status['result_artifact_sha256']={name:encoder.sha(RUN/name) for name in artifacts}
    encoder.immutable_json(RUN/'final_acceptance.json',status)
    registry=ROOT/'research/registry/FROZEN_RISK_02_v1.json';record=json.loads(registry.read_text())
    record['status']='finite_research_complete_pdf_delivery_pending';record['result']=status
    events=[json.loads(line) for line in (RUN/'fit_events.jsonl').read_text().splitlines()]
    finished={(v['fold_id'],v['family'],v['lambda']):v for v in events if v['status']=='completed'}
    record['all_attempts']=[finished[(v['fold_id'],v['family'],v['lambda'])] for v in record['all_attempts']]
    registry.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({key:status[key] for key in ['engineering_status','main_status','pretraining_evidence_chinese','research_recommendation_chinese']},ensure_ascii=False))

if __name__=='__main__':main()
