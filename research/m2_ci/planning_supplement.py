"""Post hoc observed-effect detectability scenarios; no new experiment or threshold."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'research/runs/M2_CONDITIONAL_INCREMENT_01'
MONTHS=[3,6,9,12,18,24]
CAUTION=('Post hoc selected small observed effect; conditional and optimistic normal approximation. '
         'Future stationarity and observed-effect persistence are unproven. '
         'This is neither advice to wait, a new experiment, nor a revised effect threshold. '
         'Two-sided power is detectability of the signed contrast, not evidence of beneficial R2 increment.')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def produce():
    outputs=[RUN/'metrics/future_observed_gain_scenarios.csv',RUN/'metrics/future_observed_gain_scenarios.json']
    if any(path.exists() for path in outputs): raise FileExistsError('Preserve existing observed-effect supplement')
    summary_path=RUN/'metrics/summary.json'; interval_path=RUN/'metrics/paired_intervals.csv'
    summary=json.loads(summary_path.read_text(encoding='utf-8')); intervals=pd.read_csv(interval_path,float_precision='round_trip')
    if summary['future_formal_test_approved'] is not False or summary['old_holdout_status']!='CONSUMED': raise ValueError('Future testing must remain unapproved; old holdout CONSUMED')
    rows=[]; z=norm.ppf(.975); z80=norm.ppf(.8); rate=365.25/12*3
    for contrast in ('B5_minus_mix_R2','R2_minus_B5'):
        selected=intervals[intervals.period.eq('OOF')&intervals.block_days.eq(7)&intervals.contrast.eq(contrast)]
        if len(selected)!=1: raise ValueError('Expected one OOF7day contrast: '+contrast)
        source=selected.iloc[0]; gain=float(source.point_gain); se=float(source.bootstrap_SE); n=int(source.N)
        applicable=np.isfinite(se) and se>0 and np.isfinite(gain) and gain!=0
        if contrast=='B5_minus_mix_R2': applicable=applicable and summary['screens']['R2']['alpha']>0
        n80=float(n*((z+z80)*se/abs(gain))**2) if applicable else np.nan
        months80=n80/rate if applicable else np.nan
        for months in MONTHS:
            expected_n=months*rate; future_se=se*np.sqrt(n/expected_n) if applicable else np.nan
            delta=gain/future_se if applicable else np.nan
            power=float(norm.cdf(delta-z)+norm.cdf(-delta-z)) if applicable else np.nan
            rows.append({'contrast':contrast,'months':months,'N_reference':n,
                'observed_point_gain':gain,'paired_bootstrap_SE_7day':se,
                'expected_opportunities_approx':expected_n,'planning_SE_7day':future_se,
                'power_at_observed_signed_effect_two_sided95':power,
                'nominal_N80_at_observed_effect':n80,'nominal_months80_at_observed_effect':months80,
                'planning_status':'POST_HOC_OBSERVED_EFFECT_SCENARIO' if applicable else 'NOT_APPLICABLE_ZERO_EFFECT_OR_DEGENERATE',
                'positive_contrast_means':'R2_mix_lower_loss_than_B5' if contrast=='B5_minus_mix_R2' else 'direct_R2_higher_loss_than_B5',
                'selection_conditioned_optimistic':True,'future_stationarity_unproven':True})
    frame=pd.DataFrame(rows)
    with outputs[0].open('x',encoding='utf-8',newline='') as stream: frame.to_csv(stream,index=False,float_format='%.17g')
    def clean(value):
        if isinstance(value,dict): return {k:clean(v) for k,v in value.items()}
        if isinstance(value,list): return [clean(v) for v in value]
        if isinstance(value,float) and not np.isfinite(value): return None
        return value
    result={'schema_version':1,'experiment_id':'M2_CONDITIONAL_INCREMENT_01','status':'COMPLETE',
        'at_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':sha(__file__),
        'input_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in (summary_path,interval_path)},
        'csv_path':outputs[0].relative_to(ROOT).as_posix(),'csv_sha256':sha(outputs[0]),
        'rows':clean(rows),'caution':CAUTION,'future_formal_test_approved':False,
        'changed_selection_models_or_thresholds':False,'new_fits':0}
    with outputs[1].open('x',encoding='utf-8') as stream: json.dump(result,stream,indent=2,allow_nan=False); stream.write('\n')
    print(json.dumps({'status':'COMPLETE','rows':len(rows),'csv_sha256':result['csv_sha256']}))
    return result


if __name__=='__main__': produce()
