"""Plot completed M2 DEV metrics only; no training, model selection or old test reads."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'research/runs/M2_CONDITIONAL_INCREMENT_01'
BASE = ['B5','B5_s17','B5_s29','B5_s43','R1','R2','B2','har','ewma','persistence','constant_RV']
ALPHAS = [0,.1,.25,.5,.75,1]
NOTICE = 'Development OOF: reused architecture; selected-alpha results conditional'
UNITS = '4h RV: squared natural-log return; QLIKE / Regret: dimensionless'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''): h.update(block)
    return h.hexdigest()


def exclusive(path, value):
    with Path(path).open('xb') as f: f.write(value)


def csv_output(path, table, source, metric_unit):
    table = table.copy()
    table['metric_unit'] = metric_unit
    table['RV_unit'] = '4h_squared_natural_log_return'
    table['evidence_class'] = NOTICE
    table['source_file'] = source.relative_to(ROOT).as_posix()
    table['source_file_sha256'] = sha(source)
    exclusive(path, table.to_csv(index=False,float_format='%.17g').encode())


def save_plot(fig, path):
    import matplotlib.pyplot as plt
    for fmt in ('png','svg'):
        buf = io.BytesIO()
        fig.savefig(buf,format=fmt,dpi=170,bbox_inches='tight')
        exclusive(path.with_suffix('.'+fmt),buf.getvalue())
    plt.close(fig)


def decorate(fig, axes, periods, title, ylabel):
    import matplotlib.dates as mdates
    fig.suptitle(title+'\n'+NOTICE,fontsize=14)
    for ax,(fold,start,end) in zip(axes,periods):
        ax.set_title(f'{fold}: {start:%Y-%m-%d} to {end:%Y-%m-%d} (exclusive)',fontsize=10)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('UTC time')
        ax.set_xlim(start.to_pydatetime(),end.to_pydatetime())
        locator=mdates.AutoDateLocator(minticks=3,maxticks=5)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        ax.grid(alpha=.25)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.035),ncol=4,fontsize=9)
    fig.text(.5,.008,UNITS+'; folds plotted separately, no equity curve',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.12,1,.94))


def make_charts(output_tag=None):
    # This module only opens new-run metric artifacts and its sealed configuration.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    config_path=ROOT/'research/m2_ci/config.yaml'
    config=yaml.safe_load(config_path.read_text(encoding='utf8'))
    seal=json.loads((RUN/'preflight/source_seal.json').read_text(encoding='utf8'))
    if seal['status']!='PASS' or sha(config_path)!=seal['protocol_sha256']:
        raise ValueError('Current sealed configuration required')
    periods=[(f['id'],pd.Timestamp(f['evaluation_start']),pd.Timestamp(f['evaluation_end'])) for f in config['roles']['folds']]
    periods += [('VALID',pd.Timestamp(config['roles']['original_train_end']),pd.Timestamp(config['roles']['original_validation_end']))]
    source=RUN/'metrics/row_metrics.csv'
    daily_source=RUN/'metrics/daily_paired_gains.csv'
    point_source=RUN/'metrics/point_metrics.csv'
    tail_source=RUN/'metrics/tail_metrics.csv'
    for path in (source,daily_source,point_source,tail_source,RUN/'metrics/summary.json'):
        if not path.is_file(): raise FileNotFoundError('Completed real metrics required: '+str(path))
    if output_tag and (not output_tag.replace('_','').replace('-','').isalnum()):
        raise ValueError('Output tag must contain letters, digits, hyphens or underscores')
    output=RUN/'figures'
    if output_tag: output=output/output_tag
    if output.exists() and any(output.iterdir()): raise FileExistsError('Preserve existing figures; use a new explicit output tag')
    row=pd.read_csv(source,float_precision='round_trip')
    row['decision_at']=pd.to_datetime(row.decision_at,utc=True)
    families=BASE+['mix_'+x for x in ('R1','B2','R2')]+[f'candidate_{f}_a{a:g}' for f in ('R1','B2','R2') for a in ALPHAS]
    if set(row.family)!=set(families) or set(row.fold_id)!={p[0] for p in periods}:
        raise ValueError('Expected all32 families andsix development periods')
    if row.duplicated(['opportunity_id','family']).any() or not np.isfinite(row.QLIKE_Regret).all():
        raise ValueError('Invalid real row metrics')
    if row.decision_at.max()>=pd.Timestamp('2026-04-01T00:00:00Z'):
        raise PermissionError('Consumed old test values forbidden')
    records=[]
    for fold,start,end in periods:
        section=row[row.fold_id.eq(fold)]
        if not ((section.decision_at>=start)&(section.decision_at<end)).all(): raise ValueError('Fold clock mismatch')
        counts=section.groupby('family').size()
        if len(set(counts))!=1: raise ValueError('Families use unequal opportunities')
        calendar=pd.date_range(start,end,freq='D',inclusive='left')
        for family in families:
            values=section[section.family.eq(family)].copy()
            values['day']=values.decision_at.dt.floor('D')
            daily=values.groupby('day').QLIKE_Regret.agg(['sum','count']).reindex(calendar,fill_value=0)
            total=daily['sum'].rolling(7,min_periods=1).sum()
            n=daily['count'].rolling(7,min_periods=1).sum()
            records.append(pd.DataFrame({'fold_id':fold,'family':family,'day_utc':calendar,
                'period_start':start.isoformat(),'period_end_exclusive':end.isoformat(),'period_N':len(values),
                'day_N':daily['count'].to_numpy(),'trailing_7calendar_days_N':n.to_numpy(),
                'trailing_7calendar_days_Regret_sum':total.to_numpy(),
                'trailing_7calendar_days_Regret_mean':(total/n.replace(0,np.nan)).to_numpy()}))
    rolling=pd.concat(records,ignore_index=True)
    daily=pd.read_csv(daily_source,float_precision='round_trip')
    daily['day_utc']=pd.to_datetime(daily.day_utc,utc=True)
    point=pd.read_csv(point_source,float_precision='round_trip')
    tail=pd.read_csv(tail_source,float_precision='round_trip')
    output.mkdir(parents=True,exist_ok=True)
    csv_output(output/'rolling_Regret_all32.csv',rolling,source,'dimensionless_mean_loss')
    groups={'primary':['B5','R2','R1','B2','mix_R2','mix_R1','mix_B2'],
            'historical_baselines':['B5','har','ewma','persistence','constant_RV'],
            'seed_models':['B5','B5_s17','B5_s29','B5_s43']}
    groups.update({f'alpha_{f}':[f'candidate_{f}_a{a:g}' for a in ALPHAS] for f in ('R1','B2','R2')})
    for name, models in groups.items():
        fig,axs=plt.subplots(3,2,figsize=(15,11));axes=axs.ravel()
        for ax,(fold,_,_) in zip(axes,periods):
            for family in models:
                values=rolling[rolling.fold_id.eq(fold)&rolling.family.eq(family)]
                ax.plot(values.day_utc,values.trailing_7calendar_days_Regret_mean,label=family,lw=1.2)
        decorate(fig,axes,periods,'7-calendar-day opportunity-weighted QLIKE Regret: '+name,'Mean Regret')
        save_plot(fig,output/('rolling_Regret_'+name))
    contrasts=['B5_minus_mix_R2','B5_minus_mix_R1','B5_minus_mix_B2']
    selected_daily=daily[daily.contrast.isin(contrasts)].copy()
    if selected_daily.empty: raise ValueError('Real selected fusion daily gains missing')
    csv_output(output/'selected_daily_gains.csv',selected_daily,daily_source,'dimensionless_loss_sum')
    for metric in ['gain_sum','cumulative_gain_sum']:
        fig,axs=plt.subplots(3,2,figsize=(15,11));axes=axs.ravel()
        for ax,(fold,_,_) in zip(axes,periods):
            for contrast in contrasts:
                values=selected_daily[selected_daily.fold_id.eq(fold)&selected_daily.contrast.eq(contrast)].sort_values('day_utc')
                ax.plot(values.day_utc,values[metric],label=contrast,lw=1.2)
            ax.axhline(0,color='black',lw=.7,alpha=.5)
        decorate(fig,axes,periods,'Selected fusion paired loss gains: '+metric+' (positive favors fusion)',metric)
        save_plot(fig,output/metric)
    midpoint={fold:start+(end-start)/2 for fold,start,end in periods}
    diag=point[point.family.isin(['B5','R2','R1','B2','constant_RV','ewma'])].copy()
    diag['period_midpoint_utc']=diag.fold_id.map(midpoint)
    csv_output(output/'ranking_denominator_diagnostic.csv',diag,point_source,'AUROC_or_AP_fraction')
    fig,axs=plt.subplots(1,2,figsize=(15,5))
    for ax,metric in zip(axs,['surprise_AUROC','absolute_q90_AUROC']):
        for family,values in diag.groupby('family',sort=False):
            values=values.sort_values('period_midpoint_utc');ax.plot(values.period_midpoint_utc,values[metric],marker='o',label=family)
        ax.set_title(metric);ax.set_ylim(0,1);ax.set_ylabel('AUROC');ax.set_xlabel('UTC period midpoint');ax.grid(alpha=.25)
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.suptitle('Constant RV denominator diagnostic\n'+NOTICE)
    fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.075),ncol=6)
    fig.text(.5,.018,'Lines connect period summaries only; no continuous forecast path. '+UNITS,ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.21,1,.9));save_plot(fig,output/'ranking_denominator_diagnostic')
    taildiag=tail[tail.family.isin(groups['primary']+['constant_RV'])].copy()
    taildiag['period_midpoint_utc']=taildiag.fold_id.map(midpoint)
    csv_output(output/'FIT_fixed_tail_underprediction.csv',taildiag,tail_source,'underprediction_fraction')
    fig,axs=plt.subplots(1,2,figsize=(15,5))
    for ax,subset in zip(axs,['q90','q99']):
        for family,values in taildiag[taildiag.subset.eq(subset)].groupby('family',sort=False):
            values=values.sort_values('period_midpoint_utc');ax.plot(values.period_midpoint_utc,values.under_fraction,marker='o',label=family)
        ax.set_title('FIT-fixed '+subset+': prediction / effective RV < 0.5');ax.set_ylim(0,1)
        ax.set_ylabel('Underprediction fraction');ax.set_xlabel('UTC period midpoint');ax.grid(alpha=.25)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.suptitle('Tail underprediction: each period uses its FIT threshold\n'+NOTICE)
    fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.075),ncol=4,fontsize=8)
    fig.text(.5,.018,'Subset N and thresholds in CSV; empty subsets remain missing. '+UNITS,ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.25,1,.9));save_plot(fig,output/'FIT_fixed_tail_underprediction')
    resource=RUN/'metrics/resource_benchmark.csv'
    if resource.exists(): csv_output(output/'resource_measurements.csv',pd.read_csv(resource,float_precision='round_trip'),resource,'column_specific_seconds_bytes_parameters')
    artifacts={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(output.iterdir()) if p.is_file()}
    manifest={'status':'PASS','purpose':'Real DEV metrics visualization only','configuration_sha256':sha(config_path),
        'sources_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in (source,daily_source,point_source,tail_source,Path(__file__))},
        'files_sha256':artifacts,'all32_families_rolling_csv':True,'periods':[{ 'id':f,'start':str(s),'end_exclusive':str(e)} for f,s,e in periods],
        'notice':NOTICE,'units':UNITS,'no_equity_or_economic_claims':True,'no_old_consumed_test_reads':True}
    exclusive(output/'manifest.json',(json.dumps(manifest,indent=2)+'\n').encode())
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-tag',help='Explicit new subdirectory for a retained failed or alternative rendering attempt')
    args=parser.parse_args()
    result=make_charts(args.output_tag)
    print(json.dumps({'status':result['status'],'artifacts':len(result['files_sha256'])}))


if __name__=='__main__': main()
