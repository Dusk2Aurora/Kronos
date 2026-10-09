"""Independent publication checks; reads saved generated results only."""
from pathlib import Path
import sys,json,hashlib,argparse
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score,average_precision_score
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'research/runs/FROZEN_RISK_03_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def csv(p):return pd.read_csv(p,float_precision='round_trip')
def write(p,v):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def check_aux():
 assert read(RUN/'authorization/formal_terminal.json')['state']=='CONSUMED'
 out=RUN/'metrics/publication_auxiliary';manifest=read(out/'manifest.json')
 sources={p:sha(ROOT/p) for p in manifest['input_sha256']}
 assert sources==manifest['input_sha256']
 for p,expected in manifest['artifact_sha256'].items():assert sha(out/p)==expected,p
 r=csv(RUN/'diagnostics/holdout/row_metrics_and_timeline.csv');r['decision_at']=pd.to_datetime(r.decision_at,utc=True)
 points=csv(out/'high_rv_ranking.csv');corr=csv(out/'surprise_correlations.csv');conc=csv(out/'loss_concentration.csv')
 threshold=read(RUN/'calibration/sealed_calibration.json')['thresholds']
 checks=[]
 for quarter,g in r.groupby('quarter'):
  for family,a in g.groupby('family'):
   a=a.sort_values('decision_at');y=(a.RV_effective.to_numpy()>threshold['train_effective_RV_q90'])
   point=points[(points.period==quarter)&(points.family==family)].iloc[0]
   assert point['count']==len(a) and point.positive==y.sum() and np.isclose(point.event_fraction,y.mean(),rtol=0,atol=1e-14)
   if y.any() and (~y).any():
    assert np.isclose(roc_auc_score(y,a.prediction_RV),point.AUROC,rtol=0,atol=1e-12)
    assert np.isclose(average_precision_score(y,a.prediction_RV),point.AP,rtol=0,atol=1e-12)
   else:assert np.isnan(point.AUROC) and np.isnan(point.AP)
   co=corr[(corr.period==quarter)&(corr.family==family)].iloc[0]
   x=a.observed_surprise.to_numpy();z=a.surprise_score.to_numpy();c=np.log(a.EWMA_RV.to_numpy()+1e-12)
   spe=float(spearmanr(x,z).statistic) if np.ptp(z)>1e-14 and np.ptp(x)>1e-14 else np.nan
   design=np.stack([np.ones(len(c)),c],axis=1);ix=np.linalg.pinv(design)@x;iz=np.linalg.pinv(design)@z
   ex=x-design@ix;ez=z-design@iz
   partial=float(np.corrcoef(ex,ez)[0,1]) if np.std(ex)>1e-14 and np.std(ez)>1e-14 else np.nan
   assert np.allclose([spe,partial],[co.spearman,co.partial_correlation],atol=1e-10,rtol=0,equal_nan=True)
   tail=a.RV_effective>threshold['train_effective_RV_q99']
   for _,row in conc[(conc.period==quarter)&(conc.family==family)].iterrows():
    total=a[row.loss].to_numpy().sum();ts=a.loc[tail,row.loss].to_numpy().sum()
    assert np.allclose([total,ts],[row.quarter_total_loss_sum,row.extreme_loss_sum],atol=1e-10,rtol=1e-12)
   checks.append({'check':'high_RV_sklearn_points_correlations_pseudoinverse_extreme_sums','quarter':quarter,'family':family,'status':'PASS'})
  for block in (7,3,14):
   with np.load(RUN/'bootstrap'/f'multiplicities_block{block}.npz',allow_pickle=False) as f:
    days=pd.to_datetime(f['TEST_'+quarter[-2:]+'_utc_days'],utc=True);counts=f['TEST_'+quarter[-2:]+'_counts']
   draws=csv(out/f'high_rv_draws_block{block}.csv')
   for family,a in g.groupby('family'):
    a=a.sort_values('decision_at');ids=days.get_indexer(a.decision_at.dt.floor('D'));y=a.RV_effective.to_numpy()>threshold['train_effective_RV_q90'];scores=a.prediction_RV.to_numpy()
    for attempt in (0,2499,4999):
     expanded=np.repeat(np.arange(len(a)),counts[attempt,ids].astype(int));yy=y[expanded]
     if yy.any() and (~yy).any():
      vals=[roc_auc_score(yy,scores[expanded]),average_precision_score(yy,scores[expanded])]
     else:vals=[np.nan,np.nan]
     expected=[draws.loc[attempt,f'{quarter}_{family}_{m}'] for m in ('AUROC','AP')]
     assert np.allclose(vals,expected,atol=1e-12,rtol=0,equal_nan=True)
   checks.append({'check':'saved_draw_first_middle_last_expanded_sklearn_all_nine','quarter':quarter,'block_days':block,'status':'PASS'})
 intervals=csv(out/'high_rv_paired_intervals.csv')
 for _,row in intervals.iterrows():
  d=csv(out/f'high_rv_draws_block{int(row.block_days)}.csv')
  key=f'{row.period}_R2_minus_{row.comparator}_{row.metric}'
  vals=d[key].to_numpy();valid=np.isfinite(vals);ci=np.quantile(vals[valid],[.025,.975]) if valid.any() else [np.nan,np.nan]
  assert np.allclose(ci,[row.ci_lower,row.ci_upper],atol=1e-12,rtol=0,equal_nan=True)
  assert int(valid.sum())==int(row.valid_draws) and np.isclose((~valid).mean(),row.invalid_fraction,rtol=0,atol=1e-14)
 checks.append({'check':'all_auxiliary_percentile_intervals_counts','rows':len(intervals),'status':'PASS'})
 write(RUN/'independent_review/formal_auxiliary_review.json',{'status':'PASS','checks':checks,'no_new_fits_or_random_draws':True,'independent_point_method':'sklearn; pseudoinverse regression; expanded-date samples','source_manifest_sha256':sha(out/'manifest.json'),'review_code_sha256':sha(__file__),'primary_status_unchanged':manifest['formal_primary_result_unchanged']})
 print('AUXILIARY_INDEPENDENT_REVIEW_PASS',len(checks))
def pdf_precheck():
 assert read(RUN/'authorization/formal_terminal.json')['state']=='CONSUMED'
 sys.path.insert(0,str(ROOT/'tmp/pdfs/deps'));import pymupdf as fitz
 path=ROOT/'output/pdf/Kronos_Frozen_Risk_03_Report_v2.pdf';doc=fitz.open(path)
 manifest=read(RUN/'reports/formal_pdf_generation_v2.json');assert sha(path)==manifest['pdf_sha256'] and len(doc)==manifest['pages']
 texts=[p.get_text() for p in doc];text='\n'.join(texts)
 result=read(RUN/'metrics/formal_result.json');assert result['status'] in text
 checks=[]
 for name,c in result['bootstrap']['7']['comparisons'].items():
  for value in [*c['quarter_deltas'],c['mean_delta'],c['ci_lower'],c['ci_upper']]:
   token=f'{value:.6f}' if value is not None else '未定义'
   assert token in text,(name,token)
  checks.append({'comparison':name,'quarter_mean_and_primary_CI_rendered_numeric_strings':'PASS'})
 for page,p in enumerate(doc,1):
  assert len(p.get_text().strip())>50,page
  for b in p.get_text('blocks'):
   assert b[0]>=-1 and b[1]>=-1 and b[2]<=p.rect.width+1 and b[3]<=p.rect.height+1,(page,b[:4])
 checks.append({'all_pages':len(doc),'nonempty_and_text_bounds':'PASS'})
 write(RUN/'reports/formal_pdf_text_precheck_v2.json',{'status':'PASS','pdf_sha256':sha(path),'pages':len(doc),'checks':checks,'visual_review_still_required':True,'no_scientific_rerun':True})
 print('PDF_TEXT_PRECHECK_PASS',len(doc))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['numeric-auxiliary','pdf-text-precheck']);a=p.parse_args()
 (check_aux if a.mode=='numeric-auxiliary' else pdf_precheck)()
