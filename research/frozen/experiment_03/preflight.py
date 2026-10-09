"""Complete finite engineering preflight and seal the reviewable Phase A bundle."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from research.frozen import encoder
from research.frozen.experiment_03 import guard,diagnostics
from research.frozen.experiment_03.run import read_prepared,immutable_json

def diagnostics_preview():
    x,opp,labels,matrices=read_prepared()
    pred=pd.read_csv(guard.RUN/'preflight/preview/selected_predictions.csv',float_precision='round_trip')
    def role_inputs(role):
        ids=opp.index[opp.role==role];frame=labels.loc[ids].copy()
        arrays={family:group.set_index('opportunity_id').loc[ids,'prediction'].to_numpy(float) for family,group in pred.loc[pred.role==role].groupby('family')}
        return frame,arrays
    tr,trainpred=role_inputs('train');va,valpred=role_inputs('validation')
    sealed=diagnostics.seal_calibration(tr,trainpred,guard.RUN/'calibration/training_bins')
    immutable_json(guard.RUN/'calibration/sealed_calibration.json',sealed)
    original=json.loads((guard.RUN/'calibration/training_thresholds.json').read_text())
    if original['train_q90']!=sealed['thresholds']['train_effective_RV_q90'] or original['train_q99']!=sealed['thresholds']['train_effective_RV_q99']:
        raise ValueError('Training threshold seal mismatch')
    diagnostics.analyze(va,valpred,sealed,guard.RUN/'diagnostics/development_validation',context='DEV_validation_engineering')
    immutable_json(guard.RUN/'preflight/calibration_audit.json',{'status':'PASS','train_rows':len(tr),'validation_rows':len(va),
        'sealed_train_only':True,'thresholds_match':True,'all9_selected_forecasts_preserved':len(trainpred)==9,
        'calibration_sha256':guard.sha(guard.RUN/'calibration/sealed_calibration.json'),'holdout_values_used':False})

def final_tests():
    paths=sorted((guard.ROOT/'tests').glob('test_risk03_*.py'))
    log=guard.RUN/'preflight/tests/risk03_final_tests.txt'
    if log.exists():raise ValueError('Final suite log already retained')
    result=subprocess.run([sys.executable,'-m','pytest',*[str(p) for p in paths],'-q','-s','-p','no:cacheprovider'],cwd=guard.ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
    with log.open('x',encoding='utf-8') as f:f.write(result.stdout+'\n'+result.stderr)
    immutable_json(guard.RUN/'preflight/tests/final_suites.json',{'status':'PASS' if result.returncode==0 else 'FAIL',
         'exit_code':result.returncode,'log_sha256':guard.sha(log),'files_sha256':{str(p.relative_to(guard.ROOT)).replace('\\','/'):guard.sha(p) for p in paths},'scope':'synthetic only'})
    print(result.stdout,flush=True)
    if result.returncode:raise ValueError('Final synthetic suite failed')

def seal():
    run=guard.RUN;root=guard.ROOT
    for rel in ('preflight/actual_audit.json','preflight/tests/final_suites.json','preflight/calibration_audit.json','independent_review/preflight_review.json','independent_review/preflight_supplement.json','reports/pdf_independent_review_v2.json','reports/pdf_qa.json'):
        value=json.loads((run/rel).read_text())
        if value['status']!='PASS':raise ValueError('Acceptance blocks sealing '+rel)
    if (run/'authorization/user_approval.json').exists() or (run/'authorization/execution_claim.json').exists():
        raise ValueError('Preflight must remain sealed')
    paths=list((root/'research/frozen/experiment_03').glob('*.py'))+list((root/'tests').glob('test_risk03_*.py'))
    paths += [root/'research/researchstate.py',guard.CONFIG,root/'research/environment/requirements.lock.txt',root/'research/initialization/version_manifest.json',root/'research/frozen/build_overall_pdf.py']
    for rel in ('research/frozen/encoder.py','research/baselines/ordinary_features.py','research/baselines/ordinary_features_v2.py','research/frozen/experiment_02/models.py','research/frozen/experiment_02/features.py','model/kronos.py','model/module.py'):
        paths.append(root/rel)
    sources={str(p.relative_to(root)).replace('\\','/'):guard.sha(p) for p in paths}
    for path in paths:encoder.immutable_bytes(run/'preflight/sealed_source'/path.relative_to(root),path.read_bytes())
    dependencies=json.loads((run/'provenance/dependencies.json').read_text())
    sources.update(dependencies['files_sha256']);sources.update(dependencies['wheel_sha256'])
    native=json.loads((run/'provenance/runtime_native_files.json').read_text())
    sources.update({rel:entry['sha256'] for rel,entry in native['files'].items()})
    for directory in ('features','labels','preflight/preview','calibration','provenance/official_sources','diagnostics/development_validation'):
        for path in (run/directory).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:sources[str(path.relative_to(root)).replace('\\','/')]=guard.sha(path)
    for rel in ('protocol_registration.json','provenance/protocol_locked.yaml','provenance/user_instructions.txt','provenance/dependencies.json','provenance/runtime_native_files.json','preflight/preparation.json','preflight/actual_audit.json','preflight/tests/final_suites.json','preflight/calibration_audit.json','independent_review/preflight_review.json','independent_review/preflight_supplement.json','reports/pdf_independent_review_v2.json','reports/pdf_qa.json','reports/Kronos_Frozen_Risk_03_Preflight_Report_v2.md'):
        path=run/rel;sources[str(path.relative_to(root)).replace('\\','/')]=guard.sha(path)
    for p in [root/'output/pdf/Kronos_Frozen_Risk_03_Preflight_Report_v2.pdf',root/'research/RESEARCHSTATE.md',root/'HANDOFF_FROZEN_REPRESENTATION.md',root/'research/Kronos_Crypto_Research_Roadmap.docx']:
        sources[str(p.relative_to(root)).replace('\\','/')]=guard.sha(p)
    cfg=guard.config();snapshot=root/cfg['sources']['one_hour_snapshot'];five=root/cfg['sources']['development_5m'];parent=root/cfg['sources']['parent_cache']
    for p in [snapshot/'manifest.json',snapshot/'candles.csv',five/'manifest.json',five/'candles.csv',five/'raw_csv_audit.json']:
        sources[str(p.relative_to(root)).replace('\\','/')]=guard.sha(p)
    for path in (five/'raw').rglob('*'):
        if path.is_file():sources[str(path.relative_to(root)).replace('\\','/')]=guard.sha(path)
    version=json.loads((root/'research/initialization/version_manifest.json').read_text())
    for kind in ('model','tokenizer'):
        for filename in ('config.json','model.safetensors'):
            p=root/version['weights'][kind]['local_cache']/filename;sources[str(p.relative_to(root)).replace('\\','/')]=guard.sha(p)
    for variant in ('pretrained','random_s17','random_s29','random_s43'):
        for p in (parent/variant).glob('*'):
            if p.is_file() and (p.suffix in ('.json','.npz','.safetensors')):sources[str(p.relative_to(root)).replace('\\','/')]=guard.sha(p)
    dirty=[]
    for line in subprocess.check_output(['git','-c','core.quotePath=false','status','--porcelain','--untracked-files=all'],cwd=root,text=True,encoding='utf-8').splitlines():
        p=root/line[3:]
        if p.is_file():dirty.append({'path':line[3:],'status':line[:2],'sha256':guard.sha(p)})
    bundle={'status':'PASS','experiment_id':'FROZEN_RISK_03_v1','protocol_sha256':guard.CONFIG_SHA,'sealed_at_utc':guard.now(),
       'files_sha256':sources,'canonical_code_sha256':{k:v for k,v in sources.items() if k.endswith('.py')},
       'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'dirty_files_sha256':dirty,
       'holdout':'SEALED','authorization':None,'result':'NOT_EVALUATED_HOLDOUT_SEALED','candidate_fits':28,'determinism_refits':2,
       'evidence_limits':'Metadata byte hashes of existing full 1h snapshot include test bytes, no new test numerical parsing; checkpoint-specific cutoff unproven.'}
    guard.new_json(run/'preflight/sealed_bundle.json',bundle);guard.verify_bundle()
    print(json.dumps({'status':'PASS','sealed_bundle_sha256':guard.sha(run/'preflight/sealed_bundle.json'),'files':len(sources),'holdout':'SEALED'}),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['diagnostics','tests','seal']);a=p.parse_args()
    {'diagnostics':diagnostics_preview,'tests':final_tests,'seal':seal}[a.action]()
if __name__=='__main__':main()
