"""Bind passed review, critical checks and immutable inputs before any fit."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, importlib.metadata, json, subprocess, sys
from research.m2_ci import data

ROOT, RUN = data.ROOT, data.DEFAULT_RUN
SCIENCE_NAMES = ['__init__.py','data.py','runner.py','statistics.py','benchmark.py','initialize.py','preflight.py']

def rel(p): return Path(p).resolve().relative_to(ROOT).as_posix()

def tests():
    out=RUN/'preflight/critical_tests.json'
    if out.exists(): raise FileExistsError('Preserve test attempts')
    command=[sys.executable,'-B','-m','unittest','discover','-s','tests','-p','test_m2_ci*.py','-v']
    result=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    (RUN/'preflight/critical_tests.log').write_text(result.stdout+result.stderr,encoding='utf-8')
    sources=[ROOT/'tests/test_m2_ci_oof.py',ROOT/'tests/test_m2_ci_statistics.py']
    data.write(out,{'status':'PASS' if result.returncode==0 else 'FAIL','exit_code':result.returncode,
                   'command':command,'files_sha256':{rel(p):data.sha(p) for p in sources},'scientific_fits':0})
    print(json.dumps({'status':'PASS' if result.returncode==0 else 'FAIL','exit_code':result.returncode}))
    if result.returncode: raise RuntimeError('Critical tests failed')

def seal():
    out=RUN/'preflight/source_seal.json'
    if out.exists(): raise FileExistsError('Science source already sealed')
    reviewed=RUN/'preflight/code_review_v3.json'
    tested=RUN/'preflight/critical_tests.json'
    audit=data.read(RUN/'inputs/prepare_audit.json')
    review=data.read(reviewed)
    if review['status']!='PASS' or data.read(tested)['status']!='PASS' or audit['status']!='PASS':
        raise ValueError('Current code, tests and preparation must pass')
    science=[ROOT/'research/m2_ci'/name for name in SCIENCE_NAMES]
    files=science+[data.CONFIG,
        ROOT/'tests/test_m2_ci_oof.py',ROOT/'tests/test_m2_ci_statistics.py',reviewed,tested,
        RUN/'preflight/critical_tests.log',ROOT/'research/m2_ci/phase_a_independent_review.json',
        ROOT/'research/initialization/version_manifest.json',ROOT/'research/environment/requirements.lock.txt']
    for folder in ['inputs','labels','folds']:
        files.extend(p for p in (RUN/folder).rglob('*') if p.is_file())
    for folder in ['research/frozen/experiment_03','research/frozen/experiment_02','research/frozen/experiment_03_1','research/baselines','model']:
        files.extend(p for p in (ROOT/folder).glob('*.py'))
    files += [ROOT/'research/frozen/encoder.py',ROOT/'research/configs/frozen_risk_03_v1.yaml',
              ROOT/'research/configs/frozen_risk_03_1_b5_v1.yaml',ROOT/'research/configs/initial_experiment.yaml']
    for key in ['parent_03','parent_03_1']:
        parent=ROOT/data.config()['sources'][key]
        files.extend(p for p in (parent/'models').rglob('*') if p.is_file())
    files.extend(ROOT/p for p in audit['source_files'])
    files += [RUN/'preflight/data_review_v2.json',RUN/'preflight/data_review_v3.json',
              RUN/'provenance/user_instruction.txt',RUN/'provenance/config_preregistered.yaml',
              ROOT/'research/runs/FROZEN_RISK_03_v1/formal_delivery_manifest.json',
              ROOT/'research/runs/FROZEN_RISK_03_v1/preflight/sealed_bundle.json',
              ROOT/'research/runs/FROZEN_RISK_03_1_B5_v1/delivery_manifest.json',
              ROOT/data.config()['sources']['development_5m']/'manifest.json',
              ROOT/data.config()['sources']['development_5m']/'raw_csv_audit.json']
    version=data.read(ROOT/'research/initialization/version_manifest.json')
    for item in version['weights'].values():
        for entry in item['files']:
            p=ROOT/item['local_cache']/entry['name']
            if data.sha(p)!=entry['sha256']: raise ValueError('Original model checkpoint changed')
            files.append(p)
    # Current software identity; no install, no data download or scientific fit.
    packages={k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy','scikit-learn','torch','PyYAML']}
    from research.frozen.experiment_03 import models
    lgb=models._lightgbm()
    packages['lightgbm']={'version':lgb.__version__,'module_path':str(Path(lgb.__file__).resolve()),'isolated':True}
    package_root=Path(lgb.__file__).resolve().parent
    files.extend(p for p in package_root.rglob('*') if p.is_file() and p.suffix in ['.py','.dll','.so','.dylib','.json'])
    for dist in package_root.parent.glob('lightgbm-*.dist-info'):
        files.extend(p for p in dist.iterdir() if p.is_file())
    dirty=subprocess.run(['git','-c','core.quotepath=false','status','--porcelain','-uall'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',check=True).stdout
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
    dirty_hash={}
    for line in dirty.splitlines():
        name=line[3:]; name=name.split(' -> ')[-1].strip('"')
        p=ROOT/name
        if p.is_file(): dirty_hash[name]=data.sha(p)
    snapshot=ROOT/data.config()['sources']['hourly_snapshot']
    files.extend(p for p in snapshot.iterdir() if p.is_file() and p.suffix in ['.json','.csv'])
    data.write(RUN/'provenance/runtime_preflight.json',{'at_utc':datetime.now(timezone.utc).isoformat(),
        'python':sys.version,'executable':sys.executable,'packages':packages,'git_commit':commit,
        'git_dirty_status':dirty,'dirty_files_sha256':dirty_hash,'no_push':True})
    files.append(RUN/'provenance/runtime_preflight.json')
    for p in science:
        target=RUN/'provenance/scientific_source'/p.name; target.parent.mkdir(exist_ok=True)
        with target.open('xb') as f:f.write(p.read_bytes())
        files.append(target)
    expected=data._review_hashes(data.read(ROOT/'research/m2_ci/phase_a_independent_review.json'))
    mapping={rel(p):data.sha(p) for p in files}
    for path,digest in mapping.items():
        if path in expected and digest!=expected[path]:
            raise ValueError('Reviewed historical source drift before seal: '+path)
    # Review must describe the current core scientific implementation.
    reviewed_hashes=review.get('files_sha256',review.get('source_sha256',{}))
    for name in ['data.py','runner.py','statistics.py','benchmark.py']:
        path='research/m2_ci/'+name
        if reviewed_hashes.get(path)!=mapping[path]: raise ValueError('Current science missing from passed review: '+path)
    data.write(out,{'status':'PASS','experiment_id':'M2_CONDITIONAL_INCREMENT_01',
        'sealed_at_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':data.sha(data.CONFIG),
        'code_review':{'status':'PASS','path':rel(reviewed)},'tests':{'status':'PASS','path':rel(tested)},
        'files_sha256':mapping,'no_scientific_fits_before_seal':True,'future_formal_test_approved':False,
        'old_holdout_status':'CONSUMED'})
    print(json.dumps({'status':'SOURCE_SEALED','files':len(mapping)}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['tests','seal']);a=p.parse_args()
    {'tests':tests,'seal':seal}[a.action]()
