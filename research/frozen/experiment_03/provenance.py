"""Record primary public checkpoint metadata and isolated binary provenance."""
from __future__ import annotations
import importlib.metadata
import json
import sys
import subprocess
import requests
from research.frozen.experiment_03.guard import ROOT,RUN,CONFIG_SHA,sha,now,new_json

URLS={
 'small_pinned':'https://huggingface.co/api/models/NeoQuasar/Kronos-small/revision/901c26c1332695a2a8f243eb2f37243a37bea320?blobs=true',
 'small_commits':'https://huggingface.co/api/models/NeoQuasar/Kronos-small/commits/901c26c1332695a2a8f243eb2f37243a37bea320',
 'tokenizer_pinned':'https://huggingface.co/api/models/NeoQuasar/Kronos-Tokenizer-base/revision/0e0117387f39004a9016484a186a908917e22426?blobs=true',
 'tokenizer_commits':'https://huggingface.co/api/models/NeoQuasar/Kronos-Tokenizer-base/commits/0e0117387f39004a9016484a186a908917e22426',
 'small_first_weights':'https://huggingface.co/api/models/NeoQuasar/Kronos-small/revision/ac5c409a313c4eadd1ca78201322f5cadb9c34ab?blobs=true',
 'tokenizer_first_weights':'https://huggingface.co/api/models/NeoQuasar/Kronos-Tokenizer-base/revision/9ef143b98ee3c2488eebd85404e0c215c112b46a?blobs=true',
 'small_card':'https://huggingface.co/NeoQuasar/Kronos-small/raw/901c26c1332695a2a8f243eb2f37243a37bea320/README.md',
 'gamma_source':'https://raw.githubusercontent.com/microsoft/LightGBM/v4.6.0/src/objective/regression_objective.hpp',
 'lgbm_parameters':'https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html',
 'lgbm_train':'https://lightgbm.readthedocs.io/en/v4.6.0/pythonapi/lightgbm.train.html',
}

def public_sources():
    directory=RUN/'provenance/official_sources'; directory.mkdir(parents=True,exist_ok=True)
    for name,url in URLS.items():
        body=directory/(name+'.raw'); meta=directory/(name+'.source.json')
        if body.exists() or meta.exists():
            if not(body.exists() and meta.exists()): raise ValueError('Partial immutable source '+name)
            recorded=json.loads(meta.read_text())
            if recorded['url']!=url or sha(body)!=recorded['sha256']:raise ValueError('Changed primary source')
            continue
        response=requests.get(url, timeout=45); response.raise_for_status()
        with body.open('xb') as f:f.write(response.content)
        new_json(meta,{'url':url,'collected_at_utc':now(),'sha256':sha(body),'http_status':response.status_code})
        print('Primary metadata saved '+name,flush=True)
    # The commits tie identical LFS objects to public timestamps, not training cutoff.
    from research.frozen.encoder import HASHES
    checks=[]
    for kind,prefix in [('model','small'),('tokenizer','tokenizer')]:
        for suffix in ('pinned','first_weights'):
            payload=json.loads((directory/(prefix+'_'+suffix+'.raw')).read_text())
            weights=next(x for x in payload['siblings'] if x['rfilename']=='model.safetensors')
            actual=weights['lfs']['sha256']
            if actual!=HASHES[kind]['model.safetensors']:raise ValueError('Official weight identity mismatch')
            checks.append({'kind':kind,'source':prefix+'_'+suffix,'weight_sha256':actual,'PASS':True})
    new_json(directory/'checkpoint_evidence.json',{'status':'PASS','weight_checks':checks,
        'public_same_weight_before_holdout':True,'model_first_public_utc':'2025-06-30T16:44:37Z',
        'tokenizer_first_public_utc':'2025-06-30T17:04:10Z', 'checkpoint_specific_training_cutoff_proven':False,
        'interpretation':'Public existence of the identical object before test interval; not a publisher statement of training cutoff. Early DEV not uniformly strict prospective OOS.'})

def environment():
    isolated=RUN/'provenance/python_packages'
    sys.path.insert(0,str(isolated))
    import lightgbm,numpy,scipy,sklearn,torch
    if lightgbm.__version__!='4.6.0' or not __import__('pathlib').Path(lightgbm.__file__).resolve().is_relative_to(isolated):
        raise ValueError('Isolated binary version/path mismatch')
    packages={name:importlib.metadata.version(name) for name in ('numpy','scipy','scikit-learn','pandas','torch','requests','PyYAML','threadpoolctl')}
    packages['lightgbm']='4.6.0'
    binaries={str(path.relative_to(ROOT)).replace('\\','/'):sha(path) for path in sorted(isolated.rglob('*'))
              if path.is_file() and '__pycache__' not in path.parts}
    wheels={str(path.relative_to(ROOT)).replace('\\','/'):sha(path) for path in (RUN/'provenance/wheels').glob('*.whl')}
    new_json(RUN/'provenance/dependencies.json',{'interpreter':sys.executable,'versions':packages,'files_sha256':binaries,
             'wheel_sha256':wheels,'pip_install':'isolated --target --no-deps; existing environments unchanged',
             'torch_cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'threads':1})
    (RUN/'provenance/requirements.experiment.lock.txt').write_text('\n'.join(f'{k}=={v}' for k,v in packages.items())+'\n',encoding='utf-8')

def repository_evidence():
    from research.frozen.encoder import HASHES
    parent=ROOT/'research/runs/FROZEN_RISK_02_v1'
    final=json.loads((parent/'delivery_manifest.json').read_text())
    # The latest delivery is authoritative for report/audit source updates.
    new_json(RUN/'provenance/parent_evidence.json',{'protocol_sha256':sha(ROOT/'research/configs/frozen_risk_02_v1.yaml'),
       'final_acceptance_sha256':sha(parent/'final_acceptance.json'), 'delivery_manifest_sha256':sha(parent/'delivery_manifest.json'),
       'parent_conclusion':'INCONCLUSIVE; pretraining support; independent validation required',
       'source_scope':'latest accepted delivery, no modification of parent artifacts',
       'pinned_files':HASHES, 'parent_delivery':final,
       'repository_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()})

if __name__=='__main__':
    public_sources(); environment(); repository_evidence()
