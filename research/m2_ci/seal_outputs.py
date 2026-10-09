"""Exclusive post-training artifact seal; no fitting or choice of candidates."""
from datetime import datetime, timezone
from research.m2_ci import data
from research.m2_ci.runner import verify_seal

def seal_training():
    run=data.DEFAULT_RUN
    verify_seal(run)
    out=run/'provenance/training_result_seal.json'
    if out.exists():raise FileExistsError('Preserve existing completed training seal')
    if (run/'fusion/selected_weights.json').exists():raise ValueError('Training outputs must seal before alpha choice')
    audit=data.read(run/'models/candidate_audit.json')
    if audit['status']!='PASS' or audit['fits']!=95:raise ValueError('Complete registered training required')
    identities=data.read(run/'models/selected_fold_map.json')
    for item in identities.values():
        for path,digest in item['source_sha256'].items():
            if data.sha(path)!=digest:raise ValueError('Selected fitted head drift')
        for seed,digest in item['B5_sha256'].items():
            if data.sha(item['selected']['B5'][seed])!=digest:raise ValueError('Selected fitted B5 drift')
    files=[p for p in (run/'models').rglob('*') if p.is_file()]
    files.extend([run/'predictions/oof_predictions.csv',run/'preflight/source_seal.json'])
    mapping={p.relative_to(data.ROOT).as_posix():data.sha(p) for p in files}
    data.write(out,{'status':'PASS','protocol_sha256':data.sha(data.CONFIG),
        'source_seal_sha256':data.sha(run/'preflight/source_seal.json'),
        'sealed_at_utc':datetime.now(timezone.utc).isoformat(),'files_sha256':mapping,
        'producer_source_sha256':data.sha(__file__),'new_fits':0,'alpha_selection_not_started':True})
    print({'status':'TRAINING_OUTPUTS_SEALED','files':len(mapping)})

if __name__=='__main__':seal_training()
