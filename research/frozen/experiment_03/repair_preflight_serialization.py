"""One recorded pre-fit Unicode metadata serialization repair; no values change."""
import io,json
import numpy as np
import pandas as pd
from research.frozen import encoder
from research.frozen.experiment_03.guard import RUN,sha,require_scope,new_json

def repair():
    require_scope('2026-04-01T00:00:00Z','A')
    if (RUN/'preflight/preview/fit_events.jsonl').exists():raise ValueError('Repair forbidden after any candidate fit')
    ids=pd.read_csv(RUN/'features/opportunities.csv',usecols=['opportunity_id'],dtype=str).opportunity_id.to_numpy(dtype='U20')
    old=json.loads((RUN/'features/cache_audit.json').read_text());new=[]
    for audit in old:
        variant=audit['variant'];source=RUN/'features'/f'{variant}.npz'
        if sha(source)!=audit['output_sha256']:raise ValueError('Original published cache hash changed')
        with np.load(source,allow_pickle=False) as f:values=f['features'].copy()
        if values.shape!=(len(ids),512) or values.dtype!=np.float32:raise ValueError('Unexpected original matrix')
        buffer=io.BytesIO();np.savez_compressed(buffer,ids=ids,features=values)
        target=RUN/'features'/f'{variant}_v2.npz';encoder.immutable_bytes(target,buffer.getvalue())
        with np.load(target,allow_pickle=False) as f:
            if f['ids'].tolist()!=ids.tolist() or not np.array_equal(f['features'],values):raise ValueError('Repair changed numerical values')
        new.append({**audit,'original_metadata_serialization_sha256':sha(source),'output_sha256':sha(target),
                   'metadata_encoding':'fixed Unicode U20; pickle disabled','numerical_values_unchanged':True})
    new_json(RUN/'features/cache_audit_v2.json',new)
    new_json(RUN/'preflight/serialization_repair.json',{'status':'PASS','failed_attempt':'preview001 failed before any fit/journal',
        'error':'Object metadata IDs rejected by allow_pickle=False', 'original_files_preserved':True,
        'fixed':'Explicit Unicode identifiers; retained identical all4 float32 matrices','candidate_fit_count_before_repair':0,
        'research_parameters_unchanged':True})
if __name__=='__main__':repair()
