"""Actual frozen inference resource measurements on eight shared DEV windows.

No targets are accepted or opened. Full R2 fusion executes the tokenizer and
encoder on every measured call; standalone R2-head timing uses cached hidden.
"""
from __future__ import annotations

import os
import time
import ctypes
from ctypes import wintypes
from pathlib import Path

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from research.m2_ci import data


def rss_bytes():
    """Current process working set via the Windows API; no extra dependency."""
    if os.name != 'nt':
        raise RuntimeError('Registered benchmark requires Windows RSS API')

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD),
                    ('PeakWorkingSetSize',ctypes.c_size_t),('WorkingSetSize',ctypes.c_size_t),
                    ('QuotaPeakPagedPoolUsage',ctypes.c_size_t),('QuotaPagedPoolUsage',ctypes.c_size_t),
                    ('QuotaPeakNonPagedPoolUsage',ctypes.c_size_t),('QuotaNonPagedPoolUsage',ctypes.c_size_t),
                    ('PagefileUsage',ctypes.c_size_t),('PeakPagefileUsage',ctypes.c_size_t)]

    kernel = ctypes.WinDLL('kernel32',use_last_error=True)
    psapi = ctypes.WinDLL('psapi',use_last_error=True)
    current_process = kernel.GetCurrentProcess
    current_process.argtypes = []
    current_process.restype = wintypes.HANDLE
    get_memory = psapi.GetProcessMemoryInfo
    get_memory.argtypes = [wintypes.HANDLE,ctypes.POINTER(ProcessMemoryCounters),wintypes.DWORD]
    get_memory.restype = wintypes.BOOL
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    if not get_memory(current_process(),ctypes.byref(counters),counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    if counters.WorkingSetSize <= 0:
        raise RuntimeError('Invalid Windows working set size')
    return int(counters.WorkingSetSize)


def _tree_counts(node):
    if 'leaf_index' in node or 'left_child' not in node:
        return 1, 1
    left = _tree_counts(node['left_child'])
    right = _tree_counts(node['right_child'])
    return 1 + left[0] + right[0], left[1] + right[1]


def _checkpoint_files(version, kind):
    directory = data.ROOT / version['weights'][kind]['local_cache']
    return [p for p in directory.iterdir() if p.is_file() and p.suffix in ('.bin','.pt','.safetensors')]


@threadpool_limits.wrap(limits=1)
def benchmark(run=data.DEFAULT_RUN):
    from research.m2_ci.runner import verify_seal
    run, c, _ = data.new_scope(run)
    seal = verify_seal(run)
    if Path(__file__).resolve().relative_to(data.ROOT).as_posix() not in seal['files_sha256']:
        raise PermissionError('Benchmark implementation must be bound in source seal')
    output = run / 'metrics/resource_benchmark.json'
    csv = run / 'metrics/resource_benchmark.csv'
    if output.exists() or csv.exists():
        raise FileExistsError('Preserve existing resource benchmark')
    if data.read(run / 'models/candidate_audit.json')['status'] != 'PASS':
        raise PermissionError('Complete passed OOF training required')
    selection = data.read(run / 'fusion/selected_weights.json')
    if (selection['status'] != 'FROZEN_DEVELOPMENT_SELECTION'
            or selection['protocol_sha256'] != data.sha(data.CONFIG)
            or selection['oof_predictions_sha256'] != data.sha(run / 'predictions/oof_predictions.csv')):
        raise PermissionError('Frozen fusion selection required')
    alphas = {family:float(selection['selected'][family]['alpha']) for family in ('R1','B2','R2')}
    if any(a not in c['fusion']['alpha_candidates'] for a in alphas.values()):
        raise ValueError('Unregistered selected alpha')
    features, metadata = data.load_features(run)
    ids = data.read(run / 'inputs/benchmark_ids.json')
    roles = data.fold_indices(metadata,c)
    expected = features['ids'][roles['WF01']['fit'][:8]].tolist()
    if ids != expected or len(ids) != 8:
        raise ValueError('Common eight first-fold FIT windows differ')
    lookup = {ident:i for i,ident in enumerate(features['ids'])}
    ix = np.asarray([lookup[v] for v in ids],dtype=np.int64)
    fold = data.read(run / 'models/WF05/selected_models.json')
    selected, scaler = fold['selected'], fold['scaling_audit']
    if fold['fold_id'] != 'WF05' or fold['selection_role'] != 'inner_validation' or not fold['no_refit']:
        raise ValueError('Expected selected WF05 frozen artifacts')
    from research.frozen.experiment_03 import models
    from research.frozen.experiment_03_1 import b5_model as b5
    from research.frozen import encoder
    import torch
    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('Registered CUDA benchmark unavailable; no CPU substitution')
    device = c['models']['B5_device']
    if device != 'cuda':
        raise ValueError('Registered CUDA device changed')
    windows = features['windows'][ix]
    ordinary = features['ordinary_raw'][ix]
    scaled = ((ordinary-np.asarray(scaler['mean']))/np.asarray(scaler['scale'])).astype(np.float32)
    heads = {family:data.read(selected[family]) for family in ('har','R1','R2','B2')}
    networks = {seed:b5.load_b5(path,device=device) for seed,path in selected['B5'].items()}
    tokenizer, backbone, encoder_metadata = encoder.load_models('pretrained',device=device)
    encoder_windows = []
    for i in ix:
        start = pd.Timestamp(metadata.iloc[i].history_start_at)
        end = pd.Timestamp(metadata.iloc[i].history_end_exclusive)
        timestamps = pd.date_range(start,periods=256,freq='h')
        if timestamps[-1] + pd.Timedelta(hours=1) != end:
            raise ValueError('Timestamp history identity differs')
        stamps = encoder.calc_time_stamps(pd.Series(timestamps)).to_numpy(dtype=np.float32)
        # Six normalized market channels, with unscaled known timestamps.
        encoder_windows.append((features['windows'][i,:,:6].copy(),stamps))
    records = []
    resident = {seed:sum(p.numel() for p in payload['model'].parameters()) for seed,payload in networks.items()}
    encoder_parameters = sum(p.numel() for module in (tokenizer,backbone) for p in module.parameters())
    version = data.read(data.ROOT / 'research/initialization/version_manifest.json')
    encoder_checkpoints = _checkpoint_files(version,'model') + _checkpoint_files(version,'tokenizer')
    booster = models._lightgbm().Booster(model_str=heads['B2']['model_text'])
    counts = [_tree_counts(tree['tree_structure']) for tree in booster.dump_model()['tree_info']]
    b2_nodes, b2_leaves = sum(v[0] for v in counts),sum(v[1] for v in counts)
    parameter_info = {}
    for family in ('har','R1','R2'):
        parameter_info[family] = {'trainable_parameters_at_fit':len(heads[family]['coef'])+1,
                                  'B2_nodes':0,'B2_leaves':0}
    parameter_info['B2'] = {'trainable_parameters_at_fit':None,'B2_nodes':b2_nodes,'B2_leaves':b2_leaves}

    def measure(name,function,paths,parameters,scope):
        for _ in range(3):
            function()
            torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        elapsed, rss = [], [rss_bytes()]
        for _ in range(20):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            value = function()
            torch.cuda.synchronize(device)
            elapsed.append(time.perf_counter()-started)
            rss.append(rss_bytes())
            arr = np.asarray(value)
            if arr.shape not in ((8,), (8,512)) or not np.isfinite(arr).all():
                raise ValueError('Invalid benchmark inference result')
        record = {'name':name,'batch_windows':8,'warmups':3,'repetitions':20,
            'elapsed_seconds':elapsed,'batch_mean_seconds':float(np.mean(elapsed)),
            'batch_median_seconds':float(np.median(elapsed)), 'batch_p95_seconds':float(np.quantile(elapsed,.95)),
            'per_window_mean_seconds':float(np.mean(elapsed)/8),
            'cuda_peak_allocated_bytes':int(torch.cuda.max_memory_allocated(device)),
            'cuda_peak_reserved_bytes':int(torch.cuda.max_memory_reserved(device)),
            'rss_sampled_max_bytes':int(max(rss)),'rss_before_bytes':int(rss[0]),'rss_after_bytes':int(rss[-1]),
            'checkpoint_bytes':sum(Path(p).stat().st_size for p in set(paths)),
            'checkpoint_paths':[str(p) for p in sorted(set(map(str,paths)))],
            'CPU_threads':1,'CUDA_synchronize':True,'measurement_scope':scope,**parameters}
        records.append(record)

    def single(seed):
        return b5.predict_b5(networks[seed],windows,scaled,device=device,batch_size=64)['prediction']

    def ensemble():
        return np.mean([single(seed) for seed in networks],axis=0,dtype=np.float64)

    def head(family,hidden=None):
        if family == 'har':
            x = features['har'][ix]
        elif family == 'R2':
            x = np.concatenate((ordinary,features['hidden'][ix] if hidden is None else hidden),axis=1).astype(np.float64)
        else:
            x = ordinary
        fn = models.predict_b2 if family == 'B2' else models.predict_head
        return fn(heads[family],x)

    for seed,path in selected['B5'].items():
        measure('B5_s'+seed,lambda seed=seed:single(seed),[path],
                {'trainable_parameters_at_fit':resident[seed],'B2_nodes':0,'B2_leaves':0},'loaded single B5; preprocessing excluded')
    b5paths = list(selected['B5'].values())
    b5params = sum(resident.values())
    measure('B5_ensemble',ensemble,b5paths,{'trainable_parameters_at_fit':b5params,'B2_nodes':0,'B2_leaves':0},
            'three real B5 forwards and arithmetic original-unit RV mean')
    for family in ('har','R1','B2','R2'):
        measure(family+'_head_CPU',lambda family=family:head(family),[selected[family]],parameter_info[family],
                'array-only CPU head; R2 uses cached hidden; model loading excluded')
    measure('R2_tokenizer_encoder_CUDA',lambda:encoder.encode_batch(tokenizer,backbone,encoder_windows),
            encoder_checkpoints,{'trainable_parameters_at_fit':0,'frozen_parameters':encoder_parameters,'B2_nodes':0,'B2_leaves':0},
            'actual tokenizer encode and decode_s1 last context; no cached hidden')
    for family in ('R1','B2','R2'):
        def fused(family=family):
            b = ensemble()
            hidden = encoder.encode_batch(tokenizer,backbone,encoder_windows) if family == 'R2' else None
            p = head(family,hidden)
            return (1-alphas[family])*b+alphas[family]*p
        paths = b5paths+[selected[family]]+(encoder_checkpoints if family == 'R2' else [])
        info = dict(parameter_info[family])
        info['trainable_parameters_at_fit'] = b5params + (info['trainable_parameters_at_fit'] or 0)
        info['frozen_parameters'] = encoder_parameters if family == 'R2' else 0
        measure('full_B5_'+family+'_fusion',fused,paths,info,
            f'actual B5 ensemble + {family} path + original-unit RV fusion; alpha={alphas[family]}; '
            'both paths executed even at alpha 0/1; R2 encoder uncached')
    verify_seal(run)
    result = {'status':'PASS','selected_fold':'WF05','common_input_ids':ids,'selected_alphas':alphas,
        'source_protocol_sha256':data.sha(data.CONFIG),'fusion_selection_sha256':data.sha(run / 'fusion/selected_weights.json'),
        'selected_artifacts_sha256':{str(p):data.sha(p) for p in b5paths+list(selected[k] for k in ('har','R1','R2','B2'))},
        'encoder_metadata':encoder_metadata,'resident_B5_parameters':resident,'resident_encoder_parameters':encoder_parameters,
        'records':records,'no_target_table_reads':True,'full_R2_encoder_cached_hidden_used':False,
        'definitions':{'timing':'perf_counter wall time; CUDA synchronize before and after each batch; 3 warmups and 20 repetitions',
            'CUDA_peaks':'absolute allocator peaks after warmup; includes all resident B5 seeds and tokenizer/encoder in this process; reserved includes warmed allocator cache',
            'RSS':'process resident set sampled before/after each call; sampled maximum is not a continuous allocation peak',
            'checkpoint':'bytes of selected serialized artifacts; encoder weights only; not all candidate/training storage',
            'scope':'input preprocessing and model loading excluded; full pipeline totals are measured directly, not summed component estimates',
            'training_budget':'inference benchmark does not establish matched total pretraining budgets or equal model development resources'}}
    flat = [{k:v for k,v in row.items() if not isinstance(v,(dict,list))} for row in records]
    data.frame(csv,pd.DataFrame(flat))
    data.write(output,result)
    return result
