"""Fresh-process deterministic CUDA runtime for frozen inference and review.

This wrapper performs no fitting or selection. Immutable per-action claims and
terminal records retain failures and reject automatic retries.
"""
from __future__ import annotations

import argparse
import os
import runpy
import sys
import json
from pathlib import Path
from datetime import datetime, timezone

# This must precede every Torch import, including unchanged model modules.
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

from research.m2_ci import data
from threadpoolctl import threadpool_limits, threadpool_info
from research.m2_ci.runner import verify_seal


def _now():
    return datetime.now(timezone.utc).isoformat()


def _runtime(torch):
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {'CUBLAS_WORKSPACE_CONFIG':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'cuda_matmul_allow_tf32':bool(torch.backends.cuda.matmul.allow_tf32),
        'cudnn_allow_tf32':bool(torch.backends.cudnn.allow_tf32),
        'cudnn_benchmark':bool(torch.backends.cudnn.benchmark),
        'cudnn_deterministic':bool(torch.backends.cudnn.deterministic),
        'deterministic_algorithms':bool(torch.are_deterministic_algorithms_enabled()),
        'CPU_threads':int(torch.get_num_threads()),
        'threadpools':threadpool_info(),
        'float32_matmul_precision':torch.get_float32_matmul_precision(),
        'tensor_dtype':'float32','selection_and_loss_dtype':'float64','autocast':False,
        'torch_version':str(torch.__version__),'CUDA_runtime_version':torch.version.cuda,
        'cudnn_version':torch.backends.cudnn.version(),
        'GPU':{'index':int(torch.cuda.current_device()),'name':props.name,
               'compute_capability':[props.major,props.minor],'total_memory_bytes':int(props.total_memory),
               'uuid':str(getattr(props,'uuid','not_exposed'))}}


def _check_flags(flags):
    if (flags['CUBLAS_WORKSPACE_CONFIG'] != ':4096:8' or flags['cuda_matmul_allow_tf32']
            or flags['cudnn_allow_tf32'] or flags['cudnn_benchmark']
            or not flags['cudnn_deterministic'] or not flags['deterministic_algorithms']
            or flags['CPU_threads'] != 1):
        raise RuntimeError('Runtime differs from registered deterministic float32 configuration')
    if any(pool.get('num_threads') != 1 for pool in flags['threadpools']):
        raise RuntimeError('Registered BLAS/OpenMP thread count must remain one')


def _completed_fit_journal(path):
    with Path(path).open(encoding='utf-8') as handle:
        events = [json.loads(line) for line in handle if line.strip()]
    if len(events) != 190:
        raise PermissionError('Completed 95-pair fit journal required')
    candidates = set()
    for index in range(0,len(events),2):
        started, completed = events[index:index+2]
        name = started.get('candidate')
        if (started.get('state') != 'STARTED' or completed.get('state') != 'COMPLETED'
                or not name or name != completed.get('candidate') or name in candidates
                or started.get('fit_number') != index//2+1
                or completed.get('fit_number') != started.get('fit_number')):
            raise PermissionError('Fit journal must contain distinct uninterrupted STARTED/COMPLETED pairs')
        candidates.add(name)
    return data.sha(path)


def _verify_training_result_seal(run):
    path = run / 'provenance/training_result_seal.json'
    seal = data.read(path)
    if (seal.get('status') != 'PASS' or seal.get('protocol_sha256') != data.sha(data.CONFIG)
            or seal.get('source_seal_sha256') != data.sha(run / 'preflight/source_seal.json')):
        raise PermissionError('Passed current training result seal required')
    files = seal.get('files_sha256',{})
    required = [p for p in (run / 'models').rglob('*') if p.is_file()]
    required.append(run / 'predictions/oof_predictions.csv')
    if not files or not required:
        raise PermissionError('Empty training result seal')
    for file in required:
        if file.resolve().relative_to(data.ROOT).as_posix() not in files:
            raise PermissionError('Training result seal omits model or OOF artifact: ' + str(file))
    actual = {}
    for relative,expected in files.items():
        file = (data.ROOT / relative).resolve()
        if not file.is_relative_to(run):
            raise PermissionError('Training result seal file lies outside new run: ' + relative)
        digest = data.sha(file)
        if digest != expected:
            raise PermissionError('Changed sealed training artifact: ' + relative)
        actual[relative] = digest
    return data.sha(path), actual


@threadpool_limits.wrap(limits=1)
def execute(action, run=data.DEFAULT_RUN):
    if action not in ('validation','benchmark','review'):
        raise ValueError('Only frozen inference/review actions are supported')
    run,c,_ = data.new_scope(run)
    # The unchanged review CLI owns its fixed run directory.
    if action == 'review' and run != data.DEFAULT_RUN.resolve():
        raise PermissionError('Independent review CLI requires its registered run directory')
    seal = verify_seal(run)
    audit = data.read(run / 'models/candidate_audit.json')
    if (audit.get('status') != 'PASS' or audit.get('fits') != 95 or audit.get('linear_fits') != 60
            or audit.get('B2_fits') != 20 or audit.get('B5_fits') != 15):
        raise PermissionError('Passed complete registered 95-fit audit required')
    fit_events_sha256 = _completed_fit_journal(run / 'models/fit_events.jsonl')
    training_result_seal_sha256, training_files_sha256 = _verify_training_result_seal(run)
    selection = data.read(run / 'fusion/selected_weights.json')
    if (selection['status'] != 'FROZEN_DEVELOPMENT_SELECTION'
            or selection['protocol_sha256'] != data.sha(data.CONFIG)
            or selection['oof_predictions_sha256'] != data.sha(run / 'predictions/oof_predictions.csv')):
        raise PermissionError('Frozen OOF alpha selection is required before this runtime action')
    provenance = run / 'provenance'
    claim = provenance / f'{action}_runtime_manifest.json'
    success = provenance / f'{action}_runtime_success.json'
    failure = provenance / f'{action}_runtime_failure.json'
    if any(p.exists() for p in (claim,success,failure)):
        raise FileExistsError('Existing runtime claim or outcome prohibits automatic retry')
    # Import and configure the unchanged training runtime before action imports.
    from research.frozen.experiment_03_1 import b5_model
    import torch
    b5_model.configure(seed=17,device='cuda')
    flags = _runtime(torch)
    _check_flags(flags)
    manifest = {'schema_version':1,'status':'RUNNING','action':action,'at_utc':_now(),
        'run_directory':str(run),'wrapper_sha256':data.sha(__file__),
        'source_seal_sha256':data.sha(run / 'preflight/source_seal.json'),
        'protocol_sha256':seal['protocol_sha256'],
        'selected_weights_sha256':data.sha(run / 'fusion/selected_weights.json'),
        'fit_events_sha256':fit_events_sha256,
        'training_result_seal_sha256':training_result_seal_sha256,
        'training_result_files_sha256':training_files_sha256,
        'runtime':flags,'configure_function':'unchanged b5_model.configure(seed=17,device=cuda)',
        'new_fitting':False,'new_selection':False,'automatic_retry':False}
    data.write(claim,manifest)
    try:
        if action == 'validation':
            from research.m2_ci.runner import validation
            validation(run)
        elif action == 'benchmark':
            from research.m2_ci.benchmark import benchmark
            benchmark(run)
        else:
            saved_argv = sys.argv[:]
            try:
                sys.argv = ['research.m2_ci.independent_review','review']
                try:
                    runpy.run_module('research.m2_ci.independent_review',run_name='__main__')
                except SystemExit as exc:
                    if exc.code not in (None,0):
                        raise RuntimeError(f'Independent numeric review exited {exc.code}') from exc
            finally:
                sys.argv = saved_argv
            if data.read(run / 'independent_review/numeric_review.json')['status'] != 'PASS':
                raise RuntimeError('Independent numeric review did not pass')
        actual = _runtime(torch)
        _check_flags(actual)
        verify_seal(run)
        if data.sha(run / 'fusion/selected_weights.json') != manifest['selected_weights_sha256']:
            raise RuntimeError('Frozen fusion weights changed during runtime action')
        if _completed_fit_journal(run / 'models/fit_events.jsonl') != manifest['fit_events_sha256']:
            raise RuntimeError('Fit journal changed during inference or review')
        seal_digest, files_digest = _verify_training_result_seal(run)
        if (seal_digest != manifest['training_result_seal_sha256']
                or files_digest != manifest['training_result_files_sha256']):
            raise RuntimeError('Training result seal or model/scaler/selector artifact changed during execution')
        if data.sha(__file__) != manifest['wrapper_sha256']:
            raise RuntimeError('Runtime wrapper changed during execution')
        data.write(success,{'status':'SUCCESS','action':action,'at_utc':_now(),
                            'claim_sha256':data.sha(claim),'runtime_after':actual})
    except BaseException as exc:
        data.write(failure,{'status':'FAILED','action':action,'at_utc':_now(),
                           'claim_sha256':data.sha(claim),'error':repr(exc),'automatic_retry':False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['validation','benchmark','review'])
    parser.add_argument('--run',type=Path,default=data.DEFAULT_RUN)
    args = parser.parse_args()
    execute(args.action,args.run)


if __name__ == '__main__':
    main()
