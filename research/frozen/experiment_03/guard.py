"""Fail-closed protocol, explicit user authorization and one-shot test access."""
from __future__ import annotations
import hashlib
import copy
import json
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'research/runs/FROZEN_RISK_03_v1'
CONFIG = ROOT / 'research/configs/frozen_risk_03_v1.yaml'
CONFIG_SHA = '7ed8affd7cd624cde89d23d29aaca7ef2ab660167fac7d3dc25fac274e394b7f'
_CONFIG_VALUE = None
_BUNDLE_CACHE = None

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576), b''): h.update(block)
    return h.hexdigest()

def config():
    global _CONFIG_VALUE
    if sha(CONFIG) != CONFIG_SHA: raise ValueError('Locked protocol changed')
    if _CONFIG_VALUE is None: _CONFIG_VALUE=yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    return copy.deepcopy(_CONFIG_VALUE)

def now(): return datetime.now(timezone.utc).isoformat()

def new_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False); f.write('\n')

def verify_bundle(run=RUN):
    global _BUNDLE_CACHE
    path = Path(run)/'preflight/sealed_bundle.json'
    bundle = json.loads(path.read_text(encoding='utf-8'))
    if bundle['protocol_sha256'] != CONFIG_SHA or bundle['status'] != 'PASS':
        raise PermissionError('No passed sealed preflight bundle')
    identity=(str(path.resolve()),sha(path))
    fingerprint=[]
    for rel, expected in bundle['files_sha256'].items():
        target=(ROOT/rel).resolve()
        if not target.is_relative_to(ROOT) or not target.is_file():raise PermissionError('Sealed file absent '+rel)
        st=target.stat();fingerprint.append((rel,st.st_mtime_ns,st.st_size))
    if _BUNDLE_CACHE==(identity,fingerprint):return bundle
    for rel, expected in bundle['files_sha256'].items():
        target = (ROOT/rel).resolve()
        if not target.is_relative_to(ROOT) or not target.is_file() or sha(target) != expected:
            raise PermissionError('Sealed input/source/dependency changed: '+rel)
    _BUNDLE_CACHE=(identity,fingerprint)
    return bundle

def verify_authorization(run=RUN):
    cfg = config(); run = Path(run)
    path = run/cfg['authority']['authorization_record']
    if not path.exists(): raise PermissionError('Holdout SEALED: separate explicit user approval absent')
    auth = json.loads(path.read_text(encoding='utf-8'))
    required = {'experiment_id':'FROZEN_RISK_03_v1', 'approval_text':cfg['authority']['approval_phrase'],
                'protocol_sha256':CONFIG_SHA, 'source':'subsequent_user_message'}
    if any(auth.get(k) != v for k,v in required.items()) or not auth.get('user_message_record'):
        raise PermissionError('Approval must cite the subsequent actual user message and locked protocol')
    verify_bundle(run)
    if auth.get('sealed_bundle_sha256') != sha(run/'preflight/sealed_bundle.json'):
        raise PermissionError('Approval does not bind the concrete sealed bundle')
    return auth

def claim_formal(run=RUN):
    run=Path(run); auth=verify_authorization(run)
    path=run/'authorization/execution_claim.json'
    new_json(path, {'state':'CLAIMED', 'experiment_id':'FROZEN_RISK_03_v1', 'protocol_sha256':CONFIG_SHA,
                   'authorization_sha256':sha(run/'authorization/user_approval.json'),
                   'sealed_bundle_sha256':auth['sealed_bundle_sha256'], 'created_at_utc':now()})
    return path

def require_scope(end_exclusive, phase='A', purpose='test'):
    cfg=config(); end=pd.Timestamp(end_exclusive)
    if end.tzinfo is None or end.utcoffset().total_seconds()!=0:
        raise ValueError('Explicit UTC scope required')
    if phase=='A':
        if end>pd.Timestamp(cfg['scope_guards']['phase_a_max_time_exclusive']):
            raise PermissionError('Phase A cannot read any holdout value')
    elif phase=='B':
        require_not_terminal()
        verify_authorization()
        claim=RUN/'authorization/execution_claim.json'
        if not claim.exists(): raise PermissionError('One-shot formal claim required before holdout access')
        value=json.loads(claim.read_text(encoding='utf-8'))
        if value.get('state')!='CLAIMED' or value.get('protocol_sha256')!=CONFIG_SHA:
            raise PermissionError('Formal run consumed or invalidated')
        if end>pd.Timestamp(cfg['scope_guards']['phase_b_max_time_exclusive']):
            raise PermissionError('Outside locked formal scope')
        if purpose=='test':require_formal_models()
        elif purpose!='training':raise ValueError('Unknown formal access purpose')
    else: raise ValueError('Unknown phase')
    return end

def require_formal_models():
    path=RUN/'models/formal/preview_identity_check.json'
    if not path.exists():raise PermissionError('Formal training/model identity check must precede test access')
    check=json.loads(path.read_text(encoding='utf-8'))
    if check.get('status')!='PASS' or check.get('all_selected_models_exact') is not True or not check.get('files_sha256'):
        raise PermissionError('Formal model identity evidence incomplete')
    for rel,expected in check['files_sha256'].items():
        target=(RUN/rel).resolve()
        if not target.is_relative_to(RUN.resolve()) or not target.is_file() or sha(target)!=expected:
            raise PermissionError('Formal selected model/selector changed '+rel)

def finish_formal(status, evidence):
    if status not in ('CONSUMED','INVALIDATED'): raise ValueError('Unknown terminal state')
    claim=RUN/'authorization/execution_claim.json'
    if not claim.exists():raise PermissionError('No formal claim to terminate')
    if status=='CONSUMED':require_scope(config()['roles']['holdout'][1], 'B')
    new_json(RUN/'authorization/formal_terminal.json', {'state':status,'at_utc':now(),'evidence':evidence})
    # Terminal file is independently checked below, so the immutable claim is retained.

def require_not_terminal():
    if (RUN/'authorization/formal_terminal.json').exists():
        raise PermissionError('Formal run already consumed or invalidated')
