"""Synthetic authorization fixtures only; no actual approval file is created."""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch,Mock
import pytest
from research.frozen.experiment_03 import guard,formal,inputs,data

def test_all_formal_entrypoints_reject_before_values_or_io():
    calls=[lambda:formal.execute(),lambda:formal.encode_test(None,None),
           lambda:formal.predict_test(None,None,None),lambda:formal.independent_numbers(None,None,None),
           lambda:data.collect_holdout(),lambda:data.audit_snapshot(),lambda:data.load_holdout_5m(),
           lambda:inputs.load_hourly('B'),lambda:inputs.prepare_window(None,None,'B'),
           lambda:inputs.build_features(None,None,'B'),lambda:inputs.build_labels(None,None,'B')]
    with patch.object(guard,'verify_authorization',side_effect=PermissionError('synthetic absent actual authorization')) as auth:
        for call in calls:
            with pytest.raises(PermissionError):call()
        assert auth.call_count==len(calls)

def test_phase_a_timestamp_rejection():
    assert guard.require_scope('2026-04-01T00:00:00Z','A').isoformat()=='2026-04-01T00:00:00+00:00'
    with pytest.raises(PermissionError):guard.require_scope('2026-04-01T00:00:01Z','A')
    with pytest.raises(ValueError):guard.require_scope('2026-04-01','A')

def test_claim_must_be_unique_and_models_ready_before_test():
    with tempfile.TemporaryDirectory(dir=guard.ROOT/'tmp') as tmp:
        run=Path(tmp);(run/'authorization').mkdir()
        approval=run/'authorization/user_approval.json';approval.write_text('{"context":"synthetic fixture"}')
        with patch.object(guard,'RUN',run),patch.object(guard,'verify_authorization',return_value={'sealed_bundle_sha256':'synthetic'}):
            guard.claim_formal(run)
            with pytest.raises(FileExistsError):guard.claim_formal(run)
            guard.require_scope('2026-10-01T00:00:00Z','B',purpose='training')
            with pytest.raises(PermissionError,match='must precede'):guard.require_scope('2026-10-01T00:00:00Z','B')
            model=run/'models/formal/m.json';model.parent.mkdir(parents=True);model.write_text('{"fixture":true}')
            guard.new_json(model.parent/'preview_identity_check.json',{'status':'PASS','all_selected_models_exact':True,'files_sha256':{'models/formal/m.json':guard.sha(model)}})
            guard.require_scope('2026-10-01T00:00:00Z','B')
            model.write_text('{"changed":true}')
            with pytest.raises(PermissionError,match='changed'):guard.require_scope('2026-10-01T00:00:00Z','B')

def test_invalidated_terminal_survives_broken_hash_and_blocks_retry():
    with tempfile.TemporaryDirectory(dir=guard.ROOT/'tmp') as tmp:
        run=Path(tmp);(run/'authorization').mkdir();guard.new_json(run/'authorization/execution_claim.json',{'state':'CLAIMED'})
        with patch.object(guard,'RUN',run),patch.object(guard,'verify_authorization',side_effect=PermissionError('broken hash')):
            guard.finish_formal('INVALIDATED',{'context':'synthetic','error':'broken source'})
            assert json.loads((run/'authorization/formal_terminal.json').read_text())['state']=='INVALIDATED'
            with pytest.raises(PermissionError,match='already consumed'):guard.require_scope('2026-10-01T00:00:00Z','B')
            with pytest.raises(FileExistsError):guard.finish_formal('INVALIDATED',{})

def test_concrete_bundle_file_change_is_rejected():
    with tempfile.TemporaryDirectory(dir=guard.ROOT/'tmp') as tmp:
        root=Path(tmp);run=root/'run';model=root/'immutable.txt';model.write_text('first')
        guard.new_json(run/'preflight/sealed_bundle.json',{'status':'PASS','protocol_sha256':guard.CONFIG_SHA,'files_sha256':{'immutable.txt':guard.sha(model)}})
        with patch.object(guard,'ROOT',root):
            guard.verify_bundle(run);model.write_text('second changed')
            with pytest.raises(PermissionError,match='changed'):guard.verify_bundle(run)
