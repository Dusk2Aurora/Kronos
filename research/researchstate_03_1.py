"""Append experiment 3.1 metadata; never execute research or authorize holdout access."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import researchstate_phase_b as legacy

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'research/researchstate.json'
HISTORY = ROOT / 'research/state_history'
EXPERIMENT = 'FROZEN_RISK_03_1_B5_v1'
RUN = 'research/runs/' + EXPERIMENT
CONFIG = 'research/configs/frozen_risk_03_1_b5_v1.yaml'
REGISTRY = 'research/registry/' + EXPERIMENT + '.json'
TODO = 'research/frozen/experiment_03_1/TODO.md'
DELIVERY = RUN + '/delivery_manifest.json'
PDF = 'output/pdf/Kronos_Frozen_Risk_03_1_Report_v1.pdf'
MD = RUN + '/reports/Kronos_Frozen_Risk_03_1_Report_v1.md'
SHUTDOWN_HANDOFF = 'research/frozen/experiment_03_1/HANDOFF_SHUTDOWN_20261009.md'
ANCHOR_REVISION = 11
ANCHOR_SHA = 'd7df52beb741590e349daa3a4d590133794053e2b15cb4d8c42e7104eb97ce13'
ALLOWED_CHANGED = {'current_stage', 'active_protocol', 'evidence', 'blockers',
                   'pending_actions', 'supplementary_experiments'}


def resolve(path):
    value = (ROOT / path).resolve()
    if not value.is_relative_to(ROOT):
        raise ValueError('Evidence outside project: ' + str(path))
    return value


def read(path):
    return json.loads(resolve(path).read_text(encoding='utf-8-sig'))


def sha(path):
    # Original evidence includes an immutable user attachment outside ROOT.
    # Supplement additions still pass resolve() before reaching this function.
    return legacy.sha((ROOT / path).resolve())


def anchor():
    path = HISTORY / f'{ANCHOR_REVISION:06d}_{ANCHOR_SHA}.json'
    value = read(path)
    digest = hashlib.sha256(legacy.canonical(legacy.payload(value))).hexdigest()
    if value.get('revision') != ANCHOR_REVISION or digest != ANCHOR_SHA or value.get('content_sha256') != ANCHOR_SHA:
        raise ValueError('Immutable revision 11 anchor mismatch')
    return value


def preservation_errors(state):
    old = anchor()
    errors = []
    for key, value in legacy.payload(old).items():
        if key not in ALLOWED_CHANGED and state.get(key) != value:
            errors.append('Historical field changed: ' + key)
    old_evidence = old['evidence']
    if state.get('evidence', [])[:len(old_evidence)] != old_evidence:
        errors.append('Original evidence entries/order/expected hashes changed')
    if state.get('holdout', {}).get('status') != 'CONSUMED':
        errors.append('Historical holdout must remain CONSUMED')
    for item in old_evidence:
        try:
            actual = sha(item['path'])
            if actual != item['sha256'] or (item.get('expected_sha256') and actual != item['expected_sha256']):
                errors.append('Historical evidence hash conflict: ' + item['path'])
        except OSError as exc:
            errors.append('Historical evidence unavailable: ' + item['path'] + ': ' + str(exc))
    return errors


def derive(state):
    errors = legacy.chain_errors(state) + preservation_errors(state)
    if errors:
        raise ValueError('; '.join(errors))
    registry = read(REGISTRY)
    if registry.get('experiment_id') != EXPERIMENT:
        raise ValueError('Supplement registry identity mismatch')
    protocol_sha = sha(CONFIG)
    if registry.get('protocol_sha256') != protocol_sha:
        raise ValueError('Supplement registry must bind exact protocol_sha256')
    # TODO is only hashed; no scientific values or report content are parsed.
    todo_sha = sha(TODO)
    result = copy.deepcopy(state)
    old_count = len(anchor()['evidence'])
    result['evidence'] = result['evidence'][:old_count]
    bindings = {}

    def add(path, expected=None, role='supplement_03_1_metadata'):
        normalized = resolve(path).relative_to(ROOT).as_posix()
        actual = sha(normalized)
        if expected is not None and actual != expected:
            raise ValueError('Supplement hash mismatch: ' + normalized)
        if normalized in bindings and bindings[normalized] != actual:
            raise ValueError('Conflicting binding: ' + normalized)
        bindings[normalized] = actual
        for historical in result['evidence'][:old_count]:
            if (ROOT / historical['path']).resolve() == resolve(normalized):
                if historical['sha256'] != actual:
                    raise ValueError('Supplement conflicts with historical evidence: ' + normalized)
                return
        if not any(item['path'] == normalized for item in result['evidence'][old_count:]):
            result['evidence'].append({'path': normalized, 'role': role,
                                     'expected_sha256': actual, 'sha256': actual, 'status': 'verified'})

    add(CONFIG, protocol_sha)
    add(REGISTRY)
    add(TODO, todo_sha)
    complete = False
    paused = registry.get('status') == 'paused_for_user_shutdown'
    entry = {'experiment_id': EXPERIMENT, 'status': 'IN_PROGRESS',
             'registry_status': registry.get('status'),
             'interpretation': 'POST_HOC_EXPLORATORY_SUPPLEMENT',
             'formal_status_unchanged': True, 'holdout_status': 'CONSUMED',
             'protocol': {'path': CONFIG, 'sha256': protocol_sha},
             'authority': [CONFIG, REGISTRY, TODO],
             'primary_result': 'POST_HOC_B5_SUPPLEMENT_PENDING'}
    if paused:
        handoff = registry.get('shutdown_handoff')
        if not isinstance(handoff, dict):
            raise ValueError('Paused supplement requires shutdown_handoff metadata')
        if not handoff.get('path') or resolve(handoff['path']) != resolve(SHUTDOWN_HANDOFF):
            raise ValueError('Shutdown handoff path mismatch')
        manifest_path = handoff.get('shutdown_manifest_path')
        if not manifest_path or not resolve(manifest_path).is_relative_to(resolve(RUN)):
            raise ValueError('Shutdown manifest must be within supplement run')
        for path_key, hash_key in (('path', 'sha256'),
                                   ('shutdown_manifest_path', 'shutdown_manifest_sha256')):
            expected = handoff.get(hash_key)
            if not isinstance(expected, str) or len(expected) != 64:
                raise ValueError('Shutdown handoff hash missing: ' + hash_key)
            add(handoff[path_key], expected, 'supplement_03_1_shutdown_metadata')
        entry.update(status='PAUSED', shutdown_handoff=copy.deepcopy(handoff))
        entry['authority'].extend([handoff['path'], manifest_path])
    if resolve(DELIVERY).is_file():
        if paused:
            raise ValueError('Paused registry conflicts with complete delivery manifest')
        final = read(DELIVERY)
        if final.get('experiment_id') != EXPERIMENT or final.get('protocol_sha256') != protocol_sha:
            raise ValueError('Supplement delivery identity/protocol mismatch')
        if final.get('status') != 'COMPLETE':
            raise ValueError('Present delivery manifest must be COMPLETE')
        if final.get('holdout_status', 'CONSUMED') != 'CONSUMED':
            raise ValueError('Supplement cannot change holdout status')
        for kind, required in (('pdf', PDF), ('md', MD)):
            path = final.get(kind + '_path')
            if not path or resolve(path) != resolve(required) or not final.get(kind + '_sha256'):
                raise ValueError('Required supplement report binding missing: ' + kind)
            add(path, final[kind + '_sha256'], 'supplement_03_1_report_bytes_only')
            entry[kind + '_report'] = {'path': required, 'sha256': final[kind + '_sha256']}
        artifacts = final.get('source_and_artifact_sha256')
        if not isinstance(artifacts, dict) or not artifacts:
            raise ValueError('Supplement delivery must bind source and artifact hashes')
        for path, expected in artifacts.items():
            add(path, expected, 'supplement_03_1_delivery_bytes_only')
        primary = final.get('primary_result')
        if not isinstance(primary, str) or not primary.strip():
            raise ValueError('Supplement primary_result metadata missing')
        add(DELIVERY)
        entry.update(status='COMPLETE', primary_result=primary,
                     delivery_manifest={'path': DELIVERY, 'sha256': sha(DELIVERY)})
        entry['authority'].append(DELIVERY)
        complete = True
    entry['metadata_bindings_sha256'] = bindings
    result.setdefault('supplementary_experiments', {})[EXPERIMENT] = entry
    result['current_stage'] = {'experiment_id': EXPERIMENT,
        'stage': 'frozen_risk_post_hoc_b5_supplement',
        'phase': 'Supplement03_1_completed' if complete else ('Supplement03_1_paused' if paused else 'Supplement03_1_in_progress'),
        'status': 'completed' if complete else ('paused' if paused else 'in_progress'),
        'note': 'Metadata only; original formal experiment and CONSUMED holdout preserved. Post hoc exploration has no new confirmatory status.'}
    result['active_protocol'] = {'path': CONFIG, 'sha256': protocol_sha, 'hash_verified': True}
    result['pending_actions'] = [] if complete else ['完成已授权B5事后补充、独立核验及3.1报告交付；原正式结论保持不变']
    if paused:
        result['pending_actions'] = ['用户要求关机暂停；等待用户明天恢复。先独立封存operational amendment，核验已保存的5个候选并保持不重训；未完成候选w16_wd0.001_s43须重做，不声称RAM中的checkpoint已保存。不得直接重跑原train：training_claim排他且尚无完整聚合输出。']
    result['blockers'] = []
    return result


def verify(state):
    errors = legacy.chain_errors(state) + preservation_errors(state)
    for item in state.get('evidence', []):
        try:
            actual = sha(item['path'])
            if actual != item.get('sha256') or (item.get('expected_sha256') and actual != item['expected_sha256']):
                errors.append('Evidence hash conflict: ' + item['path'])
        except (OSError, ValueError) as exc:
            errors.append('Evidence unavailable: ' + item['path'] + ': ' + str(exc))
    if EXPERIMENT in state.get('supplementary_experiments', {}):
        try:
            if legacy.payload(derive(state)) != legacy.payload(state):
                errors.append('Supplement metadata differs from current bound metadata')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(str(exc))
    elif state.get('revision') != ANCHOR_REVISION:
        errors.append('Unknown state after historical anchor')
    return errors


def save(updated, old):
    digest = hashlib.sha256(legacy.canonical(legacy.payload(updated))).hexdigest()
    if digest == old['content_sha256']:
        return old, False
    value = legacy.payload(updated)
    value.update(revision=old['revision'] + 1, updated_at_utc=datetime.now(timezone.utc).isoformat(),
                 content_sha256=digest, previous_content_sha256=old['content_sha256'])
    if read(STATE) != old:
        raise ValueError('Current state changed concurrently')
    encoded = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    temporary = STATE.with_suffix('.03_1.tmp')
    with temporary.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(encoded)
    try:
        with (HISTORY / f"{value['revision']:06d}_{digest}.json").open('x', encoding='utf-8', newline='\n') as stream:
            stream.write(encoded)
        os.replace(temporary, STATE)
    finally:
        if temporary.exists():
            temporary.unlink()
    return value, True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('preview', 'refresh', 'verify'))
    args = parser.parse_args()
    try:
        state = read(STATE)
        if args.command == 'verify':
            errors = verify(state)
            print(json.dumps({'status': 'FAIL' if errors else 'PASS', 'revision': state['revision'],
                              'errors': errors}, ensure_ascii=False, indent=2))
            return int(bool(errors))
        updated = derive(state)
        changed = legacy.payload(updated) != legacy.payload(state)
        if args.command == 'refresh':
            updated, changed = save(updated, state)
        print(json.dumps({'command': args.command, 'changed': changed, 'revision': updated['revision'],
            'current_stage': updated['current_stage'], 'holdout': updated['holdout']['status'],
            'supplement': updated['supplementary_experiments'][EXPERIMENT],
            'historical_evidence_count': len(anchor()['evidence']),
            'total_evidence_count': len(updated['evidence'])}, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({'status': 'FAIL', 'error': str(exc)}, ensure_ascii=False, indent=2))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
