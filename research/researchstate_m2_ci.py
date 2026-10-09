"""Index M2-CI metadata bytes only; never run research or grant future testing."""
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
EXPERIMENT = 'M2_CONDITIONAL_INCREMENT_01'
RUN = 'research/runs/' + EXPERIMENT
CONFIG = 'research/m2_ci/config.yaml'
REGISTRY = 'research/registry/' + EXPERIMENT + '.json'
TODO = 'research/m2_ci/TODO.md'
DELIVERY = RUN + '/delivery_manifest.json'
REPORTS = {
    'md': RUN + '/reports/M2_Conditional_Increment_Development_Report.md',
    'draft': RUN + '/reports/M2_Conditional_Increment_Future_Holdout_Protocol_DRAFT.md',
    'decision': RUN + '/reports/M2_Conditional_Increment_Project_Decision.md',
}
ANCHOR_REVISION = 15
ANCHOR_SHA = '6cfe6532d9078e68e91ae62e8a50cb2549d9bea1a38006bcbb2d813f77b55058'
ALLOWED_CHANGED = {'current_stage', 'active_protocol', 'evidence', 'blockers',
                   'pending_actions', 'development_experiments'}


def resolve(path):
    value = (ROOT / path).resolve()
    if not value.is_relative_to(ROOT):
        raise ValueError('New artifact outside project: ' + str(path))
    return value


def read(path):
    return json.loads(resolve(path).read_text(encoding='utf-8-sig'))


def sha(path):
    # Original historical and attachment source evidence may be outside ROOT.
    return legacy.sha((ROOT / path).resolve())


def anchor():
    value = read(HISTORY / f'{ANCHOR_REVISION:06d}_{ANCHOR_SHA}.json')
    digest = hashlib.sha256(legacy.canonical(legacy.payload(value))).hexdigest()
    if value.get('revision') != ANCHOR_REVISION or digest != ANCHOR_SHA or value.get('content_sha256') != ANCHOR_SHA or len(value.get('evidence', [])) != 900:
        raise ValueError('Immutable revision 15 / 900-evidence anchor mismatch')
    return value


def evidence_errors(items):
    errors = []
    for item in items:
        try:
            actual = sha(item['path'])
            if actual != item.get('sha256') or (item.get('expected_sha256') and actual != item['expected_sha256']):
                errors.append('Evidence hash conflict: ' + item['path'])
        except (OSError, ValueError) as exc:
            errors.append('Evidence unavailable: ' + item['path'] + ': ' + str(exc))
    return errors


def preservation_errors(state):
    old = anchor()
    errors = []
    for key, value in legacy.payload(old).items():
        if key not in ALLOWED_CHANGED and state.get(key) != value:
            errors.append('Historical field changed: ' + key)
    if state.get('evidence', [])[:900] != old['evidence']:
        errors.append('Historical evidence entries/order/hashes changed')
    if state.get('holdout') != old['holdout'] or state.get('holdout', {}).get('status') != 'CONSUMED':
        errors.append('Historical holdout metadata must remain exactly preserved / CONSUMED')
    for key, value in old.get('development_experiments', {}).items():
        if state.get('development_experiments', {}).get(key) != value:
            errors.append('Historical development entry changed: ' + key)
    return errors


def derive(state, check_evidence=True):
    errors = legacy.chain_errors(state) + preservation_errors(state)
    if check_evidence:
        errors += evidence_errors(state.get('evidence', [])[:900])
    if errors:
        raise ValueError('; '.join(errors))
    registry = read(REGISTRY)
    protocol_sha = sha(CONFIG)
    if registry.get('experiment_id') != EXPERIMENT or registry.get('protocol_sha256') != protocol_sha:
        raise ValueError('Registry identity/protocol SHA mismatch')
    configuration = registry.get('configuration', {})
    if configuration.get('path') != CONFIG or configuration.get('sha256') != protocol_sha:
        raise ValueError('Registry must bind exact configuration path and SHA')
    if registry.get('future_formal_test_approved') is not False:
        raise ValueError('Future formal testing must remain unapproved')
    status = registry.get('status')
    normalized_status = {'IN_PROGRESS': 'IN_PROGRESS', 'in_progress': 'IN_PROGRESS',
        'COMPLETE': 'COMPLETE', 'development_complete': 'COMPLETE',
        'completed_development_research': 'COMPLETE'}.get(status)
    if normalized_status is None:
        raise ValueError('Registry status must be IN_PROGRESS/in_progress or COMPLETE/development_complete/completed_development_research')
    result = copy.deepcopy(state)
    bindings = {}

    def add(path, expected=None, role='m2_ci_metadata_bytes_only', external=False):
        target = (ROOT / path).resolve() if external else resolve(path)
        normalized = target.as_posix() if external else target.relative_to(ROOT).as_posix()
        actual = sha(normalized)
        if expected is not None and actual != expected:
            raise ValueError('Metadata hash mismatch: ' + normalized)
        if normalized in bindings and bindings[normalized] != actual:
            raise ValueError('Conflicting metadata binding: ' + normalized)
        bindings[normalized] = actual
        for index, previous in enumerate(result['evidence']):
            if (ROOT / previous['path']).resolve() == target:
                if index < 900:
                    if previous['sha256'] != actual:
                        raise ValueError('Conflicts with historical evidence: ' + normalized)
                    return
                result['evidence'][index] = {'path': normalized, 'role': role,
                    'expected_sha256': actual, 'sha256': actual, 'status': 'verified'}
                return
        result['evidence'].append({'path': normalized, 'role': role,
            'expected_sha256': actual, 'sha256': actual, 'status': 'verified'})

    add(CONFIG, protocol_sha)
    add(REGISTRY)
    add(TODO)
    attachment = registry.get('attachment')
    if not isinstance(attachment, dict) or not attachment.get('path') or not attachment.get('sha256'):
        raise ValueError('Registry requires attachment copy path and SHA')
    if not resolve(attachment['path']).is_relative_to(resolve(RUN)):
        raise ValueError('Attachment copy must be in M2 run')
    add(attachment['path'], attachment['sha256'], 'm2_ci_original_request_copy_bytes_only')
    if attachment.get('original_path'):
        if not attachment.get('original_sha256'):
            raise ValueError('Original attachment source requires SHA')
        if attachment['sha256'] != attachment['original_sha256']:
            raise ValueError('Attachment copy/source SHA mismatch')
        add(attachment['original_path'], attachment['original_sha256'],
            'm2_ci_original_request_source_bytes_only', external=True)
    entry = {'experiment_id': EXPERIMENT, 'status': 'IN_PROGRESS',
        'registry_status': status, 'holdout_status': 'CONSUMED',
        'future_formal_test_approved': False,
        'interpretation': 'DEVELOPMENT_OOF_WITH_REUSED_ARCHITECTURE_NOT_INDEPENDENT_CONFIRMATION',
        'protocol': {'path': CONFIG, 'sha256': protocol_sha},
        'authority': [CONFIG, REGISTRY, TODO, attachment['path']],
        'primary_result': 'DEVELOPMENT_DELIVERY_PENDING'}
    complete = resolve(DELIVERY).is_file()
    if complete:
        final = read(DELIVERY)
        required = {'experiment_id': EXPERIMENT, 'protocol_sha256': protocol_sha,
                    'status': 'COMPLETE', 'holdout_status': 'CONSUMED',
                    'future_formal_test_approved': False}
        if any(final.get(key) != value for key, value in required.items()):
            raise ValueError('Delivery identity/protocol/completion/holdout/approval mismatch')
        if normalized_status != 'COMPLETE':
            raise ValueError('Complete delivery requires COMPLETE registry')
        artifacts = final.get('source_and_artifact_sha256')
        if not isinstance(artifacts, dict) or not artifacts:
            raise ValueError('Delivery requires nonempty source_and_artifact_sha256')
        for path, expected in artifacts.items():
            if not isinstance(expected, str) or len(expected) != 64:
                raise ValueError('Artifact SHA missing: ' + path)
            add(path, expected, 'm2_ci_delivery_artifact_bytes_only')
        for kind, path in REPORTS.items():
            if final.get(kind + '_path') != path or not final.get(kind + '_sha256'):
                raise ValueError('Required report binding missing: ' + kind)
            add(path, final[kind + '_sha256'], 'm2_ci_report_bytes_only')
            entry[kind + '_report'] = {'path': path, 'sha256': final[kind + '_sha256']}
        primary = final.get('primary_result')
        if not isinstance(primary, str) or not primary.strip():
            raise ValueError('Delivery primary_result metadata missing')
        add(DELIVERY)
        entry.update(status='COMPLETE', primary_result=primary,
            delivery_manifest={'path': DELIVERY, 'sha256': sha(DELIVERY)})
        entry['authority'].append(DELIVERY)
    elif normalized_status == 'COMPLETE':
        raise ValueError('COMPLETE registry requires complete delivery manifest')
    entry['metadata_bindings_sha256'] = bindings
    result.setdefault('development_experiments', {})[EXPERIMENT] = entry
    result['current_stage'] = {'experiment_id': EXPERIMENT,
        'stage': 'conditional_increment_development_only',
        'phase': 'M2_CI_completed' if complete else 'M2_CI_in_progress',
        'status': 'completed' if complete else 'in_progress',
        'note': 'Metadata bytes only; historical experiments, supplements and CONSUMED holdout preserved; future formal test unapproved.'}
    result['active_protocol'] = {'path': CONFIG, 'sha256': protocol_sha, 'hash_verified': True}
    result['blockers'] = copy.deepcopy(registry.get('blockers', []))
    if not isinstance(result['blockers'], list):
        raise ValueError('Registry blockers must be a list of actual evidence-backed blockers')
    result['pending_actions'] = [] if complete else ['完成已授权M2条件增量开发研究、独立核验与报告交付；未来正式测试仅为未批准草案']
    return result


def verify(state):
    errors = legacy.chain_errors(state) + preservation_errors(state)
    errors += evidence_errors(state.get('evidence', []))
    if EXPERIMENT in state.get('development_experiments', {}):
        try:
            if legacy.payload(derive(state, check_evidence=False)) != legacy.payload(state):
                errors.append('M2 metadata differs from currently bound metadata')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(str(exc))
    elif state.get('revision') != ANCHOR_REVISION:
        errors.append('Unknown state after revision 15 anchor')
    return errors


def save(updated, old):
    errors = preservation_errors(updated)
    if errors:
        raise ValueError('; '.join(errors))
    digest = hashlib.sha256(legacy.canonical(legacy.payload(updated))).hexdigest()
    if digest == old['content_sha256']:
        return old, False
    value = legacy.payload(updated)
    value.update(revision=old['revision'] + 1, updated_at_utc=datetime.now(timezone.utc).isoformat(),
                 content_sha256=digest, previous_content_sha256=old['content_sha256'])
    if read(STATE) != old:
        raise ValueError('Current state changed concurrently')
    encoded = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    temporary = STATE.with_suffix('.m2_ci.tmp')
    with temporary.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(encoded)
    try:
        if read(STATE) != old:
            raise ValueError('Current state changed concurrently before history write')
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
            print(json.dumps({'status': 'FAIL' if errors else 'PASS', 'revision': state['revision'], 'errors': errors}, ensure_ascii=False, indent=2))
            return int(bool(errors))
        updated = derive(state)
        changed = legacy.payload(updated) != legacy.payload(state)
        if args.command == 'refresh':
            updated, changed = save(updated, state)
        print(json.dumps({'command': args.command, 'changed': changed, 'revision': updated['revision'],
            'current_stage': updated['current_stage'], 'holdout': updated['holdout']['status'],
            'development': updated['development_experiments'][EXPERIMENT],
            'historical_evidence_count': 900, 'total_evidence_count': len(updated['evidence'])}, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({'status': 'FAIL', 'error': str(exc)}, ensure_ascii=False, indent=2))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
