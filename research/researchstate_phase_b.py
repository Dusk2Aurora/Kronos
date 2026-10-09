"""Phase B metadata index; never executes research or grants access permissions."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'research/researchstate.json'
HISTORY = ROOT / 'research/state_history'
THIRD = 'FROZEN_RISK_03_v1'
RUN = 'research/runs/' + THIRD
CONFIG = 'research/configs/frozen_risk_03_v1.yaml'
CONFIG_SHA = '7ed8affd7cd624cde89d23d29aaca7ef2ab660167fac7d3dc25fac274e394b7f'
BUNDLE = RUN + '/preflight/sealed_bundle.json'
BUNDLE_SHA = 'e0d854bc4a376194344e507a5d2d9e1a0573b778f74f82970f881fae689b3376'
LIVE = {'research/registry/FROZEN_RISK_03_v1.json', 'research/frozen/experiment_03/TODO.md'}
AUTH = RUN + '/authorization/user_approval.json'
CLAIM = RUN + '/authorization/execution_claim.json'
TERMINAL = RUN + '/authorization/formal_terminal.json'
DELIVERY = RUN + '/formal_delivery_manifest.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8-sig'))


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')


def payload(state):
    return {k: v for k, v in state.items() if k not in ('revision', 'updated_at_utc', 'content_sha256', 'previous_content_sha256')}


def chain_errors(state):
    errors = []
    if state.get('schema_version') != 2:
        errors.append('schema_version must be 2')
    revision = state.get('revision')
    if not isinstance(revision, int) or revision < 1:
        return errors + ['Missing valid revision']
    previous = None
    for number in range(1, revision + 1):
        matches = list(HISTORY.glob(f'{number:06d}_*.json'))
        if len(matches) != 1:
            errors.append(f'Expected one history file: revision {number}')
            continue
        try:
            historical = read(matches[0])
            digest = hashlib.sha256(canonical(payload(historical))).hexdigest()
            if historical.get('revision') != number or historical.get('content_sha256') != digest or matches[0].name != f'{number:06d}_{digest}.json':
                errors.append(f'History payload/filename/revision mismatch: {number}')
            if historical.get('previous_content_sha256') != previous:
                errors.append(f'History chain mismatch: {number}')
            previous = historical.get('content_sha256')
            if number == revision and historical != state:
                errors.append('Current state differs from immutable history')
        except (OSError, ValueError) as exc:
            errors.append(f'Unreadable history {number}: {exc}')
    digest = hashlib.sha256(canonical(payload(state))).hexdigest()
    if state.get('content_sha256') != digest:
        errors.append('Current payload hash mismatch')
    if any(int(p.name.split('_', 1)[0]) > revision for p in HISTORY.glob('[0-9][0-9][0-9][0-9][0-9][0-9]_*.json')):
        errors.append('History ahead of current state; reconcile before writing')
    return errors


def authority():
    """Validate metadata bindings; the recorded message remains the authority."""
    if not (ROOT / AUTH).is_file():
        raise ValueError('No subsequent user authorization; refuse Phase B update')
    auth = read(AUTH)
    if auth.get('experiment_id') != THIRD or auth.get('protocol_sha256') != CONFIG_SHA or auth.get('sealed_bundle_sha256') != BUNDLE_SHA or auth.get('source') != 'subsequent_user_message':
        raise ValueError('Authorization identity/source/protocol/bundle mismatch')
    if sha(ROOT / CONFIG) != CONFIG_SHA or sha(ROOT / BUNDLE) != BUNDLE_SHA:
        raise ValueError('Locked protocol or bundle changed')
    if auth.get('approval_text') != '授权开启 FROZEN_RISK_03_v1 holdout，按已锁定协议执行正式测试。':
        raise ValueError('Authorization scope declaration mismatch')
    message = auth.get('user_message_record')
    records = [(AUTH, sha(ROOT / AUTH)), (CONFIG, CONFIG_SHA), (BUNDLE, BUNDLE_SHA)]
    if isinstance(message, list):
        if message != ['这个时候不用生成临时pdf，你可以推进', '直到完全推进第三次实验位置，我完全同意你进行'] or auth.get('approval_text_is_normalized_scope_not_verbatim') is not True or not auth.get('interpretation'):
            raise ValueError('Actual subsequent user messages or interpretation missing/conflicting')
        # Exact original quotes are part of the hashed immutable authorization record.
    elif isinstance(message, str) and message:
        message_path = ROOT / message
        raw = message_path.read_text(encoding='utf-8-sig')
        expected = auth.get('user_message_record_sha256') or auth.get('user_message_sha256')
        if not raw.strip() or not expected or sha(message_path) != expected:
            raise ValueError('Actual user message empty or hash missing/mismatched')
        records.append((message, expected))
    else:
        raise ValueError('Authorization must preserve actual subsequent user messages')
    status = 'AUTHORISED'
    if (ROOT / CLAIM).is_file():
        claim = read(CLAIM)
        if any(claim.get(k) != v for k, v in {'state': 'CLAIMED', 'experiment_id': THIRD, 'protocol_sha256': CONFIG_SHA, 'sealed_bundle_sha256': BUNDLE_SHA, 'authorization_sha256': sha(ROOT / AUTH)}.items()):
            raise ValueError('Execution claim binding mismatch')
        records.append((CLAIM, sha(ROOT / CLAIM)))
        status = 'CLAIMED'
    if (ROOT / TERMINAL).is_file():
        if status != 'CLAIMED':
            raise ValueError('Terminal exists without a valid claim')
        terminal = read(TERMINAL)
        status = terminal.get('state')
        if status not in ('CONSUMED', 'INVALIDATED'):
            raise ValueError('Invalid formal terminal state')
        protocol = terminal.get('evidence', {}).get('protocol_sha256')
        if protocol is not None and protocol != CONFIG_SHA:
            raise ValueError('Terminal protocol conflict')
        records.append((TERMINAL, sha(ROOT / TERMINAL)))
    return status, records


def refresh(state):
    errors = chain_errors(state)
    if errors:
        raise ValueError('; '.join(errors))
    status, records = authority()
    result = copy.deepcopy(state)
    evidence = result['evidence']
    # Old immutable evidence retains its acceptance hash; only live progress can evolve.
    for item in evidence:
        if item['path'].replace('\\', '/') in LIVE:
            item.update(expected_sha256=None, sha256=sha(ROOT / item['path']), status='indexed')
        elif not item.get('expected_sha256'):
            item['expected_sha256'] = item['sha256']

    def add(path, expected, role='third_formal_authority_metadata'):
        actual = sha(ROOT / path)
        if actual != expected:
            raise ValueError('Authority hash mismatch: ' + path)
        entry = {'path': path, 'role': role, 'expected_sha256': expected, 'sha256': actual, 'status': 'verified'}
        for index, old in enumerate(evidence):
            if old['path'] == path:
                evidence[index] = entry
                break
        else:
            evidence.append(entry)

    for path, expected in records:
        add(path, expected)
    third = result['experiments'][THIRD]
    third.setdefault('phase_a_delivery_manifest', copy.deepcopy(third.get('delivery_manifest')))
    third.setdefault('phase_a_report', copy.deepcopy(third.get('report')))
    third.setdefault('phase_a_markdown_report', copy.deepcopy(third.get('markdown_report')))
    registry = read('research/registry/FROZEN_RISK_03_v1.json')
    third['registry_status'] = registry.get('status')
    third['engineering_status'] = registry.get('engineering_status')
    third['holdout_status'] = status
    third['holdout_authorized'] = True
    third['protocol_locked'] = True
    third['formal_delivery_manifest'] = None
    phase, stage_status = 'PhaseB_execution', 'in_progress'
    pending = ['完成一次正式执行及独立复核；不得重试科学执行', '正式报告、交付manifest与最终验收待完成']
    if status == 'AUTHORISED':
        pending.insert(0, '已记录授权，尚未建立唯一执行claim')
    elif status == 'CONSUMED':
        phase = 'PhaseB_publication'
        pending = ['正式执行已消耗；完成报告、交付manifest和最终验收，不重复研究']
    elif status == 'INVALIDATED':
        phase, stage_status = 'PhaseB_invalidated', 'blocked'
        pending = ['保留失败与terminal证据；禁止自动重试，等待用户决定后续研究范围']
    if (ROOT / DELIVERY).is_file():
        if status not in ('CONSUMED', 'INVALIDATED'):
            raise ValueError('Formal delivery requires a valid terminal state')
        final = read(DELIVERY)
        if final.get('experiment_id') != THIRD or final.get('protocol_sha256') != CONFIG_SHA or final.get('sealed_bundle_sha256') != BUNDLE_SHA or final.get('holdout_status', final.get('holdout')) != status:
            raise ValueError('Formal delivery identity/protocol/bundle/terminal mismatch')
        if final.get('authorization_sha256') != sha(ROOT / AUTH) or final.get('execution_claim_sha256') != sha(ROOT / CLAIM) or final.get('formal_terminal_sha256') != sha(ROOT / TERMINAL):
            raise ValueError('Formal delivery must bind authorization and terminal hashes')
        if final.get('status') != 'COMPLETE':
            raise ValueError('Formal delivery is not complete')
        bound = 0
        if final.get('scientific_acceptance_path'):
            add(final['scientific_acceptance_path'], final['scientific_acceptance_sha256'], 'third_formal_acceptance_bytes_only')
            bound += 1
        for key in ('files_sha256', 'source_and_artifact_sha256', 'final_source_and_document_sha256'):
            for path, expected in final.get(key, {}).items():
                add(path, expected, 'third_formal_delivery_bytes_only')
                bound += 1
        for kind in ('pdf', 'markdown'):
            if final.get(kind + '_path'):
                add(final[kind + '_path'], final[kind + '_sha256'], 'third_formal_report_bytes_only')
                third[kind + '_report'] = {'path': final[kind + '_path'], 'sha256': final[kind + '_sha256']}
                if kind == 'pdf':
                    third['report'] = dict(third['pdf_report'], pages=final.get('pdf_pages'))
                bound += 1
        if not bound:
            raise ValueError('Formal delivery has no bound artifacts')
        add(DELIVERY, sha(ROOT / DELIVERY))
        third['formal_delivery_manifest'] = {'path': DELIVERY, 'sha256': sha(ROOT / DELIVERY)}
        third['primary_result'] = 'INVALIDATED' if status == 'INVALIDATED' else final.get('primary_result', final.get('main_status', 'NOT_INDEXED'))
        third['engineering_status'] = final.get('engineering_status', third['engineering_status'])
        phase, stage_status, pending = ('PhaseB_completed_invalidated' if status == 'INVALIDATED' else 'PhaseB_completed'), 'completed', []
    else:
        third['primary_result'] = 'FORMAL_DELIVERY_PENDING' if status == 'CONSUMED' else 'NOT_INDEXED_PENDING_FORMAL_DELIVERY'
    third['status'] = registry.get('status', stage_status.upper() + '_' + phase.upper())
    third['authority'] = list(dict.fromkeys(third.get('authority', []) + [p for p, _ in records] + ([DELIVERY] if third['formal_delivery_manifest'] else [])))
    result['current_stage'].update(phase=phase, status=stage_status, note='Phase B metadata derived from user authorization, immutable claim/terminal and separate formal delivery; Phase A retained.')
    result['holdout'].update(status=status, authorization={'path': AUTH, 'sha256': sha(ROOT / AUTH)}, no_read_download_labels_encode_prediction_eval=status in ('CONSUMED', 'INVALIDATED'))
    result['pending_actions'] = pending
    result['blockers'] = []
    for item in evidence:
        target = ROOT / item['path']
        if not target.is_file() or sha(target) != item['sha256'] or (item.get('expected_sha256') and item['sha256'] != item['expected_sha256']):
            result['blockers'].append({'type': 'evidence_hash_conflict', 'path': item['path']})
    return result


def verify(state):
    errors = chain_errors(state)
    for item in state.get('evidence', []):
        target = ROOT / item['path']
        if not target.is_file() or sha(target) != item['sha256'] or (item.get('expected_sha256') and item['expected_sha256'] != item['sha256']):
            errors.append('Evidence hash conflict: ' + item['path'])
    if state.get('current_stage', {}).get('phase', '').startswith('PhaseB_'):
        try:
            derived = refresh(state)
            if payload(derived) != payload(state):
                errors.append('Current metadata differs from authoritative Phase B state')
        except (OSError, ValueError, KeyError, TypeError) as exc:
            errors.append(str(exc))
    return errors


def save(state, old):
    if state.get('blockers'):
        raise ValueError('Evidence blockers prevent refresh: ' + json.dumps(state['blockers']))
    digest = hashlib.sha256(canonical(payload(state))).hexdigest()
    if digest == old['content_sha256']:
        return old, False
    value = payload(state)
    value.update(revision=old['revision'] + 1, updated_at_utc=datetime.now(timezone.utc).isoformat(), content_sha256=digest, previous_content_sha256=old['content_sha256'])
    if read(STATE) != old:
        raise ValueError('Current state changed concurrently')
    encoded = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    with (HISTORY / f"{value['revision']:06d}_{digest}.json").open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(encoded)
    temporary = STATE.with_suffix('.phase_b.tmp')
    with temporary.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(encoded)
    os.replace(temporary, STATE)
    return value, True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('verify', 'preview', 'refresh'))
    args = parser.parse_args()
    state = read(STATE)
    try:
        if args.command == 'verify':
            errors = verify(state)
            print(json.dumps({'status': 'FAIL' if errors else 'PASS', 'revision': state['revision'], 'errors': errors}, ensure_ascii=False, indent=2))
            return int(bool(errors))
        updated = refresh(state)
        changed = False
        if args.command == 'refresh':
            updated, changed = save(updated, state)
        print(json.dumps({'command': args.command, 'changed': changed, 'revision': updated['revision'], 'current_stage': updated['current_stage'], 'holdout': updated['holdout']['status'], 'blockers': updated['blockers'], 'pending_actions': updated['pending_actions']}, ensure_ascii=False, indent=2))
        return int(bool(updated['blockers']))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({'status': 'FAIL', 'error': str(exc)}, ensure_ascii=False, indent=2))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

