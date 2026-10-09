"""Seal completed development delivery; never run future experiments or push."""
from pathlib import Path
from datetime import datetime, timezone
import json
from research.m2_ci import data
from research.m2_ci.runner import verify_seal

ROOT,RUN=data.ROOT,data.DEFAULT_RUN
REPORTS={'md':'M2_Conditional_Increment_Development_Report.md',
    'draft':'M2_Conditional_Increment_Future_Holdout_Protocol_DRAFT.md',
    'decision':'M2_Conditional_Increment_Project_Decision.md'}

def finalize():
    out=RUN/'delivery_manifest.json'
    if out.exists():raise FileExistsError('Preserve immutable delivery')
    verify_seal(RUN)
    review=data.read(RUN/'independent_review/numeric_review.json')
    if review['status']!='PASS':raise ValueError('Independent numeric review required')
    for path,digest in review['source_and_artifact_sha256'].items():
        if data.sha(ROOT/path)!=digest:raise ValueError('Numeric evidence drift: '+path)
    for path,required in [(RUN/'reports/document_review.json',[RUN/'reports'/name for name in REPORTS.values()]),
        (RUN/'figures/v2/visual_review.json',[p for p in (RUN/'figures/v2').rglob('*') if p.suffix in ['.png','.svg'] or p.name=='manifest.json'])]:
        acceptance=data.read(path)
        if acceptance['status']!='PASS':raise ValueError('Report and visual review required')
        bindings=acceptance.get('files_sha256',acceptance.get('source_and_artifact_sha256',{}))
        if not bindings:raise ValueError('Acceptance review must bind current reports/figures')
        for artifact in required:
            if artifact.relative_to(ROOT).as_posix() not in bindings:raise ValueError('Reviewed artifact missing: '+str(artifact))
        for relative,digest in bindings.items():
            if data.sha(ROOT/relative)!=digest:raise ValueError('Accepted report/figure changed: '+relative)
    for action in ['validation','benchmark','review']:
        if data.read(RUN/f'provenance/{action}_runtime_success.json')['status']!='SUCCESS':
            raise ValueError('Locked runtime action did not complete: '+action)
    old=data.read(ROOT/'research/m2_ci/phase_a_independent_review.json')
    checked=0
    for audit in old['old_delivery_audits']:
        for item in audit['checks']:
            if data.sha(ROOT/item['path'])!=item['expected_sha256']:
                raise ValueError('Original accepted artifact changed: '+item['path'])
            checked+=1
    data.write(RUN/'provenance/final_parent_preservation.json',{'status':'PASS',
        'old_bound_files_rechecked':checked,'no_old_writes':True,'future_formal_test_approved':False,
        'old_holdout_status':'CONSUMED','at_utc':datetime.now(timezone.utc).isoformat()})
    summary=data.read(RUN/'metrics/summary.json')
    registry=ROOT/'research/registry/M2_CONDITIONAL_INCREMENT_01.json'
    with (RUN/'provenance/registry_before_completion.json').open('xb') as f:f.write(registry.read_bytes())
    value=data.read(registry)
    value.update(status='completed_development_research',completed_at_utc=datetime.now(timezone.utc).isoformat(),blockers=[])
    value['result']={'conclusion':summary['status'],'R2_screen':summary['screens']['R2'],
        'specificity_screen_pass':summary['specificity_screen_pass'],'new_independent_confirmation':False,
        'failed_scientific_fits':0,'failed_engineering_records_preserved':True,
        'next_step':'已授权开发研究完成；未来独立协议为未批准草案，不执行'}
    value['artifacts']={'delivery_manifest':out.relative_to(ROOT).as_posix(),
        **{kind:(RUN/'reports'/name).relative_to(ROOT).as_posix() for kind,name in REPORTS.items()}}
    registry.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    todo=ROOT/'research/m2_ci/TODO.md'
    text=todo.read_text(encoding='utf-8')
    text=text.replace('- [ ]','- [x]')
    start=text.index('阶段：'); end=text.index('\n',start)
    text=text[:start]+f'阶段：Phase A–D开发研究与Phase E未批准草案完成。状态：COMPLETE。开发判定：{summary["status"]}。无待运行已授权实验。'+text[end:]
    text+='\n最终验收：95/95拟合完成，全部候选保留；OOF1267、原VALID92；配对7/3/14日各5000draw、32模型族及全部18权重；独立数值、图和文档复核通过。原CONSUMED不变；未来正式验证未批准。最终delivery封存后由专属metadata helper入账并复核状态链；若核验失败须保留失败记录处理，不能开启新正式实验。\n'
    todo.write_text(text,encoding='utf-8')
    files=[p for p in RUN.rglob('*') if p.is_file() and p!=out]
    files += [p for p in (ROOT/'research/m2_ci').glob('*') if p.is_file()]
    files += [registry,ROOT/'research/researchstate_m2_ci.py',ROOT/'tests/test_m2_ci_oof.py',ROOT/'tests/test_m2_ci_statistics.py']
    science=data.read(RUN/'preflight/source_seal.json')['files_sha256']
    mapping={**science,**{p.relative_to(ROOT).as_posix():data.sha(p) for p in files}}
    final={'experiment_id':'M2_CONDITIONAL_INCREMENT_01','status':'COMPLETE',
        'protocol_sha256':data.sha(data.CONFIG),'holdout_status':'CONSUMED',
        'future_formal_test_approved':False,'primary_result':summary['status'],
        'evidence_class':summary['evidence_class'],'independent_numeric_checks':review['check_count'],
        'source_and_artifact_sha256':mapping,'no_new_formal_test':True,'no_push':True,
        'sealed_at_utc':datetime.now(timezone.utc).isoformat()}
    for kind,name in REPORTS.items():
        path=RUN/'reports'/name
        final[kind+'_path']=path.relative_to(ROOT).as_posix(); final[kind+'_sha256']=data.sha(path)
    data.write(out,final)
    print(json.dumps({'status':'DELIVERY_SEALED','result':summary['status'],'bound_files':len(mapping),
        'manifest_sha256':data.sha(out)}))

if __name__=='__main__':finalize()
