"""Seal reviewed 3.1 publication metadata; cannot run models or alter parent."""
from pathlib import Path
import argparse, hashlib, json
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'research/runs/FROZEN_RISK_03_1_B5_v1'
PARENT=ROOT/'research/runs/FROZEN_RISK_03_v1'
EXPERIMENT='FROZEN_RISK_03_1_B5_v1'
PROTOCOL='4636e4f7cc5db9ab3a9e425b518c5f2674322a8d67293bed1ab358e8cd96741c'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rel(p):return Path(p).relative_to(ROOT).as_posix()
def write(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def finalize():
    assert sha(ROOT/'research/configs/frozen_risk_03_1_b5_v1.yaml')==PROTOCOL
    parent=read(PARENT/'formal_delivery_manifest.json')
    for name,digest in parent['source_and_artifact_sha256'].items():assert sha(ROOT/name)==digest,name
    source=read(RUN/'preflight/pretrain_review_seal.json')
    assert source['status']=='PASS' and source['protocol_sha256']==PROTOCOL
    assert source['parent_delivery_sha256']==sha(PARENT/'formal_delivery_manifest.json')
    assert source['parent_preparation_seal_sha256']==sha(RUN/'preflight/source_seal.json')
    for name,digest in source['files_sha256'].items():assert sha(ROOT/name)==digest,name
    summary=read(RUN/'metrics/summary.json');numeric=read(RUN/'independent_review/numeric_review_v2.json')
    assert summary['status']=='POST_HOC_B5_SUPPLEMENT'
    assert summary['original_primary_status']==parent['primary_result']
    assert numeric['status']=='PASS' and numeric['all_bootstrap_attempts_verified'] and numeric['all_saved_draw_matches']
    assert numeric['recovery_checks_passed'] is True
    assert (numeric['candidate_fits'],numeric['fit_attempts_including_user_interruption'],numeric['interrupted_attempts'])==(12,13,1)
    reviewed=numeric['inputs_sha256']
    assert isinstance(reviewed,dict) and reviewed
    for name,digest in reviewed.items():assert sha(ROOT/name)==digest,name
    required=['metrics/summary.json','models/selected.json','models/fit_events.jsonl','inputs/windows.npz','inputs/train_scaler.json','labels/reused_risk_labels.csv','predictions/test_predictions_only.csv','predictions/train_B5_predictions_only.csv']
    required += [f'metrics/{name}.csv' for name in ['quarter_metrics','row_metrics','underprediction','calibration']]
    required += [f'bootstrap/draws_block{b}.csv' for b in [7,3,14]]
    required += ['preflight/recovery_seal_20261009.json','models/recovery_claim_20261009.json','models/interrupted_attempt_resources.json','models/recovered_candidate_audit.json','models/candidate_audit.json','provenance/shutdown_20261009/shutdown_manifest.json','provenance/shutdown_20261009/fit_events.jsonl','provenance/pre_recovery_tool_versions/manifest.json']
    for name in required:assert rel(RUN/name) in reviewed,name
    assert rel(ROOT/'research/delivery/frozen_risk_03_1_review.py') in reviewed
    for b in [7,3,14]:assert rel(PARENT/f'bootstrap/multiplicities_block{b}.npz') in reviewed
    recovery=read(RUN/'preflight/recovery_seal_20261009.json')
    amendment=ROOT/recovery['amendment']
    assert rel(amendment) in reviewed
    assert recovery['status']=='PASS' and recovery['protocol_sha256']==PROTOCOL
    assert recovery['parent_delivery_sha256']==sha(PARENT/'formal_delivery_manifest.json')
    journal=RUN/'models/fit_events.jsonl';prefix=journal.read_bytes()[:recovery['journal_prefix_bytes']]
    assert hashlib.sha256(prefix).hexdigest()==recovery['journal_prefix_sha256']
    for name,digest in recovery['files_sha256'].items():
        assert name in reviewed,name
        actual=hashlib.sha256(prefix).hexdigest() if name==rel(journal) else sha(ROOT/name)
        assert actual==digest,name
    qa=read(RUN/'reports/final_pdf_qa.json');pub=read(RUN/'reports/publication_manifest.json')
    assert pub['publication_code_sha256']==sha(ROOT/'research/delivery/frozen_risk_03_1_report.py')
    assert qa['status']=='PASS' and qa['pdf_sha256']==pub['pdf_sha256']
    precheck=read(RUN/'reports/pdf_text_precheck.json')
    assert precheck['status']=='PASS' and precheck['pdf_sha256']==pub['pdf_sha256']
    assert precheck['publication_manifest_sha256']==sha(RUN/'reports/publication_manifest.json')
    assert qa['precheck_sha256']==sha(RUN/'reports/pdf_text_precheck.json')
    assert qa['visual_supplement_pages']==list(range(1,15))
    for relative,digest in pub['input_sha256'].items():assert sha(ROOT/relative)==digest,relative
    for name in ['preflight/recovery_seal_20261009.json','models/interrupted_attempt_resources.json']:
        assert rel(RUN/name) in pub['input_sha256'],name
    assert rel(RUN/'independent_review/numeric_review_v2.json') in pub['input_sha256']
    assert rel(RUN/'independent_review/numeric_review.json') in pub['input_sha256']
    assert read(RUN/'independent_review/numeric_review.json')['status']=='FAIL'
    assert rel(amendment) in pub['input_sha256']
    pdf=ROOT/'output/pdf/Kronos_Frozen_Risk_03_1_Report_v1.pdf';md=RUN/'reports/Kronos_Frozen_Risk_03_1_Report_v1.md'
    assert sha(pdf)==qa['pdf_sha256'] and sha(md)==pub['md_sha256']
    for path in ['preflight/inputs_audit.json','preflight/model_tests.json','preflight/inference_audit.json','models/candidate_audit.json','provenance/resources.json']:assert read(RUN/path)['status']=='PASS',path
    events=[json.loads(s) for s in (RUN/'models/fit_events.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(events)==26 and {s:sum(e['state']==s for e in events) for s in ('STARTED','COMPLETED','INTERRUPTED')}=={'STARTED':13,'COMPLETED':12,'INTERRUPTED':1}
    dest=RUN/'delivery_manifest.json';acceptance=RUN/'final_acceptance.json'
    assert not dest.exists() and not acceptance.exists()
    selected=read(RUN/'models/selected.json');reg=ROOT/'research/registry'/f'{EXPERIMENT}.json';todo=ROOT/'research/frozen/experiment_03_1/TODO.md'
    # All current files are readable before any live metadata mutation.
    for p in RUN.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:sha(p)
    write(RUN/'provenance/registry_before_completion.json',read(reg))
    with (RUN/'provenance/TODO_before_completion.md').open('xb') as f:f.write(todo.read_bytes())
    now=datetime.now(timezone.utc).isoformat()
    approved={'experiment_id':EXPERIMENT,'status':'PASS','scope':'completed_finite_post_hoc_B5_comparison_and_3_1_report','evidence_class':summary['evidence_class'],'primary_result':'POST_HOC_B5_SUPPLEMENT','original_primary_status_unchanged':parent['primary_result'],'protocol_sha256':PROTOCOL,'parent_delivery_sha256':sha(PARENT/'formal_delivery_manifest.json'),'candidate_fits':12,'fit_attempts_including_user_interruption':13,'interrupted_attempts':1,'recovery_checks_passed':True,'recovery_seal_sha256':sha(RUN/'preflight/recovery_seal_20261009.json'),'operational_amendment_sha256':sha(amendment),'all_bootstrap_attempts_verified':True,'selected_width':selected['width'],'selected_weight_decay':selected['weight_decay'],'R2_minus_B5':summary['comparisons']['7']['R2_minus_B5'],'pdf_pages':pub['total_pages'],'pdf_sha256':sha(pdf),'md_sha256':sha(md),'no_push':True,'at_utc':now}
    write(acceptance,approved)
    r=read(reg);r.update(status='completed_post_hoc_B5_supplement',completed_at_utc=now,engineering_status='PASS',holdout_status='CONSUMED')
    r['result'].update(conclusion='POST_HOC_B5_SUPPLEMENT',original_primary_status_unchanged=parent['primary_result'],R2_minus_B5=approved['R2_minus_B5'],no_new_confirmatory_conclusion=True,candidate_fits=12,fit_attempts_including_user_interruption=13,interrupted_attempts=1,operational_amendment_sha256=sha(amendment))
    r['result'].setdefault('failed_checks',[]).append({
        'check':'independent_numeric_review_attempt01',
        'classification':'resolved_engineering_implementation_failure_not_scientific_result_failure',
        'initial_status':'FAIL','resolution_status':'PASS',
        'cause':'r0_ewma auxiliary column misreferenced as a member of the 33-feature set',
        'failed_artifact':rel(RUN/'independent_review/numeric_review.json'),
        'failed_artifact_sha256':sha(RUN/'independent_review/numeric_review.json'),
        'resolved_artifact':rel(RUN/'independent_review/numeric_review_v2.json'),
        'resolved_artifact_sha256':sha(RUN/'independent_review/numeric_review_v2.json'),
        'no_refit':True,'no_reselection':True,'original_failed_attempt_preserved':True})
    publication_attempt01=RUN/'provenance/publication_attempt01/manifest.json'
    if publication_attempt01.exists():
        r['result']['failed_checks'].append({
            'check':'publication_visual_qa_attempt01',
            'classification':'resolved_engineering_visual_qa_failure_not_scientific_result_failure',
            'initial_status':'FAIL','resolution_status':'PASS',
            'cause':'tail underprediction fraction 1.0 marker clipped by y-axis upper boundary',
            'failed_artifact':rel(publication_attempt01),
            'failed_artifact_sha256':sha(publication_attempt01),
            'resolved_artifact':rel(RUN/'reports/final_pdf_qa.json'),
            'resolved_artifact_sha256':sha(RUN/'reports/final_pdf_qa.json'),
            'resolution':'display-only y-axis margin; original numerical artifacts unchanged; final visual QA PASS',
            'no_refit':True,'no_reselection':True,'no_new_random_draws':True,
            'original_failed_attempt_preserved':True})
    r['artifacts'].update(pdf_path=rel(pdf),pdf_sha256=sha(pdf),md_path=rel(md),md_sha256=sha(md),final_acceptance=rel(acceptance),delivery_manifest=rel(dest),numeric_review=rel(RUN/'independent_review/numeric_review_v2.json'),numeric_review_attempt01=rel(RUN/'independent_review/numeric_review.json'),pdf_qa=rel(RUN/'reports/final_pdf_qa.json'))
    reg.write_text(json.dumps(r,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    t=todo.read_text(encoding='utf-8').replace('- [ ]','- [x]')
    t+='\n3.1补充完成：12个候选完整拟合，13次尝试含1次用户关机中断；前5候选复用，第6从头重做，其余6按原协议完成。中断epoch未知，约430.166秒为挂起前墙钟而非GPU计算，不含停机等待。全部种子、独立预测/标签及所有配对draw复核、真实资源测量和完整PDF/MD交付通过。证据类别为事后探索；原第三轮确认性结论保持不变。没有push或自动新增搜索。\n'
    todo.write_text(t,encoding='utf-8')
    coverage=ROOT/'research/frozen/ROADMAP_BASELINE_COVERAGE.md'
    with (RUN/'provenance/roadmap_coverage_before_completion.md').open('xb') as f:f.write(coverage.read_bytes())
    ct=coverage.read_text(encoding='utf-8')
    ct=ct.replace('- [ ] B5：资源可比的小型时序模型。当前未实现、未评估；不能以普通特征 LightGBM（B2）、随机 Kronos 骨干（B4）或表征上的小型 MLP 替代。','- [x] B5：已完成3.1小型因果TCN补充及资源测量，比较为事后探索；不能冒称新的独立确认。B2、B4和向量MLP读出头不是该对照。')
    ct=ct.replace('- [ ] 后续如开展 B5，单独预登记输入、结构、训练与计算预算、选择规则及独立确认区间，并量化参数量、训练/推理时间和内存成本。','- [x] B5补充已在训练前登记输入、结构、预算、VALID选择和资源审计。使用过的Q2/Q3只作事后比较；若需新的独立确认，需新的未使用数据及另外协议。')
    ct=ct.replace('不证明相对于 B5 的优势，也不证明 Kronos 骨干不可替代。','原确认性实验不证明相对于 B5 的优势，也不证明 Kronos 骨干不可替代；新增3.1补充结果见独立报告及登记。')
    coverage.write_text(ct,encoding='utf-8')
    files=[p for p in RUN.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    files += [pdf,md,reg,todo,coverage,ROOT/'research/configs/frozen_risk_03_1_b5_v1.yaml',ROOT/'research/researchstate_03_1.py',ROOT/'research/RESEARCHSTATE_03_1.md',ROOT/'tests/test_risk03_1_b5_model.py']
    files += [amendment,ROOT/'research/frozen/experiment_03_1/HANDOFF_SHUTDOWN_20261009.md']
    files += list((ROOT/'research/frozen/experiment_03_1').glob('*.py'))+list((ROOT/'research/delivery').glob('frozen_risk_03_1_*.py'))
    bound={rel(p):sha(p) for p in sorted(set(files))}
    write(dest,{'experiment_id':EXPERIMENT,'status':'COMPLETE','engineering_status':'PASS','candidate_fits':12,'fit_attempts_including_user_interruption':13,'interrupted_attempts':1,'recovery_checks_passed':True,'recovery_seal_sha256':sha(RUN/'preflight/recovery_seal_20261009.json'),'operational_amendment_sha256':sha(amendment),'primary_result':'POST_HOC_B5_SUPPLEMENT','evidence_class':summary['evidence_class'],'original_primary_status_unchanged':parent['primary_result'],'protocol_sha256':PROTOCOL,'holdout_status':'CONSUMED','pdf_path':rel(pdf),'pdf_sha256':sha(pdf),'md_path':rel(md),'md_sha256':sha(md),'pdf_pages':pub['total_pages'],'parent_delivery_sha256':sha(PARENT/'formal_delivery_manifest.json'),'source_and_artifact_sha256':bound,'no_parent_artifact_changes':True,'no_push':True,'at_utc':now})
    print(json.dumps({'status':'COMPLETE','bound_files':len(bound),'pages':pub['total_pages']}))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seal-reviewed-publication',action='store_true',required=True);p.parse_args();finalize()
