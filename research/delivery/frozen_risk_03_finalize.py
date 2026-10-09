"""Seal completed RISK_03 publication; metadata only, no scientific rerun."""
from pathlib import Path
import hashlib,json,argparse
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'research/runs/FROZEN_RISK_03_v1'
P='7ed8affd7cd624cde89d23d29aaca7ef2ab660167fac7d3dc25fac274e394b7f'
B='e0d854bc4a376194344e507a5d2d9e1a0573b778f74f82970f881fae689b3376'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for c in iter(lambda:f.read(1048576),b''): h.update(c)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):
 with Path(p).open('x',encoding='utf-8',newline='\n') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def rel(p):return Path(p).relative_to(ROOT).as_posix()
def main():
 terminal=read(RUN/'authorization/formal_terminal.json')
 assert terminal['state']=='CONSUMED'
 result=read(RUN/'metrics/formal_result.json')
 numeric=read(RUN/'independent_review/formal_numeric.json')
 assert terminal['evidence']['status']==result['status'] and terminal['evidence']['numeric_review']=='PASS'
 assert numeric['status']=='PASS' and numeric['all_primary_attempts_verified']
 for path in ['preflight/sealed_bundle.json','models/formal/preview_identity_check.json','models/formal/fit_audit.json','data_5m/audit.json','data_5m/raw_csv_audit.json','independent_review/formal_report_review.json','independent_review/formal_auxiliary_review.json','reports/formal_pdf_qa_v2.json','reports/formal_separated_exports.json']:
  assert read(RUN/path)['status']=='PASS',path
 assert sha(ROOT/'research/configs/frozen_risk_03_v1.yaml')==P and sha(RUN/'preflight/sealed_bundle.json')==B
 bundle=read(RUN/'preflight/sealed_bundle.json')
 for path,expected in bundle['files_sha256'].items():
  target=(ROOT/path).resolve();assert target.is_relative_to(ROOT) and sha(target)==expected,path
 qa=read(RUN/'reports/formal_pdf_qa_v2.json')
 required_outputs=[RUN/'formal_final_acceptance.json',RUN/'formal_delivery_manifest.json',RUN/'authorization/formal_lifecycle_summary.json',RUN/'provenance/phase_b_metadata_before_completion',ROOT/'output/pdf/Kronos_Frozen_Risk_03_Report_v2_manifest.json']
 assert not any(p.exists() for p in required_outputs),'Preserve existing seal; refuses repeat'
 fit_events=[json.loads(x) for x in (RUN/'models/formal/fit_events.jsonl').read_text(encoding='utf-8').splitlines()]
 assert len(fit_events)==60 and (ROOT/'output/pdf/Kronos_Frozen_Risk_03_Report_v1.pdf').is_file()
 preverified_current_artifacts={rel(path):sha(path) for path in RUN.rglob('*') if path.is_file() and '__pycache__' not in path.parts}
 pdf=ROOT/'output/pdf/Kronos_Frozen_Risk_03_Report_v2.pdf'
 md=RUN/'reports/Kronos_Frozen_Risk_03_Report_v2.md'
 assert qa['pdf_sha256']==sha(pdf)
 assert md.is_file()
 sha(md);sha(ROOT/'output/pdf/Kronos_Frozen_Risk_03_Report_v1.pdf')
 assert read(RUN/'models/formal/fit_audit.json')['candidate_fits']==28
 assert read(RUN/'models/formal/fit_audit.json')['determinism_refits']==2
 acceptance={'experiment_id':'FROZEN_RISK_03_v1','status':'PASS','scope':'completed_locked_Phase_B_and_final_publication','engineering_status':'PASS','primary_result':result['status'],'holdout':'CONSUMED','protocol_sha256':P,'sealed_bundle_sha256':B,'quarter_samples_and_events':[{'quarter':q['id'],'N':q['opportunities'],'positive':q['positive'],'negative':q['negative']} for q in result['quarter_metrics']],'co_primary':result['bootstrap']['7']['comparisons'],'risk_magnitude_red_flag':result['risk_magnitude_red_flag'],'candidate_fits':28,'determinism_refits':2,'independent_numeric_checks':len(numeric['checks']),'all_primary_attempts_verified':True,'auxiliary_metrics_are_predeclared_descriptive_not_new_success_criteria':True,'new_fits_or_new_random_draws_in_publication':False,'scientific_source_unchanged':True,'checkpoint_specific_training_cutoff_proven':False,'no_push':True,'no_fourth_round':True,'pdf_path':rel(pdf),'pdf_sha256':sha(pdf),'pdf_pages':qa['pages'],'markdown_path':rel(md),'markdown_sha256':sha(md),'at_utc':datetime.now(timezone.utc).isoformat()}
 write(RUN/'formal_final_acceptance.json',acceptance)
 life={'experiment_id':'FROZEN_RISK_03_v1','authoritative_state':'CONSUMED','precedence':'formal_terminal.json is authoritative terminal; holdout_state.json is immutable historical initial SEALED snapshot, not current state','authorization_sha256':sha(RUN/'authorization/user_approval.json'),'execution_claim_sha256':sha(RUN/'authorization/execution_claim.json'),'formal_terminal_sha256':sha(RUN/'authorization/formal_terminal.json'),'scientific_retry_prohibited':True}
 write(RUN/'authorization/formal_lifecycle_summary.json',life)
 registry=ROOT/'research/registry/FROZEN_RISK_03_v1.json'
 archive=RUN/'provenance/phase_b_metadata_before_completion';archive.mkdir(exist_ok=False)
 (archive/'registry_execution.json').write_bytes(registry.read_bytes())
 todo=ROOT/'research/frozen/experiment_03/TODO.md';(archive/'TODO_execution.md').write_bytes(todo.read_bytes())
 r=read(registry);r.update(status='completed_formal_experiment',phase='PhaseB_completed',holdout_status='CONSUMED',engineering_status='PASS',completed_at_utc=acceptance['at_utc'])
 r['result'].update(main_status=result['status'],conclusion=result['status'],risk_magnitude_red_flag=result['risk_magnitude_red_flag'],co_primary=result['bootstrap']['7']['comparisons'],quarter_samples=acceptance['quarter_samples_and_events'])
 r['artifacts']['phase_a_preserved']=dict(r['artifacts'])
 r['artifacts'].update(formal_acceptance=rel(RUN/'formal_final_acceptance.json'),formal_acceptance_sha256=sha(RUN/'formal_final_acceptance.json'),formal_pdf_path=rel(pdf),formal_pdf_sha256=sha(pdf),formal_markdown_path=rel(md),formal_pdf_qa=rel(RUN/'reports/formal_pdf_qa_v2.json'),formal_delivery_manifest=rel(RUN/'formal_delivery_manifest.json'),formal_numeric_review=rel(RUN/'independent_review/formal_numeric.json'),formal_report_review=rel(RUN/'independent_review/formal_report_review.json'),formal_auxiliary_review=rel(RUN/'independent_review/formal_auxiliary_review.json'))
 r['formal_attempts']=fit_events
 registry.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 t=todo.read_text(encoding='utf-8').replace('- [ ]','- [x]')
 t=t.replace('Phase B：**用户已授权，唯一正式执行已 CLAIMED**。','Phase B：**已完成，唯一正式测试已 CONSUMED**。').replace('Phase A complete / Stage5 awaiting separate user approval。Stage6-8未执行，未push。','Phase A已验收；随后用户完整授权Phase B。Stage6-8现已完成，未push。')
 t+='\n正式结束：Phase B / Stage6-8 已完成；holdout CONSUMED；工程及独立审查 PASS。主要结论：'+result['status']+'。全部前两轮与Phase A工件原样保留。最终完整中文PDF v2、全页QA与数值底表已完成；不自动第四轮，不push。\n'
 todo.write_text(t,encoding='utf-8')
 pdfmanifest=ROOT/'output/pdf/Kronos_Frozen_Risk_03_Report_v2_manifest.json'
 write(pdfmanifest,dict(acceptance,pdf_qa_sha256=sha(RUN/'reports/formal_pdf_qa_v2.json'),formal_acceptance_sha256=sha(RUN/'formal_final_acceptance.json')))
 files=[pdf,md,pdfmanifest,registry,todo,ROOT/'research/researchstate_phase_b.py',Path(__file__).resolve(),RUN/'formal_final_acceptance.json']
 patterns=['authorization/*.json','models/formal/**/*','data_5m/**/*','features/holdout*','labels/holdout*','labels/publication_*','predictions/holdout*','predictions/publication_*','bootstrap/**/*','metrics/formal_result.json','metrics/publication_auxiliary/**/*','diagnostics/holdout/**/*','figures/formal_publication/**/*','figures/report_publication/**/*','figures/formal_report_v2/**/*','independent_review/formal*','reports/formal*','reports/Kronos_Frozen_Risk_03_Report_v*.md','provenance/formal_source/**/*','provenance/formal_source_manifest.json','provenance/formal_metadata_performance.json','provenance/formal_execution.log','provenance/phase_b_metadata_before_update/**/*','provenance/phase_b_metadata_before_completion/**/*']
 for pattern in patterns:files.extend(p for p in RUN.glob(pattern) if p.is_file())
 files.extend(p for p in (ROOT/'research/delivery').glob('frozen_risk_03_*.py'))
 files.append(ROOT/'output/pdf/Kronos_Frozen_Risk_03_Report_v1.pdf')
 bound={rel(p):sha(p) for p in sorted(set(files))}
 delivery={'experiment_id':'FROZEN_RISK_03_v1','status':'COMPLETE','engineering_status':'PASS','primary_result':result['status'],'holdout':'CONSUMED','protocol_sha256':P,'sealed_bundle_sha256':B,'authorization_sha256':life['authorization_sha256'],'execution_claim_sha256':life['execution_claim_sha256'],'formal_terminal_sha256':life['formal_terminal_sha256'],'scientific_acceptance_path':rel(RUN/'formal_final_acceptance.json'),'scientific_acceptance_sha256':sha(RUN/'formal_final_acceptance.json'),'pdf_path':rel(pdf),'pdf_sha256':sha(pdf),'pdf_pages':qa['pages'],'markdown_path':rel(md),'markdown_sha256':sha(md),'source_and_artifact_sha256':bound,'phase_a_delivery_manifest_retained':{'path':rel(RUN/'delivery_manifest.json'),'sha256':sha(RUN/'delivery_manifest.json')},'initial_SEALED_snapshot_retained_not_authoritative_current_state':True,'no_scientific_rerun':True,'no_push':True,'at_utc':acceptance['at_utc']}
 write(RUN/'formal_delivery_manifest.json',delivery)
 print(json.dumps({'status':'COMPLETE','primary_result':result['status'],'bound_files':len(bound),'pdf_pages':qa['pages']}))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seal-completed-publication',action='store_true',required=True);p.parse_args();main()
