"""Close delivery after independent PDF numbers and full visual QA have passed."""
from pathlib import Path
import sys,json
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'tmp/pdfs/deps')]
import pymupdf as fitz
from research.frozen import encoder
from research.frozen.experiment_02.run import RUN,CONFIG,config

PDF=ROOT/'output/pdf/Kronos_Frozen_Risk_02_Report_20261008_v1.pdf'
EXPECTED='5e4c41fce81d7a3bdd4c61b5a2ddda3ba12ea40db6e697b73cd06d51a3587cd5'

def main():
    config();assert encoder.sha(PDF)==EXPECTED,'Reviewed final PDF changed'
    manifest_path=PDF.with_name(PDF.stem+'_manifest.json')
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    assert manifest['pages']==35 and manifest['pdf_sha256']==EXPECTED
    for relative,digest in manifest['input_sha256'].items():assert encoder.sha(ROOT/relative)==digest
    first=json.loads((ROOT/'research/runs/FIRST_FROZEN_SYNTHESIS_20261008_v1/final_acceptance.json').read_text())
    for relative,digest in first['files_sha256'].items():assert encoder.sha(ROOT/relative)==digest
    candidate=fitz.open(ROOT/'tmp/pdfs/frozen_risk_02_v1/layout_candidate_02.pdf')
    final=fitz.open(PDF)
    unchanged=[]
    for i in range(35):
        if i==24:continue
        old=candidate[i].get_pixmap(matrix=fitz.Matrix(1,1),alpha=False)
        now=final[i].get_pixmap(matrix=fitz.Matrix(1,1),alpha=False)
        assert old.width==now.width and old.height==now.height and old.samples==now.samples,'Unreviewed visual change'
        unchanged.append(i+1)
    out_of_bounds=[]
    for number,page in enumerate(final,1):
        for block in page.get_text('blocks'):
            x0,y0,x1,y1=block[:4]
            if x0<0 or y0<0 or x1>page.rect.width+.1 or y1>page.rect.height+.1:out_of_bounds.append(number)
    assert not out_of_bounds,'Clipped PDF content'
    review={'status':'PASS','reviewer':'/root/risk_review','pdf_sha256':EXPECTED,'pages':35,
      'programmatic_numeric_matches':166,'failed_matches':0,
      'coverage':{'11':'24 train/validation/test Regret rows','13/15/17/19':'40 all-family test rows and quarterly CIs',
       '21':'4 core comparisons','22':'12 block sensitivity rows','23':'4 training thresholds and event counts',
       '24':'10 calibration decile rows / 40 fold cells','25':'4 R1 extreme rows',
       '27-30':'40 surprise rows and 16 R2 quartile sentences'},
      'conclusion_review_pages':[1,2,6,21,35],'main_result_not_success':True,
      'random_support_does_not_replace_R1':True,'no_alpha_profit_or_strict_prior_claim':True,
      'visual_QA':{'reviewer':'/root','status':'PASS','all_pages':list(range(1,36)),
                   'full_page_spot_views':[11,12,13,20,23,25,26],
                   'final_unchanged_pixels_verified_pages':unchanged,'final_changed_page_25_visually_verified':True,
                   'font_rendering_and_bounds':'PASS','no_out_of_bounds_content':True},
      'read_only':True,'reviewer_run_note':'PowerShell stdin Chinese parsing issue in an initial matching harness; corrected ASCII/AST harness all166 passed without report change'}
    encoder.immutable_json(RUN/'pdf_independent_review.json',review)
    manifest.update({'status':'completed_verified_report','independent_numeric_review':review,
                     'visual_QA_status':'PASS','rendered_pages_verified':35})
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    todo=ROOT/'research/frozen/experiment_02/TODO.md'
    text=todo.read_text(encoding='utf-8')
    text=text.replace('- [ ] 输出真实时间图表、CSV 底表及完整中文 PDF；分别给出工程、主增量、预训练证据和继续研究判断。',
                      '- [x] 输出真实时间图表、PNG/SVG/CSV底表及35页中文PDF；166项独立PDF数值核对及全部页面视觉验收PASS。')
    text=text.replace('报告数值/版面交付核验结束后关闭长期 goal；','报告数值/版面交付核验已结束，长期 goal 完成；')
    todo.write_text(text,encoding='utf-8')
    source_hashes={}
    for path in sorted((ROOT/'research/frozen/experiment_02').glob('*')):
        if not path.is_file():continue
        relative=str(path.relative_to(ROOT)).replace('\\','/')
        source_hashes[relative]=encoder.sha(path)
        encoder.immutable_bytes(RUN/'provenance/delivery_source'/path.relative_to(ROOT),path.read_bytes())
    acceptance=json.loads((RUN/'final_acceptance.json').read_text())
    for relative,digest in acceptance['result_artifact_sha256'].items():assert encoder.sha(RUN/relative)==digest
    delivery={'status':'complete_finite_research_and_verified_pdf','experiment_id':'FROZEN_RISK_02_v1',
       'pdf_path':str(PDF),'pdf_sha256':EXPECTED,'pdf_manifest_sha256':encoder.sha(manifest_path),
       'pdf_review_sha256':encoder.sha(RUN/'pdf_independent_review.json'),
       'scientific_acceptance_sha256':encoder.sha(RUN/'final_acceptance.json'),
       'final_source_and_document_sha256':source_hashes,'first_experiment_retained_hashes_unchanged':len(first['files_sha256']),
       'protocol_sha256':encoder.sha(CONFIG),'formal_budget':96,'new_holdout_access':False,'pushed':False,
       'no_pending_formal_experiments':True,'completed_at_utc':datetime.now(timezone.utc).isoformat()}
    encoder.immutable_json(RUN/'delivery_manifest.json',delivery)
    registry=ROOT/'research/registry/FROZEN_RISK_02_v1.json'
    record=json.loads(registry.read_text(encoding='utf-8'));record['status']=delivery['status']
    record['delivery']=delivery
    registry.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':delivery['status'],'pages':35,'pdf_sha256':EXPECTED,'first_experiment_unchanged':True}))

if __name__=='__main__':main()
