"""Publish accepted M2-CI evidence as a Chinese PDF; no scientific execution.

The artifact-operation marker is owned by the parent publication workflow.
This builder never mutates the sealed run, registry or central research state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import html
import sys
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT / 'tmp/pdfs/deps'))

import pandas as pd
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, A3, landscape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle
from pypdf import PdfReader
import pymupdf

RUN = ROOT / 'research/runs/M2_CONDITIONAL_INCREMENT_01'
PUB = ROOT / 'research/delivery/M2_CI_REPORT_PDF_v1'
TMP = ROOT / 'tmp/pdfs/m2_ci_report_v1'
PDF = ROOT / 'output/pdf/Kronos_M2_CI_Development_Report_v1.pdf'
INK = colors.HexColor('#183047')
BLUE = colors.HexColor('#215E8B')
GRAY = colors.HexColor('#516578')
PALE = colors.HexColor('#EAF1F6')
INPUTS, OUTLINE = {}, []
FIGURES = [
 ('rolling_Regret_primary','主要模型：7日滚动 Regret','B5、R1、R2、B2及三项选定融合；各折独立，不拼成连续确认样本。'),
 ('rolling_Regret_historical_baselines','历史基线：完整保留','HAR、EWMA、persistence、constant_RV与B5；较低Regret表示较好平均风险预测。'),
 ('rolling_Regret_seed_models','三种子与集成','B5种子17/29/43均保留；集成为原单位RV均值，不是指标均值。'),
 ('rolling_Regret_alpha_R1','全部权重：B5 + R1','六项预登记alpha均展示；选择alpha=.1，不能删去较差权重。'),
 ('rolling_Regret_alpha_B2','全部权重：B5 + B2','六项预登记alpha均展示；选择alpha=0，即保留B5。'),
 ('rolling_Regret_alpha_R2','全部权重：B5 + R2','六项预登记alpha均展示；选择alpha=.1，微小开发点效应未通过筛选。'),
 ('gain_sum','每日配对损失增量','真实UTC日期；正值有利融合。曲线表示QLIKE损失差，不是交易收益或权益。'),
 ('cumulative_gain_sum','各时期独立累计增量','各折与VALID分别从0累计，不构造伪连续全年曲线。'),
 ('ranking_denominator_diagnostic','排序与分母诊断','surprise与绝对q90分别判断；constant_RV分母效应不能解释为预测能力。'),
 ('FIT_fixed_tail_underprediction','固定FIT阈值：尾部低估','q90/q99阈值仅来自该折FIT；q99稀疏，分数变化不能证明尾部保护。'),
]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def exclusive(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False); stream.write('\n')


def source(path,delivery=None):
    path=Path(path)
    if not path.is_absolute(): path=ROOT/path
    key=path.resolve().relative_to(ROOT).as_posix()
    digest=sha(path)
    if delivery is not None and delivery.get(key)!=digest:
        raise ValueError('PDF consumer not bound by accepted delivery: '+key)
    INPUTS[key]=digest
    return path


def jread(path,delivery=None):
    return json.loads(source(path,delivery).read_text(encoding='utf-8'))


def readcsv(name,delivery):
    return pd.read_csv(source(RUN/name,delivery),float_precision='round_trip')


def display(value,precision=6):
    if pd.isna(value): return 'NA'
    if isinstance(value,(float,int)):
        return format(float(value),f'.{precision}g')
    return str(value)


class Document:
    def __init__(self,path):
        self.stream=Path(path).open('xb')
        self.c=canvas.Canvas(self.stream,pagesize=A4,pageCompression=1)
        self.c.setTitle('Kronos M2-CI 条件增量开发研究报告')
        self.c.setAuthor('Kronos Crypto Research')
        self.c.setSubject('INCONCLUSIVE；用户要求停止开发；仅出版已验收开发证据')
        self.number=0; self.x=38; self.w,self.h=A4; self.width=self.w-76
    def start(self,title,subtitle='已验收开发证据；未来正式验证未启动',figure=False):
        if self.number: self.c.showPage()
        self.number+=1
        self.w,self.h=landscape(A3) if figure else A4
        self.x=32 if figure else 38
        self.width=self.w-2*self.x
        self.c.setPageSize((self.w,self.h))
        self.c.setFillColor(INK); self.c.rect(0,self.h-12,self.w,12,fill=1,stroke=0)
        self.c.setFillColor(BLUE); self.c.setFont('CN',8)
        self.c.drawString(self.x,self.h-32,'KRONOS  /  M2-CI  /  DEVELOPMENT EVIDENCE')
        self.c.setFillColor(INK); self.c.setFont('CNBold',19)
        self.c.drawString(self.x,self.h-60,title)
        self.y=self.h-76
        self.para(subtitle,8.5,GRAY,after=8)
        self.c.setStrokeColor(colors.HexColor('#CFD9E1')); self.c.line(self.x,35,self.w-self.x,35)
        self.c.setFont('CN',7.4); self.c.setFillColor(GRAY)
        self.c.drawString(self.x,22,'2026-10-09 | 开发研究，非独立确认 | 旧Q2/Q3 CONSUMED保持')
        self.c.drawRightString(self.w-self.x,22,str(self.number))
        self.c.bookmarkPage('p'+str(self.number)); self.c.addOutlineEntry(title,'p'+str(self.number),level=0)
        OUTLINE.append({'page':self.number,'title':title,'figure_page':figure,
                        'paper':'A3 landscape' if figure else 'A4 portrait'})
    def para(self,text,size=10,color=INK,after=8,bold=False):
        style=ParagraphStyle('text',fontName='CNBold' if bold else 'CN',fontSize=size,
            leading=size*1.6,textColor=color,wordWrap='CJK',splitLongWords=True)
        p=Paragraph(html.escape(text).replace('\n','<br/>'),style)
        _,height=p.wrap(self.width,1000)
        if self.y-height<49: raise ValueError(f'Page {self.number} paragraph overflow')
        p.drawOn(self.c,self.x,self.y-height); self.y-=height+after
    def heading(self,text): self.para(text,11.5,BLUE,after=6,bold=True)
    def table(self,rows,widths=None,size=8.2):
        style=ParagraphStyle('cell',fontName='CN',fontSize=size,leading=size*1.35,
                             wordWrap='CJK',splitLongWords=True,textColor=INK)
        cells=[[Paragraph(html.escape(str(v)),style) for v in row] for row in rows]
        t=Table(cells,colWidths=widths or [self.width/len(rows[0])]*len(rows[0]))
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),PALE),('VALIGN',(0,0),(-1,-1),'TOP'),
            ('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#AFBFCC')),
            ('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#DCE4EA')),
            ('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4),
            ('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
        _,height=t.wrap(self.width,1000)
        if self.y-height<49: raise ValueError(f'Page {self.number} table overflow: {height}, available {self.y-49}')
        t.drawOn(self.c,self.x,self.y-height); self.y-=height+10
    def csvtable(self,df,columns,labels=None,size=8.2,widths=None):
        self.table([labels or columns]+[[display(v) for v in row] for row in df[columns].itertuples(index=False,name=None)],widths,size)
    def figure(self,path,caption):
        from PIL import Image
        iw,ih=Image.open(path).size
        # Reserve a fixed caption band beneath the image and above the footer.
        maxh=self.y-85
        width=self.width; height=width*ih/iw
        if height>maxh: height=maxh; width=height*iw/ih
        bottom=self.y-height
        self.c.drawImage(str(path),self.x+(self.width-width)/2,bottom,width,height,mask='auto')
        self.y=bottom-8
        self.para(caption,8.2,GRAY,after=0)
    def save(self):
        self.c.save(); self.stream.flush(); self.stream.close()


def build():
    if PDF.exists(): raise FileExistsError('Existing PDF must be preserved; use a separately approved attempt/version')
    if (PUB/'publication_manifest.json').exists(): raise FileExistsError('Existing publication manifest must be preserved')
    PUB.mkdir(parents=True,exist_ok=True); TMP.mkdir(parents=True,exist_ok=True); PDF.parent.mkdir(parents=True,exist_ok=True)
    delivery_file=source(RUN/'delivery_manifest.json')
    parent_sha=sha(delivery_file)
    delivery=json.loads(delivery_file.read_text(encoding='utf-8'))
    bound=delivery['source_and_artifact_sha256']
    if delivery['status']!='COMPLETE' or delivery['primary_result']!='INCONCLUSIVE' or delivery['holdout_status']!='CONSUMED':
        raise ValueError('Accepted M2 delivery required')
    for key,digest in bound.items():
        if sha(ROOT/key)!=digest: raise ValueError('Sealed delivery drift: '+key)
    numeric=jread(RUN/'independent_review/numeric_review.json',bound)
    if numeric['status']!='PASS': raise ValueError('Independent numeric review must pass')
    numeric_bound=numeric['source_and_artifact_sha256']
    for key,digest in numeric_bound.items():
        if sha(ROOT/key)!=digest: raise ValueError('Independent numeric binding drift: '+key)
    visual=jread(RUN/'figures/v2/visual_review.json',bound)
    fm=jread(RUN/'figures/v2/manifest.json',bound)
    if visual['status']!='PASS' or fm['status']!='PASS': raise ValueError('Accepted v2 figures required')
    for key,digest in fm['files_sha256'].items():
        if sha(ROOT/key)!=digest: raise ValueError('Accepted figure drift: '+key)
    reports=['M2_Conditional_Increment_Development_Report.md','M2_Conditional_Increment_Future_Holdout_Protocol_DRAFT.md','M2_Conditional_Increment_Project_Decision.md']
    for name in reports: source(RUN/'reports'/name,bound).read_text(encoding='utf-8')
    summary=jread(RUN/'metrics/summary.json',bound)
    audit=jread(RUN/'models/candidate_audit.json',bound)
    selection=jread(RUN/'fusion/selected_weights.json',bound)
    config_path=source(ROOT/'research/m2_ci/config.yaml',bound)
    import yaml
    cfg=yaml.safe_load(config_path.read_text(encoding='utf-8'))
    pooled=readcsv('metrics/pooled_metrics.csv',bound)
    intervals=readcsv('metrics/paired_intervals.csv',bound)
    concentration=readcsv('metrics/gain_concentration.csv',bound)
    tails=readcsv('metrics/tail_metrics.csv',bound)
    states=readcsv('metrics/state_metrics.csv',bound)
    resource=readcsv('metrics/resource_benchmark.csv',bound)
    hypothetical=readcsv('metrics/future_sample_scenarios.csv',bound)
    observed=readcsv('metrics/future_observed_gain_scenarios.csv',bound)
    direct=readcsv('metrics/future_direct_R2_B5_scenarios.csv',bound)
    denominator=readcsv('figures/v2/ranking_denominator_diagnostic.csv',bound)
    thresholds=jread(RUN/'metrics/FIT_fixed_thresholds.json',bound)
    source(__file__)
    font_paths=['C:/Windows/Fonts/msyh.ttc','C:/Windows/Fonts/msyhbd.ttc']
    font_hashes={p:sha(p) for p in font_paths}
    pdfmetrics.registerFont(TTFont('CN',font_paths[0],subfontIndex=0))
    pdfmetrics.registerFont(TTFont('CNBold',font_paths[1],subfontIndex=0))
    d=Document(PDF)
    s=summary['screens']['R2']
    d.start('M2-CI 条件增量开发研究报告','出版版 v1 | 原实验判定 INCONCLUSIVE | 当前仅出版，停止开发与新实验准备')
    d.para('现有R2输出与B5融合的开发点效应很小，跨折和区间证据不足；不支持增加当前融合系统的复杂度。',14,BLUE,bold=True,after=14)
    d.table([['主项','已验收结果','预登记要求'],['OOF Regret改善',f"{100*s['relative_Regret_improvement']:.6f}%",'至少5%'],
        ['R2融合alpha',display(s['alpha']),'六项有限权重，OOF选择'],['正向折',f"{s['positive_folds']}/5",'至少4/5'],
        ['主7日区间',f"[{s['ci_lower_7d']:.8f}, {s['ci_upper_7d']:.8f}]",'下限严格>0'],
        ['原VALID平均增量',f"{s['VALID_gain']:.9f}",'正向；仅开发检查'],['最终判定','INCONCLUSIVE','未通过实际增量筛选']],size=9)
    d.heading('当前任务状态与历史结论分别保留')
    d.para('用户现已要求停止开发，转为生成本PDF。因此本次不继续模型、数据采集、协议准备或未来实验。历史报告的下一阶段建议作为历史记录保留，不能视为本次继续执行的授权。未来正式验证未启动，旧DRAFT未执行，也未宣称新正式协议已经封存。')
    d.para('OOF共1267机会，原VALID92机会。历史B5架构曾由2026年3月VALID选择；checkpoint训练截止未证明。因此这是开发筛选，不能解释为完整嵌套或严格事前独立确认。旧2026Q2/Q3 CONSUMED终态保持。',9.2,GRAY)

    d.start('01 研究契约与五折角色')
    d.para('H1检验给定B5时现有R2正值RV输出是否互补；H2检验是否超过同样有限融合的R1/B2。H3未来独立复现仅有开发诊断，未得到确认。')
    rows=[['折','FIT右边界','内层右边界 / OOF起点','OOF右边界','FIT / 内层 / OOF N']]
    for f in cfg['roles']['folds']: rows.append([f['id'],f['fit_end'][:10],f['evaluation_start'][:10],f['evaluation_end'][:10],' / '.join(map(str,f['expected_counts']))])
    d.table(rows,[45,90,130,90,d.width-355],8.1)
    d.para('每折FIT从2024-01-01起；范围均为[start,end)。label_end和labelable_at均须严格小于角色右边界。内层验证先于外层评估，选择后不重拟合。原TRAIN2369、原VALID92、全部DEV2461。')
    d.heading('时间与标签')
    d.para('UTC04/12/20决策边界；例如04:01决策、05:00入场、09:00标签结束，09:01可标记。使用256根已完成、当时可用的1h真实OHLCVA。4h RV为入场5m open与随后48根5m close共49价格、48自然对数收益平方和；不年化。全部2461条已独立复算。')
    d.para('预测NPZ只含ID、窗口、33普通特征、512冻结hidden、HAR3、R0基线。未来RV保存在独立标签表；预测接口不接收标签、funding、profit或事件。时钟字段仅用于purge。')
    d.para('60秒可用延迟是冻结契约假设；官方数据覆盖与一致性不证明真实历史发布延迟、checkpoint事前可用性或交易所内部账本。',9,GRAY)

    d.start('02 模型身份与实际95次拟合')
    d.para('R1/B2为33普通特征；R2为相同33+冻结512。R0不进入拟合特征。HAR/R1/R2是L2 log-link QLIKE标量头，并非早期收益任务的MSE Ridge。B2沿用Gamma LightGBM。')
    d.para('B5固定width32/wd.001，种子17/29/43；256×11窗口，6市场通道及5日历通道，固定日历除数59/23/6/31/12。7个因果残差层dilation1至64，感受野257；末hidden与本折FIT标准化33特征拼接。')
    identities=[['折','HAR λ','R1 λ','R2 λ','B2叶/最小叶/迭代','B5最佳epoch 17/29/43']]
    for f in cfg['roles']['folds']:
        fold=f['id']; sm=jread(RUN/f'models/{fold}/selected_models.json',bound)['selected']
        heads={family:jread(sm[family],bound) for family in ('har','R1','R2','B2')}
        epochs=[]
        for seed in ('17','29','43'):
            meta=jread(RUN/f'models/{fold}/B5_s{seed}.json',bound)
            epochs.append(str(meta['metadata']['best_epoch']))
        identities.append([fold]+[display(heads[x]['lambda']) for x in ('har','R1','R2')]+[
            f"{heads['B2']['num_leaves']} / {heads['B2']['min_data_in_leaf']} / {heads['B2']['best_iteration']}",' / '.join(epochs)])
    d.table(identities,[45,57,57,57,135,d.width-351],8)
    d.para(f"实际拟合{audit['fits']}：线性{audit['linear_fits']}、B2 {audit['B2_fits']}、B5 {audit['B5_fits']}。B5总{audit['actual_B5_epochs']} epoch；B2总{audit['actual_B2_iterations']} iteration。无扩结构、自动重试、重新拟合或解冻。")
    d.para('每折标准化均值、尺度和目标median只来自FIT。B5最多120epoch，patience15、batch64、lr.001；各候选以内层原单位QLIKE选择，最早精确同分epoch保留。三种子集成是原单位RV算术均值。标签floor与预测clip分别为max(RV,1e-12)及[1e-12,1]。')
    d.para('具体工件路径与SHA由selected_fold_map及每折selected_models解引用；原VALID复用既有完整TRAIN冻结模型，新增拟合0。逻辑可用时点不冒充现实训练墙钟日期。',9,GRAY)

    d.start('03 全部18项融合权重')
    d.para('融合=(1-alpha)×B5_RV+alpha×比较器RV。三组各六项alpha，OOF机会加权QLIKE最小者选择，精确同分选较小alpha。权重先冻结，再检查原VALID；全部尝试保留。',9.5)
    choices=pd.DataFrame(selection['all_candidates'])
    choices['improvement_pct']=choices.relative_Regret_improvement*100
    d.csvtable(choices,['comparator','alpha','QLIKE_Regret','gain_vs_B5','improvement_pct'],
        ['比较器','alpha','平均Regret','ΔQLIKE vs B5','Regret改善%'],size=8.3)
    d.para('主增量Δ=L(B5)-L(mix)，正值有利融合。rawQLIKE=log(p)+y/p；Regret=y/p-log(y/p)-1。同标签下配对差相同，不是两个独立成功证据。rawQLIKE可为负，禁止将其百分比改善作为效应。',9.3,GRAY)

    retained=['B5','B5_s17','B5_s29','B5_s43','R1','R2','B2','har','ewma','persistence','constant_RV','mix_R1','mix_B2','mix_R2']
    for period,title in [('OOF','04 OOF完整模型对照'),('VALID','05 原VALID完整模型对照')]:
        d.start(title,'全部基模型、种子、基线与选定融合保留；损失与排序分别解释')
        table=pooled[(pooled.period==period)&pooled.family.isin(retained)].set_index('family').loc[retained].reset_index()
        d.csvtable(table,['family','N','QLIKE_Regret','logRV_MSE','surprise_AUROC','surprise_AP'],
            ['模型','N','Regret','logRV MSE','surprise AUC','surprise AP'],8.4,[97,37,91,91,102,d.width-418])
        d.para('此表展示14项基模型/种子/基线及选定融合；前页18候选全部保留。完整每折rawQLIKE、绝对q90排序及全部32模型指标仍由原MD、point_metrics.csv追溯。OOF pooled排序混合各折模型与FIT常数，仅作描述。原VALID仅92机会，曾用于历史架构选择，不能称独立确认。',9.3,GRAY)
        d.para('风险平均预测、surprise排序、绝对高RV识别、极端低估保护及交易经济价值是不同问题。constant_RV与EWMA仍必须保留；表中较好的排序不自动证明可交易盈利。',9.3,GRAY)

    d.start('06 主区间与特异性')
    d.para('每折完整UTC日历独立circular stationary block bootstrap；空日保留，折内同一抽样给所有32模型共用日权重。7日主块，3/14日敏感性，各5000次。OOF合并总损失分子/总机会数，不等权平均五折。')
    ci=intervals[(intervals.period.isin(['OOF','VALID'])) & (intervals.contrast.isin(['B5_minus_mix_R2','mix_R1_minus_mix_R2','mix_B2_minus_mix_R2']))].copy()
    ci['contrast']=ci.contrast.map({'B5_minus_mix_R2':'B5 - R2mix','mix_R1_minus_mix_R2':'R1mix - R2mix','mix_B2_minus_mix_R2':'B2mix - R2mix'})
    d.csvtable(ci,['period','contrast','block_days','point_gain','ci_lower','ci_upper'],
        ['时期','损失差，正值利R2mix','块/日','点效应','下限','上限'],7.8,[43,136,39,99,99,d.width-416])
    d.para('所选alpha在抽样中不重选，OOF同时用于权重选择；全部区间是选择条件下的开发区间，可能乐观。R2mix对R1mix/B2mix的主区间下限未严格正；未证明特异性不等于证明等效。',9.1,GRAY)
    d.para('R2mix的7日上限约为B5平均Regret的1.52%，仍低于5%门槛；保持预定INCONCLUSIVE分类，不依据已见结果改规则。',9.1,GRAY)

    d.start('07 时间集中性与H3开发诊断')
    conc=concentration[concentration.contrast=='B5_minus_mix_R2'].copy()
    d.csvtable(conc,['fold_id','N','total_gain_sum','max_day_signed_share','drop_best_positive_day_mean_gain'],
        ['时期','N','总损失增量','最佳日/有符号总量','删最佳正日后平均Δ'],8.1,[48,36,122,133,d.width-339])
    d.para('WF04及WF05最佳正日贡献约66.0%和57.1%；删除后点值仍为正。原VALID删最佳正日后转负。该诊断不重训、不重选。负总量或接近0时，有符号分母占比不可按普通百分比解读。')
    d.heading('H3：B5对直接R2')
    h3=intervals[(intervals.period.isin(['OOF','VALID'])) & (intervals.contrast=='R2_minus_B5')]
    d.csvtable(h3,['period','block_days','point_gain','ci_lower','ci_upper'],
        ['时期','块/日','R2-B5损失差','下限','上限'],8.4)
    d.para('OOF直接R2-B5差约0.104547，7日区间[0.066566,0.140267]，正值支持B5。它仍来自已见开发期，不能确认未来独立复现H3；不因此开启新正式试验。',9.2,GRAY)

    d.start('08 尾部与风险状态')
    tail=tails[tails.family.isin(['B5','mix_R2'])].copy()
    d.csvtable(tail,['fold_id','family','subset','N','under_count','under_fraction'],
        ['时期','模型','固定FIT尾部','尾部N','低估数','低估比例'],7.8)
    d.para('低估定义 prediction/effectiveRV<0.5。q90/q99阈值只来自FIT；原VALID使用原TRAIN阈值。q99事件很少，个别大幅比例变化不能证明尾部保护，需同时报告样本数。',9,GRAY)
    d.para('风险状态按FIT EWMA四分位切分，禁止在评估范围重新拟合。完整六时期、四状态、全部模型损失已保留state_metrics.csv；本PDF图与模型表提供主要关系，不删除失败或不利状态。',9,GRAY)

    d.start('09 FIT固定风险状态对照')
    state_table=states[states.family.isin(['B5','mix_R2']) & states.EWMA_bin.isin([0,3])].copy()
    d.csvtable(state_table,['fold_id','family','EWMA_bin','N','QLIKE_Regret','logRV_MSE'],
        ['时期','模型','FIT EWMA档','N','Regret','logRV MSE'],7.9)
    d.para('本表展示最低档0和最高档3，阈值来自FIT EWMA四分位。完整中间档1/2及全部32模型仍保留原state_metrics.csv，不因已见成绩移除。分档Regret按各档真实机会计算，不能将不同档或时期的点值当作独立确认。',9.2,GRAY)
    d.para('状态样本比整体更小；这里检验同一固定预测器在不同已知历史风险状态下的开发表现，不另选阈值、模型或融合权重。',9.2,GRAY)

    d.start('10 本机时延与工程代价')
    r=resource.copy(); r['ms8']=r.batch_mean_seconds*1000; r['MB']=r.checkpoint_bytes/1e6
    d.csvtable(r,['name','ms8','MB','trainable_parameters_at_fit','frozen_parameters','B2_leaves'],
        ['实际路径','8窗口ms','工件MB','拟合参数','冻结参数','B2叶'],7.7,[166,66,66,83,83,d.width-464])
    d.para('共同8个DEV窗口，WF05选定工件，3次预热与20次实跑，CPU线程1，CUDA前后同步。B5集成49.44ms，完整B5+R2为68.38ms，增加38.3%；工件约0.292MB与115.158MB。完整R2每次真实tokenizer+encoder，不用cached hidden；完整时延直接测量，不相加估算。')
    d.para('alpha0/1也实跑双路径，以显示组件代价；生产alpha0可删掉未使用路径。B2节点/叶数不同于神经可训练参数。Kronos预训练成本未在本轮重测或配平。CUDA峰值包含全部常驻模型；RSS为WorkingSet采样最大值，非连续峰值；预处理与加载不计入时延。',9,GRAY)
    d.para(f"全部95候选：B5 {audit['actual_B5_epochs']} epoch，B2 {audit['actual_B2_iterations']} iteration。完整候选实际walltime、工件大小及记录保留在candidate_audit.json；推理速度不等于全生命周期成本。",9,GRAY)

    d.start('11 条件性样本情景：不是等待建议')
    d.para('假设增量达到5%×开发B5平均Regret，目标量级约0.0168673；开发7日块SE按样本数平方根外推，每日3机会、365.25/12日/月。以下是历史条件情景，不证明未来平稳性、独立有效样本量或实际功效。')
    d.csvtable(hypothetical,['months','expected_opportunities_approx','MDE_80power_two_sided95','power_at_5pct_base_Regret_approx'],
        ['月数','约机会数','80% MDE','假设5%效应近似power'],8.2)
    d.para('“假设真实改善达到5%时，网格最短3月”不能套用到实测0.384676%。实测选后微小效应的条件诊断约需18,585机会、204个月；其点值区间跨零且选择乐观，不能视为等待建议或正式样本承诺。')
    observed_row=observed[(observed.contrast=='B5_minus_mix_R2') & (observed.months==3)]
    d.csvtable(observed_row,['observed_point_gain','paired_bootstrap_SE_7day','nominal_N80_at_observed_effect','nominal_months80_at_observed_effect'],
        ['已观测效应','开发块SE','名义N80','名义月数80'],8.3)
    d.para('直接B5对R2的同量级目标在3-24月网格内近似power不足80%，不能视为24月必然足够。未来真正起点、目标效应和方案需另行明确批准；旧未来协议文件是DRAFT，从未执行。当前用户要求停止开发和新实验准备。',9.4,GRAY)

    d.start('12 排序分母与解释边界')
    const=denominator[denominator.family.isin(['constant_RV','ewma'])]
    d.csvtable(const,['fold_id','family','N','positive','surprise_AUROC','surprise_AP','absolute_q90_AUROC'],
        ['时期','基线','N','事件N','surprise AUC','surprise AP','绝对q90 AUC'],7.5,[45,100,39,51,91,91,d.width-417])
    d.para('surprise事件=log((RV+epsilon)/(同一EWMA+epsilon))>log2；score=log((预测+epsilon)/(同一EWMA+epsilon))。constant_RV在折内是FIT标签均值，但共享EWMA分母仍可形成surprise排序。因此较好的surprise AUC不自动证明常数预测器有风险估计能力。')
    d.para('EWMA自身的surprise score为常数，AUROC=.5具有定义含义。绝对q90事件与surprise不是同一目标；折内常数的绝对事件排序仍为常数。跨折pooled常数受各FIT不同均值影响，仅作描述。')
    d.para('融合失败只约束现有R2输出与限定正值融合，不证明所有Kronos latent无信息；排序增量也不证明平均风险预测、尾部保护或交易收益。',9.2,GRAY)

    d.start('13 验收、失败记录与出版溯源')
    d.table([['证据','验收 / 范围'],['独立数值复核',f"PASS；{numeric['check_count']}检查；{len(numeric_bound)}文件绑定"],
        ['最终父交付',f"COMPLETE；{len(bound)}源/工件SHA绑定；CONSUMED保持"],['图表v2','PASS；全部10张PNG使用正式v2，初版FAIL保留'],
        ['科学候选','95次STARTED/COMPLETED；无失败科学fit或自动补训'],['出版操作','只新建PDF和独立出版manifest；不修改原run/state'],
        ['当前状态','用户要求停止开发；未来正式验证未启动；旧DRAFT未执行']],size=9)
    d.para('原code review初版/v2 FAIL、第一次prepare范围拒绝、data review过早PASS遗漏及其修正保留。图表首版图例/脚注重叠FAIL完整保留；v2图表数值与底表不变。本次PDF若出现排版失败，同样留存失败与旧输出，禁止覆写历史。')
    d.para('科学source seal、训练结果seal与后续受限运行manifest分别记录源、模型/scaler/选择器和确定性推理；核心参数未因结果更改。独立复核重载预测并验算损失/排序/选择/日块权重/判定，没有重新训练。')
    d.heading('可追溯源')
    d.para('数字：metrics/*.csv与summary.json；模型：每折selected_models、candidate_audit；图：figures/v2 PNG及其manifest/visual_review；正文依据：既有开发报告、未来DRAFT、项目决策三份Markdown。出版manifest另外记录字体、依赖、builder与实际消费者SHA。',9.4)
    d.para('父delivery_manifest SHA256：'+parent_sha,8,GRAY)
    d.para('页15-24以A3横幅展示全部真实已验收图表，使六面板文字更清晰，保持原UTC轴、单位、期间及基线。数字底表保留在原封存目录，本出版过程不再运行统计、模型或标签生成。',9.2,GRAY)

    for name,title,note in FIGURES:
        d.start(title,'正式figures/v2 | 五个开发OOF折与原VALID分别展示',figure=True)
        path=source(RUN/f'figures/v2/{name}.png',bound)
        d.figure(path,note+'\n来源：figures/v2/'+name+'.png；对应底表与来源SHA见原图manifest和本PDF出版manifest。')
    if d.number != 24:
        raise ValueError('Expected 24 explicitly bounded pages')
    d.save()
    if sha(delivery_file)!=parent_sha: raise ValueError('Parent delivery manifest changed')
    for key,digest in bound.items():
        if sha(ROOT/key)!=digest: raise ValueError('Parent source/artifact changed during PDF publication: '+key)
    reader=PdfReader(PDF); text='\n'.join(page.extract_text() or '' for page in reader.pages)
    required=['INCONCLUSIVE','0.384676','CONSUMED','停止开发','18,585','95','115.158']
    missing=[token for token in required if token not in text]
    if missing or len(reader.pages)!=24: raise ValueError('PDF text/page completeness failed: '+str(missing))
    exclusive(PUB/'page_outline.json',OUTLINE)
    doc=pymupdf.open(PDF)
    rendered=[]
    for index,page in enumerate(doc):
        dest=TMP/f'page_{index+1:02d}.png'
        if dest.exists(): raise FileExistsError('Existing rendering must be preserved: '+str(dest))
        page.get_pixmap(matrix=pymupdf.Matrix(1.5,1.5)).save(dest)
        rendered.append(str(dest.relative_to(ROOT)))
    doc.close()
    dependency_hashes={}
    for package in ('reportlab','pypdf','pymupdf','PIL'):
        directory=ROOT/'tmp/pdfs/deps'/package
        for file in directory.rglob('*'):
            if file.is_file() and '__pycache__' not in file.parts:
                dependency_hashes[file.relative_to(ROOT).as_posix()]=sha(file)
    exclusive(PUB/'publication_manifest.json',{'status':'BUILT_PENDING_VISUAL_REVIEW',
        'at_utc':datetime.now(timezone.utc).isoformat(),'PDF':str(PDF.relative_to(ROOT)),'PDF_sha256':sha(PDF),
        'page_count':len(reader.pages),'source_files_sha256':INPUTS,'font_files_sha256':font_hashes,
        'PDF_dependencies_sha256':dependency_hashes,'parent_delivery_manifest_sha256':parent_sha,
        'parent_sealed_binding_count':len(bound),'independent_numeric_binding_count':len(numeric_bound),
        'figures_v2_display_count':len(FIGURES),'parent_before_after_unchanged':True,
        'new_scientific_execution':False,'old_CONSUMED_unchanged':True,
        'current_publication_scope':'user requested stopping development and new-experiment preparation; PDF only',
        'rendered_pages':rendered})
    exclusive(PUB/'qa_draft.json',{'status':'PENDING_VISUAL_REVIEW','text_and_page_checks':'PASS',
        'missing_tokens':missing,'page_count':len(reader.pages),'all_10_figures_placed':True,
        'layout_overflow_checks':'PASS','visual_review_required':rendered,
        'note':'Rendering and text checks do not constitute visual acceptance. Root must inspect all pages before delivery.'})
    print(json.dumps({'PDF':str(PDF),'pages':len(reader.pages),'publication_manifest':str(PUB/'publication_manifest.json')},ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['build'])
    parser.parse_args()
    try: build()
    except BaseException as exc:
        PUB.mkdir(parents=True,exist_ok=True)
        index=1
        while (PUB/f'build_failure_attempt{index:02d}.json').exists(): index+=1
        exclusive(PUB/f'build_failure_attempt{index:02d}.json',{'status':'FAIL','error':repr(exc),
            'at_utc':datetime.now(timezone.utc).isoformat(),'partial_PDF_preserved':PDF.exists(),
            'partial_PDF_sha256':sha(PDF) if PDF.exists() else None})
        raise


if __name__=='__main__': main()
