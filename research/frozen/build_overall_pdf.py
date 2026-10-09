"""Publish a frozen-experiment PDF from accepted artifacts; no fitting or holdout access."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tmp/pdfs/deps'))
import json,hashlib,html,shutil
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import font_manager
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4,landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph,Table,TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfReader
import pymupdf as fitz

OUT=ROOT/'output/pdf'
TMP=ROOT/'tmp/pdfs/frozen_overall_v1'
RIDGE='FROZEN_20261008_v1'; MLP='FROZEN_MLP_20261008_v2'; QUANT='QUANT_20261008_v1'
RUNS=(RIDGE,MLP,QUANT)
FOLDS=('WF01','WF02','WF03','WF04')
PERIODS=('2025Q2','2025Q3','2025Q4','2026Q1')
DATES=pd.to_datetime(['2025-06-30','2025-09-30','2025-12-31','2026-03-31'],utc=True)
INK=colors.HexColor('#172B40'); BLUE=colors.HexColor('#215D88'); TEAL=colors.HexColor('#127568')
GRAY=colors.HexColor('#526373'); LIGHT=colors.HexColor('#EAF1F6')
INPUTS={}; PAGES=[]

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def source(rel):
    p=ROOT/rel
    if not p.exists(): raise FileNotFoundError(p)
    INPUTS[str(rel).replace('\\','/')]=sha(p)
    return p
def frame(run,filename): return pd.read_csv(source(f'research/runs/{run}/{filename}'))
def jread(rel): return json.loads(source(rel).read_text(encoding='utf-8'))
def fmt(v,d=2): return f'{float(v):+.{d}f}'
def status(v): return '通过' if v else '未通过'
def med_positive(v): return f'{np.median(v):+.2f}',f'{int((np.asarray(v)>0).sum())}/4'

class Document:
    def __init__(self,path):
        self.c=canvas.Canvas(str(path),pagesize=A4,pageCompression=1)
        self.c.setTitle('Kronos 冻结表征实验总体报告 | 2026-10-08')
        self.c.setAuthor('Kronos Crypto Research')
        self.c.setSubject('第一项冻结表征实验：Ridge、有限MLP、同坐标量化的开发期证据')
        self.number=0
    def start(self,title,subtitle='',wide=False):
        if self.number: self.c.showPage()
        self.number+=1; self.w,self.h=landscape(A4) if wide else A4
        self.c.setPageSize((self.w,self.h)); self.x=42; self.width=self.w-84
        self.c.setFillColor(INK); self.c.rect(0,self.h-13,self.w,13,fill=1,stroke=0)
        self.c.setFillColor(BLUE); self.c.setFont('CN',8)
        self.c.drawString(42,self.h-36,'KRONOS  /  FROZEN REPRESENTATIONS  /  DEVELOPMENT RESEARCH')
        self.c.setFont('CNBold',20 if not wide else 18); self.c.setFillColor(INK)
        self.c.drawString(42,self.h-66,title)
        self.y=self.h-83
        if subtitle: self.para(subtitle,9,color=GRAY,after=10)
        self.c.setStrokeColor(colors.HexColor('#CFD9E1')); self.c.line(42,35,self.w-42,35)
        self.c.setFont('CN',8); self.c.setFillColor(GRAY)
        self.c.drawString(42,22,'2026-10-08  |  已见开发季度的探索性研究  |  最终 holdout 封存')
        self.c.drawRightString(self.w-42,22,str(self.number))
        self.c.bookmarkPage('page'+str(self.number))
        self.c.addOutlineEntry(title,'page'+str(self.number),level=0)
        PAGES.append({'page':self.number,'title':title,'wide':wide})
    def para(self,text,size=10.3,color=INK,after=8,bold=False):
        style=ParagraphStyle('text',fontName='CNBold' if bold else 'CN',fontSize=size,leading=size*1.62,
                             textColor=color,wordWrap='CJK',splitLongWords=True)
        text=html.escape(text).replace('\n','<br/>')
        p=Paragraph(text,style); w,h=p.wrap(self.width,1000)
        if self.y-h<49: raise ValueError(f'Page {self.number} overflow in paragraph: {text[:50]}')
        p.drawOn(self.c,self.x,self.y-h); self.y-=h+after
    def heading(self,text): self.para(text,12,BLUE,after=6,bold=True)
    def table(self,rows,widths=None,size=9):
        if widths is None: widths=[self.width/len(rows[0])]*len(rows[0])
        style=ParagraphStyle('cell',fontName='CN',fontSize=size,leading=size*1.4,wordWrap='CJK',textColor=INK)
        data=[[Paragraph(html.escape(str(v)),style) for v in row] for row in rows]
        t=Table(data,colWidths=widths,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),LIGHT),('VALIGN',(0,0),(-1,-1),'TOP'),
            ('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#AABFCF')),
            ('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#DCE4EA')),
            ('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),
            ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
        w,h=t.wrap(self.width,1000)
        if self.y-h<49: raise ValueError(f'Page {self.number} table overflow {h:.1f}, remaining {self.y-49:.1f}')
        t.drawOn(self.c,self.x,self.y-h); self.y-=h+10
    def image(self,path,height=None):
        from PIL import Image
        iw,ih=Image.open(path).size
        height=min(height or self.width*ih/iw,self.y-68)
        width=height*iw/ih
        if width>self.width: width=self.width; height=width*ih/iw
        self.c.drawImage(str(path),self.x+(self.width-width)/2,self.y-height,width,height,mask='auto')
        self.y-=height+5
    def end(self): self.c.save()

def init_fonts():
    pdfmetrics.registerFont(TTFont('CN','C:/Windows/Fonts/msyh.ttc',subfontIndex=0))
    pdfmetrics.registerFont(TTFont('CNBold','C:/Windows/Fonts/msyhbd.ttc',subfontIndex=0))
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),
                         'axes.unicode_minus':False,'font.size':10,'svg.fonttype':'none'})

def curve_plot(curves,metrics,name,specs,cost):
    fig,axes=plt.subplots(2,2,figsize=(10.5,5.4))
    fig.subplots_adjust(left=.075,right=.96,top=.845,bottom=.09,hspace=.42,wspace=.22)
    palette=['#666666','#4477AA','#CC3311','#228833','#AA3377','#DDAA22','#66AABB']
    for ax,fold,period in zip(axes.flat,FOLDS,PERIODS):
        for i,(run,mode,label) in enumerate(specs):
            a=curves.loc[curves.run.eq(run)&curves.fold_id.eq(fold)&curves['mode'].eq(mode)&curves.cost.eq(cost)].sort_values('at')
            assert not a.empty
            metric_mode='legacy_E00_logistic_gate' if mode=='ordinary_features_gate' else mode
            expected=metrics.loc[metrics.source_run.eq(run)&metrics.fold_id.eq(fold)&metrics['mode'].eq(metric_mode)&metrics.cost.eq(cost),'net_return_pct'].item()
            assert abs(a.cumulative_return_pct.iloc[-1]-expected)<1e-8
            start=pd.Timestamp(period[:4]+'-'+{'Q1':'01','Q2':'04','Q3':'07','Q4':'10'}[period[4:]]+'-01',tz='UTC')
            end=start+pd.DateOffset(months=3)
            at=pd.to_datetime(a['at'],utc=True).tolist()+[end]
            v=a.cumulative_return_pct.tolist()+[a.cumulative_return_pct.iloc[-1]]
            ax.step(at,v,where='post',color=palette[i],lw=1.2,label=label)
        ax.set_xlim(start,end); ax.set_title(period,fontsize=11)
        ax.axhline(0,color='#888888',lw=.6); ax.grid(alpha=.2)
        ax.set_ylabel('累计净收益（%）',fontsize=9)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m',tz=timezone.utc))
        ax.tick_params(labelsize=8)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=min(4,len(labels)),frameon=False,fontsize=9)
    p=TMP/(name+'.png'); fig.savefig(p,dpi=180); plt.close(fig)
    return p

def plot_page(doc,curves,metrics,name,title,specs,cost,source_note):
    fee=7 if cost=='base' else 14
    doc.start(title,f'每边{fee}bps费用代理；实际资金另计。真实UTC时间，每季度独立重置10,000USDT。',wide=True)
    path=curve_plot(curves,metrics,name,specs,cost)
    doc.image(path,height=385)
    doc.para('闭仓事件更新的权益，不包含浮盈或持仓内盯市风险；期末延长横线仅为显示。'+source_note,8.2,GRAY,after=0)

def main():
    OUT.mkdir(parents=True,exist_ok=True); TMP.mkdir(parents=True,exist_ok=True)
    pdf=OUT/'Kronos_Frozen_Overall_Report_20261008_v1.pdf'
    if pdf.exists(): raise FileExistsError('Refuse final PDF overwrite')
    init_fonts()
    acceptance=jread('research/runs/FIRST_FROZEN_SYNTHESIS_20261008_v1/final_acceptance.json')
    assert acceptance['status']=='passed_complete_bounded_first_frozen_experiment'
    for rel,digest in acceptance['files_sha256'].items(): assert sha(ROOT/rel)==digest
    regs={run:jread(f'research/registry/{run}.json') for run in RUNS}
    for run,reg in regs.items():
        assert reg['status']=='completed_exploratory_development'
        for rel in ('analysis_report.json','paired_prediction_metrics.csv','paired_portfolio_metrics.csv','matched_random_summary.csv','paired_block_bootstrap.csv','metrics.csv'):
            assert sha(ROOT/'research/runs'/run/rel)==reg['artifacts']['final_hashes'][rel]
    analyses={r:jread(f'research/runs/{r}/analysis_report.json') for r in RUNS}
    for a in analyses.values(): assert not a['final_holdout_inspected']
    config=yaml.safe_load(source('research/configs/initial_experiment.yaml').read_text(encoding='utf-8'))
    source('research/frozen/FIRST_EXPERIMENT_CONCLUSION.md'); source('research/frozen/TODO.md')
    source('HANDOFF_FROZEN_REPRESENTATION.md'); source('research/Kronos_Crypto_Research_Roadmap.docx')
    for rel in ('frozen_comparison_v1.yaml','frozen_mlp_v2.yaml','frozen_quant_v1.yaml'): source('research/configs/'+rel)
    pred={r:frame(r,'paired_prediction_metrics.csv') for r in RUNS}
    port={r:frame(r,'paired_portfolio_metrics.csv') for r in RUNS}
    matched={r:frame(r,'matched_random_summary.csv') for r in RUNS}
    boots={r:frame(r,'paired_block_bootstrap.csv') for r in RUNS}
    all_metrics=frame(QUANT,'all_portfolio_metrics.csv')
    curves=frame(QUANT,'presentation/official_curves.csv')
    chart=jread(f'research/runs/{QUANT}/presentation/report.json')
    assert sha(ROOT/'research/runs'/QUANT/'presentation/official_curves.csv')==chart['outputs_sha256']['official_curves.csv']
    assert len(all_metrics)==512 and len(curves.groupby(['run','fold_id','mode','cost']))==512
    curves['at']=pd.to_datetime(curves['at'],utc=True)
    metric_names=[('E00R_20261008_phase4_v1','ridge19','普通19 Ridge'),(RIDGE,'ridge_pretrained','预训练531 Ridge'),
                  (MLP,'mlp19','普通19 MLP'),(MLP,'mlp_pretrained','预训练531 MLP'),
                  (QUANT,'ridge_quant_continuous','连续u39 Ridge'),(QUANT,'ridge_quant_binary','二值q39 Ridge')]
    def returns(run,family,gate,cost):
        a=all_metrics.loc[all_metrics.source_run.eq(run)&all_metrics['mode'].eq(family+'_'+gate)&all_metrics.cost.eq(cost)].set_index('fold_id').loc[list(FOLDS)]
        return a
    doc=Document(pdf)
    doc.start('冻结表征实验总体报告','第一项实验 / Roadmap 第5章与M1 / 开发期完整总结',False)
    doc.para('Kronos Crypto Research',18,BLUE,bold=True,after=16)
    doc.para('当前任务未建立稳定的信息增量、成本后经济价值或量化瓶颈。',17,bold=True,after=15)
    doc.para('工程验收完成，研究结论为证据不足。停止当前BTC单资产、1h历史、4h固定动量参与任务上的读出容量与冻结表征扩展；保留局部正向信号和全部负结果。',12,after=15)
    doc.table([['新增正式拟合','新增官方回放','保留官方曲线'],['276次','352份','512条']],size=12)
    doc.heading('三条结论')
    doc.para('01  同Ridge预训练表征相对普通19项的MSE中位改善为-2.84%，四折均未改善。优于随机骨干的部分证据不能替代普通特征增量。')
    doc.para('02  有限MLP能强烈拟合训练样本，却未形成开发增量：相对普通MLP的MSE中位改善为-54.63%。当前不支持继续扩大预测头。')
    doc.para('03  连续u与二值q在相同20坐标、相同39维Ridge输入下差异微小且不稳定；量化筛选未通过。量化主门控多季度少亏，但所有季度仍亏损。')
    doc.para('阅读提示：本报告使用此前已见的2025Q2至2026Q1开发季度。checkpoint训练截止未获证明，不能宣称严格事前样本外。最终holdout继续封存。',9.5,GRAY)
    doc.para('报告日期：2026-10-08。版本：v1。实验阶段：frozen_development。研究状态：first_frozen_experiment_completed_holdout_sealed。',9,GRAY)

    doc.start('01  固定任务与数据时序','模型只控制参与或跳过；方向、仓位、时间和成本来自固定协议。')
    doc.table([['要素','固定口径'],['市场','OKX BTC-USDT-SWAP，USDT线性永续，单资产，1h'],['输入','256根已完成历史OHLCVA；volume=BTC，amount=真实USDT成交额'],['方向','24h固定动量 sign(log(close_t / close_t-24))，零动量跳过'],['决策','每8h，UTC04/12/20起点，加60秒可用延迟'],['成交','决策后下一小时开盘入场，严格持有4h'],['成本','每边7bps基础、14bps压力费用代理；实际资金费用另计'],['组合','每季度从10,000USDT独立重置，1x杠杆'],['目标','官方参考交易基础成本已实现净收益乘10,000，单位bps']],widths=[88,doc.width-88],size=9.8)
    doc.heading('窗口和标签的因果约束')
    doc.para('输入窗口只含当时已完成、可用的行情。真实成交额来自volCcyQuote，不能用价格乘成交量替代。未来资金、已实现收益和标签放在禁止入特征的侧表。')
    doc.para('标签来自官方Freqtrade导出，保留实际结算率、原始标记价和支付/收取符号。参考标签的固定名义仓位、大额虚拟余额不能被当作投资组合业绩。')
    doc.para('对跨持有期边界、labelable_at跨界执行purge。标准化和预处理只在训练折拟合；模型超参数和门槛只由此前验证季度选择。')

    doc.start('02  时间折与统一读出预算','四个开发测试季度按时间滚动；最终holdout不参与本报告统计或选择。')
    rows=[['折','训练区间','验证季度','开发测试季度']]
    for f,period in zip(config['splits']['walk_forward_dates'],PERIODS):
        rows.append([f['id'],f['train'][0][:10]+' 至 '+f['train'][1][:10],f['validation'][0][:7]+' 至 '+f['validation'][1][:7],period])
    doc.table(rows,widths=[45,210,140,doc.width-395],size=9)
    doc.para('所有区间按[start,end_exclusive)解释，表中右界不包含；最终成员以已purge的split_index为准。',9,GRAY)
    doc.heading('三轮探索与新增正式预算')
    doc.table([['实验','主问题','新拟合','新回放'],['Ridge / E01R-E02R','骨干预训练的任务增量','64','128'],['有限MLP','简单线性读出是否限制可用性','180','160'],['同坐标量化','连续u与符号q的任务可读性','32','64'],['合计','保留全部尝试，不选最佳随机种子','276','352']],widths=[104,263,65,doc.width-432],size=9)
    doc.para('旧E00的48份及E00R的112份官方回放复用已验收工件，加上352份新增导出，共512条官方曲线。工程重算、模型加载和正规方程复核不增加正式拟合。MLP失败v1在任何拟合前停止，正式拟合为0。')
    doc.heading('Ridge与小MLP')
    doc.para('Ridge：训练折StandardScaler，带截距，SVD求解；alpha=n_train×lambda，候选0.001/0.01/0.1/1，验证MSE选择，同分选较大lambda。')
    doc.para('MLP：16单元Tanh单隐层，CPU float32；输入和目标训练折标准化；全批次AdamW，学习率0.003，固定300轮。正则0.001/0.01/0.1，头种子17/29/43预测均值为执行分数，按验证集集成MSE选择。19项头337参数，531项头8,529参数；同宽度不等于同参数量。')

    doc.start('03  特征路径与对照含义','冻结表示已经包含非线性变换；下游Ridge检验的是这些表示能否被简单读出。')
    doc.table([['变体','特征与维度','检验内容'],['普通基线','19项普通特征；13项消融保留','已显式提供的信息'],['预训练骨干','普通19+预训练末端context512=531','是否增加普通特征以外的信息'],['随机骨干','普通19+随机末端context512=531，种子17/29/43','隔离骨干预训练贡献'],['连续量化前','普通19+L2归一化quant_embed u20=39','同坐标连续可读性'],['二值量化后','普通19+BSQuantizer q20=39','相同坐标符号量化可读性']],widths=[95,233,doc.width-328],size=9.8)
    doc.heading('逐机会编码与预处理')
    doc.para('每个机会独立构建256根窗口；六字段按窗口均值和std(ddof=0)标准化，分母加1e-5，clip至[-5,5]。不能将全历史编码一次再切hidden，因为窗口归一化会改变同一根K线的输入。')
    doc.para('使用同一个固定预训练tokenizer。encode(half=True)返回显式s1与s2两组ID；decode_s1消费两组ID及时间信息，取最后有效历史位置512维context。缓存四组各2,460×512 float32。')
    doc.para('随机骨干保持相同网络结构，但冻结随机初始化权重；tokenizer仍保留预训练信息，因此不能声称移除了所有预训练信息。三种子全部保留。')
    doc.heading('量化诊断严格同坐标')
    doc.para('u是quant_embed的20坐标输出经L2归一化；q同坐标取值：u>0为+1/√20，u≤0为-1/√20。两组都取最后历史位置、拼相同普通19项、使用同Ridge候选。')
    doc.para('39维量化诊断与531维历史骨干的维度、路径及时间嵌入不同；跨路径比较只作解释参照，不能据此归因量化或骨干信息损失。')

    doc.start('04  预登记筛选与判定口径','用户在冻结结果产生前采用的研究筛选标准；信息与经济分别判断。')
    doc.table([['维度','必要条件'],['预测增量','MSE跨折中位改善≥1%，至少3/4折为正；首轮/MLP同时比较普通模型和三随机骨干损失均值'],['匹配参与数','主50%信息门控在UTC入场月份×固定方向内匹配数量，净参考收益中位增量≥1bps、至少3/4折为正'],['配对证据','关键主7日时间块95%下界>0，相关效用比较至少3/4折为正；保存3/14日敏感性'],['经济价值','相对普通模型季度中位增量≥1个百分点，至少3/4折增量为正、至少3/4季度优于现金'],['压力与风险','14bps季度中位净收益和增量均>0；基础闭仓最大回撤中位变化不更差'],['量化瓶颈','u相对q中位MSE改善≥1%、至少3/4折为正、等折主7日配对MSE差下界>0']],widths=[83,doc.width-83],size=9.5)
    doc.heading('门控与不确定性')
    doc.para('主信息门控按验证50%覆盖分位数固定门槛，25%/75%仅是预定敏感性。测试不重新选top-k、不强迫参与率；经济门控严格预测基础净收益>0。压力成本复用基础预测和动作，不再次拟合或重复扣成本。')
    doc.para('完整UTC决策日历保留零机会日；折内stationary bootstrap 5,000次，平均块长主7天、敏感性3/14天，折种子17/117/217/317，各折独立重采样再等权。参考效用单位bps/日，预测差单位bps²。')
    doc.para('匹配随机是事后诊断，不是执行策略；其抽样分位数不是组合业绩区间。时间块区间也不是复合组合收益置信区间。',9.5,GRAY)

    vals=frame('FIRST_FROZEN_SYNTHESIS_20261008_v1','prediction_loss_by_quarter.csv')
    fig,ax=plt.subplots(figsize=(10.5,5.3)); fig.subplots_adjust(left=.075,right=.98,bottom=.13,top=.77)
    for label,part in vals.groupby('variant',sort=False):
        ax.plot(DATES,part.RMSE_bps,marker='o',label=label,lw=1.4,ls='--' if '随机' in label else '-')
    ax.grid(alpha=.23); ax.set_ylabel('开发测试RMSE（bps）'); ax.set_xlabel('测试季度末（UTC）')
    ax.set_xticks(DATES); ax.set_xticklabels(PERIODS)
    fig.legend(*ax.get_legend_handles_labels(),loc='upper center',ncol=3,fontsize=9,frameon=False)
    lossplot=TMP/'prediction_loss.png';fig.savefig(lossplot,dpi=180);plt.close(fig)
    doc.start('05  四季度预测损失','目标为官方7bps基础成本净收益；每点为该季度整体损失，不是累计损失。',wide=True)
    doc.image(lossplot,height=385)
    doc.para('来源：综合prediction_loss_by_quarter.csv。随机曲线为sqrt(三个随机骨干MSE均值)，不是三策略分数集成或可执行组合。',8.5,GRAY)

    doc.start('06  预测增量：数值与判读','相对改善=(比较对象MSE-目标MSE)/比较对象MSE；跨折中位数与等折平均差是不同统计量。')
    rows=[['比较','中位MSE改善','正向折数']]
    comps=[(RIDGE,'relative_MSE_improvement_vs_ordinary','预训练Ridge vs 普通Ridge'),(RIDGE,'relative_MSE_improvement_vs_seed_mean','预训练Ridge vs 随机损失均值'),
           (MLP,'relative_MSE_improvement_vs_ordinary','预训练MLP vs 普通MLP'),(MLP,'relative_MSE_improvement_vs_seed_mean','预训练MLP vs 随机损失均值'),
           (QUANT,'relative_MSE_improvement_continuous_vs_ordinary','u39 vs 普通19 Ridge'),(QUANT,'relative_MSE_improvement_binary_vs_ordinary','q39 vs 普通19 Ridge'),
           (QUANT,'relative_MSE_improvement_continuous_vs_binary','u39 vs q39')]
    for run,col,label in comps:
        values=pred[run][col];rows.append([label,fmt(values.median()*100)+'%',f'{int((values>0).sum())}/4'])
    doc.table(rows,widths=[290,120,doc.width-410],size=9.5)
    doc.heading('各折开发测试RMSE（bps）')
    pivot=vals.pivot(index='variant',columns='fold_id',values='RMSE_bps')
    rows=[['变体',*PERIODS]]+[[name,*[f'{pivot.loc[name,f]:.2f}' for f in FOLDS]] for name in pivot.index]
    doc.table(rows,widths=[187,*[(doc.width-187)/4]*4],size=8.9)
    doc.para('Ridge的预训练表征四折均劣于普通19项，虽有优于随机骨干的信号，仍不能证明对普通基线的增量。量化两组相对普通的中位改善均小于1%，各仅2/4折为正。',9.5)

    loss=frame(MLP,'mlp_loss_chart_values.csv')
    fig,axes=plt.subplots(1,2,figsize=(10.5,5.1)); fig.subplots_adjust(left=.08,right=.98,top=.82,bottom=.15,wspace=.25)
    for ax,family,title in zip(axes,['mlp19','mlp_pretrained'],['普通19 MLP','预训练531 MLP']):
        for role,label,color in [('train','训练','#228833'),('validation','验证','#4477AA'),('test','开发测试','#CC3311')]:
            a=loss.loc[loss.family.eq(family)&loss.role.eq(role)].set_index('fold_id').loc[list(FOLDS)]
            ax.plot(DATES,a.RMSE_bps,marker='o',label=label,color=color)
        ax.set_title(title);ax.set_ylabel('RMSE（bps）');ax.set_xticks(DATES);ax.set_xticklabels(PERIODS,rotation=15);ax.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False)
    lossplot=TMP/'fit_generalization.png';fig.savefig(lossplot,dpi=180);plt.close(fig)
    doc.start('07  拟合能力与开发泛化','以测试季度对齐各折；训练、验证和测试实际覆盖不同时期。',wide=True)
    doc.image(lossplot,height=360)
    doc.para('预训练MLP训练RMSE为7.94/9.06/11.24/12.45bps，测试为121.23/115.25/144.22/147.08bps。实际执行预测为三头均值；来源：mlp_loss_chart_values.csv。',9,GRAY)

    doc.start('08  读出诊断的结论','提高样本内拟合能力没有带来当前开发任务的稳定增量。')
    doc.heading('为什么一层线性头仍有研究价值')
    doc.para('Ridge之前的冻结tokenizer与Transformer已经是非线性历史变换。线性头检验的是预训练表示是否将目标相关信息组织为简单可读的形式，而非用一层线性模型从原始行情学完所有非线性关系。')
    doc.para('Ridge直接SVD求解，没有训练轮数不足或优化器未收敛的问题。正则化与验证选择限制高维拟合自由度；训练误差降低不能单独证明可交易信息。')
    doc.heading('有限MLP给出的证据')
    doc.para('MLP在预训练531项上已显著压低训练误差，但开发MSE相对普通MLP四折均恶化。由当前预算结果，不支持继续以“线性层拟合不上”为由扩头或追加大量候选。')
    doc.para('训练与测试的时间、噪声和收益分布不同，不能只凭损失差唯一归因于过拟合或欠拟合。现有结果只支持“本次有限增强读出没有建立开发增量”。')
    doc.heading('同坐标量化诊断给出的证据')
    doc.para('u相对q中位MSE改善-0.05%，仅1/4折为正；等折平均MSE差5.26bps²的95%区间[-30.13,41.54]跨零。未达到量化瓶颈预登记筛选。')
    doc.para('无法由这一末端20坐标Ridge诊断宣称一般信息损失、tokenizer没有损失，或Kronos在其他任务上没有信息；同样不能据此自动重建tokenizer。')
    doc.heading('停止范围')
    doc.para('停止当前BTC单资产、1h历史、4h固定动量参与任务上的扩头和冻结表征扩展。新资产、独立时期、其他目标、池化、稳健损失或原生多路径生成均属新研究，需要另行登记明确问题与预算。')

    ordinary='E00R_20261008_phase4_v1'; old='E00_20261008_phase4_v1'
    for cost,costname in [('base','基础成本'),('stress','压力成本')]:
        plot_page(doc,curves,all_metrics,'ridge_'+cost,'09  同Ridge主门控 / '+costname,
            [(old,'cash','现金'),(ordinary,'ridge19_rank50','普通19'),(RIDGE,'ridge_pretrained_rank50','预训练531'),
             *[(RIDGE,f'ridge_random_s{s}_rank50',f'随机骨干{s}') for s in (17,29,43)]],cost,'首轮预训练与随机骨干保持同一个预训练tokenizer。')
    for cost,costname in [('base','基础成本'),('stress','压力成本')]:
        plot_page(doc,curves,all_metrics,'mlp_'+cost,'10  有限MLP主门控 / '+costname,
            [(old,'cash','现金'),(MLP,'mlp19_rank50','普通19'),(MLP,'mlp_pretrained_rank50','预训练531'),
             *[(MLP,f'mlp_random_s{s}_rank50',f'随机骨干{s}') for s in (17,29,43)]],cost,'每个MLP策略实际执行三头预测均值；三个随机骨干政策分别保留。')
    for cost,costname in [('base','基础成本'),('stress','压力成本')]:
        plot_page(doc,curves,all_metrics,'quant_'+cost,'11  同坐标量化主门控 / '+costname,
            [(old,'cash','现金'),(ordinary,'ridge19_rank50','普通19'),(QUANT,'ridge_quant_continuous_rank50','连续u39'),
             (QUANT,'ridge_quant_binary_rank50','二值q39'),(ordinary,'ridge13_rank50','普通13消融')],cost,'u/q同39维、同Ridge；不按测试收益选取表现更好的量化变体。')

    doc.start('12  主门控季度收益底表','累计净收益%，每季度独立重置；基础7bps与压力14bps均为每边费用代理。')
    for cost,title in [('base','基础成本'),('stress','压力成本')]:
        doc.heading(title)
        rows=[['策略',*PERIODS]]
        for run,family,label in metric_names:
            a=returns(run,family,'rank50',cost)
            rows.append([label,*[fmt(v) for v in a.net_return_pct]])
        doc.table(rows,widths=[183,*[(doc.width-183)/4]*4],size=9)
    doc.para('预训练Ridge、预训练MLP、u39和q39主门控在基础成本下均为0/4季度优于现金。缩小亏损并不等于产生正收益。',9.5)
    doc.para('u39基础季度相对普通Ridge的中位增量+7.59个百分点、3/4折为正；q39中位增量+3.25个百分点、4/4折为正。这些局部信号保留，但尚不能替代预测、配对及现金/压力盈利要求。',9.5)

    for cost,costname in [('base','基础成本'),('stress','压力成本')]:
        plot_page(doc,curves,all_metrics,'economic_'+cost,'13  量化经济门控 / '+costname,
            [(old,'cash','现金'),(ordinary,'ridge19_economic','普通19'),(QUANT,'ridge_quant_continuous_economic','连续u39'),
             (QUANT,'ridge_quant_binary_economic','二值q39'),(ordinary,'ridge13_economic','普通13消融')],cost,'经济门控固定为基础净收益预测>0，不为增加交易而降低门槛。')

    doc.start('14  经济门控季度收益底表','经济门控与50%信息门控分别评价；零交易季度按官方结果保留。')
    for cost,title in [('base','基础成本：7bps'),('stress','压力成本：14bps')]:
        doc.heading(title)
        rows=[['策略',*PERIODS]]
        for run,family,label in metric_names:
            a=returns(run,family,'economic',cost)
            rows.append([label,*[fmt(v) for v in a.net_return_pct]])
        doc.table(rows,widths=[183,*[(doc.width-183)/4]*4],size=9)
    doc.para('各新表征方案的独立经济门控均未通过预登记经济筛选；盈利个别季度不能替代跨折一致性、压力成本和回撤证据。',9.5)
    doc.para('资金费用由官方引擎按实际结算与原始标记价处理。7/14bps不是成交价格滑点、冲击或订单簿模拟；未覆盖的真实成交风险不能由闭仓曲线证明。',9.5)

    doc.start('15  配对预测损失区间','主7日时间块，5,000次重采样，四折等权；差值为比较对象MSE减目标MSE。')
    cirows=[['预测比较','差值bps²','95%区间']]
    cicomps=[(RIDGE,'MSE_improvement_vs_ordinary','Ridge预训练 vs 普通'),(RIDGE,'MSE_improvement_vs_mean_random_backbone','Ridge预训练 vs 随机均值'),
             (MLP,'MSE_improvement_vs_ordinary','MLP预训练 vs 普通'),(MLP,'MSE_improvement_vs_mean_random_backbone','MLP预训练 vs 随机均值'),
             (QUANT,'ridge_quant_continuous_MSE_vs_ordinary','u39 vs 普通'),(QUANT,'ridge_quant_binary_MSE_vs_ordinary','q39 vs 普通'),
             (QUANT,'continuous_vs_binary_MSE','u39 vs q39')]
    for run,comparison,label in cicomps:
        a=boots[run].loc[boots[run].fold_id.eq('EQUAL_FOLD_MEAN')&boots[run].cost.eq('base')&boots[run].mean_block_days.eq(7)&boots[run].comparison.eq(comparison)].iloc[0]
        cirows.append([label,fmt(a.point_difference),'['+fmt(a.lower_95)+', '+fmt(a.upper_95)+']'])
    doc.table(cirows,widths=[234,95,doc.width-329],size=9.5)
    doc.heading('区间的含义')
    doc.para('Ridge预训练相对普通模型的MSE差区间整体为负；相对随机骨干均值有正向证据。MLP相对普通的差区间同样整体为负，相对随机区间跨零。')
    doc.para('u/q相对普通以及u相对q的区间均跨零。四折MSE平均差和相对改善中位数可能具有不同符号，它们不是同一个统计量。')
    doc.para('每日汇总平方误差差后按days/n缩放，使日均换算回每机会MSE差（bps²）；保留完整零机会日历。折内近似平稳、跨折依赖可忽略、约每季度13个有效7天块构成限制。')
    doc.para('保存3/14日敏感性，未在看见区间后改块长、阈值、候选或种子。量化种子派生说明在执行后补齐，独立复核确认只增加文字元数据，计算与原执行源完全一致。',9.5,GRAY)

    doc.start('16  选择效用与局部正向信号','独立参考效用按完整UTC决策日历计算，单位bps/日；不等于组合收益。')
    rows=[['主rank50比较','差值bps/日','95%区间']]
    uc=[(RIDGE,'rank50_vs_ordinary','预训练Ridge vs 普通'),(RIDGE,'rank50_vs_matched_expected_counts','预训练Ridge vs 匹配数量'),
        (MLP,'rank50_vs_ordinary','预训练MLP vs 普通'),(MLP,'rank50_vs_matched_expected_counts','预训练MLP vs 匹配数量'),
        (QUANT,'ridge_quant_continuous_rank50_vs_ordinary','u39 vs 普通'),(QUANT,'ridge_quant_continuous_rank50_vs_matched_counts','u39 vs 匹配数量'),
        (QUANT,'ridge_quant_binary_rank50_vs_ordinary','q39 vs 普通'),(QUANT,'ridge_quant_binary_rank50_vs_matched_counts','q39 vs 匹配数量')]
    for run,comparison,label in uc:
        a=boots[run].loc[boots[run].fold_id.eq('EQUAL_FOLD_MEAN')&boots[run].cost.eq('base')&boots[run].mean_block_days.eq(7)&boots[run].comparison.eq(comparison)].iloc[0]
        rows.append([label,fmt(a.point_difference),'['+fmt(a.lower_95)+', '+fmt(a.upper_95)+']'])
    doc.table(rows,widths=[243,95,doc.width-338],size=9.2)
    rows=[['目标','匹配净增量中位bps','正向折数']]
    for run,mode,label in [(RIDGE,'ridge_pretrained_rank50','预训练Ridge'),(MLP,'mlp_pretrained_rank50','预训练MLP'),(QUANT,'ridge_quant_continuous_rank50','u39'),(QUANT,'ridge_quant_binary_rank50','q39')]:
        a=matched[run].loc[matched[run].cost.eq('base')&matched[run]['mode'].eq(mode),'exact_selection_lift_bps']
        rows.append([label,fmt(a.median()),f'{int((a>0).sum())}/4'])
    doc.heading('匹配参与数量的精确分层期望')
    doc.table(rows,widths=[170,210,doc.width-380],size=9)
    doc.para('此表使用exact_selection_lift_bps，非3,000次抽样均值列。u/q对匹配数量有正向效用区间，但相对普通模型区间跨零、预测改善不足、季度收益仍亏损，整体信息筛选仍未通过。',9.3)

    doc.start('17  全部筛选总览','任何单项通过都不能替代同时满足全部必要条件。量化不以不同维度的随机512骨干为硬筛选。')
    rows=[['目标','信息筛选','50%门控经济','经济门控经济']]
    for run,label in [(RIDGE,'预训练Ridge'),(MLP,'预训练MLP')]:
        a=analyses[run];rows.append([label,status(a['information_screen']['passed']),status(a['economic_screens']['rank50']['passed']),status(a['economic_screens']['economic']['passed'])])
    for family,label in [('ridge_quant_continuous','连续u39'),('ridge_quant_binary','二值q39')]:
        a=analyses[QUANT];rows.append([label,status(a['information_screens'][family]['passed']),status(a['economic_screens'][family]['rank50']['passed']),status(a['economic_screens'][family]['economic']['passed'])])
    doc.table(rows,widths=[148,117,130,doc.width-395],size=9.5)
    doc.para('同坐标量化可读性筛选：'+status(analyses[QUANT]['quantization_screen']['passed'])+'。',11,bold=True)
    doc.heading('通过的局部检查及未通过的关键条件')
    doc.para('预训练Ridge：相对随机骨干损失均值的预测改善、匹配参与数量的中位增量与正向折数达标；相对普通预测增量、关键配对效用和经济条件未满足。')
    doc.para('预训练MLP：相对随机均值的MSE中位改善有正向描述值，但普通预测、折一致性、匹配筛选和经济条件未满足。')
    doc.para('u/q主rank50：基础组合增量正向≥3折、中位增量≥1个百分点、压力相对普通的中位增量及闭仓回撤变化检查通过；匹配数量增量检查通过。普通预测、关键全部区间下界和现金/压力盈利要求未满足。')
    doc.para('u/q经济门控与三项量化筛选均未通过。全部逐项真假保存在all_screens.json与三轮analysis_report.json，不删除负结果，不根据收益改变主比较。')
    doc.heading('为什么当前判定为证据不足')
    doc.para('当前证据无法同时支持稳定信息增量及成本后盈利。少亏、降低暴露、个别季度盈利、高维拟合能力和一次匹配诊断均不足以晋级。停止当前任务的复杂度扩展，等待用户验收。')

    plot_page(doc,curves,all_metrics,'legacy_rules','18  原规则与旧门控基线',
        [(old,'cash','现金'),(old,'buy_hold','买入持有'),(old,'fixed_momentum','固定动量'),(old,'constant_half_exposure','半仓'),
         (old,'vol_target','波动目标'),(old,'ordinary_features_gate','旧logistic')],'base','买入持有闭仓曲线大部分平坦不能解释为持仓没有波动；原门控和固定规则保留。')

    doc.start('19  预定门控敏感性底表','各格为四个独立季度净收益的中位数（%），不是年度复利或按表现选择门控。',wide=True)
    rows=[['策略','25% / 7bps','50% / 7bps','75% / 7bps','25% / 14bps','50% / 14bps','75% / 14bps']]
    sensitivity=[]
    for run,family,label in metric_names:
        vals_row=[]
        for cost in ('base','stress'):
            for gate in ('rank25','rank50','rank75'):
                vals_row.append(fmt(returns(run,family,gate,cost).net_return_pct.median()))
        rows.append([label,*vals_row]);sensitivity.append({'run':run,'family':family,'values':vals_row})
    doc.table(rows,widths=[181,*[(doc.width-181)/6]*6],size=9.5)
    doc.para('所有随机骨干种子、普通13项消融、周期参与和随机参与对照继续保留在512条曲线和官方指标CSV中。敏感性是原先锁定的报告项，不在此处选择更好的门控或重训。',10)
    doc.para('由于验证分位数在测试期固定，25%/50%/75%是验证目标覆盖而非强制测试参与率；等分数纳入可能使实际覆盖不同。',10)
    doc.para('数据来源：QUANT_20261008_v1/all_portfolio_metrics.csv；原各轮metrics.csv和官方ZIP仍为规范证据。',9,GRAY)

    doc.start('20  工程验收与失败记录','编码、参数、官方账本和展示链路分别核验；工程通过不代表研究假设通过。')
    doc.table([['链路','已完成证据'],['编码','批次/单条与原token IDs一致、未来扰动不变、缺口/真实amount/可用时间/ID/边界拒绝、分块hash'],['缓存复算','原512维缓存独立复算；u/q首/中/末CPU float32对CUDA缓存：u最大误差1.4752e-6、q误差0，容差1e-4'],['预测头','训练折scaler、全部候选、保存/加载、验证选择、Ridge正规方程、测试预测及sidecar动作复核'],['官方回放','全部准入信号成交；方向、原价格、严格4h、7/14bps费用和实际带符号资金逐笔核验'],['图表','512条官方曲线hash和端点核验；最大端点误差约1.78e-13个百分点'],['范围','最终holdout无编码或标签业绩评估；无骨干/tokenizer训练、微调、部署或push']],widths=[83,doc.width-83],size=9.3)
    doc.heading('保留的失败和修正')
    doc.para('MLP v1：历史行索引整数契约预检失败，未开始任何正式拟合。v2只修复CSV字段解析，仍按原180次预算执行；v1原登记、工件和源码保留。')
    doc.para('MLP图表：历史配置布局预检失败，读取正确parent_configuration后生成，未增加拟合或回放。首轮原文字节/行尾hash修复历史也保留。')
    doc.para('量化bootstrap：执行源使用17+折序号×100，与先前工作流一致；计划文档未明确派生，后补说明并独立确认仅文字元数据差异。原执行源与hash仍保留。')
    doc.para('总体Markdown报告仅修正资金符号译法及二值零规则文案，原版归档；数据、阈值和结论未变。本PDF只读取既有DEV工件，不新增研究尝试。',9.5)

    doc.start('21  证据边界与下一步','当前有限研究范围已完成，没有待运行的正式实验。')
    doc.heading('结论适用范围')
    doc.para('结论适用于本次BTC-USDT永续单资产、1h窗口、4h固定动量参与任务、既定读出与选择预算。不能推广为Kronos普遍无信息，也未检验其他资产、周期、标签或原生未来路径。')
    doc.heading('时间与checkpoint')
    doc.para('开发季度此前已见，后续比较属于探索性历史研究。checkpoint训练截止尚未证明；协议预登记不使历史折重新成为未见测试，也不证明预训练严格事前可用。')
    doc.heading('holdout与数据完整性')
    doc.para('最终holdout为[2026-04-01,2026-10-01)，未编码或评估收益、标签统计与分类比例。早期来源字段审计曾显示原始OHLCV页面，已有记录；未用它选择模型或阈值。')
    doc.para('数据完整性证据限于已公布官方来源的覆盖、一致性与可追溯性，不是交易所内部账本或历史事前可用性的完整证明。60秒延迟和7/14bps费用都是保守研究假设，非实测撮合/冲击。')
    doc.heading('研究停止与重新开启')
    doc.para('当前停止扩头、tokenizer重建及骨干解冻。未来若重新探索，先明确可证伪问题，使用独立数据或新资产和有限预算另行登记；保留当前失败结果，不能在已见DEV上无限搜索。')
    doc.para('最终holdout不用于选择是否加入Kronos、晋级、模型或阈值。继续禁用骨干/tokenizer训练、微调、原生路径生成、push、部署与实盘。',10,bold=True)

    doc.start('22  复核索引与版本','本报告的所有数值均源自已验收DEV工件；本机ignored缓存与ZIP不会随git clone取得。')
    refs=[('S01','完整开发期结论','research/frozen/FIRST_EXPERIMENT_CONCLUSION.md'),('S02','冻结阶段独立TODO','research/frozen/TODO.md'),
          ('S03','综合最终验收','research/runs/FIRST_FROZEN_SYNTHESIS_20261008_v1/final_acceptance.json'),
          ('S04','首轮Ridge登记','research/registry/'+RIDGE+'.json'),('S05','有限MLP登记','research/registry/'+MLP+'.json'),('S06','量化登记','research/registry/'+QUANT+'.json'),
          ('S07','全部筛选与真假','research/runs/FIRST_FROZEN_SYNTHESIS_20261008_v1/all_screens.json'),
          ('S08','512条官方曲线','research/runs/'+QUANT+'/presentation/official_curves.csv'),
          ('S09','所有组合指标','research/runs/'+QUANT+'/all_portfolio_metrics.csv'),
          ('S10','量化独立缓存复算','research/runs/'+QUANT+'/root_quant_cache_review.json')]
    for key,label,rel in refs:
        source(rel);doc.para(f'{key}  {label}\n{rel}',8.7,after=7)
    doc.para('三轮run中另含：analysis_report.json、paired_prediction_metrics.csv、paired_portfolio_metrics.csv、matched_random_summary.csv、paired_block_bootstrap.csv、metrics.csv、provenance以及官方导出。路径均相对于D:/file/Kronos-master。',8.8)
    doc.para('固定模型：NeoQuasar/Kronos-small revision 901c26c1332695a2a8f243eb2f37243a37bea320。Tokenizer revision 0e0117387f39004a9016484a186a908917e22426。Freqtrade 2026.9，commit 1f394eaebc2f46a83d26971388628707802f8602。',8.7)
    doc.end()
    reader=PdfReader(pdf);text='\n'.join(p.extract_text() or '' for p in reader.pages)
    assert len(reader.pages)==len(PAGES)
    for expected in ['276','352','512','证据不足','最终holdout','量化','-54.63']:
        assert expected in text,expected
    render=TMP/'rendered';render.mkdir(exist_ok=True)
    opened=fitz.open(pdf)
    for i,page in enumerate(opened):
        pix=page.get_pixmap(matrix=fitz.Matrix(1.25,1.25),alpha=False)
        pix.save(str(render/f'page_{i+1:02d}.png'))
    pd.DataFrame(PAGES).to_csv(OUT/'Kronos_Frozen_Overall_Report_20261008_v1_pages.csv',index=False)
    evidence={'status':'generated_pending_visual_QA','report':str(pdf),'pages':len(PAGES),'created_at_utc':datetime.now(timezone.utc).isoformat(),
       'pdf_sha256':sha(pdf),'source_sha256':sha(Path(__file__)),'input_sha256':INPUTS,'data_scope':'accepted_development_artifacts_only',
       'new_fits':0,'new_backtests':0,'holdout_read':False,'research_protocols_changed':False,'renderer':'PyMuPDF; Poppler unavailable locally',
       'page_index':PAGES,'text_extraction_check':'required assertions passed; CJK embedded fonts','final_visual_QA':'pending'}
    (OUT/'Kronos_Frozen_Overall_Report_20261008_v1_manifest.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'pdf':str(pdf),'pages':len(PAGES),'rendered':str(render),'pdf_sha256':sha(pdf)},ensure_ascii=False))

if __name__=='__main__': main()
