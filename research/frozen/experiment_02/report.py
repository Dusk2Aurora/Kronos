"""Create the complete Chinese risk report from audited, registered DEV results."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/'tmp/pdfs/deps')]
import hashlib,json,math
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from PIL import Image,ImageDraw
from pypdf import PdfReader
import pymupdf as fitz
from research.frozen.build_overall_pdf import Document,init_fonts,PAGES

RUN=ROOT/'research/runs/FROZEN_RISK_02_v1'
OUT=ROOT/'output/pdf/Kronos_Frozen_Risk_02_Report_20261008_v1.pdf'
TMP=ROOT/'tmp/pdfs/frozen_risk_02_v1'
INPUTS={}
FOLDS=['WF01','WF02','WF03','WF04'];PERIODS=['2025Q2','2025Q3','2025Q4','2026Q1']
FAMILIES=['persistence','ewma','har','R1','R2','random_s17','random_s29','random_s43','selected_R0','selected_non_kronos']
LABELS={'persistence':'持续性','ewma':'EWMA','har':'HAR','R1':'R1 普通风险','R2':'R2 预训练',
        'random_s17':'随机17','random_s29':'随机29','random_s43':'随机43','selected_R0':'验证选R0','selected_non_kronos':'验证选非Kronos'}
PALETTE={'persistence':'#999999','ewma':'#DDAA22','har':'#44AA99','R1':'#4477AA','R2':'#CC3311',
         'random_s17':'#AA3377','random_s29':'#66AABB','random_s43':'#117733','selected_R0':'#000000','selected_non_kronos':'#EE7733'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def source(relative):
    path=RUN/relative;INPUTS[str(path.relative_to(ROOT)).replace('\\','/')]=sha(path);return path
def read(relative):return json.loads(source(relative).read_text(encoding='utf-8'))
def frame(relative):return pd.read_csv(source(relative))
def f(value,digits=4):return '—' if pd.isna(value) else f'{float(value):.{digits}f}'
def signed(value,digits=2):return '—' if pd.isna(value) else f'{float(value):+.{digits}f}'

def companion_plots(predictions,metrics,daily,surprise,concentration):
    out=RUN/'figures/report';out.mkdir(parents=True,exist_ok=True)
    test=predictions[predictions.role=='test'].copy()
    test['at']=pd.to_datetime(test.decision_boundary_at,utc=True)
    test['week']=test['at'].dt.floor('D')-pd.to_timedelta(test['at'].dt.weekday,unit='D')
    weekly=test.groupby(['fold_id','family','week'],as_index=False).agg(actual_RV=('RV_effective','mean'),predicted_RV=('prediction','mean'),N=('prediction','size'))
    weekly.to_csv(out/'weekly_RV_all_models.csv',index=False)
    daily=daily.copy();daily['day']=pd.to_datetime(daily.day,utc=True)
    daily=daily.sort_values(['fold_id','comparator','day'])
    daily['cumulative']=daily.groupby(['fold_id','comparator']).QLIKE_gain_sum.cumsum()
    daily.to_csv(out/'cumulative_gain.csv',index=False)
    for fid,period in zip(FOLDS,PERIODS):
        fig,axes=plt.subplots(2,1,figsize=(12,6.3))
        for family in FAMILIES:
            part=weekly[(weekly.fold_id==fid)&(weekly.family==family)]
            axes[0].plot(part.week,part.predicted_RV,label=LABELS[family],color=PALETTE[family],
                         linestyle='--' if family.startswith('selected') else '-',linewidth=1.2)
        part=weekly[(weekly.fold_id==fid)&(weekly.family=='R2')]
        axes[0].plot(part.week,part.actual_RV,label='实际 RV',color='#172B40',linewidth=2.5)
        axes[0].set_yscale('log');axes[0].set_ylabel('4h平方对数收益（对数轴）')
        axes[0].set_title(period+' 周均实际RV与预测；所有模型保留')
        axes[0].legend(ncol=6,fontsize=8,loc='upper center',bbox_to_anchor=(.5,1.04),framealpha=.9)
        for comp in ['R1','selected_non_kronos','selected_R0','random_average_loss']:
            part=daily[(daily.fold_id==fid)&(daily.comparator==comp)]
            axes[1].plot(part.day,part.cumulative,label=comp)
        axes[1].axhline(0,color='gray',linewidth=.7)
        axes[1].set_title('日累计配对QLIKE增益；正值支持R2，季度起点重置')
        axes[1].set_ylabel('累计损失差：控制 − R2');axes[1].legend(ncol=4,fontsize=8)
        for ax in axes:
            ax.grid(alpha=.2);ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d',tz=timezone.utc))
            ax.set_xlabel('真实UTC日期；不是收益或权益曲线')
        fig.subplots_adjust(left=.08,right=.985,top=.94,bottom=.09,hspace=.55)
        for ext in ('png','svg'):fig.savefig(out/f'time_{fid}.{ext}',dpi=180)
        plt.close(fig)
    mid=test.groupby('fold_id')['at'].agg(['min','max']);dates=[mid.loc[fid,'min']+(mid.loc[fid,'max']-mid.loc[fid,'min'])/2 for fid in FOLDS]
    fig,axes=plt.subplots(1,3,figsize=(12,4.2))
    for ax,metric in zip(axes,['QLIKE','Regret','logMSE']):
        for family in FAMILIES:
            rows=metrics[(metrics.role=='test')&(metrics.family==family)].set_index('fold_id').loc[FOLDS]
            ax.plot(dates,rows[metric],marker='o',markersize=3,label=LABELS[family],color=PALETTE[family],
                    linestyle='--' if family.startswith('selected') else '-')
        ax.set_title(metric+'：越低越好');ax.grid(alpha=.2)
        ax.set_xticks(dates);ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m',tz=timezone.utc));ax.tick_params(axis='x',rotation=20)
    axes[2].legend(fontsize=7,ncol=2,loc='upper center',bbox_to_anchor=(-.55,-.22))
    fig.subplots_adjust(left=.06,right=.99,top=.89,bottom=.28,wspace=.34)
    fig.savefig(out/'quarter_losses.png',dpi=180);fig.savefig(out/'quarter_losses.svg');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4.4))
    for ax,col in zip(axes,['q90trainAUROC','q90trainAP','logMSE']):
        for family in ['ewma','har','R1','R2','random_s17','random_s29','random_s43']:
            part=metrics[(metrics.role=='test')&(metrics.family==family)].set_index('fold_id').loc[FOLDS]
            ax.plot(dates,part[col],marker='o',label=LABELS[family],color=PALETTE[family])
        ax.set_title(col);ax.grid(alpha=.2);ax.tick_params(axis='x',rotation=20)
        ax.set_xticks(dates);ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m',tz=timezone.utc))
    axes[2].legend(fontsize=7,ncol=4,loc='upper center',bbox_to_anchor=(-.7,-.24))
    fig.subplots_adjust(left=.06,right=.99,top=.89,bottom=.28,wspace=.34)
    fig.savefig(out/'risk_ranking.png',dpi=180);fig.savefig(out/'risk_ranking.svg');plt.close(fig)
    rows=surprise.copy();rows['quarter_midpoint_utc']=rows.fold_id.map(dict(zip(FOLDS,dates)))
    rows.to_csv(out/'surprise_time.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(12,4.4))
    for ax,col in zip(axes,['surprise_MSE','surprise_AUROC','partial_correlation']):
        for family in ['ewma','har','R1','R2','random_s17','random_s29','random_s43']:
            part=rows[rows.family==family].set_index('fold_id').loc[FOLDS]
            ax.plot(dates,part[col],marker='o',label=LABELS[family],color=PALETTE[family])
        ax.set_title(col);ax.grid(alpha=.2);ax.tick_params(axis='x',rotation=20)
        ax.set_xticks(dates);ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m',tz=timezone.utc))
    axes[2].legend(fontsize=7,ncol=4,loc='upper center',bbox_to_anchor=(-.7,-.24))
    fig.subplots_adjust(left=.06,right=.99,top=.89,bottom=.28,wspace=.34)
    fig.savefig(out/'surprise_time.png',dpi=180);fig.savefig(out/'surprise_time.svg');plt.close(fig)
    return out

def build():
    OUT.parent.mkdir(parents=True,exist_ok=True);TMP.mkdir(parents=True,exist_ok=True)
    if OUT.exists():raise RuntimeError('Report already exists; preserve published version')
    init_fonts()
    cfgpath=ROOT/'research/configs/frozen_risk_02_v1.yaml';cfg=yaml.safe_load(cfgpath.read_text(encoding='utf-8'))
    INPUTS[str(cfgpath.relative_to(ROOT)).replace('\\','/')]=sha(cfgpath)
    acceptance=read('final_acceptance.json');summary=read('summary.json');reg=read('protocol_registration.json')
    if acceptance['engineering_status']!='PASS':raise RuntimeError('Engineering/independent audit blocks publication')
    if reg['protocol_sha256']!=sha(cfgpath):raise RuntimeError('Changed locked protocol')
    metrics=frame('metrics.csv');pairs=frame('paired_comparisons.csv');pred=frame('evaluated_predictions.csv')
    daily=frame('paired_daily_gains.csv');surprise=frame('surprise_diagnostics.csv');extreme=frame('extreme_gain_concentration.csv')
    calibration=frame('calibration.csv');quartiles=frame('surprise_quartiles.csv')
    selected=read('selected_models.json');selectors=read('selectors.json');clipping=read('prediction_clipping.json')
    dataaudit=read('data_5m/audit.json');rawaudit=read('data_5m/raw_csv_audit.json');dm=read('data_5m/manifest.json')
    cache=read('cache_audit.json');prep=read('preparation_audit.json');fit=read('fit_audit.json')
    tests=read('tests/actual_causal_numerical.json');pretests=read('tests/prefit_acceptance.json');independent=read('independent_review.json')
    source('provenance/formal_source_manifest.json');source('fit_events.jsonl');source('all_candidate_predictions.csv');source('labels_rv.csv')
    plotdir=companion_plots(pred,metrics,daily,surprise,extreme)
    doc=Document(OUT);doc.c.setTitle('Kronos 冻结实验2：条件波动率与风险信息增量研究')
    doc.c.setSubject('Frozen Risk Experiment 02 | future4hRV | 96 QLIKE fits | sealed holdout')
    forplot=lambda name:source('figures/report/'+name+'.png')
    core=pairs[(pairs.block_days==7)&(pairs.fold_id!='pooled')]
    pools=pairs[(pairs.block_days==7)&(pairs.fold_id=='pooled')].set_index('comparator')
    r1=core[core.comparator=='R1'].set_index('fold_id').loc[FOLDS]
    random=core[core.comparator=='random_average_loss']
    doc.start('冻结实验 2｜条件风险增量','FROZEN_RISK_02_v1 · 开发期正式预算完成 · 2026-10-08')
    doc.para('研究完成，主增量证据不足。预训练表征优于随机骨干，但尚未建立相对普通风险特征的稳定增量。',15,bold=True,after=18)
    doc.table([['验收维度','最终结果'],['工程验收',acceptance['engineering_status']],['主预测增量',summary['main_status']],
               ['预训练骨干增量证据',acceptance['pretraining_evidence_chinese']],['是否继续当前冻结表征研究',acceptance['research_recommendation_chinese']]],widths=[doc.width*.51,doc.width*.49],size=11)
    doc.para(f'R2 相对 R1 的四季度 Regret 改善中位数 {np.median(r1.regret_improvement_percent):+.2f}%，方向为 3/4 正向；合并原始 QLIKE 增益 {pools.loc["R1","QLIKE_gain"]:.5f}，95% CI [{pools.loc["R1","CI_lower"]:.5f}, {pools.loc["R1","CI_upper"]:.5f}]。区间跨零，因此主筛选未通过。',11)
    doc.para('完成 236,448 根官方 5min 数据、2460 个开发机会、96 个候选和四个季度 1091 个测试机会。最终 holdout 没有新增标签、表征、预测或评估。',11)
    doc.para('本轮检验未来风险信息，不检验交易方向 Alpha 或正期望收益。第一轮收益结论与工件保持原样。',10)
    doc.start('研究问题与边界','先判断增量，再讨论是否值得保留表征')
    doc.heading('本轮要回答什么')
    doc.para('在相同 BTC-USDT-SWAP、相同 1h 已完成 OHLCVA 信息、相同决策时刻和四折切分下，冻结预训练 Kronos context 是否比普通风险特征、持续性/EWMA/HAR 和随机冻结骨干更准确地预测入场起未来 4h 实现方差？')
    doc.heading('如何解释现在的结果')
    doc.para('相对传统风险控制和随机骨干出现正向证据，说明本轮不能宣称预训练表征完全没有风险信息。但强普通特征模型 R1 在所有验证季度都胜过传统控制；对 R1 的主比较缺乏统计确定性，并在 2026Q1 反向。')
    doc.para('有限预算已经完成，停止在当前开发季度追加头、因素、参数或更改标签。若要保留风险路线，只能设计另行授权的独立、事前风险验证；本报告不打开现有 holdout。')
    doc.heading('证据限度')
    doc.para('这些开发季度已经在第一轮研究中使用过，属于探索性复用。checkpoint 的训练截止没有得到证明，不能称为严格事前样本外。官方数据覆盖不证明交易所内部账本或历史首次公开时刻。')
    doc.start('预登记与有限研究预算','协议在目标计算前锁定；没有使用测试反馈扩展候选')
    doc.para('协议 SHA256：'+reg['protocol_sha256'],8)
    doc.table([['项目','固定口径'],['正式拟合','4 折 × 6 可训练家族 × 4 λ = 96'],['λ候选','0.001 / 0.01 / 0.1 / 1'],
               ['选择','每折验证集 raw QLIKE 最小；精确并列选更大 λ'],['比较','R2 vs R1、验证选最强非Kronos、验证选R0；另与三随机种子的逐机会损失平均比较'],
               ['筛选','每核心比较 ≥3/4 折正向；Regret 相对改善中位≥1%；合并7天块95%下限>0'],
               ['停止','可信5min缺失或因果失败阻止晋级；完成固定预算和审计后结束'],['禁止','新头/PCA搜索、微调骨干/tokenizer、额外市场状态、收益优化、实盘、push']],widths=[110,doc.width-110])
    doc.para('ALL 三个核心比较必须同时通过；随机对照的正向证据不能代替强基线。FAIL 仅按预登记的全折非正向或合并区间上限≤0；其余未满足筛选的结果为 INCONCLUSIVE。')
    doc.start('官方 5min 数据与覆盖审计','完整范围 [2024-01-01, 2026-04-01) UTC；不采集 holdout')
    doc.table([['审计项','结果'],['应有 / 实有 K线',f'{dataaudit["expected"]:,} / {dataaudit["rows"]:,}'],['官方响应页',rawaudit['raw_pages']],
               ['缺口 / 冲突重复',f'{len(dataaudit["missing"])} / {dataaudit["conflicting_duplicates"]}'],['原始与CSV逐行重构',rawaudit['status']],
               ['确认/网格/OHLC','confirm=1；唯一连续5min UTC；正价与合法OHLC'],['holdout 行数',dataaudit['holdout_rows']],
               ['成交量/成交额','volCcy（BTC） / volCcyQuote（USDT）']],widths=[180,doc.width-180])
    doc.para('来源：OKX /api/v5/market/history-candles，instId=BTC-USDT-SWAP，bar=5m，limit=300；after 游标和 before 起点同时限制请求。每页保存官方原字节、URL、UTC采集时间和 SHA256；不插值、不用1h替代5min标签。')
    doc.para('CSV SHA256：'+dm['candles_sha256'],8)
    doc.para('可用性首尾抽样只证明抽样可取得；正式结论以全开发期原响应覆盖和独立 raw→CSV 审计为据。5min 价格只出现在目标侧表，预测特征仍只使用历史1h数据。')
    doc.start('目标：未来 4h 实现方差','成交价格代理；48 段自然对数收益的平方和')
    doc.para('RV = Σ[j=1…48] log(Pj / Pj−1)²。P0 是入场时刻5min K线的 open；P1…P48 是入场起48根已完成5min K线的 close，K线开盘时刻在 [entry, entry+4h) 内。',12,bold=True)
    doc.table([['时点','例如 UTC 04 时决策'],['最后历史1h K线','03:00—04:00 完成'],['决策可用时间','04:01（固定60秒保守延迟）'],['下一可执行入场/目标起点','05:00；P0=该5min K线open'],['目标终点','09:00；P48=08:55 K线close'],['目标实际可用时间','09:01；label_end+60s']],widths=[205,doc.width-205])
    doc.para('OHLC open/close 是区间首末成交价代理，不保证存在恰好在 UTC 边界的成交 tick。没有引入盘口、标记价或成交价格插值。每标签保存49个价格、48个bar时间及原响应页/行/hash。')
    doc.para('RV_raw 保留原值；训练和损失使用 max(RV_raw,1e−12)，所有模型同规则。本次标签低于epsilon数为 '+str(int(metrics[(metrics.role=='test')&(metrics.family=='R2')].target_floor_count.sum()))+'；不裁剪或删除有效极端标签。预测统一限制 [1e−12,1] 并记录实际裁剪。')
    doc.start('四折切分、因果与封存','所有区间左闭右开；目标终点与可用时间同时 purge')
    doc.table([['折','训练终点','验证季度','测试季度','训练/验证/测试N']]+[
      [fold['id'],fold['train'][1][:10],fold['validation'][0][:7],period,
       '/'.join(str(int(metrics[(metrics.fold_id==fold['id'])&(metrics.family=='R2')&(metrics.role==role)].N.iloc[0])) for role in ('train','validation','test'))]
       for fold,period in zip(cfg['folds'],PERIODS)],size=8.5)
    doc.para('训练起点均为 2024-01-01。每个角色只包括 label_end < role_end 且 labelable_at < role_end 的机会；边界相等也剔除。季度最后一天20点机会跨界而剔除，最后一天仍有04/12点两次机会。')
    doc.para('所有模型在相同机会配对；测试日历自动检验中间缺失，Bootstrap 保留完整 UTC 日历，以每天损失和与机会计数重新计算均值，避免把末日两个机会赋予三机会日相同权重。')
    doc.para('最终 holdout [2026-04-01,2026-10-01) 未构造5min标签、未编码、预测或评估。旧1h快照包含更晚原始字节，仅计算文件hash；本轮逐记录先过滤时间，再解析 DEV 的数值。不能扩大为“历史上从未见过任何holdout原始数据”。')
    doc.start('模型矩阵与冻结表征','同一可训练头、同一目标、同一选择预算')
    doc.table([['家族','输入 / 架构','维度'],['持续性','最近4个1h close-to-close logreturn平方和','固定'],['EWMA','最近255个小时平方收益；衰减.97、权重归一化、乘4','固定'],
               ['HAR','过去4/24/168h均方风险换算4h后取log','3'],['R1','原ordinary19 + 风险14','33'],['R2','R1 + 固定预训练512 context','545'],
               ['随机17/29/43','R1 + 各随机冻结骨干512 context','各545']],widths=[100,doc.width-150,50],size=9)
    doc.para('Kronos-small 8层、d_model=512；tokenizer 固定不训练；最后有效 decode_s1 context 为表征。仅新增线性风险头参数拟合。随机对照保留相同预训练 tokenizer，因而只能分辨骨干预训练的作用。')
    doc.para('每机会历史256根真实1h OHLCVA，各字段仅在本窗口用 float64 均值与std(ddof0)标准化，epsilon1e−5、clip±5，再float32冻结编码。时间嵌入、batch8、TF32/autocast关闭与第一轮一致。')
    doc.start('有限风险特征与缓存身份','R1 的14个增补风险列预先固定；不引入5min特征')
    doc.table([['特征组','定义','数量'],['历史均方风险','log(max(4×过去h小时r²均值,eps))，h=4/24/168','3'],['历史绝对收益','log(max(过去h小时|r|均值,eps))，h=4/24/168','3'],
               ['Parkinson区间风险','log(max(4×mean(log(H/L)²/(4log2)),eps))','3'],['风险变化','4/24及24/168的log风险差','2'],['尾部比例','过去24h最大|r| / sqrt(max(mean(r²),eps))','1'],
               ['交易活动变化','log((过去4h均值+eps)/(过去24h均值+eps))；volume/amount','2']],widths=[100,doc.width-135,35],size=8.6)
    doc.para('另保留原ordinary19（6个收益、3个std、区间幅度、量/额比、动量符号及6方向交互）以固定既有基线。R0两预测列仅作为基线和surprise参考，不偷偷成为训练头的额外因素。')
    doc.para('四缓存逐chunk hash、contract、机会ID和当前窗口原值/时间hash验证，共9840个窗口；四variant各首/中/末独立再编码，全通过。每variant还重新执行单条/批次token一致性、重复和未来扰动，以及缺口/重复/amount/confirm/availability/ID/偏移拒绝。')
    doc.para('模型 revision：'+cfg['encoding']['model_revision']+'\nTokenizer revision：'+cfg['encoding']['tokenizer_revision'],8)
    doc.start('统一损失、求解与收敛','所有 HAR / R1 / R2 / R3 采用同一 float64 log-link QLIKE')
    doc.para('训练 y_scaled = max(RV,eps) / median_train_RV；z = X_train_standardized·β + b。目标 mean(z + y_scaled·exp(−z)) + λΣβ²。截距不惩罚；预测恢复原方差单位。',12,bold=True)
    doc.para('仅训练折拟合 StandardScaler 和目标中位数；scipy L-BFGS-B 使用解析梯度，beta初值0、截距log(mean(y_scaled))。maxiter3000、maxls50、ftol1e−12、gtol1e−7；要求solver成功及最终最大绝对梯度≤1e−5。训练目标不clip；溢出试步拒绝。')
    event=[json.loads(line) for line in source('fit_events.jsonl').read_text().splitlines()];done=[e for e in event if e['status']=='completed']
    doc.table([['记录','结果'],['正式拟合',f'{len(done)} / 96 完成；正式失败 '+str(len(fit['failed']))],['最大最终梯度',f(max(e['max_abs_gradient'] for e in done),8)],
               ['selected heads','6家族 × 4折 = 24'],['actual prediction clips',sum(c['clip_count'] for c in clipping)],
               ['模型容量','HAR 4 / R1 34 / R2-R3 546 个coef+intercept'],['拟合样本数',f'{min(i["train_count"] for i in selected)}—{max(i["train_count"] for i in selected)}']],widths=[160,doc.width-160])
    doc.para('高维模型可拟合性不由“只有一层”判断。本轮不假定线性头必须能拟合所有模式；同头对照、强正则、验证选择和训练/测试诊断用来分辨表征增量与容量。高维/小样本是结论边界，不用增容搜索挽救结果。')
    doc.start('验证选择与全候选保留','选择时没有读取开发测试损失')
    selection=pd.DataFrame(selected).pivot(index='family',columns='fold_id',values='lambda')
    doc.table([['家族']+FOLDS]+[[family]+[selection.loc[family,fid] for fid in FOLDS] for family in cfg['budget']['trainable_families']],size=10)
    doc.para('传统 R0 的验证选择在四折均为 HAR；最强非Kronos验证选择在四折均为 R1，因此主 R1 与 selected_non_kronos 的原始损失差完全相同。它们分别使用预登记bootstrap随机流，区间端点会有微小Monte Carlo差异，不能作为两份独立证据。')
    doc.para('全部96个模型JSON、192条running/completed事件、全部λ的训练/验证/测试预测保留。24个选择结果保留验证分数和模型路径，所有未选候选也保留。工程修复和验收重放记录另列，不计为新增搜索候选。')
    doc.start('训练、验证与测试损失','选中24个头的 Regret；全部角色的原始损失仍在 metrics.csv')
    rows=[['折','家族','训练Regret','验证Regret','测试Regret']]
    for fid in FOLDS:
        for fam in cfg['budget']['trainable_families']:
            part=metrics[(metrics.fold_id==fid)&(metrics.family==fam)].set_index('role')
            rows.append([fid,LABELS[fam]]+[f(part.loc[role,'Regret']) for role in ['train','validation','test']])
    doc.table(rows,size=8.2,widths=[50,110,117,117,doc.width-394])
    doc.para('仅验证损失选择λ；训练—测试差同时受拟合容量和时期风险分布变化影响，不宜解释为单一因果机制。545维不自动等于有效新增信息；收敛PASS也不等于跨季度预测增量PASS。',8.5)
    doc.start('主损失如何读','raw QLIKE 允许负数；百分比只用于正的 QLIKE Regret')
    doc.para('raw QLIKE = log(pred_RV) + RV_effective / pred_RV。它随方差单位而变化且可为负，不能计算“raw QLIKE改善百分比”。',12,bold=True)
    doc.para('QLIKE Regret = RV_effective/pred_RV − log(RV_effective/pred_RV) − 1 ≥ 0。对同一有效目标，其与raw QLIKE相差 log(RV_effective)+1，因此配对排序完全一致。')
    doc.para('每折相对改善 = 100×(控制平均Regret − R2平均Regret)/控制平均Regret；正值支持R2。原始QLIKE配对增益 = 控制QLIKE − R2QLIKE；置信区间与晋级使用这个损失差。')
    doc.image(forplot('quarter_losses'),height=280)
    doc.para('所有10个展示家族含两个验证选择别名；别名不是新模型。折间连线仅比较独立季度均值，横轴为真实测试季度中点，不代表连续复利或交易收益。',8.5)
    for fid,period in zip(FOLDS,PERIODS):
        doc.start(period+'｜全部模型测试损失','同机会、同标签、同成本无关风险目标；越低越好')
        rows=metrics[(metrics.fold_id==fid)&(metrics.role=='test')].set_index('family')
        doc.table([['模型','N','raw QLIKE','Regret','logRV MSE','高RV AUC','高RV AP']]+[
                  [LABELS[fam],int(rows.loc[fam,'N']),f(rows.loc[fam,'QLIKE']),f(rows.loc[fam,'Regret']),f(rows.loc[fam,'logMSE']),f(rows.loc[fam,'q90trainAUROC'],3),f(rows.loc[fam,'q90trainAP'],3)] for fam in FAMILIES],size=8,widths=[110,32,70,65,70,66,doc.width-413])
        gain=r1.loc[fid]
        doc.para(f'R2 vs R1：原始QLIKE增益 {gain.QLIKE_gain:+.5f}；Regret改善 {gain.regret_improvement_percent:+.2f}%；7天配对95%区间 [{gain.CI_lower:.5f}, {gain.CI_upper:.5f}]。')
        doc.para('高RV阈值来自该折训练RV的90%分位；评分用于风险排序，不是经校准的事件概率。AP受该季度事件比例影响，不可脱离事件计数解释。')
        doc.para('表中每项原值见 metrics.csv；不根据测试表现只保留最优模型。')
        doc.start(period+'｜真实时间诊断','周均风险及每日配对损失；本季度独立起点',wide=True)
        doc.image(forplot('time_'+fid),height=385)
        doc.para('上图：实际4h RV与全部模型预测，单位为平方自然对数收益。下图：控制−R2的日累计QLIKE损失差；零线只表相对预测优势，不是现金基线或盈亏。底表：figures/report/weekly_RV_all_models.csv 与 cumulative_gain.csv。',8.2)
    doc.start('配对统计与晋级结果','完整UTC决策日 stationary bootstrap；7天主口径，5000次')
    doc.table([['比较','正向折','Regret中位改善','均值QLIKE增益','合并95% CI','筛选']]+[
      [comp,str(int((core[core.comparator==comp].QLIKE_gain>0).sum()))+'/4',signed(core[core.comparator==comp].regret_improvement_percent.median())+'%',
       f(pools.loc[comp,'QLIKE_gain'],5),'['+f(pools.loc[comp,'CI_lower'],5)+', '+f(pools.loc[comp,'CI_upper'],5)+']',summary['comparators'][comp]]
       for comp in ['R1','selected_non_kronos','selected_R0','random_average_loss']],size=8.2,widths=[100,42,75,75,145,doc.width-437])
    doc.para('各折内按日成块成对抽样，循环续段概率1−1/块长；各折独立，合并统计对四季度均值等权。随机控制是三个种子的每机会损失平均，不是最佳种子，也不是平均预测的集成模型。')
    doc.para('R2 vs R1满足3/4正向和中位≥1%，但合并区间跨零；最强非Kronos同样未过。R2 vs selected_R0 和随机平均符合各自筛选，仍不能替代主强基线。主要结论：INCONCLUSIVE，而非主研究成功或全无增量。')
    doc.start('块长度敏感性与季度差异','预登记3/7/14天；不会用更有利口径替换主7天')
    rows=pairs[pairs.fold_id=='pooled']
    doc.table([['比较','块天数','均值增益','95%下限','95%上限']]+[
       [r.comparator,int(r.block_days),f(r.QLIKE_gain,5),f(r.CI_lower,5),f(r.CI_upper,5)] for _,r in rows.iterrows()],size=8.6)
    doc.para('R1比较在三种块长下的合并区间都跨零。2026Q1对R1和HAR均反向；前面几个季度的正向优势不能保证跨市场状态稳定。开发期只含四测试季度，块统计反映这些时期内的依赖，不能覆盖所有未来制度变化。')
    doc.start('校准与高风险排序','训练预测分位锁定分组；不在测试集重新拟合校准')
    doc.image(forplot('risk_ranking'),height=270)
    doc.para('R2四季度的高RV AUROC均比R1高，但排序变好与QLIKE均值变好是不同问题。2026Q1 logRV MSE较R1低，而QLIKE较高，体现QLIKE对低估高风险更敏感；不能用辅助指标替代主要损失。')
    thresholds=metrics[(metrics.family=='R2')&(metrics.role=='test')].set_index('fold_id')
    doc.table([['折','训练q90 RV','训练q99 RV','测试高RV数']]+[
       [fid,f'{thresholds.loc[fid,"train_q90"]:.7g}',f'{thresholds.loc[fid,"train_q99"]:.7g}',
        int(((pred.fold_id==fid)&(pred.role=='test')&(pred.family=='R2')&(pred.RV_effective>thresholds.loc[fid,'train_q90'])).sum())] for fid in FOLDS],size=8.8)
    doc.para('calibration.csv保存所有模型/角色10组的N、预测均值、实际均值及训练分界；未做测试拟合，也不将RV回归头解释为概率模型。')
    doc.start('R2 测试风险分组校准','分界只由对应训练预测形成；组间样本不保证均衡',wide=True)
    rows=[['训练预测十分位']+[fid+'：N / 预测 / 实际' for fid in FOLDS]]
    for decile in range(1,11):
        row=[decile]
        for fid in FOLDS:
            r=calibration[(calibration.fold_id==fid)&(calibration.role=='test')&(calibration.family=='R2')&(calibration.decile==decile)].iloc[0]
            row.append(f'{int(r.N)} / {r.predicted_RV:.3g} / {r.actual_RV:.3g}')
        rows.append(row)
    doc.table(rows,size=8.5)
    doc.para('数值单位均为4h平方自然对数收益；没有年化。空组均值未定义，完整边界及其他模型见 calibration.csv。',8.5)
    doc.start('极端事件与增益集中','本表为主比较 R2 vs R1；极端阈值为训练RV q99，观测全部保留')
    rows=extreme[extreme.comparator=='R1'].set_index('fold_id').loc[FOLDS]
    doc.table([['折','极端N','总增益和','极端增益和','非极端平均增益']]+[
       [fid,int(rows.loc[fid,'extreme_N']),f(rows.loc[fid,'total_QLIKE_gain_sum'],4),f(rows.loc[fid,'extreme_QLIKE_gain_sum'],4),f(rows.loc[fid,'non_extreme_mean_gain'],5)] for fid in FOLDS],size=9)
    doc.para('本表控制为 R1；总增益和=逐机会R1 QLIKE−R2 QLIKE的总和。当总和接近0或为负，极端增益占比可大于1或为负，不能当作概率。完整四个比较的集中度见 extreme_gain_concentration.csv。')
    doc.para('这张表区分“是否只靠少数高波动日”与“所有普通机会也有优势”。无论诊断有利与否，都没有删除极端观测、重新调λ或换主要判据。')
    doc.start('意外波动：超出已有EWMA的部分','辅助任务：使用已有预测，不再训练任何头')
    doc.para('真实 S = log((RV_raw+eps)/(EWMA_pred+eps))；预测 S_hat = log((model_pred+eps)/(EWMA_pred+eps))。统一使用同一历史1h EWMA参考，真实surprise事件阈值固定 S>log2。')
    doc.image(forplot('surprise_time'),height=270)
    doc.para('预测和真实S的相关、扣除logEWMA后的部分相关、超过/低于EWMA的方向准确率及事件排序分别报告；这些是统计关联诊断，不能证明因果机制。部分相关的OOS诊断回归不用于重新训练预测头。')
    doc.para('EWMA自身预测S恒0，所以相关未定义；其AUROC为无信息0.5，基准AP等于事件比例。surprise诊断不改变主增量筛选。')
    for fid,period in zip(FOLDS,PERIODS):
        doc.start(period+'｜意外波动全部对照','辅助结果保留全部模型；不替代主要 QLIKE 判定')
        rows=surprise[surprise.fold_id==fid].set_index('family')
        doc.table([['模型','S MSE','方向准确率','S>log2 AUC','AP','Spearman','部分相关']]+[
           [LABELS[fam],f(rows.loc[fam,'surprise_MSE']),f(rows.loc[fam,'direction_accuracy'],3),f(rows.loc[fam,'surprise_AUROC'],3),f(rows.loc[fam,'surprise_AP'],3),f(rows.loc[fam,'Spearman'],3),f(rows.loc[fam,'partial_correlation'],3)] for fam in FAMILIES],size=8,widths=[110,65,65,70,48,65,doc.width-423])
        qs=quartiles[(quartiles.fold_id==fid)&(quartiles.family=='R2')]
        doc.para('R2按训练预测S四分位形成测试分组：'+ '；'.join(f'第{int(r.quartile)}组 N={int(r.N)}，平均真实S={f(r.mean_true_S,3)}' for _,r in qs.iterrows())+'。')
        doc.para('score不解释为校准概率。分组、事件阈值、训练量化边界在测试前固定；小样本季度的AP可能强烈受事件数量影响。')
    doc.start('工程验收、原有测试与独立复核','严格区分可运行、可复现、无泄漏和研究增量')
    doc.table([['验收项','证据'],['全DEV raw→CSV','236448根×789页逐行一致'],['全部目标独立复算','2460标签；显式48对价格比平方和'],['缓存和原编码因果检查','四variant×2460窗口身份；batch/single、token、future与负向检查'],
               ['原可执行suite','9套：8直接PASS；E00R历史输入重放PASS'],['M0历史阶段验收','固定历史YAML和hash重放12项PASS，未修改当前配置'],['新数值/标签保护','手算QLIKE/Regret/梯度、48段、shift/gap/unconfirmed/holdout拒绝'],
               ['正式模型和确定性','24模型reload/选择/训练scaler；相同输入R2 refit逐位相等'],['独立审查','全部损失、选择与5000次配对bootstrap独立重算PASS']],widths=[180,doc.width-180],size=9)
    doc.para('工程PASS依赖实际测试、原始来源、独立复核和hash，不由训练进程正常退出或summary传入字符串决定。任何真正的因果失败均应阻止研究晋级；本轮没有放宽或删除断言。')
    doc.start('工程异常与修复记录','真实保留失败，不把验收比较误差伪装成模型改善')
    doc.para('1. 标签API序列化首次失败：Timestamp不能直接JSON序列化。统一为UTC ISO字符串，字符串与Timestamp输入等价；价格、窗口及协议未改变。')
    doc.para('2. E00R直接运行在历史阶段断言失败：现行配置已进入冻结研究。保留失败日志，并用注册表原始hash配置重放完整检查；未改旧检查或当前配置。M0同样按固定历史配置重放。')
    doc.para('3. 事后scaler验收先以pandas float32 mean对照注册的float64 mean，最大差6.10e−8。改为先float64再平均，差为0；原tol保持。')
    doc.para('4. 精确确定性验收第一次默认CSV解析与原double目标最大差约9.99e−17，不是同一输入。用round_trip解析恢复原double后系数/截距逐位相等，精确断言保持。未来扰动比较也以两个同精度原始特征路径核对。')
    doc.para('以上修复属于工程验收接口和精度修正，没有改动正式模型、目标、λ网格、统计或晋级准则；没有新增正式候选。复现检查重放不属于超参数搜索。formal_source_manifest保存首次源码，final_acceptance另记当前审计源码版本。')
    doc.start('来源、版本与工件索引','可复核的全量工件；无 Git push')
    doc.para('正式运行根目录：research/runs/FROZEN_RISK_02_v1/\n配置：research/configs/frozen_risk_02_v1.yaml\n代码与独立TODO：research/frozen/experiment_02/\n注册表：research/registry/FROZEN_RISK_02_v1.json',9)
    doc.table([['工件','内容'],['data_5m/raw + manifest/audit','官方原字节、来源、全期覆盖、逐行复核'],['labels_rv.csv','DEV未来标签、49价格、48bar与raw路径行hash'],['features.csv / cache_audit','33特征+R0列与四缓存身份证明'],['models/ + fit_events.jsonl','96全部候选、收敛和运行事件'],['all_candidate_predictions.csv','全部λ的train/validation/test预测'],['predictions / selected_models / selectors','选择后全部模型预测及验证选择依据'],['metrics / paired_comparisons','全模型指标及3/7/14天fold/pooled CI'],['surprise / calibration / extremes','辅助诊断、训练固定边界及集中度'],['tests / independent_review / acceptance','原与新检查、独立复核、hash验收'],['figures/ + report companion tables','PNG、SVG和真实数值CSV']],size=8.8,widths=[175,doc.width-175])
    doc.para('数据、run、模型和大型缓存按既有ignore规范保存；不提交环境或大型原始数据。没有push、没有修改Freqtrade策略/框架环境。报告manifest保存每个引用文件hash以及PDF页面核验。',8.5)
    doc.start('复现顺序与证据核查','使用项目明确解释器；复现不要覆写已发表run')
    doc.para('研究解释器：D:/file/Kronos-master/.venv/Scripts/python.exe；原Freqtrade测试使用D:/file/freqtrade/.venv/Scripts/python.exe。本轮没有重装研究依赖，PDF依赖位于tmp/pdfs/deps。')
    doc.para('data.py audit：校验既有官方raw和CSV；audit.py的源码展示全部原有/新检查；evaluate.py --run <run> 和 charts.py --run <run> 可从封存预测重新生成统计及图。完整collect/prepare/fit需要另登记新的复现run，不覆写本轮不可变工件。',10)
    doc.para('协议SHA固定校验在数据、训练和评估入口。正式source_manifest保存代码、模型core、配置、依赖锁、版本清单、Git commit及dirty文件hash；cache合约记录模型权重和tokenizer hash。完整源码复制位于 provenance/source/。')
    doc.para('主要外部参考：\nOKX官方 history-candles 文档：https://www.okx.com/docs-v5/en/#order-book-trading-market-data-get-candlesticks-history\nPatton (2011), Volatility forecast comparison using imperfect volatility proxies：https://public.econ.duke.edu/~ap172/Patton_vol_proxies_JoE_2011.pdf',8.5)
    doc.para('QLIKE选择是预登记的比较口径；5min RV仍是风险代理而非不可观测条件方差本身，本报告不声称已证明代理无噪声或微观结构偏差不存在。',9)
    doc.start('最终判断与研究停止边界','工程PASS；主INCONCLUSIVE；预训练支持；尚需独立验证')
    doc.para('本轮改变预测目标后，预训练冻结表征确实出现与传统风险/随机对照相比的可保留信号；但相对更强的普通风险模型R1，合并统计不确定，2026Q1方向反转。不能据此宣布Kronos已经建立稳定独特风险价值。',12,bold=True)
    doc.para('是否已有足够理由停止当前 BTC 1h 冻结研究？已有理由停止当前开发数据上的扩头、扩因素和收益路线优化。本轮风险结果并不支持“完全无信息”的绝对结论；是否保留风险路线尚需独立验证。有限实验已结束，没有下一项自动运行候选。')
    doc.para('建议：封存当前工件，停止当前开发期优化。仅在用户另行授权并预登记独立风险检验后考虑继续；可先固定R1/R2及目前候选选择方案，但不在这里打开holdout或根据本轮测试调参。')
    doc.para('风险预测成功不等于方向Alpha，也不等于参与门控、风险预算或交易策略正期望收益。若以后进入风险预算/波动目标仓位或收益回测，那是新的研究阶段。')
    doc.table([['最终维度','状态'],['工程',acceptance['engineering_status']],['主增量',summary['main_status']],['预训练骨干',acceptance['pretraining_evidence_chinese']],['继续研究',acceptance['research_recommendation_chinese']]],size=11)
    doc.end()
    reader=PdfReader(OUT);text='\n'.join(page.extract_text() or '' for page in reader.pages)
    if '\ufffd' in text or not all(word in text for word in ['INCONCLUSIVE','236,448','尚需独立验证']):raise RuntimeError('PDF text/glyph validation failed')
    render=TMP/'rendered';render.mkdir(exist_ok=True)
    pdf=fitz.open(OUT)
    for i,page in enumerate(pdf):page.get_pixmap(matrix=fitz.Matrix(1.25,1.25),alpha=False).save(render/f'page_{i+1:02d}.png')
    contacts=[]
    for start in range(0,len(pdf),8):
        paths=[render/f'page_{i+1:02d}.png' for i in range(start,min(start+8,len(pdf)))]
        canvas=Image.new('RGB',(1600,1000),'#e8edf2');draw=ImageDraw.Draw(canvas)
        for j,path in enumerate(paths):
            img=Image.open(path);img.thumbnail((380,455))
            x=(j%4)*400+(400-img.width)//2;y=(j//4)*500+24
            canvas.paste(img,(x,y));draw.text(((j%4)*400+12,(j//4)*500+6),f'Page {start+j+1}',fill='black')
        path=TMP/f'contact_{start//8+1:02d}.png';canvas.save(path);contacts.append(str(path))
    pd.DataFrame(PAGES).to_csv(OUT.with_name(OUT.stem+'_pages.csv'),index=False)
    manifest={'status':'generated_pending_visual_QA','pdf_path':str(OUT),'pdf_sha256':sha(OUT),'pages':len(reader.pages),
              'input_sha256':INPUTS,'report_source_sha256':sha(Path(__file__)),
              'font':'embedded Microsoft YaHei regular/bold; ToUnicode','renderer':'PyMuPDF fallback; Poppler unavailable in environment',
              'rendered_directory':str(render),'contact_sheets':contacts,'created_at_utc':datetime.now(timezone.utc).isoformat(),
              'holdout_access':False,'page_outline':PAGES}
    OUT.with_name(OUT.stem+'_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'pdf':str(OUT),'pages':len(reader.pages),'contacts':contacts}))

if __name__=='__main__':build()
