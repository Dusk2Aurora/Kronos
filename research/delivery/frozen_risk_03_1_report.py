"""Publish the B5 post-hoc 3.1 supplement, then append unchanged 3.0 PDF pages.

Reads generated artifacts only; no fitting, market-source reads, new thresholds,
random draws or confirmatory success criteria. Invoke only after numeric PASS.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "research/runs/FROZEN_RISK_03_1_B5_v1"
OLD = ROOT / "research/runs/FROZEN_RISK_03_v1"
CONFIG = ROOT / "research/configs/frozen_risk_03_1_b5_v1.yaml"
CONFIG_SHA = "4636e4f7cc5db9ab3a9e425b518c5f2674322a8d67293bed1ab358e8cd96741c"
FAMILIES = ["persistence","ewma","har","R1","R2","B2","random_s17","random_s29","random_s43","B5","B5_s17","B5_s29","B5_s43"]
QUARTERS = ["2026Q2","2026Q3"]
MAIN = ["R1","R2","B2","B5"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fmt(v, digits=7):
    if v is None or isinstance(v,(float,np.floating)) and not np.isfinite(v): return "未定义"
    if isinstance(v,(float,np.floating)): return f"{v:.{digits}f}"
    return str(v)


def mdtable(frame):
    def cell(v):
        if isinstance(v,(float,np.floating)) and np.isfinite(v): return format(v,".17g")
        return fmt(v).replace("|","\\|").replace("\n"," ")
    return "\n".join(["| " + " | ".join(map(str,frame.columns)) + " |", "| " + " | ".join(["---"]*len(frame.columns)) + " |"] +
                     ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False,name=None)])


def publish():
    sources = {}
    def source(p):
        p = Path(p).resolve()
        if not p.is_relative_to(ROOT): raise ValueError("Publication source outside project")
        sources[str(p.relative_to(ROOT)).replace("\\","/")] = sha(p)
        return p
    def jr(p): return json.loads(source(p).read_text(encoding="utf-8-sig"))
    if sha(source(CONFIG)) != CONFIG_SHA: raise ValueError("B5 protocol changed")
    if jr(OLD/"authorization/formal_terminal.json").get("state") != "CONSUMED": raise PermissionError("Original experiment must remain CONSUMED")
    numeric = jr(RUN/"independent_review/numeric_review_v2.json")
    numeric_attempt01_path = source(RUN/'independent_review/numeric_review.json')
    numeric_attempt01 = json.loads(numeric_attempt01_path.read_text(encoding='utf-8-sig'))
    if numeric.get('recovery_checks_passed') is not True: raise PermissionError('Independent operational recovery checks must PASS')
    recovery = jr(RUN/'preflight/recovery_seal_20261009.json')
    interruption = jr(RUN/'models/interrupted_attempt_resources.json')
    recovery_claim = jr(RUN/'models/recovery_claim_20261009.json')
    amendment_path = source(ROOT/recovery['amendment'])
    amendment_text = amendment_path.read_text(encoding='utf-8')
    if recovery['files_sha256'][recovery['amendment']] != sha(amendment_path): raise ValueError('Operational amendment changed')
    if (recovery['candidate_fits'],recovery['fit_attempts_including_user_interruption'],recovery['interrupted_attempts']) != (12,13,1): raise ValueError('Recovery attempt accounting changed')
    audits = {name:jr(RUN/path) for name,path in {"candidate":"models/candidate_audit.json","inputs":"preflight/inputs_audit.json","resources":"provenance/resources.json"}.items()}
    if numeric.get("status") != "PASS" or any(a.get("status") != "PASS" for a in audits.values()):
        raise PermissionError("B5 numeric, input, candidate and resource audits must PASS")
    summary = jr(RUN/"metrics/summary.json")
    if summary.get("status") != "POST_HOC_B5_SUPPLEMENT" or summary.get("evidence_class") != "POST_HOC_EXPLORATORY_NOT_NEW_INDEPENDENT_CONFIRMATION":
        raise ValueError("B5 is exclusively a post-hoc exploratory supplement")
    original = jr(OLD/"metrics/formal_result.json")
    if summary["original_primary_status"] != original["status"]: raise ValueError("Original primary status changed")
    selected = jr(RUN/"models/selected.json")
    frames = {}
    def csv(name):
        frame = pd.read_csv(source(RUN/("metrics/"+name+".csv")),float_precision="round_trip")
        frames[name] = frame
        return frame
    metrics, rows, tails, calibration = (csv(n) for n in ("quarter_metrics","row_metrics","underprediction","calibration"))
    parent_calibration = pd.read_csv(source(OLD/"diagnostics/holdout/model_calibration.csv"),float_precision="round_trip")
    parent_calibration = parent_calibration[(parent_calibration.family.isin(["R1","R2","B2"])) &
                                           (parent_calibration.period.isin(QUARTERS)) &
                                           (parent_calibration.bin_type=="prediction_decile")].rename(columns={
        "period":"quarter", "mean_observed_RV_raw":"mean_observed_RV", "surprise_event_fraction":"event_fraction"})
    # Publication concatenates existing TRAIN-fixed bins. It never fits cuts
    # and never writes either original scientific calibration CSV.
    calibration_comparison = pd.concat([parent_calibration,calibration],ignore_index=True,sort=False)
    rows["decision_at"] = pd.to_datetime(rows.decision_at,utc=True)
    if set(metrics.family) != set(FAMILIES) or set(rows.family) != set(FAMILIES) or set(rows.quarter) != set(QUARTERS):
        raise ValueError("All thirteen saved families and both quarters required")
    histories = [(p,pd.read_csv(source(p),float_precision="round_trip")) for p in sorted((RUN/"models").glob("history_*.csv"))]
    if len(histories) != 12 or audits["candidate"]["candidate_fits"] != 12: raise ValueError("All twelve candidate histories must be retained")
    oldpdf = source(ROOT/"output/pdf/Kronos_Frozen_Risk_03_Report_v2.pdf")
    oldmd = source(OLD/"reports/Kronos_Frozen_Risk_03_Report_v2.md")
    oldmdbytes = oldmd.read_bytes()
    resource = audits["resources"]
    pdf = ROOT/"output/pdf/Kronos_Frozen_Risk_03_1_Report_v1.pdf"
    md = RUN/"reports/Kronos_Frozen_Risk_03_1_Report_v1.md"
    manifest = RUN/"reports/publication_manifest.json"
    figures = RUN/"figures/publication"
    temp = ROOT/"tmp/pdfs/frozen_risk03_1_v1"
    if any(p.exists() for p in (pdf,md,manifest,figures,temp)): raise FileExistsError("Refuse to overwrite 3.1 publication artifacts")
    figures.mkdir(parents=True);temp.mkdir(parents=True);md.parent.mkdir(parents=True,exist_ok=True);pdf.parent.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(ROOT))
    from research.frozen.experiment_03.report import Document,pdfbase
    pdfbase.init_fonts();plt=pdfbase.plt
    colors={f:plt.get_cmap("tab20")(i) for i,f in enumerate(FAMILIES)}
    graphmeta=[]
    def save(name,fig,frame,formula):
        fig.tight_layout();p=figures/name
        fig.savefig(p.with_suffix(".png"),dpi=160);fig.savefig(p.with_suffix(".svg"));plt.close(fig)
        frame.to_csv(p.with_suffix(".csv"),index=False,float_format="%.17g")
        graphmeta.append({"name":name,"formula":formula,"evidence":"POST_HOC_EXPLORATORY","no_refit":True})
        return p.with_suffix(".png")
    fig,ax=plt.subplots(figsize=(11,6))
    for f in FAMILIES:
        d=metrics[metrics.family==f].sort_values("quarter")
        t=pd.to_datetime(d.quarter.map({"2026Q2":"2026-04-01","2026Q3":"2026-07-01"}),utc=True)
        ax.plot(t,d.AUROC,"o-",label=f,color=colors[f],linewidth=2 if f in MAIN else .8)
    ax.axhline(.5,color="gray",ls="--");ax.set(title="POST-HOC | all thirteen families",ylabel="Surprise AUROC (higher better)",xlabel="Quarter start UTC",ylim=(0,1));ax.legend(ncol=4,fontsize=8);ax.grid(alpha=.2)
    aucplot=save("quarter_AUROC",fig,metrics,"Saved within-quarter AUROC; two real quarter starts; all13 families retained")
    comparisons=[]
    for b in (7,3,14):
        for name,c in summary["comparisons"][str(b)].items(): comparisons.append({"block_days":b,"comparison":name,**{k:v for k,v in c.items() if k!="quarter_deltas"},"Q2_delta":c["quarter_deltas"][0],"Q3_delta":c["quarter_deltas"][1]})
    ci=pd.DataFrame(comparisons)
    fig,ax=plt.subplots(figsize=(11,5))
    for y,b in enumerate((7,3,14)):
        r=ci[(ci.comparison=="R2_minus_B5") & (ci.block_days==b)].iloc[0]
        ax.hlines(y,r.ci_lower,r.ci_upper,color="C0" if b==7 else "gray",lw=3)
        ax.plot(r.mean_delta,y,"o",color="C0" if b==7 else "gray")
    ax.axvline(0,color="black");ax.set(yticks=[0,1,2],yticklabels=["7d registered reference","3d sensitivity","14d sensitivity"],xlabel="Equal-quarter delta AUROC: R2 minus B5; positive favors R2",title="POST-HOC | R2 - B5 saved paired intervals");ax.grid(alpha=.2)
    ciplot=save("R2_minus_B5_CI",fig,ci,"Saved paired 95% intervals using parent day multiplicities; no new draws or confirmatory upgrade")
    timeline=rows[rows.family.isin(MAIN)].copy()
    fig,axes=plt.subplots(2,1,figsize=(11,6))
    for ax,q in zip(axes,QUARTERS):
        for f in MAIN:
            d=timeline[(timeline.quarter==q)&(timeline.family==f)].sort_values("decision_at")
            ax.plot(d.decision_at,d.prediction_RV,label=f,color=colors[f],lw=1)
        d=timeline[(timeline.quarter==q)&(timeline.family=="B5")].sort_values("decision_at")
        ax.plot(d.decision_at,d.RV_raw,color="black",label="observed RV_raw",alpha=.6,lw=.8)
        ax.set(title=q+" | POST-HOC",xlabel="Decision origin UTC",ylabel="4h squared log return (not annualized)");ax.legend(ncol=5,fontsize=8);ax.grid(alpha=.2)
    rvplot=save("RV_timeline",fig,timeline,"Saved actual/predicted RV, real decision time, quarter facets")
    daily=[];fig,axes=plt.subplots(2,1,figsize=(11,6))
    for ax,q in zip(axes,QUARTERS):
        d=rows[rows.quarter==q].pivot(index="decision_at",columns="family",values="raw_QLIKE")
        start,end=("2026-04-01","2026-07-01") if q=="2026Q2" else ("2026-07-01","2026-10-01")
        calendar=pd.date_range(pd.Timestamp(start,tz="UTC"),pd.Timestamp(end,tz="UTC"),freq="D",inclusive="left")
        for f in ("R1","B2","R2"):
            diff=d[f]-d.B5;series=diff.groupby(diff.index.floor("D")).sum().reindex(calendar,fill_value=0)
            count=diff.groupby(diff.index.floor("D")).size().reindex(calendar,fill_value=0)
            table=pd.DataFrame({"quarter":q,"day_utc":calendar,"comparison":f+"_minus_B5","N":count.to_numpy(),"daily_loss_difference_sum":series.to_numpy(),"cumulative_loss_difference_sum":series.cumsum().to_numpy()});daily.append(table)
            ax.plot(calendar,series.cumsum(),label=f+" - B5",color=colors[f])
        ax.axhline(0,color="gray");ax.set(title=q+" | reset at quarter start",xlabel="UTC calendar day",ylabel="Cumulative paired raw QLIKE difference (sum)");ax.legend();ax.grid(alpha=.2)
    dailyplot=save("daily_cumulative_QLIKE",fig,pd.concat(daily),"Sum(loss_baseline-loss_B5), accumulated independently each quarter; positive favors B5; not return/percent")
    fig,axes=plt.subplots(4,3,figsize=(14,10));historytables=[]
    for ax,(p,d) in zip(axes.flat,histories):
        required={"epoch","train_raw_qlike","valid_raw_qlike"}
        if not required.issubset(d): raise ValueError("Candidate history schema mismatch")
        ax.plot(d.epoch,d.train_raw_qlike,label="TRAIN",lw=1);ax.plot(d.epoch,d.valid_raw_qlike,label="VALID",lw=1)
        ax.set(title=p.stem.replace("history_",""),xlabel="Epoch",ylabel="Raw QLIKE",);ax.legend(fontsize=7);ax.grid(alpha=.2)
        historytables.append(d.assign(candidate=p.stem))
    historyplot=save("all12_training_curves",fig,pd.concat(historytables),"All saved12 candidate TRAIN/VALID original-unit loss trajectories; no TEST selection")
    cal=calibration_comparison[calibration_comparison.family.isin(MAIN)].copy();fig,axes=plt.subplots(1,2,figsize=(11,5))
    for ax,q in zip(axes,QUARTERS):
        for f in MAIN:
            d=cal[(cal.quarter==q)&(cal.family==f)&(cal.bin_type=="prediction_decile")].sort_values("bin_id")
            if d.empty: raise ValueError("Missing saved TRAIN-fixed calibration for "+q+"/"+f)
            ax.plot(d.bin_id+1,d.mean_observed_RV/d.mean_prediction_RV,"o-",label=f,color=colors[f])
        ax.axhline(1,color="gray",ls="--");ax.set(title=q+" | TRAIN-fixed",xlabel="Prediction decile (TRAIN cuts)",ylabel="Mean observed RV / mean prediction");ax.legend();ax.grid(alpha=.2)
    calplot=save("TRAIN_fixed_calibration",fig,cal,"Saved bin mean observed RV / mean prediction; TRAIN bins only; event_fraction retained in CSV; not event probability calibration")
    tail=tails[tails.family.isin(MAIN)].copy();fig,axes=plt.subplots(1,2,figsize=(11,5))
    for ax,q in zip(axes,QUARTERS):
        for subset,marker in [("q90","o"),("q99","s")]:
            d=tail[(tail.quarter==q)&(tail.subset==subset)].set_index("family")
            if not set(MAIN).issubset(d.index): raise ValueError("Missing saved tail subset "+q+"/"+subset)
            ax.plot(range(4),d.loc[MAIN,"underprediction_fraction"],marker+"-",label=subset)
        ax.set(title=q,xticks=range(4),xticklabels=MAIN,ylabel="Fraction pred / RV_effective < 0.5",ylim=(-.02,1.03),yticks=np.linspace(0,1,6));ax.legend(fontsize=8);ax.grid(alpha=.2)
    tailplot=save("TRAIN_fixed_tail_underprediction",fig,tail,"Saved underprediction counts/fractions, same parent TRAIN q90/q99; empty conditional fractions undefined")

    supplement=temp/"supplement.pdf";doc=Document(supplement,"B")
    lines=["# Kronos 冻结风险实验 3.1：B5 事后补充", "", "新部分全部为事后探索；原3.0确认性结论不变。"]
    outline=[]
    def page(title,sub="",wide=False):
        doc.start(title,sub,wide);outline.append({"page":doc.number,"title":title,"wide":wide})
        doc.c.setFillColor(pdfbase.colors.white);doc.c.rect(40,doc.h-43,doc.w-80,14,fill=1,stroke=0)
        doc.c.setFillColor(pdfbase.BLUE);doc.c.setFont("CN",8);doc.c.drawString(42,doc.h-36,"KRONOS 3.1 / B5 POST-HOC EXPLORATORY SUPPLEMENT")
        doc.c.setFillColor(pdfbase.colors.white);doc.c.rect(40,15,doc.w-80,16,fill=1,stroke=0)
        doc.c.setFillColor(pdfbase.GRAY);doc.c.drawString(42,22,"3.1事后补充 | 不是新的独立确认 | 真实时间UTC");doc.c.drawRightString(doc.w-42,22,str(doc.number))
        lines.extend(["","## "+str(doc.number)+". "+title,"",sub,""])
    def para(text,size=9.5,bold=False): doc.para(text,size=size,after=7,bold=bold);lines.extend([text,""])
    def table(frame,size=8,digits=7,widths=None):
        frame=frame if isinstance(frame,pd.DataFrame) else pd.DataFrame(frame)
        doc.table([list(frame.columns)]+[[fmt(v,digits) for v in r] for r in frame.itertuples(index=False,name=None)],widths=widths,size=size)
        lines.extend([mdtable(frame),""])
    def image(p,note,height=335):
        doc.image(p,height=height);para(note,8.2)
        lines.extend(["!["+note.replace("]","")+"]("+os.path.relpath(p,md.parent).replace("\\","/")+")",""])
    r2b5=summary["comparisons"]["7"]["R2_minus_B5"]
    numeric_description="R2-B5 Q2/Q3 Δ="+"/".join(fmt(v,7) for v in r2b5["quarter_deltas"])+"；等权Δ="+fmt(r2b5["mean_delta"],7)+"，7日CI=["+fmt(r2b5["ci_lower"],7)+","+fmt(r2b5["ci_upper"],7)+"]。"
    if r2b5["ci_lower"]>0 and all(v>0 for v in r2b5["quarter_deltas"]):
        b5_interpretation="两季度方向均正且CI全正，仅支持在此有限事后TCN对照下R2仍有差异；不是Kronos骨干必要性确认。未来可继续风险排序研究，但需新独立数据与另行登记。"
    elif r2b5["ci_upper"]<0:
        b5_interpretation="区间全负，构成事后描述性B5更强的反向证据，下一项应优先独立核验廉价序列基线；不回写原R1/B2两确认性比较，不新增B5确认性结论。"
    else:
        b5_interpretation="CI未全在一致正向且季度方向可能不稳，尚不能稳定区分两者；不是等效性证明。廉价序列基线值得保留，并优先在新独立数据核验。"
    page("3.1摘要：B5事后补充")
    para("证据类别：POST_HOC_EXPLORATORY_NOT_NEW_INDEPENDENT_CONFIRMATION。Q2/Q3已用于原3.0评估，本次B5比较是在这些结果已知之后设计；不能重新称独立确认。",11,True)
    para("原3.0共同主要结论保持："+original["status"]+"。B5不加入原确认性成功路径，不以本次CI升级、推翻或回写原结果。",11,True)
    table(ci[ci.block_days==7][["comparison","Q2_delta","Q3_delta","mean_delta","ci_lower","ci_upper"]],size=8)
    para(numeric_description+b5_interpretation,9.6,True)
    para("正方向按比较名称定义：R2-B5正值有利于R2；B5-R1/B2正值有利于B5。区间复用原保存7日日期权重，3/14日辅助保留；全部为事后描述性推断。")
    para("有限小型TCN可以审计便宜序列模型是否覆盖表征信号，不证明所有无界架构不可替代。当前一轮补充完成后停止，不自动扩网格、挑种子、微调或新第四轮。")
    page("动机与因果 TCN 协议")
    para("B5从头训练小型因果TCN，输入同256历史小时的6通道真实OHLCVA，加5个已知历时字段。市场窗口float64均值/总体标准差+1e-5，clip±5后float32；普通33项仅TRAIN标准化。")
    para("11输入通道；初始kernel3仅左padding2；7残差块dilation=1/2/4/8/16/32/64、kernel3、GELU；无batch normalization/dropout。感受野257，实际窗口256；取最后位置hidden拼33普通特征，线性输出log-scaled RV。")
    para("原TRAIN/VALID/两测试季度、标签、机会ID、时序与strict purge完全沿用；不重下载/重建标签。历史calendar为minute/hour/weekday/day/month，固定除数59/23/6/31/12；是当时已知时钟信息，不是未来市场值。")
    para("锁定协议SHA256："+CONFIG_SHA,8.8)
    para("TRAIN [2024-01-01,2026-03-01)，VALID [2026-03-01,2026-04-01)，事后比较Q2/Q3 [2026-04-01,2026-10-01)。B5选择只用原VALID，不能用测试季度改模型。")
    page("十二候选与 VALID 选择")
    para("width16/32×weight_decay1e-4/1e-3×seed17/29/43=12完成候选、13次尝试（1次用户关机中断）。AdamW lr=.001，batch64，最多120epoch、patience15、梯度norm clip1；float32、单线程、CUDA deterministic，TF32/autocast关闭。TRAIN effective RV中位数缩放目标。",9)
    para("复用前5个权重、历史及审计原字节；第6候选无中途检查点，按原seed从头重做，随后训练原未开始6个。操作性补充说明独立封存，原无retry协议与科学搜索范围保留；全部尝试披露，VALID-only选择不变。",8.6)
    para("每seed按最低原单位float64 VALID rawQLIKE选epoch，精确并列取早epoch。结构按三seed原单位RV预测算术均值的VALID QLIKE选择，并列较小width、较大weight_decay。B5是所选结构三seed预测均值，单seed另报；不重拟合TRAIN+VALID。")
    table(pd.DataFrame(selected["structure_scores"])[["width","weight_decay","validation_qlike"]],size=8,digits=10)
    table(pd.DataFrame([{"width":selected["width"],"weight_decay":selected["weight_decay"],"VALID_rawQLIKE":selected["validation_qlike"]}]),size=8,digits=10)
    para("选择依据："+selected["selection_basis"]+"；实际候选数="+str(audits["candidate"]["candidate_fits"])+"；全部历史保留，无TEST选择。")
    for m in selected["seed_metadata"]: para("seed="+str(m["seed"])+"，best_epoch="+str(m["best_epoch"])+"，best VALID rawQLIKE="+fmt(m["best_valid_raw_qlike"],10)+"；训练标签相对舍入最大误差="+format(m["target_rounding_max_relative_error"],".8g")+"。",8.4)
    for q in QUARTERS:
        page(q+"：十三家族完整指标")
        section=metrics[metrics.quarter==q].set_index("family").loc[FAMILIES].reset_index()
        first=section.iloc[0];para("N="+str(first.N)+"；surprise positive="+str(first.positive)+"，negative="+str(first.negative)+"；同一EWMA定义事件与评分。",9)
        table(section[["family","AUROC","AP","raw_QLIKE","QLIKE_Regret","logRV_MSE"]],size=7.8,digits=7)
        para("B5为三seed原单位RV均值的预测集成；B5_s17/s29/s43分别报告。旧三随机性能不混为预测集成。rawQLIKE可负，禁止用百分比改善解释。全部十三家族保留。")
    page("事后配对差值与有效 draw")
    table(ci[["block_days","comparison","Q2_delta","Q3_delta","mean_delta","ci_lower","ci_upper"]],size=7.7,digits=6)
    table(ci[["block_days","comparison","valid_draws","invalid_fraction"]],size=8,digits=6)
    para("共用原7/3/14日×5000次UTC日multiplicities，无新随机draw。单类季度NaN，等权要求两季有效，百分位CI；不能填0/0.5或补抽。没有B5确认性门槛或新的成功路径。",8.7)
    page("季度 AUROC 与 R2-B5 区间","事后探索；真实季度起点；十三家族",wide=True)
    image(aucplot,"全部模型AUROC；两点连线不是每日性能。quarter_AUROC.csv同时保留AP/损失/样本数。",190)
    image(ciplot,"R2-B5正值有利于R2；7日参考与3/14日敏感性均保存。R2_minus_B5_CI.csv含全部比较及有效draw。",170)
    page("真实 RV 与预测时间线","原R1/R2/B2加B5；Q2/Q3分别分面",wide=True)
    image(rvplot,"单位4h平方自然log收益，不年化；observed RV_raw保留。数值底表RV_timeline.csv逐决策可查。")
    page("日累计配对 QLIKE 差","季度重置；正值有利于B5",wide=True)
    image(dailyplot,"按UTC日sum(loss_R1/B2/R2-loss_B5)，随后季度内累计；空日零loss/count。不代表权益、收益或连续复利。")
    page("全部十二候选训练轨迹","TRAIN与VALID原始单位QLIKE；非TEST选择",wide=True)
    image(historyplot,"每个候选和seed全部保存，不只画最优。PNG/SVG/CSV及原models/history_*.csv可核查每epoch、clip计数与目标。",350)
    page("TRAIN 固定校准与幅度","同一label/epsilon；辅助描述",wide=True)
    image(calplot,"固定TRAIN预测decile；mean observed RV / mean prediction，1为箱均值相当。事件率另存CSV，不是概率校准。",290)
    para("绝对方差幅度、尾部低估与排序是不同问题。RV_eff=max(RV_raw,eps)，预测clip[1e-12,1]；原QLIKE/Regret/logRV MSE完整13家族分季度表保留。任何图都不能证明安全仓位。",8.7)
    page("TRAIN 尾部阈值与低估","q90/q99不使用TEST分位数重估",wide=True)
    image(tailplot,"pred/RV_eff<0.5为低估，TRAIN固定q90/q99。空子集条件比例/均值未定义，空集合求和0合法；稀少极端样本不等于校准安全证明。",270)
    tail_b5=tails[tails.family=="B5"]
    for r in tail_b5.itertuples(): para(str(r.quarter)+" "+str(r.subset)+"：N="+str(r.count)+"，低估count="+str(r.underprediction_count)+"，fraction="+fmt(r.underprediction_fraction)+"。",8.4)
    page("真实计算资源与容量限制")
    para("12完成候选（前5复用、第6重做、6原未开始）训练总耗时="+fmt(resource["candidate_training_seconds_total"],3)+"秒；总13尝试含1次关机中断。中断另计挂起前约"+fmt(interruption['pre_pause_partial_fit_wall_seconds_approx'],3)+"秒墙钟，不是GPU计算秒数，不含停机等待；partial epoch/GPU计算秒数未知，不并入完成候选资源。",8.5)
    inference=pd.DataFrame(resource["inference"])
    wanted=[c for c in ("family","parameters","seconds_per_batch8","peak_allocated_bytes","checkpoint_bytes") if c in inference]
    table(inference[wanted],size=7.8,digits=6)
    training=pd.DataFrame(resource["training"])
    cols=[c for c in ("width","weight_decay","seed","elapsed_seconds","parameters","epochs_run","cuda_peak_allocated_bytes") if c in training]
    table(training[cols],size=7.4,digits=4)
    para("同8TRAIN窗口、3warmup后20次基准：B5三seed集成0.048212445秒/batch8，Kronos tokenizer+backbone为0.023132335秒，前者慢2.08倍；虽参数与显存较少，不代表推理更快。Kronos基准不含R2预测头，非完整pipeline或总计算/预训练成本匹配。",8.3)
    para("CUDA peak_allocated含常驻模型/输入，非单模型增量峰值。逐seed加载累积3个TCN，单seed峰值不是独立显存需求；集成含3模型。资源原限制："+resource["resource_comparability_limit"],8.1)
    page("边界、审计与原3.0历史附录")
    para("本轮有1次用户关机中断；5候选复用、6号从头重做加原未开始6候选，共12完成、13尝试。中断epoch未知；约430.166秒为挂起前墙钟，不是GPU计算且不含停机等待。恢复seal和操作性说明hash绑定全部新证据；原对照、561项历史封存及证据类别保持。",8.5)
    doc.para("首次独立审查因实现误将辅助列r0_ewma当作33项特征成员而FAIL；修正读取后numeric_review_v2 PASS，无重训/重选。原FAIL与首次源版本不可变保留；属已解决工程检查失败，非科学结果失败。",size=8.5,after=7)
    para("B5事后补充数值独立审查PASS；输入和候选审计PASS。模型重载、因果窗口、role、标签与原机会身份、批量/单样本/重复检查及所有配对draw以原审计文件为证据；工程完成不能升级为新的独立确认。")
    para("当前有限B5补充停止。原3.0 CONFIRMED_RANKING_INCREMENT针对已锁定两主要比较保持不变；本次只帮助解释便宜序列模型是否覆盖信号。未来研究必须新独立数据、另行设计登记；不自动扩网格或使用本轮测试继续选模型。",10,True)
    para(numeric_description+b5_interpretation,9.5,True)
    para("接下来的历史附录直接拼接原3.0完整PDF，不修改任何页内容。历史页码保持原编号；其中的确认性表述只适用于原3.0，不适用于B5。新部分与历史严格分界。")
    para("原PDF SHA256："+sha(oldpdf)+"\n原MD SHA256："+sha(oldmd)+"\n新协议SHA256："+CONFIG_SHA,8.2)
    para("新出处：research/runs/FROZEN_RISK_03_1_B5_v1。所有新JSON、CSV、候选历史、seedmetadata与资源保留；原实验、报告及manifest全不覆盖。",9)
    doc.end()
    fitz=pdfbase.fitz;combined=fitz.open();newdoc=fitz.open(supplement);olddoc=fitz.open(oldpdf)
    combined.insert_pdf(newdoc);combined.insert_pdf(olddoc)
    combined.set_metadata({"title":"Kronos 3.1 - B5 post-hoc supplement with unchanged 3.0 historical appendix","author":"Kronos Research","subject":"Post-hoc exploratory B5; original primary status unchanged"})
    combined.save(pdf);total=len(combined);old_pages=len(olddoc);combined.close();newdoc.close();olddoc.close()
    lines.extend(["","## 完整新数值附录",""])
    lines.extend(["### 仅出版合并的TRAIN固定校准对照","",mdtable(calibration_comparison),""])
    for name,frame in frames.items():
        lines.extend(["### metrics/"+name+".csv",""])
        if name=="row_metrics": lines.extend(["完整逐机会CSV保留原路径；SHA256："+sha(RUN/"metrics/row_metrics.csv"),""])
        else: lines.extend([mdtable(frame),""])
    for name,value in [("summary",summary),("selected",selected),("resources",resource),("candidate_audit",audits["candidate"]),("inputs_audit",audits["inputs"]),("numeric_review",numeric),("recovery_seal",recovery),("recovery_claim",recovery_claim),("interrupted_attempt_resources",interruption)]:
        lines.extend(["### "+name,"","```json",json.dumps(value,ensure_ascii=False,indent=2),"```",""])
    lines.extend(['### 操作性恢复说明原文', '', '来源：'+str(amendment_path.relative_to(ROOT)).replace('\\','/')+'；SHA256：'+sha(amendment_path),'',amendment_text,''])
    lines.extend(['### 独立数值审查首次失败记录', '', '首次审查因脚本误将基线列 r0_ewma 索引为33项特征成员而失败；r0_ewma 保留在原完整普通表。修复读取位置后使用独占 numeric_review_v2.json 作为当前验收依据；没有重训、重新选择或修改科学指标、33特征集合与EWMA算法。原FAIL文件及首次源版本完整保留。', '', '原FAIL来源：'+str(numeric_attempt01_path.relative_to(ROOT)).replace('\\','/')+'；SHA256：'+sha(numeric_attempt01_path), '', '```json', json.dumps(numeric_attempt01,ensure_ascii=False,indent=2), '```', ''])
    lines.extend(["## 原3.0 Markdown历史附录","", "以下内容逐字decoded原3.0 MD，不回写；原字节SHA256："+sha(oldmd),"", "历史MD中的相对图片及CSV路径以原目录 `research/runs/FROZEN_RISK_03_v1/reports/` 为基准。请从[原3.0 Markdown报告]("+os.path.relpath(oldmd,md.parent).replace("\\","/")+")访问这些历史相对链接。", "",oldmdbytes.decode("utf-8"),"", "## 3.1输入来源SHA256", "",mdtable(pd.DataFrame(sorted(sources.items()),columns=["ROOT相对路径","SHA256"]))])
    with md.open("x",encoding="utf-8") as f:f.write("\n".join(lines)+"\n")
    imagepaths=[];rendered=fitz.open(pdf)
    for i,p in enumerate(rendered):
        target=temp/f"page_{i+1:02d}.png";p.get_pixmap(matrix=fitz.Matrix(1.3,1.3),alpha=False).save(target);imagepaths.append(target)
    from PIL import Image,ImageDraw
    tw,th=360,510;sheet=Image.new("RGB",(tw*4,th*((total+3)//4)),"#eeeeee")
    for i,p in enumerate(imagepaths):
        im=Image.open(p);im.thumbnail((tw-12,th-30));x,y=i%4*tw,i//4*th;sheet.paste(im,(x+(tw-im.width)//2,y+23));ImageDraw.Draw(sheet).text((x+8,y+5),str(i+1),fill="black")
    sheet.save(temp/"contact_sheet.png");rendered.close()
    for relative,digest in sources.items():
        if sha(ROOT/relative)!=digest: raise ValueError("Input changed during publication: "+relative)
    record={"status":"GENERATED_PENDING_ROOT_VISUAL_REVIEW","evidence_class":summary["evidence_class"],"original_primary_status":original["status"],"protocol_sha256":CONFIG_SHA,
            "created_at_utc":datetime.now(timezone.utc).isoformat(),"publication_code_sha256":sha(__file__),"input_sha256":sources,"pdf_sha256":sha(pdf),"md_sha256":sha(md),
            "supplement_pages":doc.number,"historical_3_0_pages":old_pages,"total_pages":total,"historical_appendix_start_page":doc.number+1,"original_pdf_sha256":sha(oldpdf),"original_md_sha256":sha(oldmd),
            "supplement_outline":outline,"charts":graphmeta,"figure_sha256":{p.name:sha(p) for p in sorted(figures.iterdir())},"render_sha256":{p.name:sha(p) for p in sorted(temp.glob("*.png"))},
            "no_refit":True,"no_new_thresholds":True,"no_new_random_draws":True,"no_new_confirmatory_success_path":True,"round_trip_csv_reads":True,"old_files_unchanged":True,"visual_QA_required":True}
    with manifest.open("x",encoding="utf-8") as f:json.dump(record,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")
    print(json.dumps({"status":record["status"],"pdf":str(pdf),"supplement_pages":doc.number,"total_pages":total},ensure_ascii=False))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--publish-generated-only",action="store_true",required=True);p.parse_args();publish()
