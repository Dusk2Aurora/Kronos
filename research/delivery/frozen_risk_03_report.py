"""Create the complete RISK_03 v2 publication from generated CONSUMED evidence.

This entry point cannot fit, redraw bootstrap, download or read market sources.
It leaves the scientific run and original v1 publication unchanged.
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
RUN = ROOT / "research/runs/FROZEN_RISK_03_v1"
PROTOCOL = "7ed8affd7cd624cde89d23d29aaca7ef2ab660167fac7d3dc25fac274e394b7f"
FAMILIES = ["persistence", "ewma", "har", "R1", "R2", "B2", "random_s17", "random_s29", "random_s43"]
MAIN = ["R1", "R2", "B2"]
QUARTERS = ["2026Q2", "2026Q3"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fmt(value, digits=6):
    if value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
        return "未定义"
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if isinstance(value, (int, np.integer)):
        return str(value)
    if isinstance(value, (float, np.floating)):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(table):
    # No optional tabulate dependency; numeric values remain sourced from frames.
    def cell(v):
        if isinstance(v, (float, np.floating)) and np.isfinite(v):
            return format(v, ".17g")
        return fmt(v, 10).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(str, table.columns)) + " |",
                       "| " + " | ".join(["---"] * len(table.columns)) + " |"] +
                      ["| " + " | ".join(cell(v) for v in row) + " |" for row in table.itertuples(index=False, name=None)])


def publish():
    sources = {}
    tables = {}

    def source(relative):
        p = RUN / relative
        sources[relative] = sha(p)
        return p

    def jread(relative):
        return json.loads(source(relative).read_text(encoding="utf-8-sig"))

    terminal = jread("authorization/formal_terminal.json")
    if terminal.get("state") != "CONSUMED":
        raise PermissionError("Report publication requires terminal CONSUMED before saved holdout reads")
    numeric = jread("independent_review/formal_numeric.json")
    if numeric.get("status") != "PASS":
        raise PermissionError("Independent formal numeric review must PASS")
    result = jread("metrics/formal_result.json")
    seal = jread("calibration/sealed_calibration.json")
    diagnostics = jread("diagnostics/holdout/manifest.json")
    chartmanifest = jread("figures/formal_publication/chartmanifest.json")
    auxiliary = jread("metrics/publication_auxiliary/manifest.json")
    separated_exports = jread("reports/formal_separated_exports.json")
    auxiliary_review = jread("independent_review/formal_auxiliary_review.json")
    report_review = jread("independent_review/formal_report_review.json")
    if separated_exports.get("status") != "PASS" or auxiliary_review.get("status") != "PASS":
        raise PermissionError("Separated exports and independent auxiliary review must PASS")
    if report_review.get("status") != "PASS" or report_review.get("formal_status") != result["status"] or report_review.get("terminal_state") != "CONSUMED":
        raise PermissionError("Saved full report numeric evidence review must PASS and match the consumed result")
    if auxiliary_review.get("source_manifest_sha256") != sources["metrics/publication_auxiliary/manifest.json"]:
        raise ValueError("Independent auxiliary review binds a different supplement")
    for mapping in (separated_exports["input_sha256"], separated_exports["output_sha256"]):
        for relative, digest in mapping.items():
            p = (ROOT / relative.replace("\\", "/")).resolve()
            if not p.is_relative_to(RUN.resolve()):
                raise ValueError("Separated export path is outside the formal run")
            if sha(source(str(p.relative_to(RUN)).replace("\\", "/"))) != digest:
                raise ValueError("Separated export/input hash mismatch: " + relative)
    if any(v.get("protocol_sha256") != PROTOCOL for v in (seal, diagnostics, auxiliary)):
        raise ValueError("Publication evidence uses a different protocol")
    if chartmanifest.get("terminal_state") != "CONSUMED" or auxiliary.get("formal_primary_result_unchanged") != result["status"]:
        raise ValueError("Publication supplements differ from consumed result")
    for folder, mapping in [("diagnostics/holdout", diagnostics["artifacts_sha256"]),
                            ("figures/formal_publication", chartmanifest["output_files_sha256"]),
                            ("metrics/publication_auxiliary", auxiliary["artifact_sha256"])]:
        for name, digest in mapping.items():
            if sha(source(folder + "/" + name)) != digest:
                raise ValueError("Generated artifact hash mismatch: " + folder + "/" + name)

    def csv(relative):
        table = pd.read_csv(source(relative), float_precision="round_trip")
        tables[relative] = table
        return table

    rows = csv("diagnostics/holdout/row_metrics_and_timeline.csv")
    rows["decision_at"] = pd.to_datetime(rows.decision_at, utc=True)
    if set(rows.family) != set(FAMILIES) or set(rows.quarter) != set(QUARTERS):
        raise ValueError("All nine families and both formal quarters required")
    calibration = csv("diagnostics/holdout/model_calibration.csv")
    under = csv("diagnostics/holdout/high_extreme_RV_underprediction.csv")
    losses = csv("diagnostics/holdout/period_losses.csv")
    state = csv("diagnostics/holdout/historical_EWMA_quartile_losses.csv")
    rolling = csv("diagnostics/holdout/calendar_rolling_7day.csv")
    rolling["day"] = pd.to_datetime(rolling.day, utc=True)
    aux = {name: csv("metrics/publication_auxiliary/" + name + ".csv") for name in
           ["high_rv_ranking", "high_rv_paired_points", "high_rv_paired_intervals", "surprise_correlations",
            "weekly_rv", "loss_concentration", "extreme_paired_QLIKE_contribution"]}
    selected = jread("models/formal/selected_models.json")
    fit_audit = jread("models/formal/fit_audit.json")
    identity = jread("models/formal/preview_identity_check.json")
    events_path = source("models/formal/fit_events.jsonl")
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    data_audit = jread("data_5m/raw_csv_audit.json")
    data_manifest = jread("data_5m/manifest.json")
    causality = jread("features/holdout_causality.json")
    encoding = {v: jread("features/holdout_" + v + "_audit.json") for v in ["pretrained", "random_s17", "random_s29", "random_s43"]}
    bundle = jread("preflight/sealed_bundle.json")
    runtime = jread("provenance/formal_source_manifest.json")
    approval = jread("authorization/user_approval.json")
    claim = jread("authorization/execution_claim.json")
    rejected = jread("authorization/rejected_encoding_before_claim.json")
    if data_audit.get("status") != "PASS" or identity.get("status") != "PASS" or fit_audit.get("status") != "PASS":
        raise ValueError("Saved formal evidence audit incomplete")

    pdf = ROOT / "output/pdf/Kronos_Frozen_Risk_03_Report_v2.pdf"
    md = RUN / "reports/Kronos_Frozen_Risk_03_Report_v2.md"
    manifest_path = RUN / "reports/formal_pdf_generation_v2.json"
    figures = RUN / "figures/formal_report_v2"
    render_dir = ROOT / "tmp/pdfs/frozen_risk03_formal_v2"
    if any(p.exists() for p in (pdf, md, manifest_path, figures, render_dir)):
        raise FileExistsError("Refuse to overwrite any v2 publication or QA artifacts")
    figures.mkdir(parents=True)
    render_dir.mkdir(parents=True)
    pdf.parent.mkdir(parents=True, exist_ok=True)

    # Import only after publication guards. Document draws saved values; no model API is called.
    sys.path.insert(0, str(ROOT))
    from research.frozen.experiment_03.report import Document, pdfbase
    pdfbase.init_fonts()
    plt = pdfbase.plt
    palette = {f: plt.get_cmap("tab10")(i) for i, f in enumerate(FAMILIES)}
    extra_plots = []

    def plot_save(name, fig, table, formula):
        fig.tight_layout()
        p = figures / name
        fig.savefig(p.with_suffix(".png"), dpi=160)
        fig.savefig(p.with_suffix(".svg"))
        plt.close(fig)
        table.to_csv(p.with_suffix(".csv"), index=False, float_format="%.17g")
        extra_plots.append({"name": name, "formula": formula, "no_refit": True, "diagnostic_only": True})
        return p.with_suffix(".png")

    weekly = aux["weekly_rv"].copy()
    weekly["week_start"] = pd.to_datetime(weekly.week_start, utc=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6))
    for ax, q in zip(axes, QUARTERS):
        for f in MAIN:
            section = weekly[(weekly.period == q) & (weekly.family == f)].sort_values("week_start")
            ax.plot(section.week_start, section.prediction_RV, label=f, color=palette[f])
        section = weekly[(weekly.period == q) & (weekly.family == "R2")].sort_values("week_start")
        ax.plot(section.week_start, section.observed_RV, color="black", label="observed RV_raw", linewidth=1.4)
        ax.set(title=q, ylabel="Mean 4h squared log return", xlabel="Monday 00:00 UTC (partial boundary weeks)")
        ax.legend(ncol=4); ax.grid(alpha=.2)
    weekly_plot = plot_save("weekly_RV", fig, weekly, "Saved opportunity-weighted weekly means; quarter boundaries grouped separately")
    fig, axes = plt.subplots(2, 1, figsize=(11, 6))
    for ax, q in zip(axes, QUARTERS):
        start, end = ("2026-04-01", "2026-07-01") if q == "2026Q2" else ("2026-07-01", "2026-10-01")
        for f in MAIN:
            d = rolling[(rolling.family == f) & (rolling.day >= pd.Timestamp(start, tz="UTC")) & (rolling.day < pd.Timestamp(end, tz="UTC"))]
            ax.plot(d.day, d.rolling_7day_raw_QLIKE_mean, label=f, color=palette[f])
        ax.set(title=q, xlabel="UTC calendar day", ylabel="7-day raw QLIKE: loss sum / count")
        ax.legend(); ax.grid(alpha=.2)
    raw_plot = plot_save("rolling_raw_QLIKE", fig, rolling[rolling.family.isin(MAIN)], "Saved trailing 7 UTC calendar-day loss sum / opportunity count; not mean of daily means; Q3 first six days may include Q2")

    deciles = calibration[(calibration.bin_type == "prediction_decile") & calibration.family.isin(MAIN) & calibration.period.isin(QUARTERS)].copy()
    fig, axes = plt.subplots(2, 2, figsize=(11, 6))
    for col, q in enumerate(QUARTERS):
        for f in MAIN:
            d = deciles[(deciles.period == q) & (deciles.family == f)].sort_values("bin_id")
            axes[0, col].plot(d.bin_id + 1, d.observed_raw_over_prediction_mean_ratio, "o-", label=f, color=palette[f])
            axes[1, col].plot(d.bin_id + 1, d.surprise_event_fraction, "o-", label=f, color=palette[f])
        axes[0, col].axhline(1, color="gray", linestyle="--")
        axes[0, col].set(title=q, ylabel="Mean observed RV_raw / mean prediction")
        axes[1, col].set(xlabel="TRAIN-fixed prediction decile", ylabel="Observed surprise event fraction", ylim=(0, 1))
        for ax in axes[:, col]: ax.legend(fontsize=8); ax.grid(alpha=.2)
    decile_plot = plot_save("train_prediction_decile_reliability", fig, deciles, "Saved mean observed RV_raw / mean prediction and event count / count; TRAIN cuts fixed; no probability calibration")
    states = state[state.family.isin(MAIN) & state.period.isin(QUARTERS)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, q in zip(axes, QUARTERS):
        for f in MAIN:
            d = states[(states.period == q) & (states.family == f)].sort_values("bin_id")
            ax.plot(d.bin_id + 1, d.QLIKE_Regret, "o-", label=f, color=palette[f])
        ax.set(title=q, xlabel="TRAIN-fixed historical EWMA quartile", ylabel="Mean QLIKE Regret (lower better)", xticks=[1, 2, 3, 4])
        ax.legend(); ax.grid(alpha=.2)
    state_plot = plot_save("historical_EWMA_state_losses", fig, states, "Saved mean QLIKE Regret in fixed TRAIN EWMA state bins; no best-state selection")

    doc = Document(pdf, "B")
    mdlines = ["# Kronos 第三轮独立冻结风险实验完整正式报告 v2", "",
               "FROZEN_RISK_03_v1；一次正式评估；holdout CONSUMED；完整证据、原始 v1 和失败历史均保留。"]
    page_meta = []

    def page(title, subtitle="", wide=False):
        doc.start(title, subtitle, wide)
        # The inherited base header describes old DEV research. Redraw only
        # this page's header to describe the independent formal publication.
        doc.c.setFillColor(pdfbase.colors.white)
        doc.c.rect(40, doc.h-43, doc.w-80, 14, fill=1, stroke=0)
        doc.c.setFillColor(pdfbase.BLUE)
        doc.c.setFont("CN", 8)
        doc.c.drawString(42, doc.h-36, "KRONOS / INDEPENDENT FROZEN RISK / FORMAL HOLDOUT CONSUMED")
        page_meta.append({"page": doc.number, "title": title, "wide": wide})
        mdlines.extend(["", "## " + str(doc.number) + ". " + title, "", subtitle, ""])

    def para(text, size=9.7, bold=False):
        doc.para(text, size=size, bold=bold, after=7)
        mdlines.extend([text, ""])

    def table(data, headers=None, size=8.3, digits=6, widths=None):
        frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data, columns=headers)
        pdf_rows = [list(frame.columns)] + [[fmt(v, digits) for v in row] for row in frame.itertuples(index=False, name=None)]
        doc.table(pdf_rows, widths=widths, size=size)
        mdlines.extend([markdown_table(frame), ""])

    def image(path, note, height=340):
        p = Path(path)
        # Every figure included in the PDF is also tracked as a publication input.
        if p.is_relative_to(RUN): sources[str(p.relative_to(RUN)).replace("\\", "/")] = sha(p)
        doc.image(p, height=height)
        para(note, 8.5)
        mdlines.extend(["![" + note.replace("]", "") + "](" + os.path.relpath(p, md.parent).replace("\\", "/") + ")", ""])

    primary = result["bootstrap"]["7"]
    comparisons = primary["comparisons"]
    status = result["status"]
    decision = {
        "INCONCLUSIVE": "当前独立证据不充分。停止当前扩模和本轮冻结表征路线扩展；未来问题应等待独立数据，另行登记，不自动启动第四轮。",
        "NOT_SUPPORTED": "共同主要增量没有得到支持。停止当前任务上的本轮路线扩展，保留失败及全部模型证据。",
        "NONLINEAR_BASELINE_COMPETITIVE": "预训练表征相对线性基线有支持，但未证明超过普通非线性基线。优先保留廉价普通基线；该结果不是统计等效性证明。停止本轮扩模。",
        "CONFIRMED_RANKING_INCREMENT": "支持继续研究风险排序/风险参与门控的方向，下一项需新的独立数据与另行登记；不自动执行第四轮。仅支持登记独立区间上的排序增量，不支持交易 alpha 或安全仓位。当前有限一轮研究完成并停止，不自动扩模。",
        "RANKING_ONLY": "仅支持风险排序，绝对方差幅度触发红旗；不能据此建议逆方差仓位。当前有限一轮研究完成并停止，不自动扩模。",
    }.get(status, "按锁定协议保留状态，停止本轮执行；不自动重跑或开启第四轮。")
    main_rows = [[k.replace("_minus_", " - "), *v["quarter_deltas"], v["mean_delta"], v["ci_lower"], v["ci_upper"]] for k, v in comparisons.items()]
    page("摘要与停止决定", "独立 Q2/Q3 | 主块7日 | 风险排序与风险幅度分别解释")
    para("正式结论：" + status, 13, True)
    table(main_rows, ["比较", "Q2 Δ", "Q3 Δ", "等权 Δ", "CI下限", "CI上限"], size=8.5)
    para("Δ为绝对AUROC差，正值有利于R2；95%CI是登记的季度分层配对 stationary bootstrap 百分位区间。共同主要终点必须分别通过全部门槛。")
    para(decision, 11, True)
    para("幅度红旗：" + str(result["risk_magnitude_red_flag"]) + "。辅助损失、相关性、q90排序和极端诊断不新增成功标准，不替换共同主要结论。")
    para("本轮不是交易收益实验。未评价资金、订单簿、成本后收益或安全风险控制；排序结果不自动转化为收益、仓位比例或实盘建议。")

    page("动机、假设与冻结协议")
    para("第一轮固定动量参与任务的Ridge、有限MLP和量化诊断没有证明稳定经济增量；第二轮条件风险研究的主要结论为INCONCLUSIVE，预训练对照保留支持信号。第三轮只检验新独立区间上的风险排序，以及普通非线性模型是否覆盖该增量。")
    table([["H1", "R2相对普通线性R1改善surprise事件排序"], ["H2", "R2相对普通非线性B2仍保留排序增量"], ["H3", "绝对RV幅度与低估风险独立审计，不能由排序替代"]], ["假设", "锁定目的"], widths=[60, doc.width-60])
    para("协议SHA256：" + PROTOCOL, 8.5)
    para("来源：research/configs/frozen_risk_03_v1.yaml；登记：research/registry/FROZEN_RISK_03_v1.json。完整协议、源码、输入、模型与依赖封存为preflight/sealed_bundle.json，审批绑定其SHA。")
    para("第三轮评价后停止有限路线；不由测试反馈新增候选、改阈值、改评分、挑随机种子、改主要块长或进入微调。")

    page("数据、时序与真实样本")
    para("TRAIN [2024-01-01,2026-03-01)，VALID [2026-03-01,2026-04-01)，独立HOLDOUT [2026-04-01,2026-10-01)。Q2/Q3各按右边界严格purge持有期和labelable_at；不保留跨季度标签。")
    para("UTC04/12/20机会钟，04:01决策，05:00入场，09:00完成4h标签，09:01可标注。特征只消费当时可用的已完成1h OHLCVA；5m只构成未来风险标签。")
    table([["正式5m行数", data_manifest["rows"]], ["原始响应页数", data_audit["raw_pages"]], ["raw/canonical一致", data_audit["exact_raw_csv_match"]], ["训练/验证机会", str(fit_audit["train"]) + " / " + str(fit_audit["validation"])], ["价格标签", "49价格 / 48个5m自然log收益平方之和"]], ["数据项", "正式证据"], widths=[165, doc.width-165])
    quarter_event = [[q["id"], q["opportunities"], q["positive"], q["negative"], q["positive"]/q["opportunities"]] for q in result["quarter_metrics"]]
    table(quarter_event, ["季度", "N", "事件", "非事件", "事件比例"], digits=6)
    para("真实volume=volCcy(BTC)，amount=volCcyQuote(USDT)。不插值、不回退1h标签、不用价格乘量替代成交额；RV_raw保留，effective RV=max(raw,1e-12)。单位为4h平方自然log收益，不年化。", 9)

    page("模型矩阵与验证期选择")
    para("6个线性家族各4个lambda=24候选；B2四结构=4候选，共28。另R2和B2各一次预登记确定性重放，共2次，不作为新候选。固定persistence与EWMA无需拟合。")
    selection_rows = []
    for s in selected:
        choice = f"leaves={s['num_leaves']}; minleaf={s['min_data_in_leaf']}; iter={s['best_iteration']}" if s["family"] == "B2" else "lambda=" + str(s["lambda"])
        selection_rows.append([s["family"], s["dimension"], choice, s["validation_qlike"]])
    table(selection_rows, ["家族", "维度", "封存所选参数", "VALID原始QLIKE"], size=8.2, widths=[87, 47, 237, doc.width-371], digits=10)
    para("HAR=3维，R1和B2=同33列，R2与三随机=33+512=545维；随机骨干保留原tokenizer。标准化和中位RV目标缩放仅TRAIN拟合，VALID原QLIKE选择，不用独立AUROC选择。")
    para("线性lambda=.001/.01/.1/1，精确并列取较大lambda。B2 leaves7/15×minleaf30/60，learning_rate=.05，最多300轮，early-stop30；并列较少叶、较大minleaf。VALID值不是独立结论。", 9)

    page("主要终点与配对统计")
    para("事件 Y=1{log((RV_raw+epsilon)/(同一EWMA预测+epsilon))>log(2)}。评分 S=log((模型预测RV+epsilon)/(同一EWMA预测RV+epsilon))。epsilon=1e-12；分数不是事件概率，无测试归一化。")
    table([["主要比较", "R2-R1与R2-B2分别通过"], ["统计量", "先季度内AUROC差，再Q2/Q3等权"], ["效应门槛", "等权ΔAUROC≥0.03，且两个季度各>0"], ["不确定性门槛", "7日主块95%CI下限>0"], ["事件数量", "每季度正/负事件各≥10"], ["重采样", "完整UTC日，季度分层circular stationary，共享日multiplicities"], ["主/敏感性", "7 / 3及14日，每块5000次尝试"], ["无效处理", "单类draw记NaN，不填0或0.5、不补抽；joint-valid CI"]], ["规则", "锁定定义"], size=8.7, widths=[117, doc.width-117])
    para("随机流=20261008+block_days×1000+quarter_index。空日保留；任何主要季度或联合无效比例>1%→INCONCLUSIVE。六个月pooled AUROC不是共同主要统计。敏感性不用于挑选主块。")

    page("九家族季度 AUROC 与 AP")
    rank_rows = []
    for q in result["quarter_metrics"]:
        for f in FAMILIES:
            m = q["models"][f]
            rank_rows.append([q["id"], f, m["AUROC"], m["AP"]])
    table(rank_rows, ["季度", "模型", "AUROC", "AP"], size=8.5, digits=8)
    for q in result["quarter_metrics"]:
        vals = q["models"]
        mean_auc = np.mean([vals[f]["AUROC"] for f in FAMILIES if f.startswith("random_")])
        mean_ap = np.mean([vals[f]["AP"] for f in FAMILIES if f.startswith("random_")])
        para(q["id"] + "三随机模型性能算术均值：AUROC=" + fmt(mean_auc, 8) + "，AP=" + fmt(mean_ap, 8) + "。这是逐模型性能均值，不是预测集成。", 8.8)

    page("全部主比较与门槛判断")
    bootstrap_rows = []
    for b in (7, 3, 14):
        item = result["bootstrap"][str(b)]
        for k, v in item["comparisons"].items():
            bootstrap_rows.append([b, k.replace("_minus_", "-"), *v["quarter_deltas"], v["mean_delta"], v["ci_lower"], v["ci_upper"]])
    table(bootstrap_rows, ["块日", "比较", "Q2 Δ", "Q3 Δ", "等权Δ", "CI下", "CI上"], size=7.8, digits=6)
    table([[b, result["bootstrap"][str(b)]["joint_valid_draws"], max(result["bootstrap"][str(b)]["invalid_fractions"].values()), result["bootstrap"][str(b)]["status"]] for b in (7,3,14)], ["块日", "联合有效draw", "最大无效比例", "状态"], size=8)
    for k, c in comparisons.items():
        gates = ["两季均正=" + str(all(v is not None and v > 0 for v in c["quarter_deltas"])), "等权≥.03=" + str(c["mean_delta"] is not None and c["mean_delta"] >= .03), "CI下限>0=" + str(c["ci_lower"] is not None and c["ci_lower"] > 0)]
        para(k + "：" + "；".join(gates) + "。", 9)
    para("事件充分=" + str(result["adequate_events"]) + "；最终状态=" + status + "。状态按封存优先级判定，辅助相关、AP和损失不能补充失败的共同主要门槛。", 9)
    for b in (7,3,14):
        para(str(b) + "日各角色无效比例：" + json.dumps(result["bootstrap"][str(b)]["invalid_fractions"], ensure_ascii=False), 8.1)

    page("季度风险排序对照", "全部九家族保留；真实UTC季度起点", wide=True)
    image(RUN/"figures/formal_publication/01_quarter_surprise_AUROC.png", "数值底表：figures/formal_publication/01_quarter_surprise_AUROC.csv。连线仅连接两个季度指标，不代表连续每日性能。", 340)
    page("两共同主要差值区间", "正方向有利于R2；7日主分析与3/14日敏感性并列", wide=True)
    image(RUN/"figures/formal_publication/02_primary_delta_AUROC_CI.png", "数值底表：02_primary_delta_AUROC_CI.csv。两比较分别判定；未按敏感性结果选择最有利块长。", 340)

    page("绝对方差损失与幅度红旗")
    table([[f, *[result["variance_losses"][f][k] for k in ("raw_QLIKE","QLIKE_Regret","logRV_MSE")]] for f in FAMILIES], ["模型(POOLED)", "原QLIKE", "Regret", "logRV MSE"], size=8.3, digits=7)
    d = losses[losses.period.isin(QUARTERS) & losses.family.isin(MAIN)]
    table(d[["period","family","raw_QLIKE","QLIKE_Regret","logRV_MSE"]], size=8, digits=7)
    ratio = result["variance_losses"]["R2"]["QLIKE_Regret"] / result["variance_losses"]["R1"]["QLIKE_Regret"]
    para("R2/R1 pooled Regret比=" + fmt(ratio, 8) + "；红旗条件>1.10；实际红旗=" + str(result["risk_magnitude_red_flag"]) + "。raw QLIKE=log(pred)+RV_eff/pred，可为负，不能用百分比改善表达。", 9)
    para("Regret=RV_eff/pred-log(RV_eff/pred)-1≥0；logRV MSE使用floor epsilon。全九模型分季度及pooled损失完整表在MD附录/diagnostics/holdout/period_losses.csv。", 8.5)

    page("日累计配对 QLIKE 差", "两季度分别从零累计；真实UTC日历", wide=True)
    image(RUN/"figures/formal_publication/04_daily_cumulative_QLIKE_difference.png", "每日sum(loss_R1-loss_R2)与sum(loss_B2-loss_R2)后季度内累计。正值有利于R2；是损失差总和，不是百分比、权益或连续复利。", 340)
    page("真实 RV 与预测时间线", "每季度独立分面；4h平方自然log收益，不年化", wide=True)
    image(RUN/"figures/formal_publication/03_RV_timeline.png", "已有正式RV_timeline逐字节复用；各模型预测与observed RV_raw全部保留。底表03_RV_timeline.csv逐决策可查。", 340)
    page("评分与滚动 Regret", "同一历史EWMA分母；7日损失sum / 机会count", wide=True)
    image(RUN/"diagnostics/holdout/surprise_score_timeline.png", "score=log((pred+eps)/(EWMA+eps))。原始评分不是事件概率；不进行TEST归一化。滚动Regret原图另见diagnostics/holdout/rolling_QLIKE_Regret.png。", 325)
    para("7日表保留完整UTC日与全部家族，滚动比值是损失总和/机会总数，而不是日均值的简单平均。Q3开始的窗口可能包含Q2末尾，这是既有正式滚动定义。", 8.5)
    page("周均 RV 与原 QLIKE", "机会加权周均值；边界周为不完整周", wide=True)
    image(weekly_plot, "底表figures/formal_report_v2/weekly_RV.csv。周起点Monday00:00 UTC，同季度单独分组；每条均值保留样本数及首末决策时间。", 175)
    image(raw_plot, "底表rolling_raw_QLIKE.csv；7UTC日sum(loss)/sum(count)，原QLIKE可负、单位非百分比；R1/R2/B2保留。", 175)

    page("TRAIN q90 高 RV 辅助排序")
    para("TRAIN effective RV q90=" + format(seal["thresholds"]["train_effective_RV_q90"], ".16g") + "。事件RV_eff>q90，以绝对预测RV为评分。该辅助终点与共同主要surprise事件不同，不能替代主要终点。", 9)
    for q in QUARTERS:
        event_row = aux["high_rv_ranking"][(aux["high_rv_ranking"].period == q) & (aux["high_rv_ranking"].family == "R2")].iloc[0]
        para(q + "：N=" + fmt(event_row["count"]) + "，positive=" + fmt(event_row.positive) + "，negative=N-positive=" + fmt(event_row["count"]-event_row.positive) + "；九模型共用同一事件与样本。", 8.3)
    table(aux["high_rv_ranking"][["period","family","count","positive","event_fraction","AUROC","AP"]], size=7.5, digits=6)
    selected_ci = aux["high_rv_paired_intervals"]
    selected_ci = selected_ci[(selected_ci.period == "equal_quarter") & (selected_ci.block_days == 7) & (selected_ci.metric == "AUROC")]
    table(selected_ci[["comparator","ci_lower","ci_upper","invalid_fraction"]], size=8)
    point_rows = aux["high_rv_paired_points"]
    point_r1 = point_rows[point_rows.comparator == "R1"].set_index("period")
    ci_r1 = selected_ci[selected_ci.comparator == "R1"].iloc[0]
    q3_events = aux["high_rv_ranking"][(aux["high_rv_ranking"].period == "2026Q3") & (aux["high_rv_ranking"].family == "R2")].iloc[0]
    para("q90辅助R2-R1 AUROC：Q2=" + fmt(point_r1.loc["2026Q2", "AUROC_R2_minus_comparator"], 7) + "，Q3=" + fmt(point_r1.loc["2026Q3", "AUROC_R2_minus_comparator"], 7) + "；等权7日CI=[" + fmt(ci_r1.ci_lower, 7) + "," + fmt(ci_r1.ci_upper, 7) + "]跨零。Q3仅" + fmt(int(q3_events.positive)) + "个阳性，不能声称稳定绝对高RV排序提升。完整7/3/14日AP/AUROC诊断CI及点差见辅助CSV；共享既有日权重，无新增抽样，不能替代主要终点。", 8.2)

    page("固定预测分箱可靠性", "R1/R2/B2 TRAIN预测decile；测试不重新划分", wide=True)
    image(decile_plot, "上：分箱mean(RV_raw)/mean(pred)。1表示该箱均值相当；下：实际surprise事件count/N，不是预测事件概率校准。底表train_prediction_decile_reliability.csv。", 320)
    for q in QUARTERS:
        d = deciles[(deciles.period == q) & (deciles.family == "R2") & (deciles["count"] > 0)].sort_values("bin_id")
        if len(d):
            a, b = d.iloc[0], d.iloc[-1]
            para(q+" R2首/末非空预测箱：N="+fmt(a["count"])+"/"+fmt(b["count"])+"，观察/预测均值比="+fmt(a.observed_raw_over_prediction_mean_ratio)+"/"+fmt(b.observed_raw_over_prediction_mean_ratio)+"，实际事件率="+fmt(a.surprise_event_fraction)+"/"+fmt(b.surprise_event_fraction)+"。", 8.3)

    page("高风险低估与极端集中", "TRAIN固定q90/q99；不按TEST重新挑极端", wide=True)
    image(RUN/"figures/formal_publication/06_underprediction_extreme_loss_concentration.png", "低估条件pred/RV_eff<0.5；损失集中为阈值子集QLIKE_Regret总和/季度总和。q90/q99嵌套，不能相加；全部九家族保留。", 265)
    summary = under[under.period.isin(QUARTERS) & under.family.isin(MAIN)]
    for q in QUARTERS:
        d = summary[(summary.period == q) & (summary.subset == "extreme_RV_train_q99")]
        text = "; ".join(str(r.family)+" N="+fmt(r.count)+", 低估="+fmt(r.underprediction_count)+"/"+fmt(r.underprediction_fraction) for r in d.itertuples())
        para(q+" q99："+text+"。", 8.1)
    pooled_under = under[(under.period == "POOLED") & (under.family == "R2")].set_index("subset")
    high_row = pooled_under.loc["high_RV_train_q90"]
    tail_row = pooled_under.loc["extreme_RV_train_q99"]
    para("R2 pooled q90 N=" + fmt(int(high_row["count"])) + "，低估" + fmt(int(high_row.underprediction_count)) + "/" + fmt(int(high_row["count"])) + "；q99 N=" + fmt(int(tail_row["count"])) + "，低估" + fmt(int(tail_row.underprediction_count)) + "/" + fmt(int(tail_row["count"])) + "。这些稀少高风险/极端样本诊断都不是校准安全证明。", 8.3)
    para("Q3 q99无极端事件时条件均值/低估率未定义；空集合损失或gain求和=0是合法描述，不概括为全部未定义。signed rawQLIKE或配对loss gain的极端占比可负或>1，零分母未定义，不是收益贡献百分比。完整表见loss_concentration.csv和extreme_paired_QLIKE_contribution.csv；无删极端。", 8.1)

    page("固定评分箱真实事件率", "TRAIN score quartile固定；R1/R2/B2至少保留", wide=True)
    image(RUN/"figures/formal_publication/05_train_fixed_score_event_frequency.png", "每箱真实事件数/N；边界由TRAIN生成，等号进右箱，重复quantile折叠，空箱未定义；没有用TEST重拟合概率。", 340)
    page("风险评分分布", "原始同EWMA评分轴；季度分面", wide=True)
    image(RUN/"figures/formal_publication/07_surprise_score_distribution.png", "Saved surprise_score经验累计分布。显示R2/R1/B2风险评分的位置和排序范围；不在TEST归一化、不把分数解释为概率。底表07_surprise_score_distribution.csv。", 340)

    page("Surprise 相关与恒等式")
    table(aux["surprise_correlations"][["period","family","count","spearman","partial_correlation"]], size=8.1, digits=7)
    para("Spearman是观察surprise与预测surprise score的平均秩相关；EWMA自身score恒定，相关未定义。partial correlation是两者分别对截距和log(EWMA+eps)做OLS后残差的Pearson相关，沿用第二轮OLS残差Pearson的方法；不是partial Spearman。第二轮控制量为log(EWMA)，本次控制log(EWMA+eps)，epsilon处理差异透明保留，不宣称计算逐字等同。", 8.6)
    para("同epsilon：observed surprise-score=log((RV_raw+eps)/(pred+eps))，故surprise MSE与additive-epsilon logRV MSE恒等。不是第二个独立证据；相关性辅助，不替代登记AUROC差/CI。", 8.6)

    page("历史 EWMA 状态诊断", "TRAIN历史EWMA quartile封存；无最优状态筛选", wide=True)
    image(state_plot, "固定历史状态中mean QLIKE Regret，仅描述R1/R2/B2；完整九家族、pooled/季度、各箱N与边界见historical_EWMA_quartile_losses.csv及MD附录。", 325)
    para("状态是已知历史EWMA，而非未来标签。所有状态箱保留；不按已见测试loss选择可交易区间或提出新过滤器。空箱均值未定义。", 9)

    page("正式审计与源码闭合")
    table([["官方5m", str(data_audit["rows"])+"行 / "+str(data_audit["raw_pages"])+"页，raw/canonical逐字段一致"], ["每个标签", "49成交价格与48时间戳，独立逐段log收益平方复算"], ["全部主要draw", "R1/R2/B2×两季×7/3/14日，正负pairwise矩阵复核每draw与CI"], ["正式训练", str(fit_audit["candidate_fits"])+"候选 + "+str(fit_audit["determinism_refits"])+"预登记重放；与preview所选模型逐位一致"], ["未来扰动因果", "首/中/尾机会："+", ".join(causality["ids"])], ["冻结编码", "四变体首/中/尾单样本、batch、重复检查及推理前后权重state SHA"], ["终态", "CONSUMED；独立数字审查PASS；禁止科学重跑"]], ["审计范围", "保存证据"], size=8.5, widths=[112, doc.width-112])
    table([[v, e["rows"], e["repeat_max_abs"], e["single_batch_max_abs"]] for v,e in encoding.items()], ["变体", "正式N", "重复最大差", "单/批最大差"], size=8.2, digits=8)
    para("independent_review/formal_numeric.json保留全部检查，不只最优结果；features/holdout_*_audit.json含model/tokenizer revision及state SHA。provenance/formal_source_manifest.json保存源码SHA、git commit与dirty状态。科学源码运行结束前重新闭合bundle。", 8.6)
    para("保存正式结果及出版辅助/图表另经独立只读复核：formal_report_review.json，" + str(len(report_review["checks"])) + "项检查PASS。该数字复核不包括本次PDF视觉验收，逐页渲染仍须单独检查。", 8.6)

    page("工程异常与真实授权")
    para("真实后续用户消息完整授权正式一次执行，并绑定协议与sealed_bundle SHA。先核验授权记录，再唯一execution_claim，随后同28训练和preview identity，通过后才打开test读取/下载。")
    para("Windows PowerShell第一次写授权metadata时Unicode编码丢失，发生在claim之前；错误记录保留于authorization/rejected_encoding_before_claim.json。已纠正metadata并保留claim前快照，没有先读holdout，没有重试科学评估。")
    table([["审批source", approval.get("source", "见原记录")], ["审批protocol", approval.get("protocol_sha256")], ["claim UTC", claim.get("created_at_utc")], ["终态 UTC", terminal.get("at_utc")], ["错误metadata记录", "authorization/rejected_encoding_before_claim.json"], ["正式fit事件总条数", len(events)]], ["记录", "证据"], size=8.3, widths=[139, doc.width-139])
    para("Phase A已保留三类拟合前工程异常：RangeIndex误判role overlap；旧M0 verifier拒绝新run路径；NPZ object ID被禁pickle读取拒绝。修复分别为ID索引、引用已验收历史证据、Unicode metadata v2且矩阵逐位不变。")
    para("原九suite中E00R当前stage断言失败原样保留，归档阶段配置按原断言重验；M0原路径保护不弱化。脚本或报告成功不替代研究验收。研究执行错误应INVALIDATED；单纯出版失败只补出版，绝不重跑科学。", 9)

    page("证据边界与容量限制")
    para("官方HF历史commit将当前相同LFS权重hash绑定至2025-06-30公开日期，支持权重在本轮保留区间之前公开；没有checkpoint-specific训练截止证明。早期DEV不能全部称严格事前样本外。")
    model_meta = encoding["pretrained"]["model"]
    para("正式推理审计中固定model revision：" + model_meta["model_revision"] + "；tokenizer revision：" + model_meta["tokenizer_revision"] + "。身份直接读取features/holdout_pretrained_audit.json的model元数据，未从结果反推。", 8.8)
    para("线性头TRAIN处理和优化float64；LightGBM原生Dataset训练标签float32，转换误差保留审计，原单位train/validation指标仍使用原始float64标签。Gamma log-link模型输出正均值，不重复exp。")
    para("B2与R1共享33列，但树容量、早停和正则语义不与线性头声称匹配。树四结构的验证选择自由度透明保留。所选模型和所有候选、失败、运行事件均可追溯。")
    para("官方raw数据覆盖和一致性不能扩大为交易所内部账本或每条历史事前可用性的证明；5m开/收成交价不是边界tick、标记价或订单簿。")
    para("多项随机、AP、q90、相关、校准、状态和极端诊断存在多重比较辅助边界；没有从它们新增成功路径。未做交易回测、收益筛选或安全仓位评估，不给风险控制比例建议。")
    para("无论主状态如何，当前有限实验已经消费独立区间，不能据此重挑模型再复用本轮测试。", 10, True)

    page("阶段完结与复现证据")
    para("正式状态："+status+"；生命周期：CONSUMED。"+decision, 11, True)
    para("完整来源链：data_5m/raw和manifest/audit → labels/holdout.csv → features/holdout_*与causality → models/formal/全部候选与selector → predictions/holdout.csv → metrics/formal_result.json与bootstrap draw/calendar → diagnostics/holdout → 独立formal_numeric → 出版辅助与图表。")
    para("出版prediction_only与surprise_scores只含预测/评分及预知日程元数据；未来RV、事件与标签严格分离到labels/publication_future_risk_labels.csv，禁止入特征。publication_sample_identities.csv另保留样本身份。原混合predictions/holdout.csv继续保留追溯；分离导出全部输入/输出SHA已核对，formal_separated_exports与独立formal_auxiliary_review均PASS。", 8.8)
    para("协议SHA："+PROTOCOL+"\nsealed_bundle文件SHA："+sources["preflight/sealed_bundle.json"]+"\nformal_result文件SHA："+sources["metrics/formal_result.json"], 8.2)
    para("v1 PDF/MD及其原始出处不覆盖。v2仅展开已生成证据：本PDF、同名MD、figures/formal_publication、figures/formal_report_v2、metrics/publication_auxiliary与formal_pdf_generation_v2.json。原数据、模型、失败历史继续保留。")
    para("MD附录完整刊印四张诊断CSV、辅助表、selector/审计与正式JSON，所有数字由round_trip读取或保存JSON派生；5000draw完整表通过路径与SHA查询，不复制成不可读PDF表。")
    para("不自动启动第四轮、骨干/tokenizer训练、微调/解冻、原生生成、git push、部署或实盘。新研究问题、独立数据及预算需未来另行登记。", 10, True)
    doc.end()

    # Complete machine-readable tables in the companion MD; no unpublished raw market reads.
    mdlines.extend(["", "## 附录 A：完整诊断与辅助数值底表", "", "以下诊断为辅助解释；保留空值与所有家族，不新增成功标准。", ""])
    for relative, frame in tables.items():
        if relative.endswith("row_metrics_and_timeline.csv") or relative.endswith("calendar_rolling_7day.csv"):
            mdlines.extend(["### " + relative, "", "逐机会/全日历大表保留原CSV，不重复打印；SHA256：`" + sources[relative] + "`。", ""])
        else:
            mdlines.extend(["### " + relative, "", markdown_table(frame), ""])
    mdlines.extend(["## 附录 B：完整正式结果", "", "```json", json.dumps(result, ensure_ascii=False, indent=2), "```", "",
                    "## 附录 C：selector、fit审计与生命周期", ""])
    for title, value in [("selected_models",selected), ("fit_audit",fit_audit), ("preview_identity",identity), ("numeric_review",numeric),
                         ("data_raw_csv_audit",data_audit), ("causality",causality), ("execution_claim",claim), ("rejected_encoding_before_claim",rejected), ("terminal",terminal),
                         ("formal_separated_exports", separated_exports), ("formal_auxiliary_review", auxiliary_review), ("formal_report_review", report_review)]:
        mdlines.extend(["### " + title, "", "```json", json.dumps(value, ensure_ascii=False, indent=2), "```", ""])
    mdlines.extend(["## 附录 D：输入来源与SHA256", "", markdown_table(pd.DataFrame(sorted(sources.items()), columns=["RUN相对路径", "SHA256"])), "",
                    "全部模型候选预测：models/formal/all_candidate_predictions.csv；fit事件：models/formal/fit_events.jsonl。",
                    "全部正式draw：bootstrap/draws_block7.csv、draws_block3.csv、draws_block14.csv；配对日期权重：对应multiplicities_block*.npz。",
                    "辅助draw：metrics/publication_auxiliary/high_rv_draws_block7.csv、block3.csv、block14.csv；无新增随机抽样。"])
    with md.open("x", encoding="utf-8") as handle: handle.write("\n".join(mdlines)+"\n")

    # Render every page with the environment-verified PyMuPDF fallback.
    fitz = pdfbase.fitz
    document = fitz.open(pdf)
    image_paths = []
    page_bounds = []
    for i, p in enumerate(document):
        target = render_dir / f"page_{i+1:02d}.png"
        p.get_pixmap(matrix=fitz.Matrix(1.4,1.4), alpha=False).save(target)
        image_paths.append(target)
        page_bounds.append({"page": i+1, "width": p.rect.width, "height": p.rect.height, "text_blocks": len(p.get_text("blocks"))})
    from PIL import Image, ImageDraw
    tw, th, cols = 360, 510, 4
    sheet = Image.new("RGB", (tw*cols, th*((len(image_paths)+cols-1)//cols)), "#eeeeee")
    for i, path in enumerate(image_paths):
        im = Image.open(path); im.thumbnail((tw-12,th-30))
        x, y = i%cols*tw, i//cols*th
        sheet.paste(im,(x+(tw-im.width)//2,y+23))
        ImageDraw.Draw(sheet).text((x+8,y+5),str(i+1),fill="black")
    sheet.save(render_dir/"contact_sheet.png")
    for relative, digest in sources.items():
        if sha(RUN/relative) != digest:
            raise ValueError("Publication input changed: " + relative)
    manifest = {"status":"GENERATED_PENDING_ROOT_VISUAL_AND_NUMERIC_REVIEW", "experiment_id":"FROZEN_RISK_03_v1", "terminal_state":"CONSUMED",
                "primary_status":status, "created_at_utc":datetime.now(timezone.utc).isoformat(), "protocol_sha256":PROTOCOL,
                "source_code_sha256":sha(__file__), "input_files_sha256":sources, "pages":doc.number, "page_outline":page_meta,
                "render_code_dependencies_sha256": {str(Path(p).relative_to(ROOT)).replace("\\", "/"): sha(p) for p in [ROOT/"research/frozen/experiment_03/report.py", ROOT/"research/frozen/build_overall_pdf.py"]},
                "pdf_path":str(pdf.relative_to(ROOT)), "pdf_sha256":sha(pdf), "md_path":str(md.relative_to(ROOT)), "md_sha256":sha(md),
                "render_method":"PyMuPDF fallback; bundled Windows environment previously verified", "render_directory":str(render_dir.relative_to(ROOT)),
                "render_sha256":{p.name:sha(p) for p in sorted(render_dir.iterdir())}, "page_bounds":page_bounds,
                "extra_figures":extra_plots, "extra_figure_sha256":{p.name:sha(p) for p in sorted(figures.iterdir())},
                "all_layout_operations_require_y_at_least":49, "round_trip_csv_reads":True, "no_refit":True, "no_bootstrap_redraw":True,
                "no_new_market_inputs":True, "no_new_hypothesis_tests_in_report":True, "no_new_success_criterion":True,
                "original_v1_preserved":True, "visual_QA_required":True, "independent_numeric_review_required":True}
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest,handle,ensure_ascii=False,indent=2,allow_nan=False);handle.write("\n")
    print(json.dumps({"status":manifest["status"], "pdf":str(pdf), "md":str(md), "pages":doc.number},ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish-generated-only", action="store_true", required=True)
    parser.parse_args()
    publish()
