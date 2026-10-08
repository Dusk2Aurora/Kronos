"""Render audited development-only official closed-trade equity; never fit or backtest."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
RUN_IDS = ("E00_20261008_phase4_v1", "E00R_20261008_phase4_v1")
OLD_MODES = ("cash", "buy_hold", "fixed_momentum", "constant_half_exposure", "vol_target", "ordinary_features_gate")
NEW_MODES = ("constant_economic", "periodic_offset0", "periodic_offset1", "random_q50_s17", "random_q50_s29", "random_q50_s43", "ridge13_economic", "ridge13_rank25", "ridge13_rank50", "ridge13_rank75", "ridge19_economic", "ridge19_rank25", "ridge19_rank50", "ridge19_rank75")
LABELS = {
    "cash": "现金", "buy_hold": "买入持有", "fixed_momentum": "固定动量",
    "constant_half_exposure": "半仓", "vol_target": "波动率目标",
    "ordinary_features_gate": "旧 logistic 门控", "ridge13_rank50": "Ridge 13 / rank50",
    "ridge19_rank50": "Ridge 19 / rank50", "ridge19_economic": "Ridge 19 / 经济门控",
    "random_q50_mean": "随机 q50 / 3种子均值", "periodic_mean": "周期 / 2相位均值",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_group(out: Path, group: str, cost: str, specifications: list, curves: dict, folds: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.subplots_adjust(left=.075, right=.97, top=.855, bottom=.335, hspace=.40, wspace=.23)
    colors = ["#454545", "#DDAA22", "#CC6677", "#4477AA", "#228833", "#AA3377", "#66AABB"]
    ends = []
    for ax, (fold, bounds) in zip(axes.flat, folds.items()):
        end_values = []
        for i, (run, mode) in enumerate(specifications):
            curve = curves[(run, fold, mode, cost)]
            ax.plot(curve["at"], curve.cumulative_return_pct, drawstyle="steps-post", label=LABELS[mode], color=colors[i], lw=1.6, linestyle="--" if "mean" in mode else "-")
            ax.scatter(curve["at"].iloc[-1], curve.cumulative_return_pct.iloc[-1], s=24, color=colors[i], clip_on=False, zorder=4)
            end_values.append(float(curve.cumulative_return_pct.iloc[-1]))
        ends.append(end_values)
        start, end = bounds
        ax.set_xlim(start, end - pd.Timedelta(hours=1))
        ax.set_title(f"{fold} · {start.year}Q{start.quarter}", fontsize=12)
        ax.set_ylabel("累计净收益（%）")
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m", tz=timezone.utc))
        ax.grid(alpha=.22)
        ax.axhline(0, color="#777777", lw=.6)
        ax.tick_params(labelsize=9)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.52, .935), ncol=4, frameon=False, fontsize=10)
    title = "原六策略" if group == "rule_baselines" else "普通预测头与无信息对照"
    rate = 7 if cost == "base" else 14
    fig.suptitle(f"{title}：开发季度已平仓权益时间线 · 每边 {rate} bps费用代理", fontsize=16, y=.982)
    fig.text(.075, .307, "真实UTC日期；各季度独立以10,000 USDT重置；仅闭仓时更新，阶梯向前保持，无浮盈插值。", fontsize=10)
    table_ax = fig.add_axes([.075, .09, .895, .195])
    table_ax.axis("off")
    table = table_ax.table(cellText=[[LABELS[mode]] + [f"{ends[f][i]:+.2f}%" for f in range(4)] for i, (_, mode) in enumerate(specifications)], colLabels=["季度末累计净收益"] + list(folds), loc="center", cellLoc="center", colWidths=[.39, .1525, .1525, .1525, .1525])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        if row == 0:
            cell.set_facecolor("#EEF1F5")
        elif col == 0:
            cell.get_text().set_color(colors[row - 1])
    note = "买入持有仅期末平仓：大部分时间的平线不代表持仓期间没有价格波动。" if group == "rule_baselines" else "虚线均值是同季度联合事件网格向前保持后的算术均值，非单一实际可执行组合。"
    fig.text(.075, .049, note, fontsize=10)
    fig.text(.075, .022, "来源：E00_20261008_phase4_v1 / E00R_20261008_phase4_v1 官方闭仓权益CSV；含实际资金费用。最终holdout未读取。", fontsize=8, color="#555555")
    for extension in ("png", "svg"):
        fig.savefig(out / f"{group}_equity_{cost}.{extension}", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="New or empty directory under research/runs")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    runs_root = (ROOT / "research/runs").resolve()
    if not out.is_relative_to(runs_root) or out == runs_root:
        raise ValueError("output-dir must be a child of research/runs")
    if out.exists() and any(out.iterdir()):
        raise ValueError("Refusing to overwrite a nonempty output directory")
    out.mkdir(parents=True, exist_ok=True)
    prov = out / "provenance"
    prov.mkdir()
    cfg_path = ROOT / "research/configs/initial_experiment.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if cfg["baselines"]["initial_portfolio_USDT"] != 10000:
        raise ValueError("Expected independent 10000 USDT wallets")
    folds = {f["id"]: tuple(pd.to_datetime(f["test"], utc=True)) for f in cfg["splits"]["walk_forward_dates"]}
    if list(folds) != ["WF01", "WF02", "WF03", "WF04"]:
        raise ValueError("Only four development folds are allowed")
    holdout_start = pd.Timestamp(cfg["splits"]["final_holdout"][0])
    if any(end > holdout_start for _, end in folds.values()):
        raise ValueError("Development folds overlap holdout")
    snapshots = [Path(__file__).resolve(), cfg_path, ROOT / "research/configs/baseline_revision_v2.yaml", ROOT / "research/initialization/TODO.md"]
    copies = {}
    for path in snapshots:
        target = prov / path.name
        shutil.copyfile(path, target)
        copies[str(path.relative_to(ROOT))] = {"copy": str(target.relative_to(out)), "sha256": digest(target)}
    inputs, registries = [], {}
    # Save provenance before loading performance CSVs or computing curves.
    for run_id in RUN_IDS:
        path = ROOT / "research/registry" / f"{run_id}.json"
        registry = json.loads(path.read_text(encoding="utf-8"))
        if not registry["status"].startswith("completed"):
            raise ValueError("Only completed, accepted development runs allowed")
        registries[run_id] = registry
        shutil.copyfile(path, prov / path.name)
        copies[str(path.relative_to(ROOT))] = {"copy": f"provenance/{path.name}", "sha256": digest(path)}
        full_text = registry["configuration"]["full_text"]
        if hashlib.sha256(full_text.encode("utf-8")).hexdigest() != registry["configuration"]["sha256"]:
            raise ValueError("Registry configuration text/hash mismatch")
        (prov / f"{run_id}_registered_config.yaml").write_text(full_text, encoding="utf-8")
        hashes = {key.replace("\\", "/"): value for key, value in registry["artifacts"]["final_hashes"].items()}
        modes = OLD_MODES if run_id == RUN_IDS[0] else NEW_MODES
        relpaths = ["metrics.csv"] + [f"engine/{fold}/{mode}/{cost}/official_closed_trade_equity.csv" for fold in folds for mode in modes for cost in ("base", "stress")]
        for rel in relpaths:
            path = ROOT / "research/runs" / run_id / rel
            actual = digest(path)
            if hashes.get(rel) != actual:
                raise ValueError(f"Final registered hash mismatch: {path}")
            inputs.append({"run": run_id, "path": rel, "sha256": actual, "registry_final_hash_verified": True})
    write_json(prov / "manifest.json", {"created_at_utc": datetime.now(timezone.utc).isoformat(), "scope": "read_only_development_plot", "no_holdout": True, "independent_fold_wallet_USDT": 10000, "copies": copies, "registered_config_sha256": {r: registries[r]["configuration"]["sha256"] for r in RUN_IDS}, "inputs": inputs})
    curves, audits, raw_tables = {}, [], []
    for run_id in RUN_IDS:
        run = ROOT / "research/runs" / run_id
        metrics = pd.read_csv(run / "metrics.csv")
        modes = OLD_MODES if run_id == RUN_IDS[0] else NEW_MODES
        expected_keys = {(fold, mode, cost) for fold in folds for mode in modes for cost in ("base", "stress")}
        actual_keys = set(metrics[["fold_id", "mode", "cost"]].itertuples(index=False, name=None))
        if actual_keys != expected_keys or len(metrics) != len(expected_keys):
            raise ValueError("Metrics scope/uniqueness mismatch")
        for row in metrics.itertuples(index=False):
            path = run / f"engine/{row.fold_id}/{row.mode}/{row.cost}/official_closed_trade_equity.csv"
            frame = pd.read_csv(path)
            frame["at"] = pd.to_datetime(frame["at"], utc=True)
            values = frame.closed_trade_equity_USDT
            start, end = folds[row.fold_id]
            if not frame["at"].is_monotonic_increasing or frame["at"].isna().any() or not np.isfinite(values).all():
                raise ValueError(f"Invalid equity curve: {path}")
            if not frame["at"].between(start, end, inclusive="left").all() or frame["at"].iloc[0] != start:
                raise ValueError("Curve is outside its independent development fold")
            if frame.groupby("at").closed_trade_equity_USDT.nunique().gt(1).any():
                raise ValueError("Conflicting values at duplicate timestamps")
            duplicates = int(frame["at"].duplicated().sum())
            frame = frame.drop_duplicates("at", keep="last").copy()
            if abs(frame.closed_trade_equity_USDT.iloc[0] - 10000) > 1e-8:
                raise ValueError("Curve must start at 10000")
            frame["cumulative_return_pct"] = (frame.closed_trade_equity_USDT / 10000 - 1) * 100
            error = abs(frame.cumulative_return_pct.iloc[-1] - row.net_return_pct)
            if error > 1e-8:
                raise ValueError("Endpoint mismatch with official metrics")
            key = (run_id, row.fold_id, row.mode, row.cost)
            curves[key] = frame
            raw_tables.append(frame.assign(run=run_id, fold_id=row.fold_id, mode=row.mode, cost=row.cost, curve_kind="official_closed_trade_equity"))
            audits.append({"run": run_id, "fold_id": row.fold_id, "mode": row.mode, "cost": row.cost, "curve_kind": "official", "points": len(frame), "identical_duplicate_timestamps_removed": duplicates, "endpoint_error_pct": error, "net_return_pct": row.net_return_pct, "trades": row.trades})
    pd.concat(raw_tables, ignore_index=True).to_csv(out / "official_curves.csv", index=False)
    mean_tables = []
    for fold in folds:
        for cost in ("base", "stress"):
            for name, members in [("random_q50_mean", [f"random_q50_s{s}" for s in (17, 29, 43)]), ("periodic_mean", ["periodic_offset0", "periodic_offset1"])]:
                source_curves = [curves[(RUN_IDS[1], fold, mode, cost)] for mode in members]
                series = [c.set_index("at").closed_trade_equity_USDT for c in source_curves]
                # The union grid preserves observed close events; ffill only holds already realized equity.
                union = series[0].index
                for s in series[1:]:
                    union = union.union(s.index)
                values = pd.concat([s.reindex(union).ffill() for s in series], axis=1)
                if values.isna().any().any():
                    raise ValueError("Mean grid has missing initial equity")
                frame = pd.DataFrame({"at": union, "closed_trade_equity_USDT": values.mean(axis=1).to_numpy()})
                frame["cumulative_return_pct"] = (frame.closed_trade_equity_USDT / 10000 - 1) * 100
                expected = np.mean([c.cumulative_return_pct.iloc[-1] for c in source_curves])
                error = abs(frame.cumulative_return_pct.iloc[-1] - expected)
                if error > 1e-8 or abs(frame.closed_trade_equity_USDT.iloc[0] - 10000) > 1e-8:
                    raise ValueError("Derived mean endpoint/start mismatch")
                curves[(RUN_IDS[1], fold, name, cost)] = frame
                mean_tables.append(frame.assign(run=RUN_IDS[1], fold_id=fold, mode=name, cost=cost, curve_kind="derived_arithmetic_equity_mean", members="|".join(members)))
                audits.append({"run": RUN_IDS[1], "fold_id": fold, "mode": name, "cost": cost, "curve_kind": "derived_mean", "endpoint_error_pct": error, "net_return_pct": expected, "members": members})
    pd.concat(mean_tables, ignore_index=True).to_csv(out / "derived_mean_curves.csv", index=False)
    pd.DataFrame(audits).to_csv(out / "endpoints_and_checks.csv", index=False)
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if not font_path.exists():
        raise FileNotFoundError("Chinese font required")
    font_manager.fontManager.addfont(str(font_path))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font_path)).get_name(), "axes.unicode_minus": False, "svg.fonttype": "none"})
    for cost in ("base", "stress"):
        plot_group(out, "rule_baselines", cost, [(RUN_IDS[0], mode) for mode in OLD_MODES], curves, folds)
        plot_group(out, "model_baselines", cost, [(RUN_IDS[0], "cash"), (RUN_IDS[0], "ordinary_features_gate")] + [(RUN_IDS[1], mode) for mode in ("ridge13_rank50", "ridge19_rank50", "ridge19_economic", "random_q50_mean", "periodic_mean")], curves, folds)
    report = {"status": "passed", "scope": "four_development_folds_only", "holdout_accessed": False, "training_or_backtesting_performed": False, "official_curves": len(raw_tables), "derived_mean_curves": len(mean_tables), "verified_input_files": len(inputs), "start_wallet_USDT": 10000, "endpoint_tolerance_percentage_points": 1e-8, "max_endpoint_error_percentage_points": max(a["endpoint_error_pct"] for a in audits), "identical_duplicate_timestamps_removed": sum(a.get("identical_duplicate_timestamps_removed", 0) for a in audits), "timestamp_duplicate_policy": "reject conflicting values; retain last identical duplicate", "mean_method": "same-fold union of observed close-event timestamps, forward hold realized equity, arithmetic mean of wallets", "limitations": ["closed-trade equity only, no hourly mark-to-market or unrealized PnL", "buy_hold stays flat until quarter-end close; this does not imply no holding-period volatility", "independent quarterly wallets; no annual compounded curve", "control means are descriptive, not an actual single executable strategy", "7/14 bps per side are fee/slippage expense proxies; actual funding included", "exploratory development results do not establish information value or live profitability"], "outputs_sha256": {p.name: digest(p) for p in out.iterdir() if p.is_file()}}
    write_json(out / "report.json", report)
    (out / "report.md").write_text("# 开发期基线时间线\n\n只读复用已验收E00/E00R官方闭仓权益，160条真实曲线和16条派生均值全部验收通过；最终holdout未读取。\n\n每折独立重置10,000 USDT，累计收益=(闭仓权益/10,000-1)×100%。横轴为真实UTC日期，阶梯向前保持，无小时盯市、浮盈插值或跨季度复利。\n\n买入持有只在期末平仓，之前的平线不能说明持仓期间没有价格波动。随机3种子/周期2相位均值按同季度联合事件网格向前保持后求算术均值，只用于对照描述，非单一可执行组合。\n\n基准每边7 bps、压力14 bps，均为费用代理，实际资金另含在官方权益内。图下表为季度末累计净收益，CSV保留全部20个mode的真实曲线；均值单独记录。\n\n追溯：provenance/manifest.json包含输入登记hash核验和源/配置副本hash；report.json及endpoints_and_checks.csv包含端点、时间边界和重复时间戳检查。来源为research/runs/E00_20261008_phase4_v1与E00R_20261008_phase4_v1。\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
