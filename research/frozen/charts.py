"""Plot verified development replay curves only; never fit, backtest or inspect holdout."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/baselines"))
from plot_baseline_time_series import OLD_MODES, NEW_MODES, LABELS as OLD_LABELS

OLD = "E00_20261008_phase4_v1"
ORDINARY = "E00R_20261008_phase4_v1"
GATES = ("rank50", "rank25", "rank75", "economic")
SEEDS = (17, 29, 43)
FROZEN_MODES = tuple(f"{family}_{gate}" for family in
                     ("ridge_pretrained", *(f"ridge_random_s{s}" for s in SEEDS))
                     for gate in GATES)
COLORS = ("#555555", "#4477AA", "#66AABB", "#CC3311", "#228833", "#AA3377", "#DDAA22")
LABELS = dict(OLD_LABELS)
for gate in GATES:
    suffix = "经济门控" if gate == "economic" else gate
    for count in (13, 19):
        LABELS[f"ridge{count}_{gate}"] = f"普通 Ridge {count} / {suffix}"
    LABELS[f"ridge_pretrained_{gate}"] = f"预训练 + 普通19 / {suffix}"
    for seed in SEEDS:
        LABELS[f"ridge_random_s{seed}_{gate}"] = f"随机骨干 s{seed} + 普通19 / {suffix}"
LABELS.update(random_q50_mean="随机参与 / 3种子均值", periodic_mean="周期参与 / 2相位均值")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def configuration(registry: dict, key: str = "configuration") -> dict:
    record = registry[key]
    text = record["full_text"]
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != record["sha256"]:
        raise ValueError("Registered configuration text/hash mismatch")
    return yaml.safe_load(text)


def panel(out, name, title, cost, specifications, curves, folds, frozen_id):
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.subplots_adjust(left=.075, right=.97, top=.85, bottom=.335, hspace=.40, wspace=.23)
    endpoints = []
    for ax, (fold, (start, end)) in zip(axes.flat, folds.items()):
        values = []
        for i, (run, mode) in enumerate(specifications):
            frame = curves[(run, fold, mode, cost)]
            # Append the last realized value at the quarter boundary for display only.
            # This holds equity; it does not add a return, trade or interpolated PnL.
            at = list(frame["at"]) + [end]
            y = list(frame.cumulative_return_pct) + [frame.cumulative_return_pct.iloc[-1]]
            ax.plot(at, y, drawstyle="steps-post", color=COLORS[i], lw=1.6,
                    linestyle="--" if "mean" in mode else "-", label=LABELS[mode])
            ax.scatter(end, y[-1], color=COLORS[i], s=22, clip_on=False, zorder=4)
            values.append(float(y[-1]))
        endpoints.append(values)
        ax.set_xlim(start, end)
        ax.set_title(f"{fold} · {start.year}Q{start.quarter}", fontsize=12)
        ax.set_ylabel("累计净收益（%）")
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m", tz=timezone.utc))
        ax.tick_params(labelsize=9)
        ax.grid(alpha=.22)
        ax.axhline(0, color="#777777", lw=.6)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.52, .935),
               ncol=3, frameon=False, fontsize=9)
    rate = 7 if cost == "base" else 14
    fig.suptitle(f"{title} · 每边 {rate} bps费用代理", fontsize=16, y=.982)
    fig.text(.075, .307, "真实UTC日期；各季度独立以10,000 USDT重置；仅闭仓时更新，阶梯向前保持，无浮盈插值。", fontsize=10)
    table_ax = fig.add_axes([.075, .092, .895, .195])
    table_ax.axis("off")
    table = table_ax.table(
        cellText=[[LABELS[mode]] + [f"{endpoints[f][i]:+.2f}%" for f in range(4)]
                  for i, (_, mode) in enumerate(specifications)],
        colLabels=["季度末累计净收益"] + list(folds), loc="center", cellLoc="center",
        colWidths=[.39, .1525, .1525, .1525, .1525])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.3)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        if row == 0:
            cell.set_facecolor("#EEF1F5")
        elif col == 0:
            cell.get_text().set_color(COLORS[row - 1])
    note = ("买入持有仅期末平仓：平线不代表持仓期间没有价格波动。" if name == "rule_baselines"
            else "全部随机骨干种子保留；虚线参与均值为联合闭仓事件网格上的算术均值，非单一可执行组合。")
    fig.text(.075, .049, note, fontsize=9)
    fig.text(.075, .023, f"来源：E00 / E00R / {frozen_id} 官方闭仓权益CSV；实际资金另计。开发期历史研究；最终holdout未评估。", fontsize=8, color="#555555")
    for ext in ("png", "svg"):
        fig.savefig(out / f"{name}_equity_{cost}.{ext}", dpi=160)
    plt.close(fig)


def run(output: Path) -> Path:
    output = output.resolve()
    runs_root = (ROOT / "research/runs").resolve()
    if not output.is_relative_to(runs_root) or output == runs_root:
        raise ValueError("output-dir must be an existing run under research/runs")
    frozen_id = output.name
    destination = output / "presentation"
    if destination.exists():
        raise ValueError("Refusing to overwrite existing presentation directory")
    registries = {}
    for run_id in (OLD, ORDINARY, frozen_id):
        path = ROOT / "research/registry" / f"{run_id}.json"
        registry = json.loads(path.read_text(encoding="utf-8"))
        if run_id != frozen_id and not registry["status"].startswith("completed"):
            raise ValueError("Ordinary replay must be completed and accepted")
        if run_id == frozen_id and registry["status"] not in (
                "engine_verified_pending_analysis", "completed_exploratory_development", "completed"):
            raise ValueError("Frozen replay has not passed engine verification")
        configuration(registry)
        registries[run_id] = registry
    cfg = configuration(registries[frozen_id], "parent_configuration")
    if cfg["baselines"]["initial_portfolio_USDT"] != 10000:
        raise ValueError("Independent wallet must be 10000 USDT")
    if cfg["evaluation_readiness"]["final_holdout_evaluation_allowed"] is not False:
        raise ValueError("Final holdout must remain sealed")
    folds = {f["id"]: tuple(pd.to_datetime(f["test"], utc=True))
             for f in cfg["splits"]["walk_forward_dates"]}
    if list(folds) != ["WF01", "WF02", "WF03", "WF04"]:
        raise ValueError("Only the four registered development folds are allowed")
    holdout_start = pd.Timestamp(cfg["splits"]["final_holdout"][0])
    if any(end > holdout_start for _, end in folds.values()):
        raise ValueError("Development fold crosses final holdout")
    old_cfg = configuration(registries[OLD])
    old_bounds = {f["id"]: tuple(pd.to_datetime(f["test"], utc=True))
                  for f in old_cfg["splits"]["walk_forward_dates"]}
    if old_bounds != folds:
        raise ValueError("Old and frozen fold boundaries differ")

    # Verify all registry inputs before writing presentation or reading performance.
    inputs = []
    scopes = {OLD: OLD_MODES, ORDINARY: NEW_MODES, frozen_id: FROZEN_MODES}
    for run_id, modes in scopes.items():
        kind = "engine_hashes" if run_id == frozen_id else "final_hashes"
        hashes = {k.replace("\\", "/"): v for k, v in registries[run_id]["artifacts"][kind].items()}
        for rel in ["metrics.csv"] + [f"engine/{fold}/{mode}/{cost}/official_closed_trade_equity.csv"
                for fold in folds for mode in modes for cost in ("base", "stress")]:
            path = ROOT / "research/runs" / run_id / rel
            actual = sha(path)
            if hashes.get(rel) != actual:
                raise ValueError(f"Registered {kind} mismatch: {path}")
            inputs.append({"run": run_id, "path": rel, "sha256": actual, "registry_hash_kind": kind})
    destination.mkdir()
    provenance = destination / "provenance"
    provenance.mkdir()
    for run_id, registry in registries.items():
        shutil.copyfile(ROOT / "research/registry" / f"{run_id}.json", provenance / f"{run_id}.json")
        for key in ("configuration", "parent_configuration"):
            if key in registry:
                configuration(registry, key)
                (provenance / f"{run_id}_{key}.yaml").write_text(registry[key]["full_text"], encoding="utf-8")
    shutil.copyfile(Path(__file__), provenance / "charts.py")
    shutil.copyfile(ROOT / "research/baselines/plot_baseline_time_series.py",
                    provenance / "plot_baseline_time_series.py")
    write_json(provenance / "manifest.json", {"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "verified_development_closed_trade_presentation", "holdout_accessed": False,
        "registered_inputs": inputs, "current_stage_configuration_used_for_old_hash_acceptance": False,
        "copies_sha256": {p.name: sha(p) for p in provenance.iterdir() if p.is_file()}})
    curves, raw, audits = {}, [], []
    for run_id, modes in scopes.items():
        run_path = ROOT / "research/runs" / run_id
        metrics = pd.read_csv(run_path / "metrics.csv")
        keys = {(fold, mode, cost) for fold in folds for mode in modes for cost in ("base", "stress")}
        if len(metrics) != len(keys) or set(metrics[["fold_id", "mode", "cost"]].itertuples(index=False, name=None)) != keys:
            raise ValueError(f"Unexpected/duplicate metrics scope: {run_id}")
        for row in metrics.itertuples(index=False):
            source = run_path / f"engine/{row.fold_id}/{row.mode}/{row.cost}/official_closed_trade_equity.csv"
            frame = pd.read_csv(source)
            frame["at"] = pd.to_datetime(frame["at"], utc=True)
            start, end = folds[row.fold_id]
            if frame.empty or not frame["at"].is_monotonic_increasing or frame["at"].isna().any() or not np.isfinite(frame.closed_trade_equity_USDT).all():
                raise ValueError(f"Invalid official curve: {source}")
            if not frame["at"].between(start, end, inclusive="left").all() or frame["at"].iloc[0] != start:
                raise ValueError("Curve outside its development quarter")
            if frame.groupby("at").closed_trade_equity_USDT.nunique().gt(1).any():
                raise ValueError("Conflicting duplicate timestamp equity")
            duplicates = int(frame["at"].duplicated().sum())
            frame = frame.drop_duplicates("at", keep="last").copy()
            if abs(frame.closed_trade_equity_USDT.iloc[0] - 10000) > 1e-8:
                raise ValueError("Quarter does not reset at 10000 USDT")
            frame["cumulative_return_pct"] = (frame.closed_trade_equity_USDT / 10000 - 1) * 100
            if not np.isfinite(row.net_return_pct):
                raise ValueError("Nonfinite official return metric")
            error = abs(frame.cumulative_return_pct.iloc[-1] - row.net_return_pct)
            if error > 1e-8:
                raise ValueError("Official endpoint differs from metrics")
            curves[(run_id, row.fold_id, row.mode, row.cost)] = frame
            raw.append(frame.assign(run=run_id, fold_id=row.fold_id, mode=row.mode, cost=row.cost, curve_kind="official_closed_trade_equity"))
            audits.append({"run": run_id, "fold_id": row.fold_id, "mode": row.mode, "cost": row.cost,
                "net_return_pct": row.net_return_pct, "trades": row.trades, "points": len(frame),
                "endpoint_error_percentage_points": error, "identical_duplicate_timestamps_removed": duplicates})
    pd.concat(raw, ignore_index=True).to_csv(destination / "official_curves.csv", index=False)
    pd.DataFrame(audits).to_csv(destination / "endpoints_and_checks.csv", index=False)
    derived = []
    for fold in folds:
        for cost in ("base", "stress"):
            for name, members in (("random_q50_mean", [f"random_q50_s{s}" for s in SEEDS]),
                                  ("periodic_mean", ["periodic_offset0", "periodic_offset1"])):
                series = [curves[(ORDINARY, fold, mode, cost)].set_index("at").closed_trade_equity_USDT for mode in members]
                union = series[0].index
                for item in series[1:]:
                    union = union.union(item.index)
                aligned = pd.concat([item.reindex(union).ffill() for item in series], axis=1)
                if aligned.isna().any().any():
                    raise ValueError("Descriptive control mean lacks initial realized equity")
                frame = pd.DataFrame({"at": union, "closed_trade_equity_USDT": aligned.mean(axis=1).to_numpy()})
                frame["cumulative_return_pct"] = (frame.closed_trade_equity_USDT / 10000 - 1) * 100
                if abs(frame.cumulative_return_pct.iloc[-1] - np.mean([(s.iloc[-1] / 10000 - 1) * 100 for s in series])) > 1e-8:
                    raise ValueError("Control mean endpoint mismatch")
                curves[(ORDINARY, fold, name, cost)] = frame
                derived.append(frame.assign(run=ORDINARY, fold_id=fold, mode=name, cost=cost,
                    curve_kind="descriptive_arithmetic_mean_not_executable", members="|".join(members)))
    pd.concat(derived, ignore_index=True).to_csv(destination / "derived_mean_curves.csv", index=False)
    font = Path("C:/Windows/Fonts/msyh.ttc")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font)).get_name(),
                         "axes.unicode_minus": False, "svg.fonttype": "none"})
    for cost in ("base", "stress"):
        for gate in GATES:
            specifications = [(OLD, "cash"), (ORDINARY, f"ridge19_{gate}"),
                (ORDINARY, f"ridge13_{gate}"), (frozen_id, f"ridge_pretrained_{gate}")]
            specifications += [(frozen_id, f"ridge_random_s{s}_{gate}") for s in SEEDS]
            title = "主信息门控 rank50" if gate == "rank50" else ("独立经济门控" if gate == "economic" else f"预定敏感性 {gate}")
            panel(destination, gate, title, cost, specifications, curves, folds, frozen_id)
        panel(destination, "rule_baselines", "固定规则与旧门控基线", cost,
              [(OLD, mode) for mode in OLD_MODES], curves, folds, frozen_id)
        panel(destination, "participation_controls", "预训练信息门控与无信息参与对照", cost,
              [(OLD, "cash"), (OLD, "ordinary_features_gate"), (ORDINARY, "ridge19_rank50"),
               (ORDINARY, "ridge13_rank50"), (frozen_id, "ridge_pretrained_rank50"),
               (ORDINARY, "random_q50_mean"), (ORDINARY, "periodic_mean")], curves, folds, frozen_id)
    report = {"status": "passed", "holdout_accessed": False, "training_or_backtesting_performed": False,
        "official_curves": len(raw), "derived_mean_curves": len(derived), "registered_input_files": len(inputs),
        "independent_quarter_wallet_USDT": 10000, "quarters": list(folds),
        "max_endpoint_error_percentage_points": max(a["endpoint_error_percentage_points"] for a in audits),
        "endpoint_tolerance_percentage_points": 1e-8,
        "plot_end_boundary": "display-only forward hold of final realized wallet; no new event or PnL",
        "mean_method": "same-fold union close-event grid, forward hold realized equity, arithmetic wallet mean",
        "limitations": ["closed-trade equity only; no unrealized PnL or hourly mark-to-market",
            "independent quarterly resets; no continuous annual compounding",
            "control means are descriptive and not a single executable portfolio",
            "7/14 bps per side are expense proxies; signed actual funding also included",
            "seen development quarters are historical exploratory research; checkpoint cutoff unproven"],
        "outputs_sha256": {p.name: sha(p) for p in destination.iterdir() if p.is_file()}}
    write_json(destination / "report.json", report)
    (destination / "report.md").write_text("# 冻结表征开发期曲线\n\n全部曲线来自registry核验的官方闭仓权益，季度独立重置为10,000 USDT。主rank50、经济门控、预定rank25/rank75敏感性及旧规则/无信息参与对照分别呈现，两种成本均保留全部随机骨干种子。\n\nCSV保存所有官方曲线、派生对照均值和季度端点；PNG/SVG中的季度末表与metrics一致。曲线只在闭仓事件变化，不包含浮盈。派生均值不代表可执行单组合。未读取holdout，未训练或回测。\n", encoding="utf-8")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Verified frozen run under research/runs; creates presentation/ and refuses overwrite")
    arguments = parser.parse_args()
    print(run(arguments.output_dir))
