"""Publication only: render saved RISK_03 formal evidence after CONSUMED.

No scientific modules, raw data, model fitting, test bin refits or new tests.
Run from the project root with the Kronos interpreter and --run PATH.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ["R1", "R2", "B2", "har", "persistence", "ewma", "random_s17", "random_s29", "random_s43"]
MAIN = ["R2", "R1", "B2"]
QUARTERS = ["2026Q2", "2026Q3"]
COLORS = dict(zip(FAMILIES, plt.get_cmap("tab10").colors))
plt.rcParams.update({"font.family": ["Microsoft YaHei", "DejaVu Sans"], "font.size": 10,
                     "axes.grid": True, "grid.alpha": .2})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render(run):
    run = Path(run).resolve()
    sources = {}

    def read_json(relative):
        p = run / relative
        sources[relative] = sha(p)
        return json.loads(p.read_text(encoding="utf-8"))

    terminal = read_json("authorization/formal_terminal.json")
    if terminal.get("state") != "CONSUMED":
        raise PermissionError("Publication requires terminal CONSUMED before saved holdout reads")
    if read_json("independent_review/formal_numeric.json").get("status") != "PASS":
        raise PermissionError("Publication requires passed independent formal numeric review")
    result = read_json("metrics/formal_result.json")
    seal = read_json("calibration/sealed_calibration.json")
    diagnostic_manifest = read_json("diagnostics/holdout/manifest.json")
    if diagnostic_manifest.get("calibration_seal_sha256") != seal.get("seal_sha256"):
        raise ValueError("Saved diagnostics and training calibration seal differ")

    def csv(name):
        relative = "diagnostics/holdout/" + name + ".csv"
        p = run / relative
        if diagnostic_manifest["artifacts_sha256"].get(p.name) != sha(p):
            raise ValueError("Saved diagnostic hash mismatch: " + name)
        sources[relative] = sha(p)
        return pd.read_csv(p, float_precision="round_trip")

    rows = csv("row_metrics_and_timeline")
    rows["decision_at"] = pd.to_datetime(rows.decision_at, utc=True)
    if set(rows.family) != set(FAMILIES) or set(rows.quarter) != set(QUARTERS):
        raise ValueError("Expected all nine saved families and both holdout quarters")
    calibration = csv("model_calibration")
    high = csv("high_extreme_RV_underprediction")
    losses = csv("period_losses")
    out = run / "figures/formal_publication"
    out.mkdir(parents=True, exist_ok=False)
    entries = []

    def save(stem, fig, table, formula, boundary):
        table.to_csv(out / (stem + ".csv"), index=False, float_format="%.17g")
        fig.tight_layout(rect=(0, .04, 1, .96))
        fig.text(.01, .01, "FROZEN_RISK_03_v1 | saved formal evidence | auxiliary plots add no success criterion", fontsize=8)
        fig.savefig(out / (stem + ".png"), dpi=120)
        fig.savefig(out / (stem + ".svg"))
        plt.close(fig)
        entries.append({"stem": stem, "formula": formula, "boundary": boundary})

    # The two points lie on their real UTC quarter starts, not equally spaced categories.
    ranking = []
    for q, quarter in zip(result["quarter_metrics"], QUARTERS):
        for f in FAMILIES:
            ranking.append({"quarter": quarter, "quarter_start_utc": "2026-04-01T00:00:00Z" if quarter == "2026Q2" else "2026-07-01T00:00:00Z",
                            "family": f, "AUROC": q["models"][f]["AUROC"], "AP": q["models"][f]["AP"],
                            "N": q["opportunities"], "events": q["positive"]})
    ranking = pd.DataFrame(ranking)
    fig, ax = plt.subplots(figsize=(10, 6))
    for f in FAMILIES:
        d = ranking[ranking.family == f]
        ax.plot(pd.to_datetime(d.quarter_start_utc, utc=True), d.AUROC, "o-", label=f, color=COLORS[f])
    ax.axhline(.5, color="gray", linestyle="--", linewidth=1)
    ax.set(title="Surprise AUROC | all nine families", xlabel="Quarter start (UTC)", ylabel="AUROC (unitless; higher is better)", ylim=(0, 1))
    ax.legend(ncol=3, fontsize=9)
    save("01_quarter_surprise_AUROC", fig, ranking, "Saved within-quarter AUROC", "All families retained; connecting quarter summaries does not imply daily metrics")

    ci = pd.DataFrame([{"block_days": int(b), "comparison": k, **v}
                       for b, item in result["bootstrap"].items() for k, v in item["comparisons"].items()])
    ci = ci.drop(columns=["quarter_deltas"]).sort_values(["comparison", "block_days"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharex=True)
    for ax, comparison in zip(axes, ["R2_minus_R1", "R2_minus_B2"]):
        d = ci[ci.comparison == comparison].set_index("block_days").loc[[7, 3, 14]]
        for y, (block, r) in enumerate(d.iterrows()):
            ax.hlines(y, r.ci_lower, r.ci_upper, color="C0" if block == 7 else "gray", linewidth=3 if block == 7 else 2)
            ax.plot(r.mean_delta, y, "o", color="C0" if block == 7 else "gray")
        ax.axvline(0, color="black", linewidth=1)
        ax.set(yticks=[0, 1, 2], yticklabels=["7d primary", "3d sensitivity", "14d sensitivity"],
               title=comparison.replace("_minus_", " - "), xlabel="Equal-quarter delta AUROC; positive favors R2", ylim=(-.6, 2.6))
    save("02_primary_delta_AUROC_CI", fig, ci, "Saved equal-quarter delta and registered 95% percentile CI", "Co-primary comparisons kept separate; sensitivity intervals are auxiliary, no new multiplicity claim")

    # Reuse the scientific run's existing RV chart exactly; only supply its numeric table.
    stem = "03_RV_timeline"
    for suffix in ("png", "svg"):
        relative = "diagnostics/holdout/RV_timeline." + suffix
        p = run / relative
        sources[relative] = sha(p)
        if diagnostic_manifest["artifacts_sha256"][p.name] != sources[relative]:
            raise ValueError("RV figure hash mismatch")
        shutil.copyfile(p, out / (stem + "." + suffix))
    rows[["decision_at", "quarter", "family", "RV_raw", "prediction_RV"]].to_csv(out / (stem + ".csv"), index=False, float_format="%.17g")
    entries.append({"stem": stem, "formula": "Byte-identical saved RV_timeline figures", "boundary": "4h squared natural log return; not annualized"})

    daily_tables = []
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5))
    for ax, quarter in zip(axes, QUARTERS):
        d = rows[rows.quarter == quarter].pivot(index="decision_at", columns="family", values="raw_QLIKE")
        start = pd.Timestamp("2026-04-01", tz="UTC") if quarter == "2026Q2" else pd.Timestamp("2026-07-01", tz="UTC")
        end = pd.Timestamp("2026-07-01", tz="UTC") if quarter == "2026Q2" else pd.Timestamp("2026-10-01", tz="UTC")
        calendar = pd.date_range(start, end, freq="D", inclusive="left")
        for f in ("R1", "B2"):
            pair = d[f] - d.R2
            daily = pair.groupby(pair.index.floor("D")).sum().reindex(calendar, fill_value=0)
            counts = pair.groupby(pair.index.floor("D")).size().reindex(calendar, fill_value=0)
            table = pd.DataFrame({"quarter": quarter, "day_utc": calendar, "comparison": f + "_minus_R2", "N": counts.to_numpy(), "daily_raw_QLIKE_difference_sum": daily.to_numpy(), "cumulative_raw_QLIKE_difference_sum": daily.cumsum().to_numpy()})
            daily_tables.append(table)
            ax.plot(calendar, daily.cumsum(), label=f + " - R2")
        ax.axhline(0, color="gray", linewidth=1)
        ax.set(title=quarter + " | reset at quarter start", ylabel="Cumulative paired loss difference (sum)", xlabel="UTC calendar day")
        ax.legend()
    save("04_daily_cumulative_QLIKE_difference", fig, pd.concat(daily_tables), "Daily sum(raw_QLIKE_baseline - raw_QLIKE_R2), then within-quarter cumulative sum", "Positive favors R2; not a percentage or investment return; calendar missing days contribute zero loss/count")

    bins = calibration[(calibration.bin_type == "surprise_score_quartile") & calibration.family.isin(MAIN) & calibration.period.isin(QUARTERS)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
    for ax, quarter in zip(axes, QUARTERS):
        for f in MAIN:
            d = bins[(bins.period == quarter) & (bins.family == f)].sort_values("bin_id")
            ax.plot(d.bin_id + 1, d.surprise_event_fraction, "o-", label=f, color=COLORS[f])
        ax.set(title=quarter, xlabel="TRAIN-fixed score bin (low to high)", ylabel="Observed surprise event fraction", ylim=(0, 1), xticks=[1, 2, 3, 4])
        ax.legend()
    save("05_train_fixed_score_event_frequency", fig, bins, "Saved surprise event count / saved bin count", "Family-specific TRAIN score cuts unchanged; event is log((RV_raw+eps)/(EWMA+eps)) > log(2); score is not probability; empty bins undefined")

    concentration = high[high.period.isin(QUARTERS)].copy()
    totals = losses[losses.period.isin(QUARTERS)][["period", "family", "count", "QLIKE_Regret"]].rename(columns={"count": "total_count", "QLIKE_Regret": "total_mean_QLIKE_Regret"})
    concentration = concentration.merge(totals, on=["period", "family"], validate="many_to_one")
    concentration["subset_QLIKE_Regret_sum"] = concentration["count"] * concentration.QLIKE_Regret
    concentration["total_QLIKE_Regret_sum"] = concentration.total_count * concentration.total_mean_QLIKE_Regret
    concentration["QLIKE_Regret_loss_share"] = concentration.subset_QLIKE_Regret_sum / concentration.total_QLIKE_Regret_sum.replace(0, np.nan)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    for col, quarter in enumerate(QUARTERS):
        for subset, marker in [("high_RV_train_q90", "o"), ("extreme_RV_train_q99", "s")]:
            d = concentration[(concentration.period == quarter) & (concentration.subset == subset)].set_index("family").loc[FAMILIES]
            axes[0, col].plot(range(9), d.underprediction_fraction, marker + "-", label=subset.replace("_RV_train_", " "))
            axes[1, col].plot(range(9), d.QLIKE_Regret_loss_share, marker + "-", label=subset.replace("_RV_train_", " "))
        for r in range(2):
            axes[r, col].set(xticks=range(9), xticklabels=FAMILIES, ylim=(0, 1), title=quarter)
            axes[r, col].tick_params(axis="x", labelrotation=45)
            axes[r, col].legend(fontsize=8)
        axes[0, col].set_ylabel("Underprediction fraction: pred / RV_eff < 0.5")
        axes[1, col].set_ylabel("Subset share of total QLIKE Regret (0–1)")
    save("06_underprediction_extreme_loss_concentration", fig, concentration, "Underprediction fraction from saved table; (subset count * subset mean QLIKE_Regret) / (total count * total mean QLIKE_Regret)", "TRAIN q90/q99 thresholds; RV_eff > threshold; nested groups, not additive; all nine families, auxiliary descriptive concentration")

    score_table = rows[rows.family.isin(MAIN)][["quarter", "decision_at", "family", "surprise_score"]].copy()
    curves = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), sharex=True, sharey=True)
    for ax, quarter in zip(axes, QUARTERS):
        for f in MAIN:
            d = score_table[(score_table.quarter == quarter) & (score_table.family == f)].sort_values("surprise_score", kind="stable").copy()
            d["empirical_cumulative_fraction"] = np.arange(1, len(d) + 1) / len(d)
            curves.append(d)
            ax.step(d.surprise_score, d.empirical_cumulative_fraction, where="post", label=f, color=COLORS[f])
        ax.set(title=quarter, xlabel="Saved log((pred+eps)/(same EWMA+eps))", ylabel="Empirical cumulative fraction", ylim=(0, 1))
        ax.legend()
    save("07_surprise_score_distribution", fig, pd.concat(curves), "Empirical CDF of saved surprise_score", "Same saved EWMA reference and epsilon; raw score axis shared across quarters; no TEST normalization or quantile refit; not event probability")

    for relative, expected in sources.items():
        if sha(run / relative) != expected:
            raise ValueError("Source changed during publication: " + relative)
    manifest = {"experiment_id": "FROZEN_RISK_03_v1", "terminal_state": "CONSUMED", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "publication_code_sha256": sha(__file__), "source_files_sha256": sources,
                "output_files_sha256": {p.name: sha(p) for p in sorted(out.iterdir())}, "charts": entries,
                "no_refit": True, "no_new_hypothesis_tests": True, "no_raw_source_reads": True,
                "no_new_success_criteria": True, "registered_result_status": result["status"],
                "calibration_seal_sha256": seal["seal_sha256"], "numeric_csv_read": "pandas float_precision=round_trip"}
    (out / "chartmanifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PUBLICATION_CHARTS_COMPLETE", "output": str(out), "charts": len(entries)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "research/runs/FROZEN_RISK_03_v1")
    args = parser.parse_args()
    render(args.run)
