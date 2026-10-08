"""Build outcome sidecars from official Freqtrade reference exports only.

No model fitting, representation extraction, backtest, or synthetic equity ledger.
Every outcome/funding field is forbidden as a model input. Source OHLCVA remains
in the immutable snapshot; input windows are positional references into that CSV.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import zipfile

import pandas as pd
import yaml

HOUR = pd.Timedelta(hours=1)
PAIR = "BTC/USDT:USDT"
STRATEGY = "KronosFixedMomentum"
BASIS = "independent_reference_trade_net_return_greater_than_zero"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stamp(value):
    result = pd.Timestamp(value)
    require(result.tzinfo is not None and result.utcoffset().total_seconds() == 0,
            f"Explicit UTC timestamp required: {value}")
    return result.tz_convert("UTC")


def iso(value):
    return value.isoformat().replace("+00:00", "Z")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(actual, expected, description, *, absolute=1e-10):
    require(math.isfinite(float(actual)) and math.isfinite(float(expected))
            and math.isclose(float(actual), float(expected), rel_tol=1e-11, abs_tol=absolute),
            f"{description}: mismatch {actual} versus {expected}")


def read_snapshot(snapshot, config):
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads((snapshot / "quality_report.json").read_text(encoding="utf-8"))
    require(manifest["status"] == "collected" and report["ready_for_labeling"], "Snapshot not ready")
    require(manifest["snapshot_id"] == report["snapshot_id"], "Snapshot ID mismatch")
    for name, digest in manifest["files"].items():
        path = (snapshot / name).resolve()
        require(path.is_relative_to(snapshot) and path.is_file() and sha256(path) == digest,
                f"Snapshot hash mismatch: {name}")
    start, end = (stamp(config["market"][key]) for key in ["history_start", "history_end_exclusive"])
    require(start == stamp(report["evaluation_start"]) and end == stamp(report["evaluation_end_exclusive"]),
            "Snapshot/config range mismatch")
    candles = pd.read_csv(snapshot / "candles.csv")
    candles["date"] = pd.to_datetime(candles.bar_open_at, utc=True)
    require(list(candles.date) == list(pd.date_range(start - 256 * HOUR, end, freq="h", inclusive="left")),
            "Candle grid must be exact, unique and continuous with 256 warmup bars")
    for field in ["open", "high", "low", "close", "volume", "amount"]:
        require(candles[field].map(lambda x: math.isfinite(float(x))).all(), f"Nonfinite {field}")
    require((candles[["open", "high", "low", "close"]] > 0).all().all(), "Nonpositive price")
    require((candles[["volume", "amount"]] >= 0).all().all(), "Negative volume/amount")
    require((candles.confirm == 1).all(), "Unconfirmed candle")
    require((pd.to_datetime(candles.bar_close_at, utc=True) == candles.date + HOUR).all(), "Wrong close time")
    require((pd.to_datetime(candles.available_at, utc=True) == candles.date + HOUR + pd.Timedelta(seconds=60)).all(),
            "Availability must equal candle close plus 60 seconds")
    marks = pd.read_csv(snapshot / "mark_prices.csv")
    marks["date"] = pd.to_datetime(marks.bar_open_at, utc=True)
    require(list(marks.date) == list(pd.date_range(start, end, freq="h", inclusive="left")), "Mark grid mismatch")
    require((marks.confirm == 1).all(), "Unconfirmed mark")
    funding = pd.read_csv(snapshot / "funding.csv")
    funding["date"] = pd.to_datetime(funding.funding_at, utc=True)
    require(list(funding.date) == list(pd.date_range(start, end, freq="8h", inclusive="left")), "Funding grid mismatch")
    require((funding.instrument == "BTC-USDT-SWAP").all(), "Funding instrument mismatch")
    return manifest, candles, marks, funding, start, end


def opportunities(candles, start, end):
    by_time = {ts: i for i, ts in enumerate(candles.date)}
    records = []
    for boundary in pd.date_range(start + 4 * HOUR, end, freq="8h", inclusive="left"):
        pos = by_time[boundary]
        history = candles.iloc[pos - 256:pos]
        require(len(history) == 256 and history.date.iloc[-1] == boundary - HOUR,
                f"Incomplete causal history at {boundary}")
        decision = boundary + pd.Timedelta(seconds=60)
        require((pd.to_datetime(history.available_at, utc=True) <= decision).all(), "Unavailable history")
        momentum = float(history.close.iloc[-1]) / float(history.close.iloc[-25]) - 1.0
        direction = 1 if momentum > 0 else -1 if momentum < 0 else 0
        entry, exit_at = boundary + HOUR, boundary + 5 * HOUR
        labelable = exit_at + pd.Timedelta(seconds=60)
        reason = "zero_momentum" if direction == 0 else "research_right_boundary" if labelable >= end or exit_at >= end else "labelable"
        records.append({"opportunity_id": iso(boundary), "decision_boundary_at": iso(boundary),
                        "decision_at": iso(decision), "history_start_row": pos - 256,
                        "history_end_row_exclusive": pos, "history_start_at": iso(history.date.iloc[0]),
                        "history_end_exclusive": iso(boundary), "momentum_24h": momentum,
                        "direction": direction, "entry_at": iso(entry), "exit_at": iso(exit_at),
                        "labelable_at": iso(labelable), "reason": reason})
    return records


def load_export(path, fee):
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
        configs = [name for name in members if name.endswith("_config.json")]
        require(len(configs) == 1, "Expected one embedded Freqtrade config")
        config = json.loads(archive.read(configs[0]))
        close(config["fee"], fee, "Embedded fee")
        require(config.get("strategy") == STRATEGY and config.get("timeframe") == "1h"
                and config.get("trading_mode") == "futures" and config.get("margin_mode") == "isolated",
                "Unexpected reference configuration")
        require(config.get("stake_amount") == 1000 and float(config.get("dry_run_wallet", 0)) >= 1_000_000,
                "Reference must use fixed 1000 USDT stake and large paper wallet")
        require(config.get("max_open_trades") == 1, "Reference must have one position")
        require(any(name.endswith(f"_{STRATEGY}.py") for name in members), "Missing embedded strategy")
        results = []
        for name in members:
            if name.endswith(".json") and name not in configs:
                data = json.loads(archive.read(name))
                if "strategy" in data:
                    require(set(data["strategy"]) == {STRATEGY}, "Unexpected strategy export")
                    results.append(data["strategy"][STRATEGY])
        require(len(results) == 1, "Expected exactly one official trade export")
        trades = results[0]["trades"]
    return trades, config, members


def match_trades(records, trades, fee, candles, marks, funding):
    expected = {stamp(row["entry_at"]): row for row in records if row["reason"] == "labelable"}
    require(len(trades) == len(expected), "Reference count mismatch: missing/unexpected trades cannot be dropped")
    prices = candles.set_index("date")["open"].to_dict()
    mark_prices = marks.set_index("date")["open"].to_dict()
    matched = {}
    for trade in trades:
        entry, exit_at = stamp(trade["open_date"]), stamp(trade["close_date"])
        require(entry in expected and entry not in matched, f"Unexpected or duplicate reference trade at {entry}")
        row = expected[entry]
        require(exit_at == stamp(row["exit_at"]) and exit_at - entry == 4 * HOUR
                and trade["trade_duration"] == 240 and trade["exit_reason"] == "fixed_hold_4h"
                and trade.get("is_open") is False, f"Unexpected exit at {entry}")
        require(trade["pair"] == PAIR and trade["is_short"] == (row["direction"] < 0), "Wrong pair/direction")
        require(trade["enter_tag"] == f"momentum_24h_{'short' if row['direction'] < 0 else 'long'}", "Wrong entry tag")
        require(trade["open_timestamp"] == int(entry.timestamp() * 1000)
                and trade["close_timestamp"] == int(exit_at.timestamp() * 1000), "Trade timestamp mismatch")
        for key in ["fee_open", "fee_close"]:
            close(trade[key], fee, key)
        close(trade["leverage"], 1, "Leverage")
        close(trade["open_rate"], prices[entry], "Entry price")
        close(trade["close_rate"], prices[exit_at], "Exit price")
        require(float(trade["amount"]) > 0, "Invalid amount")
        events = funding.loc[(funding.date >= entry) & (funding.date <= exit_at)]
        require(len(events) == 1, "Scheduled trade must cross exactly one settlement")
        raw_payment = sum(mark_prices[event.date] * float(event.realized_rate) for event in events.itertuples())
        expected_funding = -row["direction"] * float(trade["amount"]) * raw_payment
        close(trade["funding_fees"], expected_funding, "Raw mark/funding signed payment", absolute=1e-8)
        for key in ["profit_ratio", "profit_abs", "funding_fees"]:
            require(math.isfinite(float(trade[key])), f"Nonfinite official {key}")
        matched[entry] = trade
    require(set(matched) == set(expected), "Missing reference trades")
    return matched


def segments(config):
    splits = config["splits"]
    for fold in splits["walk_forward_dates"]:
        for role in ["train", "validation", "test"]:
            yield fold["id"], role, fold[role]
    for role in ["train", "validation"]:
        yield splits["final_fit"].get("id", "final_fit"), role, splits["final_fit"][role]
    yield "final_holdout", "holdout", splits["final_holdout"]


def split_rows(records, config):
    rows = []
    for fold, role, segment in segments(config):
        start, end = stamp(segment[0]), stamp(segment[1])
        require(start < end, "Invalid split range")
        for row in records:
            if start <= stamp(row["decision_at"]) < end:
                reason = row["reason"]
                if reason == "labelable" and (stamp(row["labelable_at"]) >= end or stamp(row["exit_at"]) >= end):
                    reason = "purged_segment_right_boundary"
                rows.append({"fold_id": fold, "role": role, "opportunity_id": row["opportunity_id"],
                             "segment_start": iso(start), "segment_end_exclusive": iso(end),
                             "included": reason == "labelable", "reason": reason})
    return rows


def write_csv(path, rows):
    require(bool(rows), f"No rows for {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build(config_path, snapshot, base_result, stress_result, output):
    snapshot, output = snapshot.resolve(), output.resolve()
    require(output != snapshot and not output.is_relative_to(snapshot), "Output may not modify snapshot")
    require(not output.exists() or (output.is_dir() and not any(output.iterdir())), "Output must be new/empty")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    require(config["timing"]["lookback_bars"] == 256 and config["timing"]["decision_hours_utc"] == [4, 12, 20]
            and config["timing"]["availability_delay_seconds"] == 60
            and config["timing"]["holding_horizon_bars"] == 4
            and config["fixed_strategy"]["momentum_lookback_bars"] == 24, "Unsupported causal protocol")
    close(config["labels_and_costs"]["freqtrade_fee_per_side"], 0.0007, "Protocol base fee")
    close(config["labels_and_costs"]["freqtrade_stress_fee_per_side"], 0.0014, "Protocol stress fee")
    require(config["fixed_strategy"]["leverage"] == 1.0, "Unsupported protocol leverage")
    manifest, candles, marks, funding, start, end = read_snapshot(snapshot, config)
    records = opportunities(candles, start, end)
    matched = {}
    archives = {}
    for name, path, fee in [("base", base_result, 0.0007), ("stress", stress_result, 0.0014)]:
        trades, embedded, members = load_export(path, fee)
        matched[name] = match_trades(records, trades, fee, candles, marks, funding)
        archives[name] = (path, members)
    labels = []
    for row in records:
        if row["reason"] != "labelable":
            continue
        label = {key: row[key] for key in ["opportunity_id", "entry_at", "exit_at", "labelable_at", "direction"]}
        label["label_basis"] = BASIS
        base_trade = matched["base"][stamp(row["entry_at"])]
        stress_trade = matched["stress"][stamp(row["entry_at"])]
        close(base_trade["amount"], stress_trade["amount"], "Base/stress amount")
        close(base_trade["stake_amount"], stress_trade["stake_amount"], "Base/stress stake")
        close(base_trade["funding_fees"], stress_trade["funding_fees"], "Base/stress funding")
        require(stress_trade["profit_ratio"] <= base_trade["profit_ratio"] + 1e-12,
                "Stress official net return exceeds base")
        for name in ["base", "stress"]:
            trade = matched[name][stamp(row["entry_at"])]
            for key in ["profit_ratio", "profit_abs", "funding_fees", "amount", "stake_amount", "fee_open", "fee_close"]:
                label[f"{name}_{key}"] = trade[key]
            label[f"{name}_target"] = int(trade["profit_ratio"] > 0)
        labels.append(label)
    indices = split_rows(records, config)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "opportunities.csv", records)
    write_csv(output / "labels_forbidden_as_features.csv", labels)
    write_csv(output / "split_index.csv", indices)
    provenance = output / "provenance"
    provenance.mkdir()
    shutil.copy2(config_path, provenance / "experiment_config.yaml")
    shutil.copy2(Path(__file__), provenance / "build_reference_labels.py")
    for name, (path, members) in archives.items():
        destination = provenance / name
        destination.mkdir()
        shutil.copy2(path, destination / path.name)
        with zipfile.ZipFile(path) as archive:
            for member in members:
                if member.endswith("_config.json") or member.endswith(f"_{STRATEGY}.py"):
                    (destination / Path(member).name).write_bytes(archive.read(member))
    audit = {"schema_version": 1, "status": "official_reference_labels_verified",
             "snapshot_id": manifest["snapshot_id"], "opportunity_rows": len(records), "label_rows": len(labels),
             "reference_trade_rows": {name: len(value) for name, value in matched.items()},
             "split_index_rows": len(indices), "split_included_rows": sum(row["included"] for row in indices),
             "checks": ["snapshot_hashes", "256_completed_available_history", "zero_momentum_skip",
                        "strict_unique_reference_matching", "4h_fixed_exit", "open_prices", "fees_and_leverage",
                        "signed_original_mark_funding", "right_boundary_purge"],
             "label_basis": BASIS, "labelable_at_policy": "exit_plus_60_seconds",
             "outcome_fields_forbidden_as_features": True, "funding_usage": "outcome_audit_only",
             "input_source": str(snapshot / "candles.csv"), "input_fields": ["open", "high", "low", "close", "volume", "amount"],
             "reference_capital": "fixed_stake_large_paper_wallet_for_independent_labels_not_strategy_performance",
             "performance_summary_computed": False,
             "sources": {"snapshot": str(snapshot), "snapshot_manifest_sha256": sha256(snapshot / "manifest.json"),
                         "snapshot_csv_sha256": {name: sha256(snapshot / name) for name in ["candles.csv", "mark_prices.csv", "funding.csv"]},
                         "config_sha256": sha256(config_path), "script_sha256": sha256(Path(__file__)),
                         "result_sha256": {name: sha256(path) for name, (path, _) in archives.items()}},
             "artifact_sha256": {str(path.relative_to(output)): sha256(path) for path in output.rglob("*") if path.is_file()}}
    (output / "manifest.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return {key: audit[key] for key in ["status", "opportunity_rows", "label_rows", "reference_trade_rows", "split_index_rows", "split_included_rows"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["config", "snapshot", "base-result", "stress-result", "output-dir"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.config, args.snapshot, args.base_result, args.stress_result, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
