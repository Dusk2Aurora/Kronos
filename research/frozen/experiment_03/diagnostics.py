"""Pure supplied-data risk diagnostics; no source reads, fitting or test bin refits.

Absolute variance reliability is observed RV versus predicted RV, never an event
probability calibration. Callers must enforce the formal authorization guard.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-12
PROTOCOL_SHA256 = "7ed8affd7cd624cde89d23d29aaca7ef2ab660167fac7d3dc25fac274e394b7f"
LOSSES = ("raw_QLIKE", "QLIKE_Regret", "logRV_MSE",
          "additive_epsilon_logRV_MSE", "surprise_MSE")
CONTEXTS = ("DEV_validation_engineering", "authorized_holdout")


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _hash(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _inputs(frame, predictions):
    frame = frame.copy()
    if not len(frame) or not {"decision_at", "RV_raw"}.issubset(frame.columns):
        raise ValueError("Nonempty frame requires decision_at and RV_raw")
    times = pd.to_datetime(frame.decision_at, utc=True, errors="raise")
    if times.isna().any() or times.duplicated().any():
        raise ValueError("Unique finite UTC decision origins required")
    y = np.asarray(frame.RV_raw, dtype=np.float64)
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("RV_raw must be finite and nonnegative")
    if "ewma" not in predictions:
        raise ValueError("Same sealed EWMA reference prediction required")
    checked = {}
    for family, prediction in predictions.items():
        p = np.asarray(prediction, dtype=np.float64)
        if (p.shape != y.shape or not np.isfinite(p).all()
                or (p < EPS).any() or (p > 1.).any()):
            raise ValueError("Predictions must be aligned and already absolutely clipped")
        checked[str(family)] = p.copy()
    frame["decision_at"] = times
    frame["RV_raw"] = y
    return frame, checked


def quantile_cuts(values, probabilities):
    """Only finite inner cuts are stored; repeated quantiles collapse identically."""
    return np.unique(np.quantile(np.asarray(values, dtype=float), probabilities)).tolist()


def assign_bins(values, inner_cuts):
    """Bins [lower,upper), equality goes right; terminal bin includes +infinity.

    Outer infinity bounds are conceptual and never serialized into JSON.
    """
    values, cuts = np.asarray(values, dtype=float), np.asarray(inner_cuts, dtype=float)
    if (not np.isfinite(values).all() or not np.isfinite(cuts).all()
            or cuts.ndim != 1 or (np.diff(cuts) <= 0).any()):
        raise ValueError("Finite values and strictly ascending unique cuts required")
    return np.searchsorted(cuts, values, side="right")


def seal_calibration(training_frame, predictions, output_dir):
    """Seal training-only effective RV thresholds and all selected-family bins."""
    frame, predictions = _inputs(training_frame, predictions)
    times = frame.decision_at
    if ((times < pd.Timestamp("2024-01-01", tz="UTC"))
            | (times >= pd.Timestamp("2026-03-01", tz="UTC"))).any():
        raise ValueError("Calibration accepts only registered training origins")
    effective = np.maximum(frame.RV_raw.to_numpy(), EPS)
    ewma = predictions["ewma"]
    thresholds = {"train_effective_RV_q90": float(np.quantile(effective, .9)),
                  "train_effective_RV_q99": float(np.quantile(effective, .99))}
    bins = {family: {
        "prediction_decile_inner_cuts": quantile_cuts(p, np.arange(1, 10) / 10),
        "surprise_score_quartile_inner_cuts": quantile_cuts(
            np.log((p + EPS) / (ewma + EPS)), [.25, .5, .75])}
        for family, p in predictions.items()}
    state = {"EWMA_quartile_inner_cuts": quantile_cuts(ewma, [.25, .5, .75])}
    # Hash all supplied frame columns as well as explicit numeric RV/predictions.
    frame_hash = hashlib.sha256(frame.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()
    payload = {"schema_version": 1, "context": "training_calibration_seal",
               "protocol_sha256": PROTOCOL_SHA256, "training_rows": len(frame),
               "training_frame_sha256": frame_hash,
               "training_RV_raw_sha256": hashlib.sha256(np.asarray(frame.RV_raw, dtype="<f8").tobytes()).hexdigest(),
               "training_predictions_sha256": {k: hashlib.sha256(np.asarray(p, dtype="<f8").tobytes()).hexdigest()
                                                for k, p in predictions.items()},
               "thresholds": thresholds, "model_bins": bins, "historical_state_bins": state,
               "bin_convention": "searchsorted right; [lower,upper); equality goes right; outer bounds implicit infinity",
               "unit": "squared_log_return_4h_not_annualized", "epsilon": EPS,
               "reliability": "observed RV versus predicted RV; not event probability"}
    payload["threshold_and_bins_sha256"] = _hash({"thresholds": thresholds, "model_bins": bins,
                                                  "historical_state_bins": state})
    payload["seal_sha256"] = _hash(payload)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "calibration.json").open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return payload


def _check_seal(sealed):
    copy = dict(sealed)
    claimed = copy.pop("seal_sha256", None)
    if claimed != _hash(copy) or copy.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("Training calibration seal has changed")
    subset = {k: copy[k] for k in ("thresholds", "model_bins", "historical_state_bins")}
    if copy["threshold_and_bins_sha256"] != _hash(subset):
        raise ValueError("Training thresholds/bin hash has changed")


def row_metrics(frame, predictions):
    """One row per decision/family; the two additive-epsilon MSEs are identical."""
    frame, predictions = _inputs(frame, predictions)
    raw = frame.RV_raw.to_numpy()
    effective = np.maximum(raw, EPS)
    ewma = predictions["ewma"]
    event_surprise = np.log((raw + EPS) / (ewma + EPS))
    results = []
    for family, p in predictions.items():
        score = np.log((p + EPS) / (ewma + EPS))
        ratio = effective / p
        additive = (np.log((raw + EPS) / (p + EPS))) ** 2
        surprise_mse = (event_surprise - score) ** 2
        if not np.allclose(additive, surprise_mse, rtol=1e-12, atol=1e-12):
            raise ValueError("Same-epsilon surprise MSE identity failed")
        data = {"decision_at": frame.decision_at.to_numpy(), "family": family,
                "RV_raw": raw, "RV_effective": effective, "prediction_RV": p,
                "EWMA_RV": ewma, "surprise_score": score,
                "observed_surprise": event_surprise, "surprise_event": event_surprise > np.log(2.),
                "prediction_over_actual_effective": p / effective,
                "raw_QLIKE": np.log(p) + ratio,
                "QLIKE_Regret": ratio - np.log(ratio) - 1.,
                "logRV_MSE": (np.log(effective) - np.log(p)) ** 2,
                "additive_epsilon_logRV_MSE": additive, "surprise_MSE": surprise_mse}
        for key in ("opportunity_id", "role"):
            if key in frame:
                data[key] = frame[key].to_numpy()
        results.append(pd.DataFrame(data))
    result = pd.concat(results, ignore_index=True)
    result["quarter"] = result.decision_at.dt.year.astype(str) + "Q" + result.decision_at.dt.quarter.astype(str)
    return result.sort_values(["decision_at", "family"]).reset_index(drop=True)


def calendar_rolling(rows):
    """Full UTC calendar, rolling sum(loss)/sum(count), never daily-mean averages."""
    rows = rows.copy()
    rows["day"] = pd.to_datetime(rows.decision_at, utc=True).dt.floor("D")
    calendar = pd.date_range(rows.day.min(), rows.day.max(), freq="D", tz="UTC")
    output = []
    for family, group in rows.groupby("family", sort=True):
        daily = group.groupby("day")[list(LOSSES)].sum().reindex(calendar, fill_value=0.)
        counts = group.groupby("day").size().reindex(calendar, fill_value=0)
        roll_count = counts.rolling("7D").sum()
        table = pd.DataFrame({"day": calendar, "family": family,
                              "daily_count": counts.to_numpy(), "rolling_7day_count": roll_count.to_numpy()})
        for loss in LOSSES:
            sums = daily[loss].rolling("7D").sum()
            table[f"daily_{loss}_sum"] = daily[loss].to_numpy()
            table[f"rolling_7day_{loss}_sum"] = sums.to_numpy()
            table[f"rolling_7day_{loss}_mean"] = (sums / roll_count.replace(0, np.nan)).to_numpy()
        output.append(table)
    return pd.concat(output, ignore_index=True)


def _summary(group):
    if not len(group):
        return {"count": 0, "mean_observed_RV_raw": None, "mean_observed_RV_effective": None,
                "surprise_event_count": 0, "surprise_event_fraction": None,
                "mean_prediction_RV": None, "observed_raw_over_prediction_mean_ratio": None,
                "observed_effective_over_prediction_mean_ratio": None,
                **{loss: None for loss in LOSSES}}
    pred = float(group.prediction_RV.mean())
    return {"count": len(group), "mean_observed_RV_raw": float(group.RV_raw.mean()),
            "surprise_event_count": int(group.surprise_event.sum()),
            "surprise_event_fraction": float(group.surprise_event.mean()),
            "mean_observed_RV_effective": float(group.RV_effective.mean()),
            "mean_prediction_RV": pred,
            "observed_raw_over_prediction_mean_ratio": float(group.RV_raw.mean()) / pred,
            "observed_effective_over_prediction_mean_ratio": float(group.RV_effective.mean()) / pred,
            **{loss: float(group[loss].mean()) for loss in LOSSES}}


def diagnostic_tables(rows, sealed):
    """Training-fixed prediction/score/state bins and thresholds, all rows kept."""
    _check_seal(sealed)
    calibration, state, high, loss = [], [], [], []
    periods = [("POOLED", rows)] + list(rows.groupby("quarter", sort=True))
    for period, frame in periods:
        for family, group in frame.groupby("family", sort=True):
            if family not in sealed["model_bins"]:
                raise ValueError("Family has no sealed training calibration")
            loss.append({"period": period, "family": family, **_summary(group)})
            definitions = [("prediction_decile", group.prediction_RV.to_numpy(),
                            sealed["model_bins"][family]["prediction_decile_inner_cuts"]),
                           ("surprise_score_quartile", group.surprise_score.to_numpy(),
                            sealed["model_bins"][family]["surprise_score_quartile_inner_cuts"]),
                           ("historical_EWMA_quartile", group.EWMA_RV.to_numpy(),
                            sealed["historical_state_bins"]["EWMA_quartile_inner_cuts"])]
            for name, values, cuts in definitions:
                ids = assign_bins(values, cuts)
                for i in range(len(cuts) + 1):
                    entry = {"period": period, "family": family, "bin_type": name, "bin_id": i,
                             "lower_bound": None if i == 0 else cuts[i - 1],
                             "upper_bound": None if i == len(cuts) else cuts[i],
                             **_summary(group.iloc[np.flatnonzero(ids == i)])}
                    (state if name == "historical_EWMA_quartile" else calibration).append(entry)
            for label, key in (("high_RV_train_q90", "train_effective_RV_q90"),
                               ("extreme_RV_train_q99", "train_effective_RV_q99")):
                threshold = sealed["thresholds"][key]
                subset = group[group.RV_effective > threshold]
                high.append({"period": period, "family": family, "subset": label,
                             "threshold_effective_RV": threshold, "threshold_comparator": ">",
                             "underprediction_comparator": "prediction/effective_RV < 0.5",
                             "underprediction_count": int((subset.prediction_over_actual_effective < .5).sum()),
                             "underprediction_fraction": float((subset.prediction_over_actual_effective < .5).mean()) if len(subset) else None,
                             "diagnostic_only": True, **_summary(subset)})
    return {"model_calibration": pd.DataFrame(calibration), "historical_EWMA_quartile_losses": pd.DataFrame(state),
            "high_extreme_RV_underprediction": pd.DataFrame(high), "period_losses": pd.DataFrame(loss)}


def _plots(rows, rolling, directory, context):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = ["Microsoft YaHei", "DejaVu Sans"]
    quarters = sorted(rows.quarter.unique())
    label = "DEV validation engineering; holdout sealed" if context == "DEV_validation_engineering" else "AUTHORIZED HOLDOUT; registered diagnostics"
    units = "4h squared log return; not annualized"
    for filename, source, time_column, value_column, ylabel in (
        ("RV_timeline", rows, "decision_at", "prediction_RV", units),
        ("surprise_score_timeline", rows, "decision_at", "surprise_score", "log((pred+eps)/(EWMA+eps)); not probability"),
        ("rolling_QLIKE_Regret", rolling, "day", "rolling_7day_QLIKE_Regret_mean", "7 UTC calendar days: loss sum / opportunity count")):
        fig, axes = plt.subplots(len(quarters), 1, figsize=(13, 4 * len(quarters)), squeeze=False)
        for ax, quarter in zip(axes[:, 0], quarters):
            period = rows[rows.quarter == quarter]
            start = period.decision_at.min().floor("D")
            end = period.decision_at.max().floor("D") + pd.Timedelta(days=1)
            subset = source[(source[time_column] >= start) & (source[time_column] < end)]
            for family, group in subset.groupby("family", sort=True):
                ordered = group.sort_values(time_column)
                ax.plot(ordered[time_column], ordered[value_column], label=family, linewidth=1.)
            if filename == "RV_timeline":
                observed = period.drop_duplicates("decision_at").sort_values("decision_at")
                ax.plot(observed.decision_at, observed.RV_raw, label="observed RV_raw", color="black", alpha=.5, linewidth=.8)
            ax.set_title(f"{quarter} | n={len(period.drop_duplicates('decision_at'))} | {label}")
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Original decision origin / calendar day (UTC)")
            ax.grid(alpha=.25)
            ax.legend(ncol=4, fontsize=8)
        fig.suptitle("Source: supplied aligned frame/predictions; sealed train calibration; no costs or revenues", fontsize=10)
        fig.tight_layout()
        fig.savefig(directory / f"{filename}.png", dpi=160)
        fig.savefig(directory / f"{filename}.svg")
        plt.close(fig)


def analyze(frame, predictions, sealed_calibration, outdir, *, context):
    """Analyze DEV validation or caller-authorized holdout, never recompute cuts.

    Explicit context is provenance, not authorization: the root guard must run
    before the caller supplies any holdout values to this pure-data function.
    """
    if context not in CONTEXTS:
        raise ValueError("Explicit DEV_validation_engineering/authorized_holdout context required")
    frame, predictions = _inputs(frame, predictions)
    _check_seal(sealed_calibration)
    lower, upper = (("2026-03-01", "2026-04-01") if context == CONTEXTS[0]
                    else ("2026-04-01", "2026-10-01"))
    if ((frame.decision_at < pd.Timestamp(lower, tz="UTC"))
            | (frame.decision_at >= pd.Timestamp(upper, tz="UTC"))).any():
        raise ValueError("Supplied decision origins fall outside declared diagnostic context")
    rows = row_metrics(frame, predictions)
    tables = diagnostic_tables(rows, sealed_calibration)
    rolling = calendar_rolling(rows)
    directory = Path(outdir)
    directory.mkdir(parents=True, exist_ok=False)
    rows.to_csv(directory / "row_metrics_and_timeline.csv", index=False)
    rolling.to_csv(directory / "calendar_rolling_7day.csv", index=False)
    for name, table in tables.items():
        table.to_csv(directory / f"{name}.csv", index=False)
    _plots(rows, rolling, directory, context)
    manifest = {"context": context, "rows": len(frame), "families": sorted(predictions),
                "analysis_frame_sha256": hashlib.sha256(frame.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest(),
                "analysis_predictions_sha256": {k: hashlib.sha256(np.asarray(p, dtype="<f8").tobytes()).hexdigest()
                                                 for k, p in predictions.items()},
                "protocol_sha256": PROTOCOL_SHA256, "calibration_seal_sha256": sealed_calibration["seal_sha256"],
                "threshold_and_bins_sha256": sealed_calibration["threshold_and_bins_sha256"],
                "thresholds_refit": False, "diagnostic_only": True, "epsilon": EPS,
                "unit": "squared_log_return_4h_not_annualized", "raw_QLIKE_percentage": False,
                "rolling_definition": "7 trailing UTC calendar days; sum losses/sum opportunities",
                "sample_retention": "every supplied valid row retained; no test bin refit",
                "same_epsilon_surprise_MSE_identity": True,
                "artifacts_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                     for p in sorted(directory.iterdir()) if p.is_file()}}
    with (directory / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return {"manifest": manifest, "row_metrics": rows, "rolling": rolling, **tables}
