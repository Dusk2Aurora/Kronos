"""Locked risk-03 heads; B2 uses only the 33 R1 columns and no test selection.

Callers own registration, role filtering and fit journals. This module never
reads market data. Synthetic tests are engineering checks, not study fits.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml

from research.frozen.experiment_02.models import (
    EPSILON, PREDICTION_MAX, LAMBDA_CANDIDATES, fit_head, predict_head,
    predict_head_with_stats, clip_predictions, qlike, regret, log_mse,
    select_candidate, objective_and_gradient, _effective_y,
)

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "research/configs/frozen_risk_03_v1.yaml"
PROTOCOL_SHA256 = "7ed8affd7cd624cde89d23d29aaca7ef2ab660167fac7d3dc25fac274e394b7f"
DEPENDENCY_PATH = ROOT / "research/runs/FROZEN_RISK_03_v1/provenance/python_packages"
B2_WIDTH = 33
B2_GRID = ((7, 30), (7, 60), (15, 30), (15, 60))


def _locked_b2():
    raw = CONFIG.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PROTOCOL_SHA256:
        raise RuntimeError("Risk-03 scientific protocol hash changed")
    return yaml.safe_load(raw)["B2"]


def _lightgbm():
    """No fallback to an unrecorded environment/global LightGBM installation."""
    path = DEPENDENCY_PATH.resolve()
    if not (path / "lightgbm/__init__.py").is_file():
        raise RuntimeError("Install the registered isolated LightGBM dependency first")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    lgb = importlib.import_module("lightgbm")
    if lgb.__version__ != "4.6.0" or not Path(lgb.__file__).resolve().is_relative_to(path):
        raise RuntimeError("B2 requires isolated LightGBM 4.6.0")
    return lgb


def _b2_x(x):
    values = np.asarray(x, dtype=np.float64)
    if (values.ndim != 2 or values.shape[1] != B2_WIDTH
            or not len(values) or not np.isfinite(values).all()):
        raise ValueError("B2 requires a nonempty finite matrix of exactly 33 R1 columns")
    return values


def _restore_log_prediction(raw_score, scale):
    """Gamma raw score is log(mean scaled RV); clip in absolute log units."""
    scores = np.asarray(raw_score, dtype=np.float64)
    scale = float(scale)
    if scores.ndim != 1 or not np.isfinite(scores).all() or not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid Gamma raw scores or training target scale")
    log_absolute = scores + np.log(scale)
    low, high = np.log(EPSILON), np.log(PREDICTION_MAX)
    lower_count, upper_count = int((log_absolute < low).sum()), int((log_absolute > high).sum())
    prediction = np.clip(np.exp(np.clip(log_absolute, low, high)), EPSILON, PREDICTION_MAX)
    return prediction, {"rows": len(scores), "lower_clip_count": lower_count,
                        "upper_clip_count": upper_count,
                        "clip_count": lower_count + upper_count}


def fit_b2(x_train, y_train, x_val, y_val, num_leaves, min_data_in_leaf):
    """One locked Gamma candidate, selected iteration by validation raw QLIKE.

    LightGBM's built-in Gamma objective supplies already exponentiated means
    to feval, unlike custom objectives. Both role targets are median-scaled;
    the metric restores *original supplied* effective targets to avoid label
    float32 rounding inside Dataset. Training metrics never select an iteration.
    """
    train_x, val_x = _b2_x(x_train), _b2_x(x_val)
    train_y, val_y = _effective_y(y_train), _effective_y(y_val)
    if len(train_y) != len(train_x) or len(val_y) != len(val_x):
        raise ValueError("B2 feature/target lengths differ")
    if (num_leaves, min_data_in_leaf) not in B2_GRID:
        raise ValueError("Unregistered B2 structure")
    cfg = _locked_b2()
    lgb = _lightgbm()
    scale = float(np.median(train_y))
    params = dict(cfg["params"], num_leaves=int(num_leaves),
                  min_data_in_leaf=int(min_data_in_leaf))
    train = lgb.Dataset(train_x, label=train_y / scale, params=params, free_raw_data=False)
    valid = lgb.Dataset(val_x, label=val_y / scale, reference=train,
                        params=params, free_raw_data=False)

    def raw_qlike(predictions, dataset):
        predictions = np.asarray(predictions, dtype=np.float64)
        if not np.isfinite(predictions).all() or (predictions <= 0).any():
            raise RuntimeError("Nonpositive/nonfinite Gamma mean supplied to metric")
        restored, _ = _restore_log_prediction(np.log(predictions), scale)
        targets = train_y if dataset is train else val_y
        return "raw_QLIKE", float(np.mean(qlike(targets, restored))), False

    history = {}
    booster = lgb.train(params, train, num_boost_round=cfg["num_boost_round_max"],
                        valid_sets=[train, valid], valid_names=["train", "validation"],
                        feval=raw_qlike,
                        callbacks=[lgb.record_evaluation(history),
                                   lgb.early_stopping(cfg["early_stopping_rounds"],
                                       first_metric_only=cfg["first_metric_only"],
                                       min_delta=cfg["early_stopping_min_delta"], verbose=False)])
    best = int(booster.best_iteration)
    if best <= 0 or best > cfg["num_boost_round_max"]:
        raise RuntimeError("Invalid B2 best iteration")
    model_text = booster.model_to_string(num_iteration=best)
    # The model dump includes native defaults in addition to caller parameters.
    effective_text = model_text.split("parameters:\n", 1)[1].split("end of parameters", 1)[0]
    effective = {}
    for line in effective_text.splitlines():
        if line.startswith("[") and line.endswith("]") and ": " in line:
            key, value = line[1:-1].split(": ", 1)
            effective[key] = value
    model = {"kind": "B2", "eligible": True, "feature_count": B2_WIDTH,
             "library_version": lgb.__version__, "protocol_sha256": PROTOCOL_SHA256,
             "model_text": model_text, "scale": scale, "median": scale,
             "num_leaves": int(num_leaves), "min_data_in_leaf": int(min_data_in_leaf),
             "best_iteration": best, "history": history,
             "training_iterations": len(history["train"]["raw_QLIKE"]),
             "requested_params": params, "effective_params": effective,
             "training_label_precision": {"library_dataset_dtype": str(train.get_label().dtype),
                 "original_target_dtype": "float64", "scaled_label_max_abs_rounding":
                 float(np.max(np.abs(np.asarray(train.get_label(), dtype=np.float64) - train_y / scale))),
                 "validation_metric_targets": "original float64 supplied targets; no Dataset rounding"},
             "effective_params_text": effective_text,
             "target_floor_count": int((np.asarray(y_train) < EPSILON).sum()),
             "prediction_clip": {"minimum": EPSILON, "maximum": PREDICTION_MAX}}
    predicted, stats = predict_b2_with_stats(model, val_x)
    model["validation_qlike"] = float(np.mean(qlike(val_y, predicted)))
    model["validation_prediction_stats"] = stats
    if not np.isclose(model["validation_qlike"], history["validation"]["raw_QLIKE"][best - 1],
                      rtol=1e-12, atol=1e-12):
        raise RuntimeError("B2 transformed metric/raw-score reload mismatch")
    return model


def predict_b2_with_stats(model, x):
    if (not model.get("eligible", False) or model.get("kind") != "B2"
            or model.get("feature_count") != B2_WIDTH
            or model.get("library_version") != "4.6.0"
            or model.get("protocol_sha256") != PROTOCOL_SHA256):
        raise ValueError("Invalid or ineligible B2 model")
    x = _b2_x(x)
    booster = _lightgbm().Booster(model_str=model["model_text"])
    if booster.num_feature() != B2_WIDTH:
        raise ValueError("Serialized B2 model width mismatch")
    raw_score = booster.predict(x, raw_score=True, num_iteration=model["best_iteration"], num_threads=1)
    return _restore_log_prediction(raw_score, model["scale"])


def predict_b2(model, x):
    return predict_b2_with_stats(model, x)[0]


def select_b2(candidates):
    """Return supplied candidate; support direct models and model wrappers."""
    def model(candidate):
        return candidate.get("model", candidate)
    eligible = [c for c in candidates if model(c).get("eligible", False)
                and np.isfinite(c["validation_qlike"])]
    if not eligible:
        raise ValueError("No eligible B2 candidate")
    if any((model(c)["num_leaves"], model(c)["min_data_in_leaf"]) not in B2_GRID for c in eligible):
        raise ValueError("Unregistered B2 selection candidate")
    return min(eligible, key=lambda c: (c["validation_qlike"], model(c)["num_leaves"],
                                        -model(c)["min_data_in_leaf"]))


def save_b2(model, path):
    """Exclusive creation preserves historical candidate evidence."""
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(model, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def load_b2(path):
    model = json.loads(Path(path).read_text(encoding="utf-8"))
    if model.get("kind") != "B2" or not model.get("eligible", False):
        raise ValueError("Saved model is not an eligible B2")
    return model
