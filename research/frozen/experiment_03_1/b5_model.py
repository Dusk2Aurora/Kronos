"""Array-only B5 causal TCN. Callers own TRAIN preprocessing and split roles."""
from __future__ import annotations

import copy
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


FLOOR = 1e-12
DILATIONS = (1, 2, 4, 8, 16, 32, 64)


def configure(seed: int, device: str) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if str(device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA is unavailable")
        torch.cuda.manual_seed_all(seed)


class CausalConv(nn.Conv1d):
    def forward(self, x):
        return super().forward(F.pad(x, (2 * self.dilation[0], 0)))


class B5TCN(nn.Module):
    """RF=257, with all 256 history positions available to the final hidden."""
    def __init__(self, width: int, initial_bias: float = 0.0):
        super().__init__()
        if width <= 0:
            raise ValueError("width must be positive")
        self.width = int(width)
        self.input_conv = CausalConv(11, width, 3)
        self.blocks = nn.ModuleList([CausalConv(width, width, 3, dilation=d) for d in DILATIONS])
        self.head = nn.Linear(width + 33, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, initial_bias)

    def hidden(self, x):
        h = self.input_conv(x.transpose(1, 2))
        for conv in self.blocks:
            h = h + F.gelu(conv(h))
        return h.transpose(1, 2)

    def forward(self, x, ordinary):
        return self.head(torch.cat((self.hidden(x)[:, -1], ordinary), dim=1)).squeeze(-1)


def _inputs(x, ordinary):
    x = np.asarray(x, dtype=np.float32)
    ordinary = np.asarray(ordinary, dtype=np.float32)
    if x.ndim != 3 or x.shape[1:] != (256, 11) or ordinary.shape != (len(x), 33) or not len(x):
        raise ValueError("Expected nonempty history (N,256,11) and ordinary (N,33)")
    if not np.isfinite(x).all() or not np.isfinite(ordinary).all():
        raise ValueError("Nonfinite inputs")
    return np.ascontiguousarray(x), np.ascontiguousarray(ordinary)


def raw_qlike(y, prediction):
    """Original-unit QLIKE in float64, without the irrelevant target constant."""
    y = np.asarray(y, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if y.shape != prediction.shape or not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError("Invalid loss arrays")
    if (y < 0).any() or (prediction <= 0).any():
        raise ValueError("QLIKE requires nonnegative target and positive prediction")
    return float(np.mean(np.log(prediction) + np.maximum(y, FLOOR) / prediction, dtype=np.float64))


def predict_b5(model_or_payload, x, ordinary, *, scale=None, device=None, batch_size=64):
    if isinstance(model_or_payload, dict):
        model = model_or_payload["model"]
        if scale is None:
            scale = model_or_payload["metadata"]["target_scale"]
    else:
        model = model_or_payload
    if scale is None or not np.isfinite(scale) or scale <= 0 or batch_size <= 0:
        raise ValueError("Positive TRAIN target scale and batch size required")
    x, ordinary = _inputs(x, ordinary)
    device = device or str(next(model.parameters()).device)
    model.to(device=device, dtype=torch.float32)
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            z = model(torch.from_numpy(x[start:start + batch_size]).to(device),
                      torch.from_numpy(ordinary[start:start + batch_size]).to(device))
            if not torch.isfinite(z).all():
                raise FloatingPointError("Nonfinite B5 scores")
            scores.append(z.cpu().numpy())
    score = np.concatenate(scores)
    log_rv = score.astype(np.float64) + np.log(float(scale))
    low, high = log_rv < np.log(FLOOR), log_rv > 0
    prediction = np.exp(np.clip(log_rv, np.log(FLOOR), 0))
    prediction = np.clip(prediction, FLOOR, 1.0)
    return {"prediction": prediction, "raw_score": score,
            "clip_counts": {"lower": int(low.sum()), "upper": int(high.sum())}}


def fit_b5(xtrain, ordinary_train, ytrain, xval, ordinary_val, yval,
           width, weight_decay, seed, device, max_epochs=120, patience=15, batch_size=64, lr=0.001):
    configure(seed, device)
    xtrain, ordinary_train = _inputs(xtrain, ordinary_train)
    xval, ordinary_val = _inputs(xval, ordinary_val)
    ytrain = np.asarray(ytrain, dtype=np.float64)
    yval = np.asarray(yval, dtype=np.float64)
    for y, x in ((ytrain, xtrain), (yval, xval)):
        if y.shape != (len(x),) or not np.isfinite(y).all() or (y < 0).any():
            raise ValueError("Invalid target array")
    if max_epochs <= 0 or patience <= 0 or batch_size <= 0 or weight_decay < 0:
        raise ValueError("Invalid training budget")
    effective_y = np.maximum(ytrain, FLOOR)
    scale = float(np.median(effective_y))
    scaled64 = effective_y / scale
    scaled = scaled64.astype(np.float32)
    if not np.isfinite(scaled).all():
        raise FloatingPointError("Nonfinite float32 targets")
    rounding = float(np.max(np.abs(scaled.astype(np.float64) - scaled64) / scaled64))
    model = B5TCN(width, np.log(np.mean(scaled64))).to(device=device, dtype=torch.float32)
    weights, biases = [], []
    for name, parameter in model.named_parameters():
        (biases if name.endswith("bias") else weights).append(parameter)
    optimizer = torch.optim.AdamW([{"params": weights, "weight_decay": weight_decay},
                                  {"params": biases, "weight_decay": 0.0}], lr=lr)
    tx, to, ty = (torch.from_numpy(a).to(device) for a in (xtrain, ordinary_train, scaled))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best_loss, best_epoch, best_state, stale = np.inf, 0, None, 0
    history = []
    started = time.perf_counter()
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(tx), generator=generator)
        objective_sum = 0.0
        for start in range(0, len(tx), batch_size):
            idx = order[start:start + batch_size].to(device)
            optimizer.zero_grad(set_to_none=True)
            z = model(tx[idx], to[idx])
            loss = (z + ty[idx] * torch.exp(-z)).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite gradient")
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError("Nonfinite parameters")
            objective_sum += float(loss.detach()) * len(idx)
        train_prediction = predict_b5(model, xtrain, ordinary_train, scale=scale, batch_size=batch_size)
        val_prediction = predict_b5(model, xval, ordinary_val, scale=scale, batch_size=batch_size)
        train_loss = raw_qlike(ytrain, train_prediction["prediction"])
        val_loss = raw_qlike(yval, val_prediction["prediction"])
        history.append({"epoch": epoch, "train_scaled_objective": objective_sum / len(tx),
                        "train_raw_qlike": train_loss, "valid_raw_qlike": val_loss,
                        "train_clip_counts": train_prediction["clip_counts"],
                        "valid_clip_counts": val_prediction["clip_counts"]})
        if val_loss < best_loss:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    metadata = {"width": int(width), "target_scale": scale, "seed": int(seed),
                "weight_decay": float(weight_decay), "lr": float(lr), "batch_size": int(batch_size),
                "max_epochs": int(max_epochs), "patience": int(patience), "best_epoch": best_epoch,
                "best_valid_raw_qlike": best_loss, "target_rounding_max_relative_error": rounding,
                "receptive_field": 257, "history_shape": [256, 11], "ordinary_dim": 33,
                "input_channels": "six_market_plus_five_known_calendar",
                "calendar_fixed_scales": [59, 23, 6, 31, 12],
                "calendar_window_normalized": False, "dtype": "float32", "selection_loss_dtype": "float64", "finite_checks_passed": True,
                "torch_version": str(torch.__version__), "device": str(device)}
    resources = {"elapsed_seconds": time.perf_counter() - started,
                 "parameters": sum(p.numel() for p in model.parameters()),
                 "epochs_run": len(history), "threads": torch.get_num_threads(),
                 "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if str(device).startswith("cuda") else 0}
    return {"model": model, "state_dict": best_state, "metadata": metadata,
            "history": history, "resources": resources}


def save_b5(payload, path):
    """Save only weights plus primitive metadata; never pickle a model object."""
    state = {k: v.detach().cpu().clone() for k, v in payload["model"].state_dict().items()}
    torch.save({"state_dict": state, "metadata": copy.deepcopy(payload["metadata"])}, Path(path))


def load_b5(path, *, device="cpu"):
    stored = torch.load(Path(path), map_location="cpu", weights_only=True)
    model = B5TCN(stored["metadata"]["width"])
    model.load_state_dict(stored["state_dict"], strict=True)
    model.to(device=device, dtype=torch.float32).eval()
    return {"model": model, "state_dict": stored["state_dict"], "metadata": stored["metadata"]}

