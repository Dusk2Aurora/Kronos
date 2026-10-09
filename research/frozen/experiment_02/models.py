"""One deterministic float64 L2 log-link QLIKE head for every feature family."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

EPSILON = 1e-12
PREDICTION_MAX = 1.0
LAMBDA_CANDIDATES = (.001, .01, .1, 1.)


def _matrix(x):
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Features must be a finite 2D matrix")
    return values


def _effective_y(y):
    values = np.asarray(y, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Variance targets must be finite nonnegative 1D values")
    return np.maximum(values, EPSILON)


def objective_and_gradient(theta, x_scaled, y_scaled, lam):
    """No target winsorization or objective clipping; reject overflow trial steps."""
    theta = np.asarray(theta, dtype=np.float64)
    x_scaled = np.asarray(x_scaled, dtype=np.float64)
    y_scaled = np.asarray(y_scaled, dtype=np.float64)
    beta, intercept = theta[:-1], theta[-1]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        score = x_scaled @ beta + intercept
        exponent = -score
        if (not np.isfinite(exponent).all()
                or (exponent > np.log(np.finfo(np.float64).max)).any()
                or (exponent < np.log(np.nextafter(np.float64(0), np.float64(1)))).any()):
            return float("inf"), np.zeros_like(theta)
        inverse = np.exp(exponent)
        ratio = y_scaled * inverse
        value = np.mean(score + ratio) + lam * np.dot(beta, beta)
        residual = 1 - ratio
        gradient = np.r_[x_scaled.T @ residual / len(y_scaled) + 2 * lam * beta,
                         residual.mean()]
    if not np.isfinite(value) or not np.isfinite(gradient).all():
        return float("inf"), np.zeros_like(theta)
    return float(value), gradient


def fit_head(train_x, train_y, lam):
    """Fit only supplied training rows; return a JSON-serializable audit model.

    A failed solver or gradient guard returns eligible=False. Such a candidate
    must be recorded and may not be promoted or used for prediction.
    """
    x, y = _matrix(train_x), _effective_y(train_y)
    lam = float(lam)
    if len(y) != len(x) or not len(y) or lam not in LAMBDA_CANDIDATES:
        raise ValueError("Wrong training length or unregistered lambda")
    scaler = StandardScaler().fit(x)
    mean, scale = scaler.mean_, scaler.scale_
    scaled_x = scaler.transform(x)
    median = float(np.median(y))
    scaled_y = y / median
    initial = np.r_[np.zeros(x.shape[1]), np.log(scaled_y.mean())]
    options = {"maxiter": 3000, "maxls": 50, "ftol": 1e-12, "gtol": 1e-7}
    result = minimize(objective_and_gradient, initial,
                      args=(scaled_x, scaled_y, lam), jac=True,
                      method="L-BFGS-B", options=options)
    objective, gradient = objective_and_gradient(result.x, scaled_x, scaled_y, lam)
    finite = bool(np.isfinite(objective) and np.isfinite(result.x).all())
    max_abs_gradient = float(np.max(np.abs(gradient))) if finite else None
    eligible = bool(result.success and finite and max_abs_gradient <= 1e-5)
    return {"mean": mean.tolist(), "scale": scale.tolist(),
            "coef": result.x[:-1].tolist(), "intercept": float(result.x[-1]),
            "median": median, "lambda": lam, "eligible": eligible,
            "target_floor_count": int((np.asarray(train_y, dtype=np.float64) < EPSILON).sum()),
            "solver": {"method": "L-BFGS-B", "options": options,
                       "success": bool(result.success), "status": int(result.status),
                       "message": str(result.message), "iterations": int(result.nit),
                       "function_evaluations": int(result.nfev),
                       "gradient_evaluations": int(result.njev),
                       "objective": objective if finite else None,
                       "max_abs_gradient": max_abs_gradient},
            "prediction_clip": {"minimum": EPSILON, "maximum": PREDICTION_MAX}}


def predict_head_with_stats(model, x):
    """Return clipped absolute variance and counts, without changing the model."""
    if not model.get("eligible", False):
        raise ValueError("Failed candidate may not be promoted or predicted")
    x = _matrix(x)
    mean, scale, coef = (np.asarray(model[k], dtype=np.float64)
                         for k in ("mean", "scale", "coef"))
    if x.shape[1] != len(coef):
        raise ValueError("Prediction feature width mismatch")
    score = (x - mean) / scale @ coef + model["intercept"]
    # Evaluate the identical fixed variance clip in log space, avoiding overflow.
    log_prediction = np.log(model["median"]) + score
    lower, upper = np.log(EPSILON), np.log(PREDICTION_MAX)
    low_count, high_count = int((log_prediction < lower).sum()), int((log_prediction > upper).sum())
    prediction = np.exp(np.clip(log_prediction, lower, upper))
    prediction = np.clip(prediction, EPSILON, PREDICTION_MAX)
    return prediction, {"rows": len(x), "clip_count": low_count + high_count,
                        "lower_clip_count": low_count, "upper_clip_count": high_count}


def predict_head(model, x):
    return predict_head_with_stats(model, x)[0]


def clip_predictions(prediction):
    """Apply the same absolute variance clip to persistence/EWMA predictions."""
    prediction = np.asarray(prediction, dtype=np.float64)
    if prediction.ndim != 1 or not np.isfinite(prediction).all() or (prediction < 0).any():
        raise ValueError("Raw R0 predictions must be finite nonnegative 1D values")
    low_count = int((prediction < EPSILON).sum())
    high_count = int((prediction > PREDICTION_MAX).sum())
    return np.clip(prediction, EPSILON, PREDICTION_MAX), {
        "rows": len(prediction), "clip_count": low_count + high_count,
        "lower_clip_count": low_count, "upper_clip_count": high_count}


def _metric_inputs(y, p):
    y = _effective_y(y)
    p = np.asarray(p, dtype=np.float64)
    if p.shape != y.shape or not np.isfinite(p).all() or (p <= 0).any():
        raise ValueError("Predictions must be finite positive and aligned")
    return y, p


def qlike(y, p):
    y, p = _metric_inputs(y, p)
    return np.log(p) + y / p


def regret(y, p):
    y, p = _metric_inputs(y, p)
    ratio = y / p
    return ratio - np.log(ratio) - 1


def log_mse(y, p):
    y, p = _metric_inputs(y, p)
    return (np.log(y) - np.log(p)) ** 2


def select_candidate(candidates):
    """Candidates supply model and validation_qlike; exact ties favor larger L2."""
    eligible = [c for c in candidates if c["model"]["eligible"]
                and np.isfinite(c["validation_qlike"])]
    if not eligible:
        raise ValueError("No converged candidate is eligible")
    return min(eligible, key=lambda c: (c["validation_qlike"], -c["model"]["lambda"]))
