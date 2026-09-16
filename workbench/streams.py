"""Causal numerical-stream learning with observable feedback and explicit baselines."""

import csv
import io
import math
from datetime import datetime

import numpy as np

WINDOWS = (8, 16, 32, 64)
METHODS = ("persistence", "drift", *(f"ar2_{n}" for n in WINDOWS))


def parse_csv(text):
    if not isinstance(text, str) or len(text) > 250_000:
        raise ValueError("Use a CSV smaller than 250 KB with timestamp,value columns")
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    if reader.fieldnames != ["timestamp", "value"]:
        raise ValueError("CSV header must be timestamp,value")
    rows = []
    for row in reader:
        if None in row or not row["timestamp"] or len(row["timestamp"]) > 80:
            raise ValueError("Each row needs a timestamp and one value")
        try:
            value = float(row["value"])
        except (TypeError, ValueError) as error:
            raise ValueError("Values must be finite numbers") from error
        if not math.isfinite(value) or abs(value) > 1e12:
            raise ValueError("Values must be finite and have magnitude at most 1e12")
        rows.append({"timestamp": row["timestamp"], "value": value})
    if not 8 <= len(rows) <= 1024 or len({r["timestamp"] for r in rows}) != len(rows):
        raise ValueError("Supply 8–1024 rows with distinct timestamps, ordered at a regular interval")
    try:
        coordinates = [float(row["timestamp"]) for row in rows]
    except ValueError:
        try:
            coordinates = [datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00")).timestamp() for row in rows]
        except ValueError as error:
            raise ValueError("Timestamps must be ISO dates/times or numeric observation indices") from error
    differences = np.diff(coordinates)
    if not np.isfinite(coordinates).all() or np.any(differences <= 0) or not np.allclose(differences, differences[0], rtol=1e-5, atol=1e-6):
        raise ValueError("Timestamps must increase at a regular interval; resample irregular data explicitly")
    return rows


def fit(values, method):
    y = np.asarray(values, dtype=np.float64)
    if y.ndim != 1 or len(y) < 3 or not np.isfinite(y).all():
        raise ValueError("A finite observed history is required")
    if method == "persistence":
        return np.array([1., 0., 0.])
    if method == "drift":
        return np.array([1., 0., float(np.mean(np.diff(y[-6:])))])
    if method not in METHODS:
        raise ValueError("Unknown update method")
    y = y[-int(method.split("_")[1]):]
    location = float(y.mean())
    scale = max(float(y.std()), 1e-9)
    z = (y-location)/scale
    x = np.column_stack([z[1:-1], z[:-2], np.ones(len(z)-2)])
    a, b, offset = np.linalg.solve(x.T@x+np.diag([.2, .2, .01]), x.T@z[2:])
    return np.array([a, b, offset*scale+location*(1-a-b)])


def predictions(values):
    models = np.array([fit(values, method) for method in METHODS])
    y = np.asarray(values)
    pred = models@np.array([y[-1], y[-2], 1.])
    scale = max(float(np.std(y[-32:])), 1e-6)
    # Supplied operational range guard; it is not a learned law or calibrated bound.
    return np.clip(pred, y[-1]-4*scale, y[-1]+4*scale), models


def features(values, predictions_, loss_history):
    y = np.asarray(values)
    scale = max(float(np.std(y[-32:])), 1e-6)
    loss = np.array(loss_history, dtype=float).reshape(-1, len(METHODS))
    columns = [np.ones(len(METHODS)), np.arange(len(METHODS))/len(METHODS)]
    for count in (1, 4, 12):
        columns.append(np.log1p(np.mean(loss[-count:], axis=0)/scale**2) if len(loss) else np.zeros(len(METHODS)))
    columns.extend([np.abs(predictions_-y[-1])/scale,
                    np.full(len(METHODS), abs(float(np.mean(np.diff(y[-8:]))))/scale),
                    np.full(len(METHODS), min(10., float(np.std(np.diff(y[-8:])))/scale))])
    return np.clip(np.stack(columns, axis=1), -20., 20.)


def choose(values, pred, losses, eta=None):
    if eta is not None:
        return int(np.argmin(features(values, pred, losses)@np.asarray(eta))), "learned_update_selection"
    if len(losses) < 4:
        return 0, "persistence_until_feedback"
    recent = np.asarray(losses[-12:])
    weights = .8**np.arange(len(recent)-1, -1, -1)
    return int(np.argmin(np.average(recent, axis=0, weights=weights))), "observed_error_selection"


def analyze(rows, horizon=12, *, eta=None):
    if type(horizon) is not int or not 1 <= horizon <= 48:
        raise ValueError("Forecast horizon must be 1–48 observations")
    values = [row["value"] for row in rows]
    losses, records = [], []
    for i in range(6, len(values)):
        history = values[:i]
        pred, _ = predictions(history)
        selection, procedure = choose(history, pred, losses, eta)
        residual = values[i]-float(pred[selection])
        prior_errors = np.array([r["error"] for r in records[-32:]])
        residual_scale = (float(np.median(np.abs(prior_errors-np.median(prior_errors))))*1.4826
                          if len(prior_errors) >= 8 else None)
        threshold = max(4*residual_scale, .05*max(float(np.std(history[-32:])), 1e-6)) if residual_scale is not None else None
        records.append({"index": i, "timestamp": rows[i]["timestamp"], "actual": values[i],
                        "prediction": float(pred[selection]), "persistence": float(pred[0]),
                        "error": residual, "method": METHODS[selection],
                        "flagged": threshold is not None and abs(residual) > threshold,
                        "threshold": threshold})
        losses.append(np.square(pred-values[i]).tolist())
    pred, models = predictions(values)
    choice, procedure = choose(values, pred, losses, eta)
    coefficients = models[choice]
    history = list(values)
    forecasts = []
    base_scale = max(float(np.std(values[-32:])), 1e-6)
    for step in range(horizon):
        value = float(coefficients@np.array([history[-1], history[-2], 1.]))
        value = float(np.clip(value, values[-1]-4*base_scale, values[-1]+4*base_scale))
        forecasts.append({"step": step+1, "value": value})
        history.append(value)
    return {"rows": len(rows), "last_timestamp": rows[-1]["timestamp"], "last_value": values[-1],
            "method": METHODS[choice], "procedure": procedure, "coefficients": coefficients.tolist(),
            "forecast": forecasts, "prequential": records,
            "mae": float(np.mean([abs(r["error"]) for r in records])),
            "persistence_mae": float(np.mean([abs(r["actual"]-r["persistence"]) for r in records])),
            "flags": sum(r["flagged"] for r in records), "fits": 4*(len(records)+1),
            "assumptions": "Ordered scalar observations at a regular interval; forecasts are conditional estimates. Flags indicate surprise, not a diagnosed cause. No calibrated prediction interval."}
