"""Finite, bounded empirical system identification; supplied candidate mechanisms."""
import numpy as np
from scipy.optimize import lsq_linear

from .data import DT

TAUS = (.1, .15, .25, .4, .65, 1., 1.6)


def design(v, u, kind, tau=None):
    columns = [u, np.ones(len(v))]
    if kind != "coarse":
        columns.append(-v)
    if kind == "memory":
        rho, z = np.exp(-DT / tau), 0.
        memory = []
        for value in v:
            memory.append(z)
            z = rho * z - (1 - rho) * value
        columns.extend([np.array(memory), rho ** np.arange(len(v))])
    return np.stack(columns, axis=1)


def fit_candidate(values, masks, controls, kind, tau=None, end=48):
    v, u = values[:, 1], np.asarray(controls)
    x = design(v[:-1], u, kind, tau)
    y = np.diff(v) / DT + 1
    valid = masks[:-1, 1] & masks[1:, 1]
    train = valid & (np.arange(len(u)) < end)
    if train.sum() < 20:
        raise ValueError("At least twenty observed velocity transitions are required")
    lo, hi = [0.3, -.5], [2., .5]
    if kind != "coarse":
        lo += [0.]; hi += [1.]
    if kind == "memory":
        lo += [0., -2.]; hi += [2., 2.]
    # A tiny fixed ridge prevents exact singular fits; no final-set tuning.
    xx = np.concatenate((x[train], np.eye(x.shape[1]) * 1e-4))
    yy = np.concatenate((y[train], np.zeros(x.shape[1])))
    fit = lsq_linear(xx, yy, bounds=(lo, hi), tol=1e-10)
    if not fit.success or not np.isfinite(fit.x).all():
        raise ValueError("Finite bounded identification did not converge")
    validation = valid & (np.arange(len(u)) >= 48) & (np.arange(len(u)) < 64)
    if validation.sum() < 6:
        raise ValueError("Insufficient independent observed validation transitions")
    return dict(kind=kind, tau=tau, coef=fit.x.tolist(),
                validation_mse=float(np.mean((x[validation] @ fit.x - y[validation]) ** 2)),
                observed_transitions=int(train.sum()), validation_transitions=int(validation.sum()),
                design_rank=int(np.linalg.matrix_rank(x[train])), optimality=float(fit.optimality))


def fit(values, masks, controls, force=None):
    values, masks = np.asarray(values), np.asarray(masks, dtype=bool)
    candidates = [fit_candidate(values, masks, controls, "coarse"),
                  fit_candidate(values, masks, controls, "instant")]
    candidates += [fit_candidate(values, masks, controls, "memory", tau) for tau in TAUS]
    eligible = candidates if force is None else [c for c in candidates if c["kind"] == force]
    best = min(c["validation_mse"] for c in eligible)
    near = [c for c in eligible if c["validation_mse"] <= best * 1.1 + .0025]
    chosen = min(near, key=lambda c: (len(c["coef"]), c["validation_mse"]))
    final = fit_candidate(values, masks, controls, chosen["kind"], chosen["tau"], end=len(controls))
    final["selection_validation_mse"] = chosen["validation_mse"]
    final["validation_mse_after_refit_is_not_independent"] = final.pop("validation_mse")
    final["candidates"] = candidates
    return final


def initial_force(model, history):
    if model["kind"] != "memory":
        return 0.
    rho = np.exp(-DT / model["tau"])
    f = model["coef"][4]
    for v in np.asarray(history)[:-1, 1]:
        f = rho * f - (1 - rho) * model["coef"][3] * v
    return float(f)


def predict(model, history, commands, independent=False):
    coef = model["coef"]
    b, w = coef[:2]
    d = coef[2] if len(coef) >= 3 else 0.
    c = coef[3] if model["kind"] == "memory" else 0.
    rho = np.exp(-DT / model["tau"]) if model["kind"] == "memory" else 0.
    h, v = np.asarray(history)[-1]
    f, rows = initial_force(model, history), []
    if independent and model["kind"] == "memory":
        vs = np.asarray(history)[:-1, 1]
        f = rho ** len(vs) * coef[4] - (1-rho) * c * np.dot(rho ** np.arange(len(vs)-1, -1, -1), vs)
    for command in commands:
        offset = -1 + b * command + w
        if independent:
            matrix = np.array([[1, DT*(1-DT*d), DT*DT, DT*DT*offset],
                               [0, 1-DT*d, DT, DT*offset],
                               [0, -(1-rho)*c, rho, 0], [0, 0, 0, 1]])
            h, v, f, _ = matrix @ [h, v, f, 1]
        else:
            nv = v + DT * (offset - d * v + f)
            f = rho * f - (1-rho) * c * v
            h, v = h + DT * nv, nv
        rows.append([float(h), float(v)])
    return np.asarray(rows)
