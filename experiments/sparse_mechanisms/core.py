"""Common supplied expressions, factual evidence and competing recovery procedures."""

import hashlib
import itertools
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

from experiments.generative_memory.core import Observations, fit_linear, term_value


def fingerprint():
    inherited = Path(__file__).parents[1]/"generative_memory/core.py"
    return hashlib.sha256(Path(__file__).read_bytes()+inherited.read_bytes()).hexdigest()


def dictionary(p):
    if p not in (7, 17, 33):
        raise ValueError("Use a declared finite dictionary")
    return [["one"]]+[[op, 2*k] for k in range(1, (p+1)//2) for op in ("cos", "sin")]


def matrix(terms, t):
    # The existing AST uses pi*t; integer cycle k is therefore encoded as 2*k.
    return np.column_stack([term_value(term, np.asarray(t, float))*(1 if term[0] == "one" else np.sqrt(2)) for term in terms])


def relative(prediction, truth):
    return float(np.linalg.norm(np.asarray(prediction)-truth)/max(np.linalg.norm(truth), 1e-12))


def observations(t, y, role):
    return Observations(np.asarray(t), np.asarray(y), role, "observed")


def diagnostics(a):
    norms = np.linalg.norm(a, axis=0)
    keep = norms > 1e-10
    b = a[:, keep]/norms[keep]
    gram = b.T@b
    np.fill_diagonal(gram, 0)
    duplicate_pairs = np.argwhere(np.triu(np.abs(gram) > 1-1e-9, 1))
    ids = np.flatnonzero(keep)
    return {"rank": int(np.linalg.matrix_rank(a)), "zero_columns": np.flatnonzero(~keep).tolist(),
            "coherence": float(abs(gram).max(initial=0)),
            "aliased_pairs": [[int(ids[i]), int(ids[j])] for i, j in duplicate_pairs],
            "certificate": False}


def l1(a, y, epsilon, *, prior=()):
    """Basis pursuit / bounded-infinity-noise LP; prior means unpenalized support."""
    p = a.shape[1]
    penalty = np.ones(p)
    penalty[list(prior)] = 0
    doubled = np.c_[a, -a]
    coefficients, metadata = [], []
    for column in y.T:
        args = {"A_eq": doubled, "b_eq": column} if epsilon == 0 else {
            "A_ub": np.r_[doubled, -doubled], "b_ub": np.r_[column+epsilon, -column+epsilon]}
        result = linprog(np.r_[penalty, penalty], bounds=(0, None), method="highs",
                         options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9}, **args)
        if not result.success:
            raise ValueError(f"Sparse LP failed: {result.status}: {result.message}")
        estimate = result.x[:p]-result.x[p:]
        if epsilon == 0:
            dual = result.eqlin.marginals
            gap = abs(penalty@abs(estimate)-column@dual)
            violation = max(0., float((abs(a.T@dual)-penalty).max()))
        else:
            dual = result.ineqlin.marginals
            gap = abs(penalty@abs(estimate)-args["b_ub"]@dual)
            violation = max(0., float((args["A_ub"].T@dual-np.r_[penalty, penalty]).max()), float(dual.max()))
        metadata.append({"iterations": int(result.nit), "status": int(result.status),
                         "objective_gap": float(gap), "dual_violation": violation,
                         "dual": dual.tolist(), "epsilon": float(epsilon), "primal": estimate.tolist()})
        coefficients.append(estimate)
    return np.column_stack(coefficients), metadata


def omp(a, y, count):
    """Orthogonal matching pursuit; support count is selected from a fixed grid."""
    norm = np.linalg.norm(a, axis=0)
    normalized = a/np.maximum(norm, 1e-15)
    result = np.zeros((a.shape[1], y.shape[1]))
    for output in range(y.shape[1]):
        chosen, residual = [], y[:, output].copy()
        for _ in range(count):
            scores = abs(normalized.T@residual)
            scores[chosen] = -1
            scores[norm < 1e-12] = -1
            index = int(scores.argmax())
            if scores[index] <= 1e-12:
                break
            chosen.append(index)
            weights = np.linalg.lstsq(a[:, chosen], y[:, output], rcond=None)[0]
            residual = y[:, output]-a[:, chosen]@weights
        if chosen:
            result[chosen, output] = weights
    return result


def fit(method, support, selection, terms, noise_bound, precision="float64"):
    support.require("support")
    selection.require("selection")
    if set(support.t) & set(selection.t):
        raise ValueError("Support and selection overlap")
    if precision not in ("float32", "float64") or noise_bound < 0:
        raise ValueError("Invalid precision/noise contract")
    start = time.perf_counter()
    cast = np.float32 if precision == "float32" else np.float64
    a, y = matrix(terms, support.t).astype(cast).astype(float), support.y.astype(cast).astype(float)
    selection_a = matrix(terms, selection.t)
    candidates, certificates = [], []
    epsilon = noise_bound+(5e-7*max(1., float(abs(y).max())) if precision == "float32" else 0.)
    if method == "basis_pursuit":
        value, certificates = l1(a, y, epsilon)
        candidates.append(("fixed-noise-bound", value, 0.))
    elif method == "omp":
        for k in range(1, min(6, len(support.t)-1)+1):
            candidates.append((k, omp(a, y, k), 0.))
    elif method == "ridge":
        for alpha in (0., 1e-6, .001, .1, 1.):
            value = np.linalg.lstsq(a, y, rcond=None)[0] if alpha == 0 else np.linalg.solve(a.T@a+alpha*np.eye(len(terms)), a.T@y)
            candidates.append((alpha, value, 0.))
    elif method == "sera_subset":
        if len(terms) != 7:
            raise ValueError("Exhaustive existing-route control is restricted to seven terms")
        for k in range(4):
            for subset in itertools.combinations(range(1, len(terms)), k):
                ids = [0, *subset]
                spec = {"kind": "expression", "name": "common-dictionary-subset", "terms": [terms[i] for i in ids]}
                model = fit_linear(spec, support, max(noise_bound, 1e-5))
                value = np.zeros((len(terms), 2))
                value[ids] = np.array(model["mean"]).reshape(-1, 2)/np.array([1, *([np.sqrt(2)]*k)])[:, None]
                candidates.append((ids, value, 1e-4*len(model["mean"])))
    else:
        raise ValueError("Unknown recovery method")
    trace = []
    for setting, value, penalty in candidates:
        score = float(np.mean((selection_a@value-selection.y)**2))
        trace.append({"setting": setting, "mse": score, "penalty": penalty})
    selected = min(range(len(trace)), key=lambda i: trace[i]["mse"]+trace[i]["penalty"])
    value = candidates[selected][1].astype(cast).astype(float)
    artifact = {"schema": "sera.sparse-mechanism.1", "terms": terms, "coefficients": value.tolist(),
                "method": method, "source": fingerprint(), "precision": precision,
                "support": {"t": support.t.tolist(), "y": support.y.tolist()},
                "selection": {"t": selection.t.tolist(), "y": selection.y.tolist()},
                "normalization": "constant=1; every sinusoid=sqrt(2); phase is supplied in cycles",
                "measurement_diagnostics": diagnostics(a), "fit_trace": trace, "selected": selected,
                "certificates": certificates, "noise_bound": noise_bound, "effective_epsilon": epsilon,
                "work": {"candidate_fits": len(candidates), "scalar_support_values": support.y.size,
                         "scalar_selection_values": selection.y.size, "wall_seconds": time.perf_counter()-start}}
    return artifact


def predict(artifact, t):
    if artifact["source"] != fingerprint():
        raise ValueError("Sparse interpreter changed; explicit migration required")
    return matrix(artifact["terms"], t)@np.asarray(artifact["coefficients"])


def guard(artifact, calibration):
    calibration.require("calibration")
    previous = set(artifact["support"]["t"]) | set(artifact["selection"]["t"])
    if previous & set(calibration.t):
        raise ValueError("Calibration overlaps earlier evidence")
    error = relative(predict(artifact, calibration.t), calibration.y)
    diag = artifact["measurement_diagnostics"]
    accepted = error <= .05 and not diag["zero_columns"] and not diag["aliased_pairs"]
    return {"accepted": bool(accepted), "relative_calibration_error": error,
            "scalar_calibration_values": calibration.y.size,
            "certificate": False, "scope": "Empirical residual and alias diagnostics; not a global adequacy or causal certificate."}
