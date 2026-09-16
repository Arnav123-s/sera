"""Finite numerical task acquisition. No evaluator coefficients enter this module."""

import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import beta

from experiments.sparse_mechanisms.core import l1, omp

EXPONENTS = sorted((p for p in itertools.product(range(4), repeat=3) if sum(p) <= 3),
                   key=lambda p: (sum(p), p))


def fingerprint():
    inherited = Path(__file__).parents[1]/"sparse_mechanisms/core.py"
    return hashlib.sha256(Path(__file__).read_bytes()+inherited.read_bytes()).hexdigest()


def features(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all() or (abs(x) > 1).any():
        raise ValueError("Use finite normalized inputs with three columns in [-1, 1]")
    values = [np.ones_like(x), x, (3*x*x-1)/2, (5*x*x*x-3*x)/2]
    return np.column_stack([np.prod([values[p[j]][:, j]*np.sqrt(2*p[j]+1)
                                    for j in range(3)], axis=0) for p in EXPONENTS])


def evidence(x, y, role):
    x, y = np.asarray(x, float), np.asarray(y, float)
    features(x)
    if y.shape != (len(x),) or not np.isfinite(y).all() or abs(y).max(initial=0) > 1e9:
        raise ValueError("One finite bounded scalar output is required per example")
    if role not in ("fit", "selection", "calibration", "observation"):
        raise ValueError("Unknown evidence role")
    keys = [tuple(v) for v in x]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate inputs cannot count as independent evidence")
    return {"x": x.tolist(), "y": y.tolist(), "role": role}


def disjoint(*batches):
    seen = set()
    for batch in batches:
        keys = set(map(tuple, batch["x"]))
        if keys & seen:
            raise ValueError("Evidence roles must have disjoint inputs")
        seen |= keys


def learned_basis(coefficients):
    """Learn a rank-at-most-three task subspace from previously admitted readouts."""
    if not coefficients:
        return np.empty((20, 0)), []
    a = np.asarray(coefficients, float).T
    if a.shape[0] != 20 or not np.isfinite(a).all():
        raise ValueError("Invalid previous learned readouts")
    u, s, _ = np.linalg.svd(a, full_matrices=False)
    rank = min(3, int(np.sum(s > max(1e-10, s[0]*1e-7))))
    return u[:, :rank], s.tolist()


def candidate_models(fit, selection, basis=None, noise=0.):
    if fit["role"] != "fit" or selection["role"] != "selection":
        raise ValueError("Fitting and selection roles are required")
    disjoint(fit, selection)
    if not 4 <= len(fit["x"]) <= 256 or not 8 <= len(selection["x"]) <= 128:
        raise ValueError("Use 4..256 fit and 8..128 selection examples")
    if not np.isfinite(noise) or not 0 <= noise <= 1e3:
        raise ValueError("Declare a finite nonnegative observation noise bound")
    a, y = features(fit["x"]), np.asarray(fit["y"])
    v, target = features(selection["x"]), np.asarray(selection["y"])
    result = []

    def add(method, setting, producer, solves):
        start = time.perf_counter()
        try:
            coefficient, certificate = producer()
            result.append({"method": method, "setting": setting, "coefficients": coefficient.tolist(),
                           "selection_mse": float(np.mean((v@coefficient-target)**2)),
                           "certificate": certificate, "solves": solves,
                           "seconds": time.perf_counter()-start, "status": "OK"})
        except ValueError as error:
            result.append({"method": method, "setting": setting, "status": "FAILED",
                           "reason": str(error), "seconds": time.perf_counter()-start, "solves": solves})

    for alpha in (0., .0001, .01, .1, 1.):
        def ridge(alpha=alpha):
            w = (np.linalg.lstsq(a, y, rcond=None)[0] if alpha == 0 else
                 np.linalg.solve(a.T@a+alpha*np.eye(20), a.T@y))
            return w, []
        add("ridge", alpha, ridge, 1)
    for k in range(1, min(8, len(y)-1)+1):
        add("omp", k, lambda k=k: (omp(a, y[:, None], k)[:, 0], []), k)
    def sparse():
        w, cert = l1(a, y[:, None], noise)
        return w[:, 0], cert
    add("sparse", noise, sparse, 1)
    if basis is not None and basis.shape[1]:
        basis = np.asarray(basis, float)
        if basis.shape[0] != 20 or not np.isfinite(basis).all():
            raise ValueError("Invalid acquired task basis")
        add("subspace", basis.shape[1],
            lambda: (basis@np.linalg.lstsq(a@basis, y, rcond=None)[0], []), 1)
        def innovation():
            augmented = np.c_[a@basis, a]
            code, cert = l1(augmented, y[:, None], noise, prior=range(basis.shape[1]))
            return basis@code[:basis.shape[1], 0]+code[basis.shape[1]:, 0], cert
        add("innovation", noise, innovation, 1)
    return result


POLICIES = {"scratch": {"ridge", "omp", "sparse"},
            "transfer": {"ridge", "omp", "sparse", "subspace", "innovation"},
            "ridge": {"ridge"}, "sparse": {"sparse"}, "omp": {"omp"},
            "subspace": {"subspace"}, "innovation": {"innovation"}}


def choose(candidates, policy):
    viable = [c for c in candidates if c["status"] == "OK" and c["method"] in POLICIES[policy]]
    if not viable:
        return None
    # Selection outcomes are used once. Calibration is absent from this interface.
    return min(viable, key=lambda c: c["selection_mse"])


def calibrate(coefficients, calibration, tolerance):
    if calibration["role"] != "calibration" or not 64 <= len(calibration["x"]) <= 512:
        raise ValueError("Use 64..512 fresh calibration examples")
    if not np.isfinite(tolerance) or not 1e-8 <= tolerance <= 1e6:
        raise ValueError("Declare a positive finite absolute error tolerance")
    errors = abs(features(calibration["x"])@np.asarray(coefficients)-calibration["y"])
    failures = int(np.sum(errors > tolerance))
    upper = 1. if failures == len(errors) else float(beta.ppf(.95, failures+1, len(errors)-failures))
    return {"accepted": upper <= .05, "n": len(errors), "failures": failures,
            "max_error": float(errors.max()), "mean_error": float(errors.mean()),
            "risk_upper_95": upper, "tolerance": tolerance,
            "claim": "Per fixed model, 95% one-sided binomial upper bound on the probability of absolute error exceeding tolerance; conditional on fresh iid calibration and the same stationary deployment distribution. Not a per-query proof, simultaneous guarantee, or guarantee under shift."}


def acquire(fit, selection, calibration, *, basis=None, noise=0., tolerance=.05, policy="transfer"):
    disjoint(fit, selection, calibration)
    candidates = candidate_models(fit, selection, basis, noise)
    selected = choose(candidates, policy)
    if selected is None:
        raise ValueError("All acquisition candidates failed")
    gate = calibrate(selected["coefficients"], calibration, tolerance)
    batches = {"fit": fit, "selection": selection, "calibration": calibration}
    return {"schema": "sera.task-readout.1", "source": fingerprint(), "selected": selected,
            "candidates": candidates, "gate": gate, "batches": batches,
            "evidence_sha256": hashlib.sha256(json.dumps(batches, sort_keys=True).encode()).hexdigest(),
            "labels": sum(len(b["x"]) for b in batches.values()), "policy": policy,
            "noise_bound": noise, "domain": [[-1., 1.]]*3}
