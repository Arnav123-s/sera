"""Independent probability calculations, inverse fits and qualification.

This assessor uses NumPy/SciPy, not the learner's forward or optimizer.
"""

import copy
import math

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

from experiments.gap_inquiry import digest
from experiments.intervention_model import KINDS, checked_observation, features, rows_arrays


def predict(programs, weights):
    # Deliberately reconstruct signed intervals rather than call owner.forward.
    values = []
    for program in programs:
        boundaries = [0, *program["pulses"], program["ticks"]]
        area = sum((right-left) * (-1)**i for i, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True))) / 4
        values.append([program["ticks"]/4, abs(area), len(program["pulses"])])
    return .5 + .5 * np.exp(-np.asarray(values)[:, :len(weights)] @ np.asarray(weights))


def nll(rows, weights):
    groups = [g for row in rows for g in row["groups"]]
    p = np.clip(predict([g["applied"] for g in groups], weights), 1e-12, 1-1e-12)
    n = np.asarray([g["shots"] for g in groups])
    k = np.asarray([g["plus"] for g in groups])
    return float(-(k*np.log(p)+(n-k)*np.log1p(-p)).sum()/n.sum())


def independent_fit(rows, kind):
    size = 2 if kind == "temporal" else 3
    x, n, k = rows_arrays(rows, size)

    def objective(w):
        e = .5*np.exp(-x@w)
        p = np.clip(.5+e, 1e-12, 1-1e-12)
        loss = -(k*np.log(p)+(n-k)*np.log1p(-p)).sum()/n.sum()
        grad = x.T @ (e * (k/p-(n-k)/(1-p))) / n.sum()
        return loss, grad

    result = minimize(objective, np.array([.2, .2, .03][:size]), jac=True, method="L-BFGS-B",
                      bounds=[(0., 10.)]*size, options={"maxiter": 300, "ftol": 1e-14, "gtol": 1e-9, "maxls": 40})
    if not np.isfinite(result.x).all():
        raise ValueError("Independent optimizer failed")
    return result.x, {"success": bool(result.success), "message": str(result.message), "iterations": int(result.nit), "nll": float(result.fun)}


def qualification(subject, record, weights, selection, audit):
    selection = [checked_observation(r, subject, record["source"], "selection") for r in selection]
    audit = [checked_observation(r, subject, record["source"], "adequacy") for r in audit]
    all_ids = [r["id"] for r in record["observations"] + selection + audit]
    if len(all_ids) != len(set(all_ids)) or len(selection) != 12 or len(audit) != 20:
        raise ValueError("Acquisition, selection and adequacy must be independent declared banks")
    scores = {k: nll(selection, weights[k]) for k in KINDS}
    selected = "pulse_loss" if scores["temporal"] - scores["pulse_loss"] > .0005 else "temporal"
    w = np.asarray(weights[selected])
    x, n, _ = rows_arrays(record["observations"], len(w))
    p = .5+.5*np.exp(-x@w)
    derivative = p-.5
    fisher = x.T @ ((n*derivative**2/np.maximum(p*(1-p), 1e-12))[:, None]*x)
    rank = int(np.linalg.matrix_rank(x))
    covariance = np.linalg.pinv(fisher, rcond=1e-10)
    xa, na, ka = rows_arrays(audit, len(w))
    pa = .5+.5*np.exp(-xa@w)
    prediction_variance = (pa-.5)**2 * np.einsum("ij,jk,ik->i", xa, covariance, xa)
    variance = na*pa*(1-pa) + na**2*prediction_variance
    z = (ka-na*pa) / np.sqrt(np.maximum(variance, 1e-9))
    statistic = float(np.sum(z*z))
    probability = float(chi2.sf(statistic, len(z)))
    adequate = bool(probability >= .001 and np.max(np.abs(z)) <= 5.0)
    identified = rank == len(w)
    fitted_shots = int(n.sum())
    independent, fit_report = independent_fit(record["observations"], selected)
    probe = [g["applied"] for r in selection + audit for g in r["groups"]]
    difference = float(np.max(np.abs(predict(probe, w)-predict(probe, independent))))
    fit_qualified = difference <= .002
    requested = sum(r["monitor"]["requested_pulse_shots"] for r in record["observations"])
    applied = sum(r["monitor"]["applied_pulse_shots"] for r in record["observations"])
    accepted = bool(adequate and identified and fit_qualified)
    q = {"subject": subject, "goal_id": record["goal_id"], "source": record["source"], "revision": record["revision"],
         "weight_identity": digest(weights), "evidence_identity": digest(record["observations"]),
         "selected": selected, "selection_scores": scores, "selection": selection, "audit": audit,
         "acquisition_shots": fitted_shots, "selection_shots": sum(g["shots"] for r in selection for g in r["groups"]),
         "adequacy_shots": int(na.sum()), "accepted": accepted,
         "adequacy": {"passed": adequate, "pearson_with_local_fit_variance": statistic,
                      "diagnostic_tail_probability": probability, "max_abs_standardized_error": float(np.max(np.abs(z))),
                      "group_count": len(z), "scope": "finite independent controls; a diagnostic gate, not completeness of the model family"},
         "identifiability": {"design_rank": rank, "parameters": len(w), "identified_within_candidate": identified},
         "independent_fit": {"weights": independent.tolist(), "max_prediction_difference": difference, "qualified": fit_qualified, **fit_report},
         "actuation": {"requested_pulse_shots": requested, "applied_pulse_shots": applied,
                       "observed_application_fraction": applied/requested if requested else None,
                       "meaning": "outcomes conditioned on monitor-recorded applied histories; commands are not assumed successful"},
         "next_action": "Return the qualified conditional answer to the original goal" if accepted else
                        "Acquire distinguishing controls" if not identified else "Investigate model inadequacy or optimizer discrepancy",
         "mathematical_parent_proved": False, "unique_physical_mechanism_proved": False}
    q["receipt"] = digest(q)
    return q


def same_qualification(a, b):
    def same(left, right):
        if isinstance(left, float) and isinstance(right, (float, int)):
            return math.isclose(left, right, rel_tol=1e-8, abs_tol=1e-10)
        if isinstance(left, dict) and isinstance(right, dict):
            return left.keys() == right.keys() and all(same(left[k], right[k]) for k in left)
        if isinstance(left, list) and isinstance(right, list):
            return len(left) == len(right) and all(same(x, y) for x, y in zip(left, right, strict=True))
        return left == right
    # Keep the stored receipt exact; tolerate only re-evaluated floating diagnostics.
    if a.get("receipt") != digest({k: v for k, v in a.items() if k != "receipt"}):
        return False
    aa, bb = copy.deepcopy(a), copy.deepcopy(b)
    aa.pop("receipt", None)
    bb.pop("receipt", None)
    return same(aa, bb)


def choose_program(session, subject, bank, policy, step):
    record = session.subjects[subject]
    if policy == "passive":
        choices = [p for p in bank if not p["pulses"]]
        return choices[step % len(choices)]
    if policy == "balanced":
        return bank[step % len(bank)]
    if policy != "information":
        raise ValueError("Unknown fixed acquisition policy")
    rows = record["observations"]
    x, n, _ = rows_arrays(rows, 3)
    weights = session.model(subject, "pulse_loss").weight.detach().numpy()
    p = .5 + .5*np.exp(-x@weights)
    gram = x.T @ ((n*(p-.5)**2/np.maximum(p*(1-p), 1e-12))[:, None]*x) + np.eye(3)*.1
    inverse = np.linalg.inv(gram)
    candidates = np.array([features(p) for p in bank])
    pp = .5+.5*np.exp(-candidates@weights)
    variance = (pp-.5)**2 * np.einsum("ij,jk,ik->i", candidates, inverse, candidates)
    score = variance/np.maximum(pp*(1-pp), 1e-12)
    return bank[int(np.argmax(score))]
