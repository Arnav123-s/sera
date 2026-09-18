"""Independent scalar program evaluation and evidence-bound, non-repeatable credit."""

import math

import numpy as np

from experiments.verified_completion.common import digest


def evaluate(candidate, times):
    """Deliberately does not call the fitting module's matrix/basis evaluator."""
    program, weights = candidate["program"], candidate["weights"]
    values = []
    for time in times:
        u = (float(time) - candidate["center"]) / candidate["scale"]
        cuts = [(c - candidate["center"]) / candidate["scale"] for c in program["cuts"]]
        def polynomial(z):
            return [sum(c * z ** power for power, c in enumerate(coefficients))
                    for coefficients in candidate["basis"][:program["degree"] + 1]]
        if not cuts:
            features = polynomial(u)
        elif program["kind"] == "segments":
            branch = sum(u >= c for c in cuts)
            features = []
            for j in range(len(cuts) + 1):
                features.extend(polynomial(u - (cuts[j - 1] if j else 0.)) if j == branch else [0.] * (program["degree"] + 1))
        else:
            features = polynomial(u)
            for cut in cuts:
                features.extend(polynomial(max(0., u - cut))[1 if program["kind"] == "continuous" else 2:])
        if len(features) != len(weights):
            raise ValueError("Coefficient count changed")
        values.append(sum(w * f for w, f in zip(weights, features, strict=True)))
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite imagined consequence")
    return values


def assess_development(candidate, public):
    prediction = evaluate(candidate, public["time"])
    discrepancy = float(np.max(np.abs(np.array(prediction) - candidate["fit_predictions"])))
    if discrepancy > 1e-7:
        raise ValueError("Independent scalar execution differs")
    cv = np.array(candidate["development_predictions"])
    mse = float(np.mean((cv - public["value"]) ** 2))
    # Fixed mild complexity penalty; no final row influences it.
    score = mse * (1 + candidate["parameters"] / len(cv))
    return {"mse": mse, "score": score, "independent_execution_max_difference": discrepancy,
            "purpose": "development_search_only", "factual_credit": 0}


def qualify(candidate, independent, baseline, tolerance):
    values = evaluate(candidate, independent["time"])
    errors = np.array(values) - independent["value"]
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    baseline_rmse = float(np.sqrt(np.mean((np.array(baseline) - independent["value"]) ** 2)))
    return {"candidate": candidate["id"], "ids": independent["ids"], "predictions": values,
            "rmse": rmse, "baseline_rmse": baseline_rmse,
            "accepted": rmse < .5 * baseline_rmse and rmse < tolerance,
            "scope": "reserved measurements from the retained source; empirical interpolation",
            "causal_mechanism_identified": False}


def credit(candidate, independent, baseline, tolerance, used):
    result = qualify(candidate, independent, baseline, tolerance)
    credited, points = [], 0.
    if result["accepted"]:
        for identifier, y, old, new in zip(independent["ids"], independent["value"], baseline, result["predictions"], strict=True):
            if identifier not in used and abs(new - y) < tolerance and (new - y) ** 2 < (old - y) ** 2:
                credited.append(identifier)
                points += min(1., ((old - y) ** 2 - (new - y) ** 2) / max(tolerance ** 2, (old - y) ** 2))
    receipt = {"candidate": candidate["id"], "goal": candidate["goal"], "source": candidate["source"],
               "predictor": digest(candidate["weights"]), "evidence": digest(independent),
               "credited_ids": credited, "points": points, "assessment": result,
               "purpose": "independent_credit", "qualification": "empirical_prediction"}
    receipt["identity"] = digest(receipt)
    return receipt


def validate_receipt(candidate, receipt):
    if receipt["identity"] != digest({k: v for k, v in receipt.items() if k != "identity"}):
        raise ValueError("Forged credit")
    if receipt["candidate"] != candidate["id"] or receipt["source"] != candidate["source"] or receipt["goal"] != candidate["goal"]:
        raise ValueError("Credit belongs to a different proposal, source or goal")
    if receipt["predictor"] != digest(candidate["weights"]) or receipt["purpose"] != "independent_credit":
        raise ValueError("Stale or self-confirming credit")
    if set(receipt["credited_ids"]) & set(candidate["input_ids"]):
        raise ValueError("Fitting evidence cannot credit itself")
    if not math.isfinite(receipt["points"]) or receipt["points"] < 0:
        raise ValueError("Invalid credit")
