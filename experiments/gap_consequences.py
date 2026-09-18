"""Executable residual consequences of the learner's own retained programs."""

import math


def derivative(candidate, time, order):
    """Differentiate the represented polynomial; branch boundaries stay explicit."""
    if order not in {0, 1, 2} or not math.isfinite(time):
        raise ValueError("A finite coordinate and derivative order zero, one or two are required")
    program = candidate["program"]
    if any(abs(time - cut) < 1e-10 for cut in program["cuts"]):
        raise ValueError("A conditional boundary needs one-sided consequences")
    u = (time - candidate["center"]) / candidate["scale"]
    cuts = [(c - candidate["center"]) / candidate["scale"] for c in program["cuts"]]

    def powers(z):
        return [sum(coefficient * math.factorial(power) / math.factorial(power - order) * z ** (power - order)
                    for power, coefficient in enumerate(p) if power >= order) / candidate["scale"] ** order
                for p in candidate["basis"][:program["degree"] + 1]]

    if not cuts:
        features = powers(u)
    elif program["kind"] == "segments":
        branch = sum(u >= c for c in cuts)
        features = []
        for j in range(len(cuts) + 1):
            features.extend(powers(u - (cuts[j - 1] if j else 0.)) if j == branch else [0.] * (program["degree"] + 1))
    else:
        features = powers(u)
        start = 1 if program["kind"] == "continuous" else 2
        for cut in cuts:
            features.extend(powers(u - cut)[start:] if u > cut else [0.] * (program["degree"] + 1 - start))
    return sum(w * x for w, x in zip(candidate["weights"], features, strict=True))


def residual_consequences(session, time):
    record = next(iter(session.base.evidence.values()))
    # This binding is inherited from the earlier independently checked route.
    if record["evaluation"]["used_columns"] != ["Time (s)", "Position (m)"]:
        raise ValueError("A retained time-position binding is required")
    old_acceleration = float(session.owner.observed_parameters[record["id"]][2])
    rows = []
    for key, candidate in session.models.items():
        owned = {**candidate, "weights": session.owner.gap_models[key].detach().tolist()}
        try:
            acceleration = derivative(owned, time, 2)
            rows.append({"model": key, "position": derivative(owned, time, 0),
                         "velocity": derivative(owned, time, 1), "acceleration": acceleration,
                         "additional_acceleration_relative_to_retained_model": acceleration - old_acceleration,
                         "assumptions": candidate["program"]})
        except ValueError as error:
            rows.append({"model": key, "status": "BOUNDARY_NEEDS_SEPARATE_CHECK", "reason": str(error)})
    return {"time": time, "retained_acceleration": old_acceleration, "alternatives": rows,
            "kind": "MODEL_CONDITIONAL_MISSING_INFLUENCE", "measured_force": False,
            "derivation": "Differentiate each learned position program twice, then subtract the retained acceleration.",
            "next_obligation": "Observe an independent explanatory variable to distinguish physical causes."}
