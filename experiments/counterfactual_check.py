"""Independent conditional checks and concealed simulated evidence.

Reference coefficients and inverse solves are never exposed to the proposer.
An observation receipt establishes a simulated measurement, not a physical event.
"""

import json
from fractions import Fraction as Q
from functools import lru_cache

import numpy as np
import sympy as sp

from experiments.gap_inquiry import digest
from experiments.self_chosen import equations as eq

Z = sp.Symbol("z", real=True)


def _q(v):
    f = Q(str(v))
    return sp.Rational(f.numerator, f.denominator)


@lru_cache(maxsize=128)
def inverse(domain, pivot, observed):
    world = eq.symbolic_world(domain)
    target = sp.Symbol(observed, real=True)
    solutions = sp.solve(world[observed] - target, world[pivot])
    if len(solutions) != 1:
        raise ValueError("Independent model needs a separately qualified inverse branch")
    return solutions[0]


def reference_world(spec, context, z):
    world = _reference_world(json.dumps(spec, sort_keys=True), json.dumps(context, sort_keys=True))
    if z == Z:
        return world
    return {n: v.subs(Z, z) for n, v in world.items()}


@lru_cache(maxsize=8192)
def _reference_world(spec_json, context_json):
    spec, context = json.loads(spec_json), json.loads(context_json)
    return structural_world(spec, {n: _q(context[n]) for n in spec["model"]["controls"]}, Z)


def structural_world(spec, inputs, z):
    model = spec["model"]
    symbols = {n: sp.Symbol(n, real=True) for n in eq.layout(model["domain"])}
    inputs = dict(inputs)
    inputs[spec["axis"]] *= z
    other, power = spec["companion"], spec["power"]
    if other:
        inputs[other] *= (2 - z) if power == "reverse" else z ** power
    if model["pivot"]:
        formula = inverse(model["domain"], model["pivot"], model["controlled_observation"])
        inputs[model["pivot"]] = formula.subs({symbols[n]: v for n, v in inputs.items()}, simultaneous=True)
    world = eq.symbolic_world(model["domain"])
    return {n: sp.cancel(v.subs({symbols[k]: x for k, x in inputs.items()}, simultaneous=True)) for n, v in world.items()}


def unpack(record):
    numerator = sum(_q(c) * Z ** i for i, c in enumerate(record["n"]))
    denominator = sum(_q(c) * Z ** i for i, c in enumerate(record["d"]))
    return numerator / denominator


def check_candidate(question, candidate):
    try:
        world = reference_world(candidate["spec"], question["context"], Z)
        expected = world[question["target"]]
        residual = sp.cancel(expected - unpack(candidate["curve"]))
        derivative_residual = sp.cancel(sp.diff(expected, Z) - unpack(candidate["analysis"]["derivative"]))
        actual_limit = sp.limit(expected, Z, 0, dir="+")
        limit = "+infinity" if actual_limit == sp.oo else "-infinity" if actual_limit == -sp.oo else str(actual_limit)
        positive = [unpack(r) for r in candidate["conditions"]["positive"]]
        required = [world[n] for n in ("t", "m") if n in world]
        guard_ok = len(positive) == len(required) and all(sp.cancel(a - b) == 0 for a, b in zip(positive, required, strict=True))
        integers = candidate["conditions"]["integer"]
        guard_ok = guard_ok and (len(integers) == 1 and sp.cancel(unpack(integers[0]) - world["t"]) == 0
                                 if question["domain"] == "polynomials" else not integers)
        model = candidate["spec"]["model"]
        nonzero = candidate["conditions"]["nonzero"]
        if model["pivot"]:
            denominator = model["route"]["denominator"]
            expected_guard = sum(_q(c) * sp.prod(world[n] ** p for n, p in term)
                                 for c, term in zip(denominator["weights"], denominator["terms"], strict=True))
            guard_ok = guard_ok and len(nonzero) == 1 and sp.cancel(unpack(nonzero[0]) - expected_guard) == 0
        else:
            guard_ok = guard_ok and not nonzero
        world_ok = all(sp.cancel(world[n] - unpack(r)) == 0 for n, r in candidate.get("observables", {}).items())
        def roots(poly):
            if poly == 0 or not sp.Poly(poly, Z).degree():
                return []
            return [{"interval": [str(a), str(b)], "multiplicity": count}
                    for (a, b), count in sp.Poly(poly, Z).intervals(eps=sp.Rational(1, 100000))
                    if b >= sp.Rational(1, 16) and a <= 16]
        n, d = sp.fraction(sp.cancel(expected))
        dn, _ = sp.fraction(sp.cancel(sp.diff(expected, Z)))
        thresholds_ok = all(candidate["analysis"][key] == roots(poly)
                            for key, poly in (("zeros", n), ("poles", d), ("stationary_points", dn)))
        accepted = bool(residual == 0 and derivative_residual == 0 and limit == candidate["analysis"]["zero_right_limit"]
                        and guard_ok and world_ok and thresholds_ok)
        return {"accepted": accepted, "candidate": candidate["id"], "residual": str(residual),
                "derivative_residual": str(derivative_residual), "zero_right_limit": limit,
                "domain_guards_checked": bool(guard_ok), "all_observables_checked": bool(world_ok), "thresholds_checked": thresholds_ok,
                "proof": "Independent structural substitution and rational identity; derivative, roots and one-sided limit checked separately",
                "kind": "CONDITIONAL_INTERVENTION_PROGRAM", "physical_occurrence": False}
    except (ValueError, TypeError, NotImplementedError) as error:
        return {"accepted": False, "candidate": candidate["id"], "reason": str(error)}


def validate_event(event, predictor):
    core = {k: v for k, v in event.items() if k != "commitment"}
    if digest(core) != event["commitment"] or event["predictor"] != predictor:
        raise ValueError("Changed counterfactual commitment or predictor")
    proofs = [check_candidate(event["question"], c) for c in event["candidates"]]
    receipt = {"commitment": event["commitment"], "question": event["question"]["id"], "predictor": predictor,
               "proofs": proofs, "accepted": bool(proofs) and all(p["accepted"] for p in proofs),
               "scope": "Each alternative has its own explicit mechanism; no alternative is selected as a factual cause by algebra"}
    return receipt | {"id": digest(receipt)}


class Simulation:
    """The hidden mechanism is chosen only after the whole proposal set is sealed."""
    def __init__(self, event, seed, omitted=False):
        self.question = event["question"]
        rng = np.random.default_rng(seed)
        candidates = [c for c in event["candidates"] if c["spec"]["companion"]]
        candidate = (candidates or event["candidates"])[int(rng.integers(len(candidates or event["candidates"]))) ]
        self.spec = dict(candidate["spec"])
        if omitted:
            if not self.spec["companion"]:
                choices = [n for n in self.spec["model"]["controls"] if n != self.spec["axis"]]
                self.spec["companion"] = choices[0] if choices else None
            self.spec["power"] = 3
        self.expression = reference_world(self.spec, self.question["context"], Z)[self.question["target"]]
        self.world = reference_world(self.spec, self.question["context"], Z)
        self.cache = {}
        self.seed = seed
        self.noise_bound = .002 * (1 + abs(float(Q(self.question["context"][self.question["target"]]))))
        self.identity = digest({"question": self.question["id"], "seed": seed, "spec": self.spec,
                                "noise_bound": self.noise_bound, "evidence_kind": "BOUNDED_NOISE_SIMULATION"})

    def world_at(self, z):
        if str(z) not in self.cache:
            self.cache[str(z)] = {n: v.subs(Z, _q(z)) for n, v in self.world.items()}
        return self.cache[str(z)]

    def observe(self, z, index, variable=None):
        variable = variable or self.question["target"]
        world = self.world_at(z)
        value = world[variable]
        valid = all(v.is_finite is True for v in world.values())
        valid = valid and world["t"] > 0 and ("m" not in world or world["m"] > 0)
        valid = valid and (self.question["domain"] != "polynomials" or world["t"].is_integer)
        if not valid:
            return {"status": "OUTSIDE_SIMULATOR_DOMAIN", "z": str(z), "source": self.identity}
        rng = np.random.default_rng(self.seed + int(digest([str(z), variable])[:8], 16))
        noise_bound = .002 * (1 + abs(float(Q(self.question["context"][variable]))))
        observed = float(value) + float(rng.uniform(-noise_bound, noise_bound))
        evidence = {"status": "SIMULATED_OBSERVATION", "question": self.question["id"], "z": str(z),
                    "variable": variable, "value": observed, "absolute_error_bound": noise_bound, "source": self.identity,
                    "index": index, "physical_observation": False}
        return evidence | {"id": digest(evidence)}

    def expected(self, z, variable=None):
        world = self.world_at(z)
        value = world[variable or self.question["target"]]
        if value.is_finite is not True:
            return None
        if world["t"] <= 0 or "m" in world and world["m"] <= 0:
            return None
        if self.question["domain"] == "polynomials" and not world["t"].is_integer:
            return None
        return float(value)


def check_generalization(claim):
    spec = claim["spec"]
    controls = {n: sp.Symbol("b_" + n, real=True) for n in spec["model"]["controls"]}
    initial = structural_world(spec, controls, sp.Integer(1))[claim["target"]]
    changed = structural_world(spec, controls, Z)[claim["target"]]
    residual = sp.cancel(changed - initial * unpack(claim["shape"]))
    return {"accepted": residual == 0 and initial != 0, "claim": digest(claim), "residual": str(residual),
            "scope": "Symbolic identity for every allowed background of the specified mechanism; initial target nonzero"}


def replay_investigation(result, source, matrices):
    """Independently reproduce decisions, source receipts and evidence-conditioned weights."""
    from experiments.counterfactual_loop import investigate
    if digest({k: v for k, v in result.items() if k != "id"}) != result["id"]:
        raise ValueError("Changed investigation record")
    fresh = investigate(result["events"][0], matrices, source, result["policy"], result["weights"], result["seed"], lambda _: None)
    if fresh != result:
        raise ValueError("Independent investigation replay differs")
    return {"accepted": True, "investigation": result["id"], "source": source.identity,
            "observations": len(result["observations"]), "original_goal": result["original_goal"],
            "predictor": result["predictor"], "kind": "SIMULATOR_RECEIPTS_AND_PROCEDURE_REPLAY"}


def evaluate_investigation(result, source):
    """Sealed queries never enter fitting, procedure selection or discovery reward."""
    from experiments.counterfactual_core import admitted, at
    from experiments.counterfactual_loop import EVALUATION_PROBES
    candidates = [c for c in result["events"][-1]["candidates"] if c["id"] in result["survivors"]]
    rows = []
    for z in EVALUATION_PROBES:
        expected = source.expected(z)
        if expected is None:
            continue
        predictions = [float(at(c["curve"], z)) for c in candidates if admitted(c, z)]
        scale = 1. + abs(expected)
        tolerance = 1e-8 * scale
        lower, upper = (min(predictions), max(predictions)) if predictions else (None, None)
        consensus = bool(predictions) and upper - lower <= tolerance
        rows.append({"z": str(z), "expected": expected, "lower": lower, "upper": upper,
                     "covered": bool(predictions) and lower - tolerance <= expected <= upper + tolerance,
                     "consensus": consensus, "wrong_consensus": consensus and abs(lower - expected) > tolerance,
                     "normalized_absolute_error": abs(float(np.median(predictions)) - expected) / scale if predictions else None})
    return {"goal": result["original_goal"], "policy": result["policy"], "rows": rows,
            "cases": len(rows), "covered": sum(r["covered"] for r in rows),
            "correct_consensus": sum(r["consensus"] and not r["wrong_consensus"] for r in rows),
            "wrong_consensus": sum(r["wrong_consensus"] for r in rows),
            "errors": [r["normalized_absolute_error"] for r in rows if r["normalized_absolute_error"] is not None],
            "observations": len(result["observations"]), "remaining": len(candidates),
            "expanded": len(result["events"]) > 1, "unresolved": not candidates,
            "scope": "Held-out interventions on a concealed simulated mechanism"}


def after_discovery_words(candidate):
    """Terminology follows checked structure; it supplies no proposal coefficients."""
    n, d = list(map(Q, candidate["curve"]["n"])), list(map(Q, candidate["curve"]["d"]))
    terms = []
    if len(n) == len(d) == 1:
        terms.append("invariance under the declared intervention")
    if len(n) == 1 and len(d) == 2 and d[0] == 0:
        terms.append("inverse proportionality")
    if len(n) == 2 and n[0] == 0 and len(d) == 1:
        terms.append("direct proportionality")
    if candidate["analysis"]["stationary_points"]:
        terms.append("stationary point of the conditional response")
    if candidate["analysis"]["poles"] or "infinity" in candidate["analysis"]["zero_right_limit"]:
        terms.append("singular model limit")
    return {"terms": terms, "source": "https://ftp.cs.ucla.edu/pub/stat_ser/r350.pdf",
            "source_scope": "structural intervention and counterfactual interpretation",
            "naming_supplied_after_verification": True, "new_physical_law_claimed": False}
