"""Construct alternative interventions from the actual learner's retained programs.

No reference-world equations, source retrieval or teacher answers enter this file.
The supplied mechanism grammar controls what may be proposed, not which proposal
is true. Every intervention preserves its own structural and domain assumptions.
"""

import itertools
import json
from collections import defaultdict
from fractions import Fraction as Q
from functools import lru_cache

import numpy as np
import sympy as sp

from experiments.gap_inquiry import digest
from experiments.self_chosen import equations as eq

Z = sp.Symbol("z", real=True)
DOMAINS = ("motion_0", "motion_1", "motion_2", "polynomials")
POWERS = (0, -1, 1, 2)
EXPANSIONS = (3, 4)
PROBES = tuple(map(Q, ("1/16", "1/8", "1/4", "1/2", "3/4", "7/8", "15/16", "1", "17/16", "9/8", "5/4", "3/2", "2", "3", "4", "6", "8", "12", "16")))


def rational(value):
    value = Q(str(value))
    return sp.Rational(value.numerator, value.denominator)


def primitives(domain):
    return tuple(eq.inputs(domain, 4500, 1)[0])


def part(record, values, coefficients=None):
    coefficients = record["weights"] if coefficients is None else coefficients
    return sum(rational(c) * sp.prod(values[n] ** p for n, p in term)
               for c, term in zip(coefficients, record["terms"], strict=True))


def decode_route(owner, candidate):
    weights = [str(Q(float(v)).limit_denominator(1000000)) for v in owner.solution_models[candidate["id"]]]
    if weights != candidate["numerator"]["weights"] + candidate["denominator"]["weights"]:
        raise ValueError("Changed qualified owner coefficients")
    n = len(candidate["numerator"]["weights"])
    return {**candidate, "numerator": {**candidate["numerator"], "weights": weights[:n]},
            "denominator": {**candidate["denominator"], "weights": weights[n:]}}


def operators(owner):
    matrices = {}
    for index, kind in enumerate(("integral", "sum")):
        cap = int(owner.autonomous_denominators[index])
        matrices[kind] = [[str(Q(float(v)).limit_denominator(cap)) for v in row]
                          for row in owner.study_maps[kind].weight.detach().tolist()]
    return matrices


def transform(matrices, kind, polynomial):
    matrix = matrices[kind]
    if len(polynomial) > len(matrix):
        raise ValueError("The retained operator does not represent this degree")
    result = [sum(c * rational(matrix[i][j]) for i, c in enumerate(polynomial)) for j in range(len(matrix[0]))]
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return result


def value(polynomial, time):
    return sum(c * time ** i for i, c in enumerate(polynomial))


def forward(matrices, domain, inputs):
    row, time = dict(inputs), inputs["t"]
    if domain == "polynomials":
        p = [row[f"c{k}"] for k in range(3)]
        row.update(p=value(p, time), i=value(transform(matrices, "integral", p), time),
                   s=value(transform(matrices, "sum", p), time))
    else:
        a = [row[f"a{k}"] for k in range(int(domain[-1]) + 1)]
        velocity = transform(matrices, "integral", a)
        velocity[0] += row["v0"]
        position = transform(matrices, "integral", velocity)
        position[0] += row["x0"]
        row.update(f=row["m"] * value(a, time), v=value(velocity, time), x=value(position, time))
    return row


def model_bank(session):
    """Generate alternate causal parameterizations from already owned inverses."""
    records = [decode_route(session.owner, r["candidate"]) for r in session.records.values()]
    result = {}
    for domain in DOMAINS:
        names = set(primitives(domain))
        bank = [{"domain": domain, "pivot": None, "controlled_observation": None, "route": None,
                 "controls": sorted(names), "assumption": "The retained primitive inputs are independently controlled"}]
        available = defaultdict(list)
        for c in records:
            external = set(c["requires"]) - names
            if c["domain"] == domain and c["target"] in names and len(external) == 1:
                observed = next(iter(external))
                available[c["target"], observed].append(c)
        for (pivot, observed), options in sorted(available.items()):
            c = min(options, key=lambda r: (r["cost"], r["id"]))
            bank.append({"domain": domain, "pivot": pivot, "controlled_observation": observed, "route": c,
                         "controls": sorted(names - {pivot} | {observed}),
                         "assumption": f"Control {observed}; {pivot} adjusts through the retained checked inverse"})
        for model in bank:
            model["id"] = digest(model)
        result[domain] = bank
    return result


def baseline(owner, domain, seed):
    for row in eq.imagine(owner, domain, seed, count=128):
        if all(row.values()) and (domain != "polynomials" or row["t"].denominator == 1):
            return {k: str(v) for k, v in row.items()}
    raise ValueError("No nonsingular retained context in the declared imagination block")


def curve(expression):
    n, d = sp.fraction(sp.cancel(expression))
    if not d:
        raise ValueError("Undefined imagined expression")
    scale = sp.Poly(d, Z).LC()
    return {"n": [str(c / scale) for c in reversed(sp.Poly(n, Z).all_coeffs())],
            "d": [str(c / scale) for c in reversed(sp.Poly(d, Z).all_coeffs())]}


def symbolic(record):
    return value(list(map(rational, record["n"])), Z) / value(list(map(rational, record["d"])), Z)


def at(record, z):
    z = Q(z)
    n = sum(Q(c) * z ** i for i, c in enumerate(record["n"]))
    d = sum(Q(c) * z ** i for i, c in enumerate(record["d"]))
    return None if not d else n / d


def mechanisms(model, axis, powers=POWERS):
    yield {"model": model, "axis": axis, "companion": None, "power": 0}
    for companion in model["controls"]:
        if companion != axis:
            for power in powers:
                if power:
                    yield {"model": model, "axis": axis, "companion": companion, "power": power}
            if 2 in powers:
                yield {"model": model, "axis": axis, "companion": companion, "power": "reverse"}


def imagine(matrices, spec, context):
    return _imagine(json.dumps(matrices, sort_keys=True), json.dumps(spec, sort_keys=True), json.dumps(context, sort_keys=True))


@lru_cache(maxsize=8192)
def _imagine(matrix_json, spec_json, context_json):
    matrices, spec, context = map(json.loads, (matrix_json, spec_json, context_json))
    model = spec["model"]
    controls = {n: rational(context[n]) for n in model["controls"]}
    controls[spec["axis"]] *= Z
    if spec["companion"]:
        power = spec["power"]
        controls[spec["companion"]] *= (2 - Z) if power == "reverse" else Z ** power
    row = dict(controls)
    guards = []
    if model["pivot"]:
        route = model["route"]
        denominator = part(route["denominator"], controls)
        if denominator == 0:
            raise ValueError("Inverse is undefined in this background context")
        guards.append(curve(denominator))
        row[model["pivot"]] = part(route["numerator"], controls) / denominator
    row = forward(matrices, model["domain"], {n: row[n] for n in primitives(model["domain"])})
    if model["controlled_observation"]:
        observed = model["controlled_observation"]
        if sp.cancel(row[observed] - controls[observed]) != 0:
            raise ValueError("Controlled observation conflicts with the learned forward execution")
    conditions = {"positive": [curve(row[n]) for n in ("t", "m") if n in row],
                  "integer": [curve(row["t"])] if model["domain"] == "polynomials" else [], "nonzero": guards}
    return {"curves": {n: curve(v) for n, v in row.items()}, "conditions": conditions}


def admitted(candidate, z):
    for name, tests in candidate["conditions"].items():
        for test in tests:
            v = at(test, z)
            if v is None or name == "positive" and v <= 0 or name == "integer" and v.denominator != 1 or name == "nonzero" and not v:
                return False
    return at(candidate["curve"], z) is not None


@lru_cache(maxsize=4096)
def describe(numerator, denominator):
    record = {"n": list(numerator), "d": list(denominator)}
    expression = symbolic(record)
    derivative = sp.cancel(sp.diff(expression, Z))
    def roots(poly):
        if poly == 0 or not sp.Poly(poly, Z).degree():
            return []
        return [{"interval": [str(a), str(b)], "multiplicity": count}
                for (a, b), count in sp.Poly(poly, Z).intervals(eps=sp.Rational(1, 100000))
                if b >= sp.Rational(1, 16) and a <= 16]
    n, d = sp.fraction(expression)
    dn, _ = sp.fraction(derivative)
    low_n = next((i for i, x in enumerate(numerator) if Q(x)), None)
    low_d = next(i for i, x in enumerate(denominator) if Q(x))
    if low_n is None or low_n > low_d:
        limit = "0"
    elif low_n == low_d:
        limit = str(Q(numerator[low_n]) / Q(denominator[low_d]))
    else:
        limit = "+infinity" if Q(numerator[low_n]) / Q(denominator[low_d]) > 0 else "-infinity"
    return {"expression": str(expression), "derivative": curve(derivative), "zeros": roots(n),
            "stationary_points": roots(dn), "poles": roots(d), "zero_right_limit": limit,
            "limit_scope": "algebraic one-sided consequence of this mechanism; the endpoint must separately pass domain guards"}


def generate(session, inventory, seed=45001):
    bank, matrices = model_bank(session), operators(session.owner)
    language = defaultdict(list)
    for row in inventory["grounded_bindings"]:
        language[row["binding"]["candidate"]].append(row["entry"])
    roles = {"m": "mass", "f": "force", "v": "velocity", "v0": "velocity", "a0": "acceleration", "x": "position", "x0": "position"}
    questions = []
    for domain in DOMAINS:
        context = baseline(session.owner, domain, seed + DOMAINS.index(domain))
        variables = sorted(set(itertools.chain.from_iterable(m["controls"] for m in bank[domain])))
        for axis in variables:
            for target in eq.layout(domain):
                if target == axis:
                    continue
                eligible = [m for m in bank[domain] if axis in m["controls"]]
                if not eligible:
                    continue
                labels = {}
                for n in (axis, target):
                    entries = language.get(roles.get(n), [])
                    labels[n] = min(entries, key=lambda r: (len(r["term"]), r["id"])) if entries else None
                core = {"domain": domain, "axis": axis, "target": target}
                def label(n):
                    return labels[n]["term"] if labels[n] else n
                questions.append({**core, "id": digest(core), "context": context, "models": eligible,
                                  "word_bindings": labels,
                                  "question": f"What changes in {label(target)} when {label(axis)} changes, under different compatible mechanisms?"})
    return questions, matrices


def propose(question, matrices, powers=POWERS):
    candidates, aliases, failures = {}, defaultdict(list), []
    generated = 0
    for model in question["models"]:
        for spec in mechanisms(model, question["axis"], powers):
            generated += 1
            try:
                imagined = imagine(matrices, spec, question["context"])
                response = imagined["curves"][question["target"]]
                core = {"curve": response, "observables": imagined["curves"], "conditions": imagined["conditions"]}
                signature = digest(core)
                spec_id = digest(spec)
                if signature in candidates:
                    aliases[signature].append({"id": spec_id, "spec": spec})
                    continue
                candidate = {**core, "id": signature, "spec": spec, "spec_id": spec_id,
                             "analysis": describe(tuple(response["n"]), tuple(response["d"]))}
                if sum(admitted(candidate, z) for z in PROBES) < 3:
                    failures.append({"spec": spec_id, "reason": "Insufficient admissible interventions in the declared probe pool"})
                    continue
                candidates[signature] = candidate
            except (ValueError, ZeroDivisionError, sp.PolynomialError) as error:
                failures.append({"spec": digest(spec), "reason": str(error)})
    event = {"question": question, "candidates": list(candidates.values()), "aliases": dict(aliases),
             "failures": failures, "generated": generated, "external_information_used": 0,
             "powers": list(powers), "proposal_source": "actual retained coefficient weights and supplied compositional intervention grammar"}
    event["commitment"] = digest(event)
    return event


def probe_features(event, alive, z, used, variable=None):
    variable = variable or event["question"]["target"]
    values = [at(c.get("observables", {}).get(variable, c["curve"]), z)
              for c in event["candidates"] if c["id"] in alive and admitted(c, z)]
    counts = defaultdict(int)
    for v in values:
        counts[str(v)] += 1
    n = max(len(values), 1)
    proportions = np.array(list(counts.values()), dtype=float) / n
    entropy = float(-(proportions * np.log(proportions)).sum()) if values else 0.
    numeric = [float(v) for v in values]
    scale = max(1., max(map(abs, numeric), default=0.))
    spread = (max(numeric) - min(numeric)) / scale if numeric else 0.
    distance = min((abs(float(z - u)) for u in used), default=1.)
    return [1., entropy, spread, len(counts) / n, len(values) / max(len(alive), 1),
            np.log1p(abs(float(z))), min(distance, 16.) / 16., 1. / (1 + len(used))]


def generalized_shape(matrices, candidate, target):
    """Propose a compact invariant across symbolic backgrounds, from owned weights."""
    spec, model = candidate["spec"], candidate["spec"]["model"]
    controls = {n: sp.Symbol("b_" + n, real=True) for n in model["controls"]}

    def execute(z):
        row = dict(controls)
        row[spec["axis"]] *= z
        if spec["companion"]:
            row[spec["companion"]] *= 2 - z if spec["power"] == "reverse" else z ** spec["power"]
        if model["pivot"]:
            route = model["route"]
            row[model["pivot"]] = part(route["numerator"], row) / part(route["denominator"], row)
        return forward(matrices, model["domain"], {n: row[n] for n in primitives(model["domain"])})[target]
    initial, changed = execute(sp.Integer(1)), execute(Z)
    if initial == 0:
        return None
    ratio = sp.cancel(changed / initial)
    if ratio.free_symbols - {Z}:
        return None
    return {"shape": curve(ratio), "target": target, "spec": spec,
            "condition": "Nonzero initial target; all original mechanism/domain guards hold in both worlds",
            "tautological_control": target in model["controls"],
            "scope": "All symbolic backgrounds in this declared mechanism"}
