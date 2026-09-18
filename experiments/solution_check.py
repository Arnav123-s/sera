"""Independent exact certificates and literal reference replay for proposed answers."""

import json
from fractions import Fraction as Q
from functools import lru_cache

import sympy as sp

from experiments.gap_inquiry import digest
from experiments.self_chosen import equations as eq


def symbolic(part, world):
    return sum(sp.Rational(c) * sp.prod(world[n] ** p for n, p in t)
               for t, c in zip(part["terms"], part["weights"], strict=True))


@lru_cache(maxsize=16384)
def _identity(domain, target, numerator_json, denominator_json):
    world = eq.symbolic_world(domain)
    numerator = symbolic(json.loads(numerator_json), world)
    denominator = sp.expand(symbolic(json.loads(denominator_json), world))
    residual = sp.expand(world[target] * denominator - numerator)
    return bool(denominator != 0 and residual == 0), str(residual), str(denominator)


def certify(question, candidate, commitment, predictor):
    known = set(eq.layout(question["domain"])) - {question["target"], *(question["missing"] or [])}
    core = {k: candidate[k] for k in ("domain", "target", "numerator", "denominator", "requires")}
    dependencies = sorted({n for part in (candidate["numerator"], candidate["denominator"])
                           for term in part["terms"] for n, power in term if power})
    if digest(core) != candidate["id"] or dependencies != candidate["requires"]:
        raise ValueError("Candidate identity or dependencies changed")
    if candidate["domain"] != question["domain"] or candidate["target"] != question["target"] or not set(candidate["requires"]) <= known:
        raise ValueError("The proposed answer uses unavailable information")
    accepted, residual, denominator = _identity(question["domain"], question["target"],
                                               json.dumps(candidate["numerator"]), json.dumps(candidate["denominator"]))
    receipt = {"question": question["id"], "candidate": candidate["id"], "commitment": commitment,
               "predictor": predictor, "accepted": bool(accepted), "kind": "EXACT_CONDITIONAL_RATIONAL_IDENTITY",
               "residual": str(residual), "substituted_denominator": str(denominator),
               "proof": "Substitute the independently specified domain laws; expand target*D - N; verify zero. Divide only when D != 0.",
               "assumptions": "retained polynomial-model laws; t > 0; m > 0 for motion; integer t for finite sums; explicit denominator guard",
               "original_goal": question["question"], "physical_fact_added": False}
    return receipt | {"id": digest(receipt)}


def validate(receipt, question, candidate, commitment, predictor):
    expected = certify(question, candidate, commitment, predictor)
    if receipt != expected or not receipt["accepted"]:
        raise ValueError("Changed or unverified solution credit")


def execute_literal(candidate, row):
    """Independent scalar evaluator: no proposal matrix or learned decoder."""
    def part(record):
        result = Q(0)
        for term, coefficient in zip(record["terms"], record["weights"], strict=True):
            value = Q(coefficient)
            for name, power in term:
                value *= row[name] ** power
            result += value
        return result
    n, d = part(candidate["numerator"]), part(candidate["denominator"])
    return None if not d else n / d


def final_check(candidate, seed, count=128):
    rows = eq.reference_rows(candidate["domain"], seed, count)
    correct, guard, wrong = 0, 0, []
    for row in rows:
        result = execute_literal(candidate, row)
        if result is None:
            guard += 1
        elif result == row[candidate["target"]]:
            correct += 1
        else:
            wrong.append({"row": {k: str(v) for k, v in row.items()}, "actual": str(result)})
    return {"accepted": correct >= 16 and not wrong, "candidate": candidate["id"], "correct": correct,
            "guarded": guard, "n": count, "wrong": wrong, "seed": seed,
            "inputs": digest([{k: str(v) for k, v in r.items()} for r in rows])}
