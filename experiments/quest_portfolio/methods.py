"""Supplied typed mechanics programs; names are not method identities."""

import math
from fractions import Fraction

import numpy as np

from .common import digest

SCOPES = ("time", "impulse", "work_positive", "work_negative")
STOP = 6
SPECS = (
    {"op": "integrate", "requires": ["a", "t", "v0"], "guards": ["constant_acceleration"], "scope": "time", "cost": 3.},
    {"op": "momentum_balance", "requires": ["j", "m", "v0"], "guards": ["constant_mass"], "scope": "impulse", "cost": 1.},
    {"op": "energy_balance", "requires": ["w", "m", "v0"], "guards": ["constant_mass", "positive_final_velocity"], "scope": "work_positive", "cost": 2.},
    {"op": "energy_balance", "requires": ["w", "m", "v0"], "guards": ["constant_mass", "negative_final_velocity"], "scope": "work_negative", "cost": 2.},
    {"op": "wrong_sign", "requires": ["w", "m", "v0"], "guards": ["constant_mass", "negative_final_velocity"], "scope": "work_negative", "cost": .5},
    {"op": "unrelated_zero", "requires": [], "guards": [], "scope": "all", "cost": .1},
)


def canonical(spec):
    return digest({k: v for k, v in spec.items() if k not in {"name", "cost"}})


def registry():
    rows = [dict(s, name=s["op"]+"/"+s["scope"]) for s in SPECS]
    return rows + [dict(rows[0], name=f"integration-alias-{i}") for i in range(24)]


def distinct(rows):
    kept = {}
    for row in rows:
        kept.setdefault(canonical(row), row)
    return list(kept.values())


def eligible(spec, case):
    if canonical(spec) not in {canonical(s) for s in SPECS}:
        raise ValueError("Closed supplied program registry required")
    if case.get("mechanism") != "constant_mechanics":
        return False
    if case.get("units") != "SI" or case.get("origin") != "SUPPLIED_CONDITIONAL_MECHANICS":
        raise ValueError("Explicit conditional source and SI units required")
    assumptions = case.get("assumptions", {})
    if assumptions.get("positive_final_velocity") and assumptions.get("negative_final_velocity"):
        raise ValueError("Contradictory sign assumptions")
    for key in ("a", "t", "v0", "m", "j", "w"):
        if key in case:
            value = str(case[key])
            if len(value) > 64 or abs(Fraction(value)) > 1_000_000:
                raise ValueError("Bounded finite rational input required")
    if "m" in case and Fraction(str(case["m"])) <= 0:
        raise ValueError("Positive mass required")
    if "t" in case and Fraction(str(case["t"])) < 0:
        raise ValueError("Nonnegative elapsed time required")
    return (spec["scope"] in {case.get("scope"), "all"}
            and all(k in case for k in spec["requires"])
            and all(assumptions.get(k) is True for k in spec["guards"]))


def execute(spec, case, owner_session=None):
    if not eligible(spec, case):
        return None
    def f(k):
        return Fraction(str(case[k]))
    if spec["op"] == "integrate":
        if owner_session is not None:
            result = owner_session.grounded.base.exact_motion([case["a"]], case["t"], "0", case["v0"])
            if result["status"] != "CERTIFIED_ALGEBRA":
                raise ValueError("Retained integral operator did not certify its program")
            return float(Fraction(result["result"]["velocity"]))
        return float(f("v0")+f("a")*f("t"))
    if spec["op"] == "momentum_balance":
        return float(f("v0")+f("j")/f("m"))
    if spec["op"] in {"energy_balance", "wrong_sign"}:
        square = f("v0")**2+2*f("w")/f("m")
        if square < 0:
            return None
        sign = -1 if spec["scope"] == "work_negative" and spec["op"] != "wrong_sign" else 1
        return sign*math.sqrt(float(square))
    if spec["op"] == "unrelated_zero":
        return 0.
    raise ValueError("Closed program grammar; arbitrary code is not an eligible method")


def cases(seed, n=256, *, omitted=False):
    rng = np.random.default_rng(seed)
    rows, answers = [], []
    for _ in range(n):
        scope = int(rng.integers(4))
        m = Fraction(int(rng.integers(1, 11)), 2)
        v0 = Fraction(int(rng.integers(-8, 9)), 2)
        vf = Fraction(int(rng.integers(1, 13)), 2)*(1 if scope != 3 else -1)
        common = {"mechanism": "constant_mechanics", "assumptions": {"constant_mass": True},
                  "scope": SCOPES[scope], "v0": str(v0), "units": "SI", "origin": "SUPPLIED_CONDITIONAL_MECHANICS"}
        if scope == 0:
            a, t = Fraction(int(rng.integers(-6, 7)), 6), Fraction(int(rng.integers(1, 25)), 6)
            common.update(a=str(a), t=str(t))
            common["assumptions"]["constant_acceleration"] = True
            vf = v0+a*t
        elif scope == 1:
            common.update(j=str(m*(vf-v0)), m=str(m))
        else:
            common.update(w=str(m*(vf**2-v0**2)/2), m=str(m))
            common["assumptions"]["positive_final_velocity" if scope == 2 else "negative_final_velocity"] = True
        if omitted:
            common = {"mechanism": "cubic_drag", "assumptions": {}, "scope": "omitted", "units": "SI",
                      "origin": "SUPPLIED_CONDITIONAL_MECHANICS", "v0": str(v0), "m": str(m)}
        rows.append(common)
        answers.append(float(vf))
    return rows, np.asarray(answers)


def screen(seed=33100):
    rows, answers = cases(seed, 256)
    profiles = np.zeros((6, 4))
    errors = []
    for i, spec in enumerate(SPECS):
        predictions = [execute(spec, c) for c in rows]
        returned = [(j, p) for j, p in enumerate(predictions) if p is not None]
        wrong = [j for j, p in returned if abs(p-answers[j]) > 1e-8]
        errors.append(wrong)
        if not wrong:
            for j, _ in returned:
                profiles[i, SCOPES.index(rows[j]["scope"])] = 1
    return profiles, errors


def contexts(n, seed, shifted=False):
    rng = np.random.default_rng(seed)
    w = rng.dirichlet([.6, 1., 2., 3.] if shifted else [3., 1., .7, .4], size=n)
    for row in w:
        row[rng.choice(4, size=int(rng.integers(0, 3)), replace=False)] = 0
    return w/w.sum(1, keepdims=True)


def features(weights, covered, visits, step):
    claimed = np.zeros((7, 4))
    claimed[:4] = np.eye(4)
    claimed[4, 3] = 1
    claimed[5] = 1
    n = len(weights)
    return np.concatenate((np.broadcast_to((weights*(1-covered))[:, None, :], (n, 7, 4)),
                           np.broadcast_to(claimed, (n, 7, 4)),
                           np.broadcast_to(np.array([s["cost"] for s in SPECS]+[0.])[None, :, None], (n, 7, 1)),
                           (visits > 0)[..., None], np.full((n, 7, 1), step/5),
                           np.broadcast_to((np.arange(7) == STOP)[None, :, None], (n, 7, 1))), axis=-1)


def deploy(indices, case, owner_session=None):
    if any(type(i) is not int or not 0 <= i < STOP for i in indices):
        raise ValueError("Finite canonical method indices required")
    allowed = sorted(set(indices), key=lambda i: (SPECS[i]["cost"], canonical(SPECS[i])))
    for i in allowed:
        if eligible(SPECS[i], case):
            return execute(SPECS[i], case, owner_session), i
    return None, None
