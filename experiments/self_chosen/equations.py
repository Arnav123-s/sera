"""Fit relationships to learned executions; no reference answers enter proposal fitting."""

import itertools
from fractions import Fraction as Q
from math import comb

import numpy as np
import torch

from .common import digest


def layout(domain):
    if domain == "polynomials":
        return {k: (0, 0, 0) for k in ("t", "c0", "c1", "c2", "p", "i", "s")}
    degree = int(domain[-1])
    return {"t": (0, 1, 0), "x0": (1, 0, 0), "v0": (1, -1, 0), "m": (0, 0, 1),
            **{f"a{k}": (1, -2-k, 0) for k in range(degree+1)},
            "f": (1, -2, 1), "x": (1, 0, 0), "v": (1, -1, 0)}


def questions(domain):
    names = list(layout(domain))
    return [{"domain": domain, "target": target, "missing": omitted,
             "id": digest([domain, target, omitted]),
             "question": f"Can retained {domain} executions predict {target} "
                         f"when {omitted or 'no additional variable'} is unavailable?"}
            for target in names for omitted in [None, *[n for n in names if n != target]]]


def features(question):
    units = layout(question["domain"])
    names = [n for n in units if n not in {question["target"], question["missing"]}]
    terms = {()}
    def add(powers):
        term = tuple(sorted((n, p) for n, p in powers.items() if p))
        dimension = tuple(sum(units[n][j]*p for n, p in term) for j in range(3))
        if dimension == units[question["target"]]:
            terms.add(term)
    for n in names:
        add({n: 1})
        for power in (-2, -1, 2, 3, 4, 5):
            if n == "t" or power in (-1, 2):
                add({n: power})
        if "t" in names and n != "t":
            for power in (-3, -2, -1, 1, 2, 3, 4):
                add({n: 1, "t": power})
    for left, right in itertools.combinations_with_replacement(names, 2):
        pair = {left: 1}
        pair[right] = pair.get(right, 0)+1
        add(pair)
        for denominator in names:
            powers = dict(pair)
            powers[denominator] = powers.get(denominator, 0)-1
            add(powers)
    if units[question["target"]] != (0, 0, 0):
        terms.discard(())
    return sorted(terms, key=lambda t: (sum(abs(p) for _, p in t), len(t), t))[:256]


def value(term, row):
    result = Q(1)
    for name, power in term:
        result *= row[name]**power
    return result


def learned_map(owner, kind, p):
    padded = list(p)+[Q(0)]*(13-len(p))
    x = torch.tensor(list(map(float, padded)), dtype=torch.float64)
    with torch.no_grad():
        raw = x @ owner.study_maps[kind].weight
    cap = int(owner.autonomous_denominators[0 if kind == "integral" else 1])
    return [Q(float(v)).limit_denominator(cap) for v in raw]


def evaluate(p, t):
    result = Q(0)
    for c in reversed(p):
        result = result*t+c
    return result


def inputs(domain, seed, count):
    rng = np.random.default_rng(seed)
    result = []
    for _ in range(count):
        row = {"t": Q(int(rng.integers(1, 17)), 4)}
        if domain == "polynomials":
            row.update({f"c{k}": Q(int(rng.integers(-9, 10)), 3) for k in range(3)})
        else:
            row.update(x0=Q(int(rng.integers(-8, 9)), 3), v0=Q(int(rng.integers(-8, 9)), 3),
                       m=Q(int(rng.integers(1, 13)), 3))
            row.update({f"a{k}": Q(int(rng.integers(-6, 7)), 3) for k in range(int(domain[-1])+1)})
        result.append(row)
    return result


def imagine(owner, domain, seed, count=48):
    rows = inputs(domain, seed, count)
    for row in rows:
        if domain == "polynomials":
            p = [row[f"c{k}"] for k in range(3)]
            row.update(p=evaluate(p, row["t"]), i=evaluate(learned_map(owner, "integral", p), row["t"]),
                       s=evaluate(learned_map(owner, "sum", p), row["t"]))
        else:
            a = [row[f"a{k}"] for k in range(int(domain[-1])+1)]
            v = learned_map(owner, "integral", a)
            v[0] += row["v0"]
            x = learned_map(owner, "integral", v)
            x[0] += row["x0"]
            row.update(v=evaluate(v, row["t"]), x=evaluate(x, row["t"]),
                       f=row["m"]*evaluate(a, row["t"]))
    return rows


def canonical(target, terms, coefficients):
    items = {((target, 1),): Q(-1)}
    for term, coefficient in zip(terms, coefficients, strict=True):
        items[tuple(map(tuple, term))] = items.get(tuple(map(tuple, term)), Q(0))+Q(coefficient)
    items = {t: c for t, c in items.items() if c}
    if not items:
        raise ValueError("A tautology is not a discovery")
    names = sorted({n for t in items for n, _ in t})
    # Remove a common Laurent monomial: multiplying by time is not a new law.
    common = {n: min(dict(t).get(n, 0) for t in items) for n in names}
    shifted = {tuple((n, dict(t).get(n, 0)-common[n]) for n in names
                     if dict(t).get(n, 0) != common[n]): c for t, c in items.items()}
    first = shifted[sorted(shifted)[0]]
    return [[list(map(list, t)), str(c/first)] for t, c in sorted(shifted.items())]


def propose(question, rows, method):
    terms = features(question)
    # Denominators are explicit applicability conditions; exclude features
    # that were undefined in this imagined sample rather than imputing values.
    terms = [t for t in terms if all(all(row[n] or p >= 0 for n, p in t) for row in rows)]
    if not terms:
        return {"method": method, "status": "OPEN", "reason": "No defined typed feature"}
    matrix = np.array([[float(value(t, row)) for t in terms] for row in rows])
    y = np.array([float(row[question["target"]]) for row in rows])
    scale = np.linalg.norm(matrix, axis=0)
    permitted = scale > 1e-12
    terms, matrix, scale = [t for t, good in zip(terms, permitted, strict=True) if good], matrix[:, permitted], scale[permitted]
    if not terms:
        return {"method": method, "status": "OPEN", "reason": "No informative imagined feature"}
    z = matrix/scale
    calls = 0
    if method == "dense":
        coef = np.linalg.lstsq(z, y, rcond=1e-10)[0]/scale
        calls += 1
    elif method == "sparse":
        active, residue = [], y.copy()
        coef = np.zeros(len(terms))
        for _ in range(min(10, len(terms))):
            correlations = np.abs(z.T @ residue)
            correlations[active] = -1
            chosen = int(correlations.argmax())
            if chosen in active:
                break
            active.append(chosen)
            fitted = np.linalg.lstsq(z[:, active], y, rcond=1e-10)[0]
            calls += 1
            residue = y-z[:, active] @ fitted
            coef[active] = fitted/scale[active]
            if np.max(np.abs(residue)) < 1e-8:
                break
    else:
        raise ValueError("Unknown proposal procedure")
    rational = [Q(float(c)).limit_denominator(720) if abs(c) > 1e-8 else Q(0) for c in coef]
    selected = [(t, c) for t, c in zip(terms, rational, strict=True) if c]
    error = max(abs(sum(c*value(t, row) for t, c in selected)-row[question["target"]]) for row in rows)
    if not selected or error > Q(1, 1000000):
        return {"method": method, "status": "OPEN", "reason": "No compact exact imagined fit",
                "imagined_error": str(error), "linear_solves": calls, "feature_count": len(terms)}
    chosen_terms, weights = zip(*selected, strict=True)
    conditions = sorted({n for t in chosen_terms for n, p in t if p < 0})
    return {"method": method, "status": "CONJECTURE", "target": question["target"],
            "terms": [list(map(list, t)) for t in chosen_terms], "coefficients": list(map(str, weights)),
            "requires": sorted({n for t in chosen_terms for n, _ in t}), "nonzero": conditions,
            "canonical": canonical(question["target"], chosen_terms, weights),
            "imagined_error": str(error), "linear_solves": calls, "feature_count": len(terms),
            "execution_cost": sum(1+sum(abs(p) for _, p in t) for t in chosen_terms)}


def symbolic_world(domain):
    """Assessor-only reference: symbolic laws, never the learner's learned matrices."""
    import sympy as sp

    names = layout(domain)
    symbols = {n: sp.Symbol(n, real=True) for n in names}
    t = symbols["t"]
    if domain == "polynomials":
        c = [symbols[f"c{k}"] for k in range(3)]
        symbols.update(p=sum(c[k]*t**k for k in range(3)),
                       i=sum(c[k]*t**(k+1)/sp.Integer(k+1) for k in range(3)),
                       s=c[0]*t+c[1]*t*(t-1)/2+c[2]*t*(t-1)*(2*t-1)/6)
    else:
        a = [symbols[f"a{k}"] for k in range(int(domain[-1])+1)]
        symbols.update(v=symbols["v0"]+sum(a[k]*t**(k+1)/sp.Integer(k+1) for k in range(len(a))),
                       x=symbols["x0"]+symbols["v0"]*t+sum(a[k]*t**(k+2)/sp.Integer((k+1)*(k+2)) for k in range(len(a))),
                       f=symbols["m"]*sum(a[k]*t**k for k in range(len(a))))
    return symbols


def certify(domain, proposal):
    import sympy as sp

    if proposal["status"] != "CONJECTURE":
        return {"accepted": False, "kind": "OPEN_SEARCH", "reason": proposal["reason"]}
    world = symbolic_world(domain)
    result = -world[proposal["target"]]
    for term, coefficient in zip(proposal["terms"], proposal["coefficients"], strict=True):
        q = Q(coefficient)
        result += sp.Rational(q.numerator, q.denominator)*sp.prod(world[n]**p for n, p in term)
    numerator, denominator = sp.fraction(sp.cancel(result))
    good = sp.expand(numerator) == 0
    return {"accepted": bool(good), "kind": "EXACT_CONDITIONAL_IDENTITY",
            "nonzero": proposal["nonzero"], "residual_numerator": str(sp.expand(numerator)),
            "denominator": str(denominator), "scope": domain,
            "assumptions": "polynomial inputs; real time > 0, positive mass for motion; natural time for sums; declared nonzero denominators",
            "observed_event": False}


def reference_rows(domain, seed, count=64):
    """Independent exact execution with elementary sums and coefficient integration."""
    rows = inputs(domain, seed, count)
    for row in rows:
        t = row["t"]
        if domain == "polynomials":
            # Natural time makes the independent sum a literal finite loop.
            row["t"] = t = Q(int(4*t))
            c = [row[f"c{k}"] for k in range(3)]
            row.update(p=sum(c[k]*t**k for k in range(3)),
                       i=sum(c[k]*t**(k+1)/(k+1) for k in range(3)),
                       s=sum(sum(c[j]*Q(k)**j for j in range(3)) for k in range(int(t))))
        else:
            a = [row[f"a{k}"] for k in range(int(domain[-1])+1)]
            row.update(v=row["v0"]+sum(a[k]*t**(k+1)/(k+1) for k in range(len(a))),
                       x=row["x0"]+row["v0"]*t+sum(a[k]*t**(k+2)/(2*comb(k+2, 2)) for k in range(len(a))),
                       f=row["m"]*sum(a[k]*t**k for k in range(len(a))))
    return rows


def check_execution(proposal, rows):
    checked, failures, skipped = 0, [], 0
    for row in rows:
        if any(not row[n] for n in proposal["nonzero"]):
            skipped += 1
            continue
        prediction = sum(Q(c)*value(t, row) for t, c in zip(proposal["terms"], proposal["coefficients"], strict=True))
        checked += 1
        if prediction != row[proposal["target"]]:
            failures.append({"input": {k: str(v) for k, v in row.items()}, "prediction": str(prediction)})
    return {"checked": checked, "skipped_outside_scope": skipped, "failures": failures, "accepted": checked > 0 and not failures}
