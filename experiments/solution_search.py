"""Invent rational answers from owned imagined executions, before feedback."""

import itertools
from collections import defaultdict
from fractions import Fraction as Q

import numpy as np
import scipy.linalg as la
import sympy as sp

from experiments.gap_inquiry import digest
from experiments.self_chosen import equations as eq

DEGREES = (1, 2, 3, 4)
ORDERS = ("simple", "reverse", "permuted")


def monomials(names, degree):
    result = [()]
    for size in range(1, degree + 1):
        for sequence in itertools.combinations_with_replacement(sorted(names), size):
            result.append(tuple((n, sequence.count(n)) for n in sorted(set(sequence))))
    return result


def dimension(term, units):
    return tuple(sum(units[n][i] * p for n, p in term) for i in range(3))


def expression(terms, weights, symbols):
    return sum(sp.Rational(str(c)) * sp.prod(symbols[n] ** p for n, p in term)
               for term, c in zip(terms, weights, strict=True))


def polynomial_record(value, symbols):
    poly = sp.Poly(value, *[symbols[n] for n in sorted(symbols)])
    pairs = []
    for exponents, c in poly.terms():
        if c:
            pairs.append(([[n, p] for n, p in zip(sorted(symbols), exponents, strict=True) if p], str(c)))
    return {"terms": [p[0] for p in pairs], "weights": [p[1] for p in pairs]}


def canonical(question, numerator, denominator, method, degree):
    symbols = {n: sp.Symbol(n) for n in eq.layout(question["domain"])}
    n = expression(numerator["terms"], numerator["weights"], symbols)
    d = expression(denominator["terms"], denominator["weights"], symbols)
    if not d:
        raise ValueError("An identically zero denominator is not an answer")
    n, d = sp.fraction(sp.cancel(n / d))
    # Fix scale/sign without using any domain equations.
    leading = sp.Poly(d, *symbols.values()).LC()
    n, d = sp.expand(n / leading), sp.expand(d / leading)
    core = {"domain": question["domain"], "target": question["target"],
            "numerator": polynomial_record(n, symbols), "denominator": polynomial_record(d, symbols),
            "requires": sorted(str(s) for s in n.free_symbols | d.free_symbols)}
    if question["target"] in core["requires"] or set(question["missing"] or []) & set(core["requires"]):
        raise ValueError("Answer depends on unavailable information")
    core["id"] = digest(core)
    return {**core, "method": method, "degree": degree,
            "expression": f"{question['target']} = {sp.sstr(n / d)}", "guard": f"{sp.sstr(d)} != 0",
            "status": "CONJECTURE", "cost": len(core["numerator"]["terms"]) + len(core["denominator"]["terms"])}


def evaluate(candidate, row, weights=None):
    values = candidate["numerator"]["weights"] + candidate["denominator"]["weights"] if weights is None else weights
    count = len(candidate["numerator"]["weights"])
    n = sum(Q(c) * eq.value(t, row) for c, t in zip(values[:count], candidate["numerator"]["terms"], strict=True))
    d = sum(Q(c) * eq.value(t, row) for c, t in zip(values[count:], candidate["denominator"]["terms"], strict=True))
    if not d:
        return None
    return n / d


def implicit(question, rows, checks, degree, order, values=None, seen=None):
    """QR dependencies among dimensionally compatible N and target*D columns.

    All columns are generated from the variable layout; their values come only
    from the actual owner's forward executions. No physical answer is encoded.
    """
    units = eq.layout(question["domain"])
    available = set(units) - {question["target"], *(question["missing"] or [])}
    groups = defaultdict(list)
    for side, maximum in ((0, degree), (1, degree - 1)):
        for term in monomials(available, maximum):
            dim = dimension(term, units)
            dim = tuple(d + side * t for d, t in zip(dim, units[question["target"]], strict=True))
            groups[dim].append((side, term))
    attempts, candidates = [], []
    for dim, columns in sorted(groups.items()):
        if {s for s, _ in columns} != {0, 1}:
            continue
        if order == "reverse":
            columns.reverse()
        elif order == "permuted":
            rng = np.random.default_rng(int(question["id"][:8], 16) + degree)
            columns = [columns[i] for i in rng.permutation(len(columns))]
        target_values = np.array([float(row[question["target"]]) for row in rows])
        matrix = np.column_stack([(values[t] if values is not None else np.array([float(eq.value(t, row)) for row in rows]))
                                  * (target_values if side else 1.) for side, t in columns])
        scale = np.linalg.norm(matrix, axis=0)
        good = scale > 1e-12
        matrix, scale = matrix[:, good], scale[good]
        columns = [c for c, g in zip(columns, good, strict=True) if g]
        if len(columns) < 2:
            continue
        z = matrix / scale
        # Non-pivoted alternatives would mistake order-dependent zero diagonals
        # for rank. Pivoting is mandatory; permutations break equal norm ties.
        _, triangular, pivot = la.qr(z, mode="economic", pivoting=True, check_finite=False)
        rank = int(np.sum(abs(np.diag(triangular)) > 1e-9))
        if rank == len(columns):
            attempts.append({"dimension": dim, "columns": len(columns), "rank": rank, "status": "NO_DEPENDENCY"})
            continue
        basis = pivot[:rank]
        dependent = pivot[rank:]
        fitted = la.solve_triangular(triangular[:rank, :rank], triangular[:rank, rank:], check_finite=False)
        for index, anchor in enumerate(dependent):
            coef = np.zeros(len(columns))
            coef[anchor] = 1. / scale[anchor]
            coef[basis] = -fitted[:, index] / scale[basis]
            denom = [abs(c) for c, (side, _) in zip(coef, columns, strict=True) if side and abs(c) > 1e-9]
            event = {"dimension": dim, "columns": len(columns), "rank": rank, "anchor": int(anchor)}
            if not denom:
                attempts.append(event | {"status": "NO_TARGET_DEPENDENCE"})
                continue
            coef /= max(denom)
            rational = [Q(float(c)).limit_denominator(720) if abs(c) > 1e-8 else Q(0) for c in coef]
            terms = {0: [], 1: []}
            weights = {0: [], 1: []}
            for (side, term), c in zip(columns, rational, strict=True):
                if c:
                    terms[side].append(term)
                    weights[side].append(str(c if side else -c))
            raw_d = {"terms": terms[1], "weights": weights[1]}
            denominator_values = [sum(Q(c) * eq.value(t, row) for t, c in zip(terms[1], weights[1], strict=True)) for row in checks[:16]]
            if not any(denominator_values):
                attempts.append(event | {"status": "VANISHING_IMAGINED_DENOMINATOR"})
                continue
            signature = digest(sorted((side, tuple(t), str(c)) for side in (0, 1)
                                      for t, c in zip(terms[side], weights[side], strict=True)))
            if seen is not None and signature in seen:
                candidate = seen[signature]
                if candidate is not None:
                    candidates.append(candidate | {"method": "implicit_" + order, "degree": degree})
                attempts.append(event | {"status": "PREVIOUSLY_IMAGINED", "candidate": candidate["id"] if candidate else None})
                continue
            try:
                candidate = canonical(question, {"terms": terms[0], "weights": weights[0]},
                                      raw_d, "implicit_" + order, degree)
                predictions = [evaluate(candidate, row) for row in checks]
                defined = [(p, row[question["target"]]) for p, row in zip(predictions, checks, strict=True) if p is not None]
                if len(defined) < 16 or any(a != b for a, b in defined):
                    attempts.append(event | {"status": "IMAGINED_CHECK_REJECTED", "candidate": candidate})
                    if seen is not None:
                        seen[signature] = None
                    continue
                candidates.append(candidate)
                if seen is not None:
                    seen[signature] = candidate
                attempts.append(event | {"status": "PROPOSED", "candidate": candidate["id"]})
            except (ValueError, ZeroDivisionError) as error:
                attempts.append(event | {"status": "INVALID_PROGRAM", "reason": str(error)})
    return {"method": "implicit_" + order, "degree": degree, "attempts": attempts, "proposals": candidates}


def investigate(owner, question):
    """Exhaust this finite question grammar with no independent feedback calls."""
    from experiments.discovery_frontier import fit

    seed = int(question["id"][:8], 16) + 440000
    rows = eq.imagine(owner, question["domain"], seed, count=256)
    checks = eq.imagine(owner, question["domain"], seed + 1, count=48)
    attempts = []
    for method in ("sparse", "dense"):
        old = fit(question, rows, method)
        candidates = []
        if old["status"] == "CONJECTURE":
            candidates = [canonical(question, {"terms": old["terms"], "weights": old["coefficients"]},
                                    {"terms": [[]], "weights": ["1"]}, method, 0)]
        attempts.append({"method": method, "degree": 0, "attempts": [old], "proposals": candidates})
    available = set(eq.layout(question["domain"])) - {question["target"], *(question["missing"] or [])}
    numeric = {n: np.array([float(r[n]) for r in rows]) for n in available}
    values = {t: np.prod([numeric[n] ** p for n, p in t], axis=0) if t else np.ones(len(rows))
              for t in monomials(available, max(DEGREES))}
    seen = {}
    for degree in DEGREES:
        for order in ORDERS:
            attempts.append(implicit(question, rows, checks, degree, order, values, seen))
    proposals = {}
    for attempt in attempts:
        for candidate in attempt["proposals"]:
            entry = proposals.setdefault(candidate["id"], {"candidate": candidate, "derivations": []})
            entry["derivations"].append({"method": attempt["method"], "degree": attempt["degree"]})
    return {"question": question, "seed": seed, "imagined_fit": digest([{n: str(v) for n, v in r.items()} for r in rows]),
            "imagined_check": digest([{n: str(v) for n, v in r.items()} for r in checks]),
            "attempts": attempts, "proposals": list(proposals.values()), "finite_search_exhausted": True,
            "new_external_information": 0, "independent_feedback_during_search": 0}
