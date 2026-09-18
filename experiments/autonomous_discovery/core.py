"""Generated investigations, two proposal methods and separate exact verification."""

import itertools
from fractions import Fraction as Q
from functools import lru_cache
from math import comb

import numpy as np
import torch

from experiments.self_study.algebra import SIZE, check, vector

TOKENS = {"integral": "I", "sum": "S"}
DEPTH = 3
DEGREE = 3
WORDS = ("",) + tuple("".join(p) for n in range(1, DEPTH+1) for p in itertools.product("IMS", repeat=n))


class AcquisitionGap(ValueError):
    pass


def canonical(values):
    values = list(map(Q, values))
    if len(values) != len(WORDS):
        raise ValueError("Declared program vocabulary required")
    first = next((v for v in values if v), None)
    if first is None:
        raise ValueError("The zero relation carries no credit")
    return tuple(v/first for v in values)


class Span:
    """Exact incremental elimination, also used to identify covered knowledge."""
    def __init__(self):
        self.rows = []

    def add(self, row):
        row = list(map(Q, row))
        for pivot, basis in self.rows:
            scale = row[pivot]
            if scale:
                row = [a-scale*b for a, b in zip(row, basis, strict=True)]
        pivot = next((i for i, v in enumerate(row) if v), None)
        if pivot is None:
            return False
        scale = row[pivot]
        self.rows.append((pivot, [v/scale for v in row]))
        return True


def reference_step(op, p):
    """Checker semantics, independent of the learned operator matrices."""
    p = vector(list(p))
    if p[-1]:
        raise ValueError("Polynomial overflow")
    if op == "I":
        return [Q(0)] + [v/(i+1) for i, v in enumerate(p[:-1])]
    if op == "M":
        return [Q(0)]+p[:-1]
    if op != "S":
        raise ValueError("Unknown retained primitive")
    # Solve q(n+1)-q(n)=p(n), q(0)=0, from highest degree downward.
    residue, result = p[:], [Q(0)]*SIZE
    for degree in range(SIZE-2, -1, -1):
        c = residue[degree]/(degree+1)
        result[degree+1] = c
        for j in range(degree+1):
            residue[j] -= c*comb(degree+1, j)
    return result


def reference(word, p):
    p = vector(p)
    if word not in WORDS or any(p[DEGREE+1:]):
        raise ValueError("Declared degree and grammar required")
    for op in reversed(word):
        p = reference_step(op, p)
    return p


@lru_cache(maxsize=1)
def proof_signatures():
    return {w: tuple(v for degree in range(DEGREE+1)
                     for v in reference(w, [0]*degree+[1])) for w in WORDS}


def certify(values):
    values = canonical(values)
    signatures = proof_signatures()
    nonzero = [(c, signatures[w]) for c, w in zip(values, WORDS, strict=True) if c]
    residual = [sum(c*row[j] for c, row in nonzero) for j in range((DEGREE+1)*SIZE)]
    return {"accepted": not any(residual), "degree": DEGREE,
            "scope": "all rational polynomials of degree at most three; zero-anchored integration and discrete summation",
            "argument": "exact basis coefficients and linearity; sum uses the difference identity and zero boundary",
            "residual": list(map(str, residual))}


def consequence_span(records):
    """Reward excludes certified compositions of already retained relations."""
    span = Span()
    for record in records:
        coefficients = canonical(record["coefficients"])
        terms = [(w, c) for w, c in zip(WORDS, coefficients, strict=True) if c]
        spare = DEPTH-max(len(w) for w, _ in terms)
        contexts = [w for w in WORDS if len(w) <= spare]
        for left in contexts:
            for right in contexts:
                if len(left)+len(right) > spare:
                    continue
                row = [Q(0)]*len(WORDS)
                for word, coefficient in terms:
                    row[WORDS.index(left+word+right)] += coefficient
                # A degree-bounded certificate cannot automatically be applied
                # to a higher-degree inner composition. Recheck each context.
                if certify(row)["accepted"]:
                    span.add(row)
    return span


def execute(owner, word, p, cache, counts):
    p = vector(p)
    if word not in WORDS or any(p[DEGREE+1:]):
        raise ValueError("Declared degree and grammar required")
    key = (word, tuple(p))
    if key in cache:
        return cache[key]
    if not word:
        result = p
    else:
        inner = execute(owner, word[1:], p, cache, counts)
        if inner[-1]:
            raise ValueError("Unrepresented tail")
        op = word[0]
        if op == "M":
            result = [Q(0)]+inner[:-1]
        else:
            domain = next(k for k, v in TOKENS.items() if v == op)
            proposed = owner.study_maps[domain].propose(list(map(str, inner)))
            if not check(domain, list(map(str, inner)), proposed)["accepted"]:
                counts["old_decoder_rejections"] = counts.get("old_decoder_rejections", 0)+1
                index = tuple(TOKENS).index(domain)
                previous = int(owner.autonomous_denominators[index])
                learning = counts.get("learning", False)
                caps = [120*2**n for n in range(1, 9)] if learning else [previous]
                caps = [c for c in caps if c >= previous]
                x = torch.tensor(list(map(float, inner)), dtype=torch.float64)
                raw = (x @ owner.study_maps[domain].weight).detach().tolist()
                repaired = False
                for cap in caps:
                    candidate = [str(Q(v).limit_denominator(cap)) for v in raw]
                    counts["reconstruction_attempts"] = counts.get("reconstruction_attempts", 0)+1
                    if check(domain, list(map(str, inner)), candidate)["accepted"]:
                        if learning and cap > previous:
                            owner.autonomous_denominators[index] = cap
                            counts.setdefault("procedure_updates", []).append({
                                "domain": domain, "input": list(map(str, inner)), "old_candidate": proposed,
                                "new_candidate": candidate, "previous_limit": previous, "new_limit": cap,
                                "old_correct": False, "new_correct": True})
                        proposed, repaired = candidate, True
                        break
                if not repaired:
                    raise AcquisitionGap("Retained operator has no verified reconstruction in its current search budget")
            result = list(map(Q, proposed))
        counts["primitive_calls"] += 1
    cache[key] = result
    return result


def signatures(owner, depth, seed):
    rng = np.random.default_rng(seed)
    imagined = rng.integers(-4, 5, size=(8, DEGREE+1)).tolist()
    names = [w for w in WORDS if len(w) <= depth]
    cache, counts = {}, {"primitive_calls": 0, "learning": True}
    admitted, columns, gaps, attempts = [], [], [], 0
    for word in names:
        values = []
        try:
            for p in imagined:
                attempts += 1
                values.extend(execute(owner, word, p, cache, counts))
        except AcquisitionGap as error:
            gaps.append({"word": word, "reason": str(error)})
            continue
        admitted.append(word)
        columns.append(values)
    return admitted, columns, {"imagined": imagined, **counts, "unresolved_execution_gaps": gaps,
                           "attempted_program_evaluations": attempts,
                           "completed_program_columns": len(admitted), "shared_prefix_cache": True}


def numerical_proposals(names, columns):
    design = torch.tensor(np.array(columns, dtype=float).T, dtype=torch.float64)
    pivots, proposals = [], []
    for col in range(len(names)):
        if not pivots:
            pivots.append(col)
            continue
        x, y = design[:, pivots], design[:, col]
        fitted = torch.linalg.lstsq(x, y, driver="gelsd", rcond=1e-10).solution
        if float(torch.linalg.vector_norm(x @ fitted-y)) > 1e-7:
            pivots.append(col)
            continue
        row = [Q(0)]*len(WORDS)
        row[WORDS.index(names[col])] = Q(1)
        for i, c in zip(pivots, fitted, strict=True):
            row[WORDS.index(names[i])] = -Q(float(c)).limit_denominator(120)
        proposals.append(canonical(row))
    return proposals


def elimination_proposals(names, columns):
    rows, proposals = [], []
    for name, values in zip(names, columns, strict=True):
        residue = list(values)
        combination = [Q(0)]*len(WORDS)
        combination[WORDS.index(name)] = Q(1)
        for pivot, basis, expression in rows:
            scale = residue[pivot]
            if scale:
                residue = [a-scale*b for a, b in zip(residue, basis, strict=True)]
                combination = [a-scale*b for a, b in zip(combination, expression, strict=True)]
        pivot = next((i for i, v in enumerate(residue) if v), None)
        if pivot is None:
            proposals.append(canonical(combination))
        else:
            scale = residue[pivot]
            rows.append((pivot, [v/scale for v in residue], [v/scale for v in combination]))
    return proposals
