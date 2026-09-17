"""Small exact certificate contract, independent of learned numerical synthesis."""
import re
from fractions import Fraction as Q
from math import comb

SIZE = 13


def vector(values):
    if not isinstance(values, (list, tuple)) or not 1 <= len(values) <= SIZE:
        raise ValueError("Use 1..13 rational polynomial coefficients, constant first")
    out = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (str, int, Q)):
            raise ValueError("Exact integer/rational coefficients required")
        if isinstance(value, str) and (len(value) > 80 or
                re.fullmatch(r"[+-]?[0-9]+(?:/[1-9][0-9]*)?", value) is None):
            raise ValueError("Use bounded integer or numerator/denominator syntax")
        q = Q(value)
        if max(abs(q.numerator).bit_length(), q.denominator.bit_length()) > 128:
            raise ValueError("Coefficient exceeds the bounded arithmetic contract")
        out.append(q)
    return out + [Q(0)] * (SIZE - len(out))


def check(domain, p, q):
    p, q = vector(p), vector(q)
    residue = [Q(0)] * SIZE
    if domain == "sum":
        for power, coefficient in enumerate(q):
            for j in range(power):
                residue[j] += coefficient * comb(power, j)
        scope = "q(n)=sum(p(k), k=0..n-1) for every natural n, by polynomial identity and induction"
    elif domain == "integral":
        for j in range(1, SIZE):
            residue[j - 1] = j * q[j]
        scope = "q'(x)=p(x) for every real x and q(0)=0"
    else:
        raise ValueError("Unsupported certificate domain")
    return {"accepted": q[0] == 0 and residue == p, "base_zero": q[0] == 0,
            "residual": [str(a - b) for a, b in zip(residue, p)], "scope": scope}


def evaluate(p, x):
    x = vector([x])[0]
    result = Q(0)
    for c in reversed(vector(p)):
        result = result * x + c
    return result


def independent(domain, p, q):
    """Degree-bounded exact evaluation at 13 points, using a different algorithm.

    A polynomial of degree at most 12 with 13 distinct rational roots is zero.
    This is an exact identity argument, not arbitrary finite evidence for RH.
    """
    p, q = vector(p), vector(q)
    if q[0] != 0:
        return False
    for n in range(SIZE):
        if domain == "sum":
            left = evaluate(q, n + 1) - evaluate(q, n)
        elif domain == "integral":
            left = sum(j * q[j] * Q(n) ** (j - 1) for j in range(1, SIZE))
        else:
            return False
        if left != evaluate(p, n):
            return False
    return True
