"""Declared mathematical readers; exact source equations, never prose commands."""
import ast
import hashlib
import re
from fractions import Fraction as Q

from .algebra import SIZE, vector
from .sources import Maths


def mul(a, b):
    c = [Q(0)] * SIZE
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            if x * y:
                if i + j >= SIZE:
                    raise ValueError("Source expression exceeds degree 12")
                c[i + j] += x * y
    return c


def expression(text, variable):
    """Parse a bounded arithmetic AST; no eval/sympify or generated execution."""
    if len(text) > 256:
        raise ValueError("Formula too long")
    text = re.sub(r"\^\{(\d+)\}", r"**\1", text)
    text = re.sub(r"(\d)N", r"\1*N", text)
    if re.search(r"[^0-9N+*/()\- ]", text):
        raise ValueError("Unsupported formula token")
    root = ast.parse(text, mode="eval")

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) is int and abs(node.value) < 100000:
            return vector([node.value])
        if isinstance(node, ast.Name) and node.id == "N":
            return vector(variable)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return [-v for v in visit(node.operand)]
        if isinstance(node, ast.BinOp):
            a, b = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return [x + y for x, y in zip(a, b)]
            if isinstance(node.op, ast.Sub):
                return [x - y for x, y in zip(a, b)]
            if isinstance(node.op, ast.Mult):
                return mul(a, b)
            if isinstance(node.op, ast.Div) and not any(b[1:]) and b[0]:
                return [x / b[0] for x in a]
            if isinstance(node.op, ast.Pow) and not any(b[1:]) and b[0].denominator == 1 and 0 <= b[0] <= 6:
                result = vector([1])
                for _ in range(int(b[0])):
                    result = mul(result, a)
                return result
        raise ValueError("Formula outside the arithmetic reader contract")

    return visit(root.body)


def knuth(meta, raw):
    parser = Maths()
    parser.feed(raw.decode("utf-8"))
    tables = [f for f in parser.formulas if r"1^{1}+2^{1}" in f and r"N=(n^{2}+n)/2" in f]
    if len(tables) != 1:
        raise ValueError("Expected one identified power-sum table and its N definition")
    lessons = []
    for row in tables[0].split(r"\cr")[:6]:
        match = re.search(r"1\^\{(\d+)\}.*?&=(.*)", row)
        if not match:
            raise ValueError("Power-sum row not recognized")
        power = int(match[1])
        rhs = match[2].split(r"\,")[0]
        p = [0] * SIZE
        p[power] = 1
        # The source sums 1..n; replace n by n-1 for our 0..n-1 contract.
        q = expression(rhs, [0, Q(-1, 2), Q(1, 2)])
        lessons.append({"id": f"knuth-odd-{power}", "domain": "sum", "input": p,
                        "output": list(map(str, q)), "source": meta["url"], "source_sha256": meta["sha256"],
                        "equation_sha256": hashlib.sha256(row.encode()).hexdigest(),
                        "assumptions": ["natural upper endpoint", "positive odd monomial exponent",
                                        "source n replaced by n-1", "exact rational arithmetic"],
                        "interpretation": "engineered bounded MathML/TeX reader"})
    return lessons


def calculus(source):
    """Practice from an explicitly supplied power-rule curriculum, not web reading.

    Its provenance is kept separate from the automatically extracted arXiv table.
    The independent checker controls acceptance and the installed rule is a control.
    """
    return [{"id": f"power-rule-{k}", "domain": "integral",
             "input": [int(i == k) for i in range(SIZE)],
             "output": [str(Q(1, k + 1)) if i == k + 1 else "0" for i in range(SIZE)],
             "source": source["url"], "source_sha256": source["sha256"],
             "assumptions": ["real polynomial", "zero integration constant", "integer exponent 0..5"],
             "interpretation": "supplied power-rule schema generates verified practice"} for k in range(6)]
