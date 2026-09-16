"""Supplied English/finite-math teaching mechanisms with disjoint semantic cases."""

import re

import numpy as np

WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")
TEMPLATES = (
    ("multiply x by {a} then subtract {b} to get {c}",
     "take {a} times x subtract {b} and obtain {c}",
     "subtract {b} from the product of {a} and x equals {c}",
     "the product of x and {a} minus {b} equals {c}"),
    ("subtract {b} from x then multiply by {a} to get {c}",
     "take x minus {b} multiply by {a} and obtain {c}",
     "the product of {a} and the difference of x and {b} equals {c}",
     "multiply the difference of x and {b} by {a} equals {c}"),
)
TOKENS = ["<pad>", "<unknown>"]+sorted(set(" ".join(t for family in TEMPLATES for t in family).replace("{a}", "").replace("{b}", "").replace("{c}", "").split()) | set(WORDS) | set(map(str, range(11))) | {"modulo", "eleven", "scale"})
VOCAB = {word: index for index, word in enumerate(TOKENS)}
MAX_LENGTH = 32


def tokens(text):
    if not isinstance(text, str) or len(text) > 300:
        raise ValueError("A bounded mathematical instruction is required")
    words = re.findall(r"[a-z]+|\d+", text.lower())
    if not 1 <= len(words) <= MAX_LENGTH:
        raise ValueError("Instruction length outside the learned route")
    ids = [VOCAB.get(word, 1) for word in words]
    return ids+[0]*(MAX_LENGTH-len(ids)), words


def partition(a, b, c):
    return (a*121+b*11+c) % 5


def examples(seed, count, split, *, wording="known", lesson=False):
    if split not in ("train", "development", "final", "calibration"):
        raise ValueError("Unknown semantic split")
    rng = np.random.default_rng(seed)
    permitted = {"train": {2, 3, 4}, "development": {1}, "calibration": {1}, "final": {0}}[split]
    rows = []
    while len(rows) < count:
        mode, a, b, c = map(int, [rng.integers(2), *rng.integers(0, 11, 3)])
        if partition(a, b, c) not in permitted:
            continue
        template = 3 if wording == "novel" else int(rng.integers(3))
        representations = {name: (WORDS[value] if rng.random() < .5 else str(value)) for name, value in (("a", a), ("b", b), ("c", c))}
        text = TEMPLATES[mode][template].format(**representations)+" modulo eleven"
        if lesson:
            if "multiply" not in text:
                continue
            text = text.replace("multiply", "scale")
        rows.append({"text": text, "labels": [mode, a, b, c], "template": template,
                     "partition": partition(a, b, c), "provenance": "supplied finite relation and sentence-generation mechanism"})
    return rows


def independent_solutions(labels):
    mode, a, b, c = labels
    if mode not in (0, 1) or any(type(v) is not int or not 0 <= v < 11 for v in (a, b, c)):
        raise ValueError("Unsupported interpreted statement")
    return [x for x in range(11) if ((a*x-b if mode == 0 else a*(x-b))-c) % 11 == 0]


def execute(labels):
    """Algebraic candidate plus independent exhaustive substitution; not a language proof."""
    mode, a, b, c = labels
    if a == 0:
        candidate = list(range(11)) if ((-b if mode == 0 else 0)-c) % 11 == 0 else []
    else:
        candidate = [((b+c)*pow(a, -1, 11) if mode == 0 else b+c*pow(a, -1, 11)) % 11]
    checked = independent_solutions(labels)
    if candidate != checked:
        raise ValueError("Formal execution failed independent verification")
    return {"solutions": candidate, "formal_statement": f"{a}*x-{b}={c} (mod 11)" if mode == 0 else f"{a}*(x-{b})={c} (mod 11)",
            "formal_verification": True, "checked_domain": list(range(11)),
            "boundary": "Checks the interpreted statement. A correct formal answer can still follow an incorrect language interpretation."}
