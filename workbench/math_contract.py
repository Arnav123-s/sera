"""Supplied finite-language contract; verifies a model's interpretation, never trains it."""

import re

NUMERAL = "(?:[0-9]+|zero|one|two|three|four|five|six|seven|eight|nine|ten)"
NAMES = "zero one two three four five six seven eight nine ten".split()


def interpretation_contract(text):
    text = " ".join(text.lower().split())
    fields = {name: f"(?P<{name}>{NUMERAL})" for name in ("a", "b", "c")}
    a, b, c = (fields[name] for name in ("a", "b", "c"))
    productions = (
        (0, f"(?:multiply|scale) x by {a} then subtract {b} to get {c}"),
        (0, f"take {a} times x subtract {b} and obtain {c}"),
        (0, f"subtract {b} from the product of {a} and x equals {c}"),
        (0, f"the product of x and {a} minus {b} equals {c}"),
        (1, f"subtract {b} from x then (?:multiply|scale) by {a} to get {c}"),
        (1, f"take x minus {b} (?:multiply|scale) by {a} and obtain {c}"),
        (1, f"the product of {a} and the difference of x and {b} equals {c}"),
        (1, f"(?:multiply|scale) the difference of x and {b} by {a} equals {c}"),
    )
    for mode, pattern in productions:
        matched = re.fullmatch(pattern+" modulo eleven", text)
        if matched:
            values = [NAMES.index(matched[name]) if matched[name] in NAMES else int(matched[name]) for name in ("a", "b", "c")]
            if all(0 <= value <= 10 for value in values):
                return [mode, *values]
    raise ValueError("Instruction is outside the supplied finite-language contract")


def check_translation(text, prediction):
    if prediction["status"] != "ACCEPTED":
        return prediction
    expected = interpretation_contract(text)
    if prediction["labels"] != expected:
        return {"status": "NEEDS_LESSON", "reason": "The learned interpretation failed the independent task contract",
                "rejected_interpretation": prediction, "translation_verification": False}
    return {**prediction, "translation_verification": True,
            "translation_check_scope": "Exact supplied sentence productions and number meanings; not unrestricted English semantics."}
