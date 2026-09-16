import itertools

import pytest

from experiments.grounded_language.data import TEMPLATES, WORDS
from workbench.math_contract import check_translation, interpretation_contract


def test_independent_contract_covers_all_coefficients_and_ordered_productions():
    for mode, family in enumerate(TEMPLATES):
        for a, b, c in itertools.product(range(11), repeat=3):
            for template in family:
                for values in ({"a": a, "b": b, "c": c}, {"a": WORDS[a], "b": WORDS[b], "c": WORDS[c]}):
                    text = template.format(**values)+" modulo eleven"
                    assert interpretation_contract(text) == [mode, a, b, c]
                    if "multiply" in text:
                        assert interpretation_contract(text.replace("multiply", "scale")) == [mode, a, b, c]


def test_formally_correct_mistranslation_is_withheld():
    text = "multiply x by three then subtract two to get four modulo eleven"
    # The candidate really solves a different equation; that is insufficient.
    prediction = {"status": "ACCEPTED", "labels": [1, 3, 2, 4], "solutions": [7], "formal_verification": True}
    assert check_translation(text, prediction)["status"] == "NEEDS_LESSON"
    assert "solutions" not in check_translation(text, prediction)
    correct = {"status": "ACCEPTED", "labels": [0, 3, 2, 4], "solutions": [2], "formal_verification": True}
    assert check_translation(text, correct)["translation_verification"]
    for unsupported in ("multiply x by -3 then subtract two to get four modulo eleven", text.replace("three", "33"), text.replace("eleven", "twelve")):
        with pytest.raises(ValueError):
            interpretation_contract(unsupported)
