from fractions import Fraction as Q

import pytest

from experiments.operator_discovery.core import WORDS, canonical, certify, exact, rational_rank


def relation():
    values = ["0"]*len(WORDS)
    for word, value in (("II", "1"), ("IM", "1"), ("MI", "-1")):
        values[WORDS.index(word)] = value
    return values


def test_exact_independent_operator_certificate():
    assert certify(relation())["accepted"]
    wrong = relation()
    wrong[WORDS.index("MI")] = "1"
    assert not certify(wrong)["accepted"]


def test_scalar_alias_and_spanned_relation_cannot_inflate_novelty():
    row = relation()
    scaled = [str(7*Q(v)) for v in row]
    assert canonical(row) == canonical(scaled)
    assert rational_rank([row, scaled]) == 1
    with pytest.raises(ValueError):
        canonical([0]*len(WORDS))


def test_degree_and_grammar_bounds_are_enforced():
    with pytest.raises(ValueError):
        exact("I", [0, 0, 0, 0, 1])
    with pytest.raises(ValueError):
        exact("IIII", [1])
    with pytest.raises(ValueError):
        exact("eval", [1])
    assert exact("II", [2])[2] == 1
