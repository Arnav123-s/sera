from fractions import Fraction as Q

import pytest

from experiments.gap_inquiry import digest
from experiments.solution_check import certify, validate
from experiments.solution_search import canonical, evaluate, implicit


def question(target="m"):
    return {"id": digest(["test", target]), "domain": "motion_0", "target": target,
            "missing": ["x", "v"], "question": "Recover the unavailable value"}


def candidate():
    return canonical(question(), {"terms": [[["f", 1]]], "weights": ["2"]},
                     {"terms": [[["a0", 1]]], "weights": ["2"]}, "test", 2)


def test_denominator_guard_and_scale_alias():
    c = candidate()
    alias = canonical(question(), {"terms": [[["f", 1]]], "weights": ["1"]},
                      {"terms": [[["a0", 1]]], "weights": ["1"]}, "other", 2)
    assert c["id"] == alias["id"]
    assert evaluate(c, {"f": Q(12), "a0": Q(3)}) == 4
    assert evaluate(c, {"f": Q(12), "a0": Q(0)}) is None


def test_proof_is_bound_and_forged_credit_rejected():
    q, c = question(), candidate()
    receipt = certify(q, c, "commitment", "owner")
    assert receipt["accepted"] and receipt["residual"] == "0"
    validate(receipt, q, c, "commitment", "owner")
    with pytest.raises(ValueError, match="Changed"):
        validate(receipt | {"predictor": "other"}, q, c, "commitment", "owner")
    with pytest.raises(ValueError, match="Changed"):
        validate(receipt, q, c, "different", "owner")


def test_wrong_answer_has_nonzero_proof_residual():
    q = question()
    c = canonical(q, {"terms": [[["f", 1]]], "weights": ["2"]},
                  {"terms": [[["a0", 1]]], "weights": ["1"]}, "wrong", 2)
    assert not certify(q, c, "frozen", "owner")["accepted"]


def test_declared_dependencies_and_candidate_hash_cannot_be_forged():
    c = candidate()
    with pytest.raises(ValueError, match="identity or dependencies"):
        certify(question(), c | {"requires": []}, "committed", "owner")
    changed = {**c, "numerator": {"terms": [[["x", 1]]], "weights": ["1"]}}
    with pytest.raises(ValueError, match="identity or dependencies"):
        certify(question(), changed, "committed", "owner")


def test_no_target_or_missing_variable_in_answer():
    for name in ("m", "x"):
        with pytest.raises(ValueError, match="unavailable"):
            canonical(question(), {"terms": [[[name, 1]]], "weights": ["1"]},
                      {"terms": [[]], "weights": ["1"]}, "leak", 1)


def test_implicit_search_uses_only_rows(monkeypatch):
    from experiments.self_chosen import equations as eq
    def forbidden(*args):
        raise AssertionError("Proposer accessed reference world")
    monkeypatch.setattr(eq, "symbolic_world", forbidden)
    q = {"id": digest("generic-test"), "domain": "polynomials", "target": "p",
         "missing": ["c1", "c2", "i", "s"], "question": "Infer a generic rational relationship"}
    rows = [{"t": Q(t), "c0": Q(c), "p": Q(c, t+1)} for t in range(1, 9) for c in range(-3, 4)]
    result = implicit(q, rows, rows, 2, "simple")
    assert result["proposals"]
    assert any(evaluate(c, {"t": Q(20), "c0": Q(63)}) == 3 for c in result["proposals"])
