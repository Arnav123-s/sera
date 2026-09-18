from fractions import Fraction as Q
from types import SimpleNamespace

import pytest
import torch

from experiments.self_chosen import equations, teaching
from experiments.self_chosen.runtime import reward_for


def proposal(target="v0", terms=None, coefficients=None):
    terms = terms or [[["v", 1]], [["a0", 1], ["t", 1]]]
    coefficients = coefficients or ["1", "-1"]
    return {"status": "CONJECTURE", "target": target, "terms": terms, "coefficients": coefficients,
            "requires": sorted({n for t in terms for n, _ in t}), "nonzero": [],
            "canonical": equations.canonical(target, terms, coefficients), "execution_cost": 5}


def test_three_examples_transfer_to_new_process_contexts():
    owner = SimpleNamespace(discovery_action=torch.nn.Linear(3, 3, dtype=torch.float64))
    torch.nn.init.zeros_(owner.discovery_action.weight)
    torch.nn.init.zeros_(owner.discovery_action.bias)
    result = teaching.teach(owner)
    assert len(result["examples"]) == 3 and result["transfer_correct"] == 12
    assert teaching.action(owner, "counterexample") == "revise"


def test_independent_proof_rejects_wrong_sign_and_fresh_execution_agrees():
    good = proposal()
    bad = proposal(coefficients=["1", "1"])
    assert equations.certify("motion_0", good)["accepted"]
    assert not equations.certify("motion_0", bad)["accepted"]
    assert equations.check_execution(good, equations.reference_rows("motion_0", 41900))["accepted"]
    assert not equations.check_execution(bad, equations.reference_rows("motion_0", 41900))["accepted"]


def test_alias_and_dominated_route_receive_no_repeat_credit():
    q = {"domain": "motion_0", "target": "v0"}
    p = proposal()
    receipt = equations.certify("motion_0", p)
    first = reward_for(q, p, receipt, [])
    record = {"question": q, "proposal": p, "receipt": receipt, "relation": first["relation"], "route": first["route"]}
    assert first["total"] > 0
    assert reward_for(q, p, receipt, [record])["total"] == 0
    alias = dict(p, method="renamed", execution_cost=6, requires=p["requires"]+["m"])
    assert reward_for(q, alias, receipt, [record])["total"] == 0


def test_native_forward_formula_is_preserved_knowledge_not_new_discovery():
    p = proposal("v", [[["v0", 1]], [["a0", 1], ["t", 1]]], ["1", "1"])
    q = {"domain": "motion_0", "target": "v"}
    assert reward_for(q, p, equations.certify("motion_0", p), [])["total"] == 0


def test_laurent_factor_canonicalization_removes_aliases():
    first = equations.canonical("v0", [[["v", 1]], [["a0", 1], ["t", 1]]], ["1", "-1"])
    reordered = equations.canonical("v0", [[["a0", 1], ["t", 1]], [["v", 1]]], ["-1", "1"])
    assert first == reordered


def test_division_scope_and_nonfinite_questions():
    p = proposal("a0", [[["v", 1], ["t", -1]], [["v0", 1], ["t", -1]]], ["1", "-1"])
    p["nonzero"] = ["t"]
    result = equations.check_execution(p, [{"v": Q(2), "v0": Q(1), "a0": Q(1), "t": Q(1)},
                                          {"v": Q(1), "v0": Q(1), "a0": Q(1), "t": Q(0)}])
    assert result["checked"] == result["skipped_outside_scope"] == 1
    assert result["accepted"]
    with pytest.raises(ValueError):
        equations.canonical("x", [[["x", 1]]], ["1"])


def test_proposals_fit_observations_without_reference_semantics(monkeypatch):
    q = {"domain": "motion_0", "target": "v0", "missing": "x"}
    rows = equations.reference_rows("motion_0", 41922, 48)
    def forbidden(*args):
        raise AssertionError("Proposal accessed assessor semantics")
    monkeypatch.setattr(equations, "symbolic_world", forbidden)
    result = equations.propose(q, rows, "sparse")
    assert result["status"] == "CONJECTURE"
    assert equations.check_execution(result, equations.reference_rows("motion_0", 41923))["accepted"]
