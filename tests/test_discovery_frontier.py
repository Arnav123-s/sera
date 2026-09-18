from fractions import Fraction as Q
from types import SimpleNamespace

import pytest
import torch

from experiments import discovery_frontier as frontier
from experiments.discovery_frontier import compose, credit, equation_record, fit
from experiments.self_chosen import equations as eq


def test_known_force_route_receives_no_new_discovery_points():
    q = {"domain": "motion_0", "target": "a0", "missing": ["v", "x"]}
    p = equation_record(q, [(('f', 1), ('m', -1))], [Q(1)], "test")
    receipt = eq.certify(q["domain"], p)
    assert receipt["accepted"]
    assert credit(q, p, receipt, [])["total"] == 0


def test_two_missing_observables_leave_a_genuine_fit_problem():
    q = {"domain": "motion_0", "target": "a0", "missing": ["v", "f"]}
    p = fit(q, eq.reference_rows("motion_0", 42900, 48), "sparse")
    assert p["status"] == "CONJECTURE"
    assert not set(p["requires"]) & {"v", "f", "a0"}
    assert eq.certify("motion_0", p)["accepted"]


def test_composition_uses_owned_rule_parameters_and_records_dependencies():
    q = {"domain": "motion_0", "target": "v", "missing": ["a0"]}
    outer = equation_record(q, [(('v0', 1),), (('a0', 1), ('t', 1))], [Q(1), Q(1)], "parent")
    inner = equation_record({**q, "target": "a0"}, [(('f', 1), ('m', -1))], [Q(1)], "parent")
    records = [{"id": "outer", "question": q, "proposal": outer},
               {"id": "inner", "question": {**q, "target": "a0"}, "proposal": inner}]
    owner = SimpleNamespace(self_discoveries={"outer": torch.tensor([1., 1.]), "inner": torch.tensor([1.])})
    result = compose(q, records, owner)
    assert result["status"] == "CONJECTURE" and result["dependencies"] == ["outer", "inner"]
    assert eq.certify("motion_0", result)["accepted"]


def test_unrepresentable_composition_remains_open():
    result = equation_record({"target": "v"}, [(('a0', 1),)], [Q(1, 1009)], "compose")
    assert result["status"] == "OPEN"


def test_learned_process_action_can_defer_external_assessment(monkeypatch):
    session = frontier.FrontierSession.__new__(frontier.FrontierSession)
    question = {"domain": "en-US", "target": "alarm_set", "missing": None, "id": "12345678"}
    session.choose = lambda arm: (question, torch.zeros(15))
    session.events, session.visits, session.records = [], {}, []
    session.public, session.goal = {"en-US": []}, "preserved original goal"
    session.discovery_parent = {"owner": "fixed-parent"}
    session.owner = SimpleNamespace(self_question_policy=torch.nn.Linear(15, 1))
    monkeypatch.setattr(frontier, "sha", lambda path: "fixed-source")
    monkeypatch.setattr(frontier.teaching, "action", lambda owner, status: "diversify")
    def forbidden(*args, **kwargs):
        raise AssertionError("Deferred action accessed external evidence")
    monkeypatch.setattr(frontier, "assess", forbidden)
    committed = []
    with pytest.raises(ValueError, match="deferred"):
        session.step("learned", committed.append)
    assert len(committed) == 1 and not session.records and not session.events
