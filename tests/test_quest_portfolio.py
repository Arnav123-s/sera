import copy

import numpy as np
import pytest
import torch

from experiments.quest_portfolio.assessor import Issuer, lower_bound
from experiments.quest_portfolio.board import Board
from experiments.quest_portfolio.common import digest
from experiments.quest_portfolio.methods import (
    SPECS,
    canonical,
    cases,
    deploy,
    distinct,
    execute,
    registry,
    screen,
)
from experiments.quest_portfolio.runtime import QuestSession
from sera.session_state import model_identity


def test_method_aliases_preserve_derivations_and_assumptions():
    assert len(registry()) == 30
    assert len(distinct(registry())) == 6
    assert canonical(SPECS[2]) != canonical(SPECS[3])
    profiles, wrong = screen()
    np.testing.assert_array_equal(profiles[:4], np.eye(4))
    assert wrong[4] and wrong[5]
    assert not profiles[4:].any()


def test_actual_guarded_deployment_and_unmodeled_abstention():
    rows, truth = cases(101, 32)
    for row, expected in zip(rows, truth, strict=True):
        value, _ = deploy([0, 1, 2, 3], row)
        assert value == pytest.approx(expected)
    assert all(deploy([0, 1, 2, 3], c) == (None, None) for c in cases(102, 8, omitted=True)[0])
    with pytest.raises(ValueError):
        deploy([-1], rows[0])
    for bad in (dict(rows[0], units="imperial"), dict(rows[0], m="0"), dict(rows[0], v0="NaN")):
        with pytest.raises(ValueError):
            deploy([0, 1, 2, 3], bad)


def test_cosmetic_points_scope_and_alternative_prerequisites():
    board = Board()
    key = board.open("one", "Compare mechanics routes")
    assert board.open("renamed", "Compare these methods") == key
    board.append("points", {"amount": 1_000_000})
    assert board.quests[key]["state"] == "OPEN"
    assert not board.eligible("one", "weights", "time")
    routes = board.quests[key]["contract"]["prerequisite_routes"]
    assert ["impulse", "constant_mass"] in routes
    assert ["work", "constant_mass", "final_velocity_sign"] in routes
    with pytest.raises(ValueError):
        board.open("one", "Changed scope", ["impulse"])


def checked_practice(issuer, board, key, method, sequence):
    decision = {"owner": "owner", "quest": key, "action": method, "original_goal": board.quests[key]["original_goal"], "sequence": sequence}
    request = {"mode": "practice", "quest": key, "candidate": canonical(SPECS[method]), "owner": "owner", "decision": digest(decision)}
    challenge = issuer.begin(request)
    receipt = issuer.grade({"id": challenge["id"], "owner": "owner", "candidate": request["candidate"],
                            "predictions": [execute(SPECS[method], c) for c in challenge["cases"]]})
    event = {"quest": key, "method": method, "receipt": receipt, "reward": board.reward(key, method, receipt),
             "decision": decision, "decision_id": digest(decision), "updated": False}
    board.append("practice", event)
    return event


def test_global_high_water_aliases_repeats_and_restore(tmp_path):
    board, issuer = Board(), Issuer(tmp_path)
    key = board.open("original", "Find routes")
    first = checked_practice(issuer, board, key, 0, 0)
    second = checked_practice(issuer, board, key, 0, 1)
    assert first["reward"] == .25
    assert second["reward"] == -.03
    board.open("alias", "Another name")
    assert board.quests[key]["remaining_steps"] == 3
    other = board.open("subset", "Solve impulse only", ["impulse"])
    third = checked_practice(issuer, board, other, 0, 0)
    assert third["reward"] == -.03
    assert Board.restore(board.snapshot()).snapshot() == board.snapshot()
    with pytest.raises(ValueError):
        board.reward(key, 0, first["receipt"])
    forged = board.snapshot()
    forged["view"][key]["state"] = "QUALIFIED"
    with pytest.raises(ValueError):
        Board.restore(forged)


def test_independent_assessor_fresh_attempts_and_global_spending(tmp_path):
    issuer = Issuer(tmp_path)
    retention = {"owner": "frozen", "protected_equal": True, "language_max_error": 0., "probes": 128, "old_definition_gate": False}
    request = {"mode": "boss", "quest": "q1", "candidate": "portfolio", "owner": "frozen", "retention": retention, "decision": "attempt1"}
    first = issuer.begin(request)
    receipt = issuer.grade({"id": first["id"], "owner": "frozen", "candidate": "portfolio",
                            "predictions": [deploy([0, 1, 2, 3], c)[0] for c in first["cases"]]})
    assert receipt["qualified"] and receipt["correct"] == 256
    assert receipt["alpha"] == .0125
    assert issuer.recover(request)["receipt"] == receipt
    issuer.verify([receipt])
    with pytest.raises(KeyError):
        issuer.grade({"id": first["id"]})
    forged = copy.deepcopy(receipt)
    forged["owner"] = "changed"
    with pytest.raises(ValueError):
        issuer.verify([forged])
    second = issuer.begin(dict(request, decision="attempt2"))
    third = issuer.begin(dict(request, quest="q2", decision="attempt3"))
    assert second["inputs_sha256"] != first["inputs_sha256"]
    assert issuer.state["pending"][second["id"]]["alpha"] == pytest.approx(.05/12)
    assert issuer.state["pending"][third["id"]]["alpha"] == pytest.approx(.05/12)
    assert sum(.05/(j*(j+1)*t*(t+1)) for j in range(1, 100) for t in range(1, 100)) < .05
    assert lower_bound(256, 256, .0125) > .95
    assert lower_bound(250, 256, .0125) < .95


@pytest.fixture(scope="module")
def owner_session(tmp_path_factory):
    torch.set_num_threads(1)
    return QuestSession(authority=tmp_path_factory.mktemp("quest-issuer"))


def test_real_owner_retention_projection_and_persistence(owner_session):
    session = owner_session
    assert session.owner is session.base.owner is session.base.grounded.base.study.owner
    session.board.open("test", "Find multiple checked ways to solve mechanics")
    result = session.practice("test", controller="balanced")
    assert result["receipt"]["wrong"] == 0
    before = session.snapshot()
    restored = QuestSession(before)
    assert restored.snapshot() == before
    retention = restored.retention()
    assert retention["protected_equal"] and retention["language_max_error"] == 0
    assert not retention["old_definition_gate"]


def test_owner_staleness_and_received_credit_integrity(owner_session):
    session = owner_session
    saved = session.snapshot()
    event = next(e for e in saved["board"]["events"] if e["kind"] == "practice")
    event["data"]["reward"] += 100
    with pytest.raises(ValueError):
        QuestSession(saved)
    before = model_identity(session.owner)
    session.base._mutation(lambda: session.owner.quest_policy.load_state_dict(session.owner.quest_policy.state_dict()))
    assert model_identity(session.owner) == before


def test_reserved_exploration_cannot_be_spent_by_stop():
    board = Board()
    key = board.open("stops", "Test bounded exploration")
    for step in range(4):
        board.append("stop", {"quest": key, "step": step, "owner": "owner"})
    assert board.quests[key]["remaining_steps"] == 1
    with pytest.raises(ValueError):
        board.append("stop", {"quest": key, "step": 4, "owner": "owner"})


def test_interruption_resumes_same_decision_random_state_and_credit(owner_session, monkeypatch):
    from experiments.quest_portfolio import runtime
    session = owner_session
    original = runtime.call_assessor

    def interrupt(path, request):
        if request["action"] == "grade":
            raise RuntimeError("simulated boundary")
        return original(path, request)

    with monkeypatch.context() as patch:
        patch.setattr(runtime, "call_assessor", interrupt)
        with pytest.raises(RuntimeError, match="simulated boundary"):
            session.practice("test", controller="learned")
    saved = session.snapshot()
    assert saved["pending"] is not None
    restored = QuestSession(saved)
    assert restored.practice("test") == session.practice("test")
    assert restored.snapshot() == session.snapshot()


def test_qualification_stales_and_a_counterexample_revokes(tmp_path):
    board, issuer = Board(), Issuer(tmp_path)
    key = board.open("q", "Multiple checked mechanics routes")
    for method in range(4):
        checked_practice(issuer, board, key, method, method)
    retention = {"owner": "owner", "protected_equal": True, "language_max_error": 0., "probes": 128, "old_definition_gate": False}
    request = {"mode": "boss", "quest": key, "candidate": digest([0, 1, 2, 3]), "owner": "owner", "retention": retention, "decision": "first"}
    challenge = issuer.begin(request)
    receipt = issuer.grade({"id": challenge["id"], "owner": "owner", "candidate": request["candidate"],
                            "predictions": [deploy([0, 1, 2, 3], c)[0] for c in challenge["cases"]]})
    board.append("assessment", {"quest": key, "receipt": receipt})
    assert board.eligible("q", "owner", "time")
    assert not board.eligible("q", "owner", "arbitrary_language")
    board.append("owner", {"identity": "new"})
    assert board.quests[key]["state"] == "STALE"
    fresh = issuer.begin(dict(request, owner="new", retention=dict(retention, owner="new"), decision="second"))
    failed = issuer.grade({"id": fresh["id"], "owner": "new", "candidate": request["candidate"], "predictions": [999.] * 256})
    board.append("assessment", {"quest": key, "receipt": failed})
    assert board.quests[key]["state"] == "NEEDS_WORK"
    assert not board.eligible("q", "new", "time")
