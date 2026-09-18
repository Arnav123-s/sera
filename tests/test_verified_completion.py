import copy

import pytest
import torch

from experiments.verified_completion.common import digest, model_hash
from experiments.verified_completion.credit import Credit
from experiments.verified_completion.model import Investigator, JointCompletion
from experiments.verified_completion.runtime import CompletionSession
from experiments.verified_completion.task import PracticeSource
from sera.session_state import model_identity


@pytest.fixture(scope="module")
def session():
    torch.set_num_threads(1)
    return CompletionSession(load_selected=False)


def test_actual_owner_branch_consistency_and_fact_isolation(session):
    source = PracticeSource(32410)
    before = copy.deepcopy(session.grounded.base.subjects)
    result = session.begin("unit-gap", "Learn which check helps this motion prediction", source)
    assert session.owner is session.grounded.owner is session.grounded.base.study.owner
    assert result["status"] == "COMPLETION_CANDIDATE"
    assert not result["autonomous_definition_admission"]
    for branch in result["branches"]:
        from fractions import Fraction
        a = Fraction(branch["acceleration"])
        assert Fraction(branch["consequences"][1]["velocity"]) == 2*a
        assert Fraction(branch["consequences"][1]["position"]) == 2*a
    assert session.grounded.base.subjects == before
    assert session.grounded.gate["autonomous"] is False


def test_credit_changes_policy_and_keeps_joint_evidence(session):
    source = PracticeSource(32410)
    decision = session.decide("unit-gap")
    session.acquire("unit-gap", source)
    cloud_before = model_hash(session.owner.completion_clouds["unit-gap"])
    policy_before = model_hash(session.owner.completion_policy)
    result = session.verify_and_credit("unit-gap", source)
    assert result["receipt"]["decision"] == decision["identity"]
    assert model_hash(session.owner.completion_clouds["unit-gap"]) == cloud_before
    if result["receipt"]["reward"] != 0:
        assert model_hash(session.owner.completion_policy) != policy_before
    with pytest.raises(ValueError):
        session.verify_and_credit("unit-gap", source)


def test_goal_substitution_and_repeated_source_credit(session):
    source = PracticeSource(32410)
    with pytest.raises(ValueError, match="original goal"):
        session.begin("unit-gap", "A substituted question", source)
    session.begin("renamed-source", "An alias of an already checked world", source)
    session.decide("renamed-source")
    session.acquire("renamed-source", source)
    before = model_identity(session.owner)
    with pytest.raises(ValueError, match="Duplicate"):
        session.verify_and_credit("renamed-source", source)
    assert model_identity(session.owner) == before


def test_stale_policy_and_predictor_cannot_receive_credit(session):
    source = PracticeSource(32412)
    session.begin("stale-gap", "Check stale predictor rejection", source)
    session.decide("stale-gap")
    session.acquire("stale-gap", source)
    cloud = session.owner.completion_clouds["stale-gap"]
    previous = cloud.mean.detach().clone()
    with torch.no_grad():
        cloud.mean[0].add_(.01)
    with pytest.raises(ValueError, match="Stale predictor"):
        session.verify_and_credit("stale-gap", source)
    with torch.no_grad():
        cloud.mean.copy_(previous)


def test_imagination_cannot_be_relabelled_as_an_external_source(session):
    class Forged(PracticeSource):
        pass
    with pytest.raises(ValueError, match="source boundary"):
        session.begin("forged", "Forged factual evidence", Forged(32410))


def test_frozen_predictions_and_live_goal_are_bound_to_credit(session):
    source = PracticeSource(32412)
    pending = session.goals["stale-gap"]["pending"]
    before = pending["before"][0]
    pending["before"][0] += 1
    try:
        with pytest.raises(ValueError, match="prediction before"):
            session.verify_and_credit("stale-gap", source)
    finally:
        pending["before"][0] = before
    goal = session.goals["stale-gap"]
    question = goal["contract"]["question"]
    goal["contract"]["question"] = "Another goal"
    try:
        with pytest.raises(ValueError, match="Original completion goal"):
            session.verify_and_credit("stale-gap", source)
    finally:
        goal["contract"]["question"] = question


def test_exact_resume_between_decision_observation_and_reward():
    original = CompletionSession(load_selected=False)
    source = PracticeSource(32413)
    original.begin("resume-gap", "Resume the same original goal", source)
    original.decide("resume-gap")
    restored = CompletionSession(original.snapshot())
    assert restored.acquire("resume-gap", source) == original.acquire("resume-gap", source)
    restored = CompletionSession(restored.snapshot())
    assert restored.verify_and_credit("resume-gap", source) == original.verify_and_credit("resume-gap", source)
    assert restored.snapshot() == original.snapshot()


@pytest.mark.parametrize("edit", ["unit", "question", "origin", "reward"])
def test_saved_contract_corruption_is_rejected(session, edit):
    saved = session.snapshot()
    goal = saved["goals"]["unit-gap"]
    if edit == "question":
        goal["contract"]["question"] = "different"
    elif edit == "reward":
        saved["credit"]["receipts"][0]["reward"] += 1
    else:
        goal["contract"]["public"][edit] = "OBSERVED" if edit == "origin" else "kg"
        goal["goal_hash"] = digest(goal["contract"])
    with pytest.raises(ValueError):
        CompletionSession(saved)


def test_joint_covariance_and_credit_scoring_reject_invalid_values():
    with pytest.raises(ValueError):
        JointCompletion([0, 0, 0], [[1, 2, 0], [2, 1, 0], [0, 0, 1]])
    with pytest.raises(ValueError):
        Credit.reward([0, 1], [0, 0], 0, 0)
    policy = Investigator()
    controller = Credit(policy)
    assert Credit(policy, controller.snapshot()).snapshot() == controller.snapshot()
