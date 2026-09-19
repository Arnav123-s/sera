"""Credit binding and compiled-weight use independent of large optional fixtures."""

import copy

import pytest

from experiments.counterfactual_owner import InquirySession
from experiments.gap_inquiry import ROOT, digest, read


def bare_session():
    session = InquirySession.__new__(InquirySession)
    session.records, session.credits, session.rules = {}, {}, {}
    session.pending = None
    return session


def receipt_result():
    result = {"original_goal": "goal", "predictor": "predictor", "observations": [{"source": "source"}], "events": []}
    result["id"] = digest(result)
    replay = {"accepted": True, "investigation": result["id"], "predictor": "predictor", "original_goal": "goal", "source": "source"}
    return result, replay


@pytest.mark.parametrize("field,replacement", [("predictor", "other"), ("original_goal", "other"), ("source", "other")])
def test_independent_credit_cannot_move_to_another_goal_predictor_or_source(field, replacement):
    session = bare_session()
    result, replay = receipt_result()
    replay[field] = replacement
    with pytest.raises(ValueError):
        session.retain(result, [], replay, {}, {})


def test_changed_progress_record_cannot_receive_credit():
    session = bare_session()
    result, replay = receipt_result()
    result["observations"][0]["value"] = 9
    with pytest.raises(ValueError, match="commitment"):
        session.retain(result, [], replay, {}, {})


def test_pending_predictor_is_checked_before_learning():
    session = bare_session()
    session.pending = {"predictor": "new-owner"}
    result, replay = receipt_result()
    with pytest.raises(ValueError, match="predictor changed"):
        session.retain(result, [], replay, {}, {})


def test_repeated_investigation_receives_no_fresh_credit():
    session = bare_session()
    result, replay = receipt_result()
    session.records[result["original_goal"]] = {"points": 0}
    before = copy.deepcopy(session.records)
    with pytest.raises(ValueError, match="Repeated"):
        session.retain(result, [], replay, {}, {})
    assert session.records == before and not session.credits


def test_published_owner_qualification_and_all_completed_questions():
    out = ROOT / "research-continuation/45_counterfactual_inquiry"
    if not (out / "audit.json").exists():
        pytest.skip("Counterfactual owner qualification not yet published")
    audit = read(out / "audit.json")
    assert audit["passed"] and audit["exact_owner_restore"]
    assert audit["retained_parent_tensors"] == 609 and not audit["changed_parent_tensors"]
    assert audit["questions"] == 228
    assert audit["earlier_task_results_retained"] == 11
    assert audit["exact_investigation_replays"] == 228
    assert audit["background_transfer_checks"] > 0
    assert audit["shared_language_owner"] and not audit["source_gate"]["autonomous"]
