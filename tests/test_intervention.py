"""Independent contracts for measurement semantics and temporal obligations."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from experiments.gap_inquiry import digest
from experiments.intervention_assess import independent_fit, predict, same_qualification
from experiments.intervention_events import MeasurementEvent, measurement_branches
from experiments.intervention_model import (
    InterventionSession,
    checked_observation,
    features,
    program,
)
from experiments.intervention_source import Source, bank
from scripts.intervention_study import GOAL, ORIGINAL, investigate
from sera.r2 import ControlledInstrument


def fixture_session():
    session = InterventionSession.__new__(InterventionSession)
    session.base = SimpleNamespace()
    session.parent_owner = "test-fixture-only"
    session.owner = nn.Module()
    session.owner.intervention_models = nn.ModuleDict()
    session.owner.export_config = lambda: {"type": "fixture"}
    session.subjects = {}
    return session


def passive():
    session = fixture_session()
    source = Source("mixed", 47001, "subject")
    session.start("subject", GOAL, source.identity, ORIGINAL)
    for control in bank("initial"):
        d = session.commit_probe("subject", control)
        session.observe("subject", source.observe(d))
    session.fit("subject")
    return session, source


def completed(tmp_path, family="pulse_loss", policy="information"):
    session = fixture_session()
    source = Source(family, 47001, "subject")
    investigate(session, "subject", source, policy, tmp_path / "progress.json")
    return session, source


def test_no_measurement_and_unread_measurement_differ():
    model = ControlledInstrument()
    eye = torch.eye(4, dtype=torch.complex64)
    effects = torch.stack([torch.outer(v, v.conj()) for v in eye])
    operators = (None, effects)
    plus = torch.tensor([2**-.5, 2**-.5, 0, 0], dtype=torch.complex64)
    rho = torch.outer(plus, plus.conj())[None]
    probabilities = {}
    for occurrence in ("omitted", "performed", "unknown"):
        event = MeasurementEvent("test", "Z measurement", 1., occurrence, None, "fixture", "HYPOTHETICAL")
        branches = measurement_branches(model, rho, event, operators)
        probabilities[occurrence] = {k: float((plus.conj()@v[0]@plus).real) for k, v in branches.items()}
        assert event.record()["outcome_available"] is False
    assert probabilities["omitted"]["omitted"] == pytest.approx(1.)
    assert probabilities["performed"]["performed_unread"] == pytest.approx(.5)
    assert len(probabilities["unknown"]) == 2
    legacy, _ = model.observe(rho, torch.tensor([-1]), operators)
    assert torch.equal(legacy, measurement_branches(model, rho, MeasurementEvent("test", "Z", 1., "performed", None, "fixture", "HYPOTHETICAL"), operators)["performed_unread"])


@pytest.mark.parametrize("occurrence,outcome", [("omitted", 0), ("unknown", 1), ("performed", -1), ("unavailable", None)])
def test_invalid_measurement_meanings_rejected(occurrence, outcome):
    with pytest.raises(ValueError):
        MeasurementEvent("s", "Z", 0., occurrence, outcome, "source", "MEASURED_OBSERVATION")


def test_control_features_and_independent_probability():
    assert features({"ticks": 8, "pulses": []}) == [2., 2., 0.]
    assert features({"ticks": 8, "pulses": [4]}) == [2., 0., 1.]
    assert predict([{"ticks": 8, "pulses": [4]}], [0., .6])[0] == 1.
    with pytest.raises(ValueError):
        program({"ticks": 8, "pulses": [4, 4]})


def test_sealed_final_controls_are_unique_and_unseen():
    final = [digest(p) for p in bank("final")]
    known = {digest(p) for name in ("initial", "candidates", "selection", "adequacy") for p in bank(name)}
    assert len(final) >= 60 and len(final) == len(set(final))
    assert not known.intersection(final)


def test_assessment_cannot_be_reused_after_new_learning(tmp_path):
    session, source = completed(tmp_path)
    old = copy.deepcopy(session.subjects["subject"]["qualification"])
    d = session.commit_probe("subject", {"ticks": 14, "pulses": [7]})
    session.observe("subject", source.observe(d))
    session.fit("subject")
    with pytest.raises(ValueError, match="Fresh independent"):
        session.qualify("subject", old["selection"], old["audit"])
    assert session.subjects["subject"]["history"][-1] == old


def test_invalid_start_is_transactional():
    session = fixture_session()
    before = session.state()
    with pytest.raises(ValueError):
        session.start("subject", GOAL, "source", {"ticks": 0, "pulses": []})
    assert session.state() == before


def test_passive_data_keeps_alias_fiber_and_original_goal():
    session, _ = passive()
    answer = session.answer("subject")
    assert answer["status"] == "INVESTIGATION_REQUIRED"
    assert answer["original_goal"] == GOAL
    assert answer["alternatives"]["type"] == "PASSIVE_EQUIVALENCE_CLASS"
    assert np.ptp(answer["alternatives"]["conditional_probabilities"]) > .1
    assert answer["model_confidence"]["local_standard_error"] is None


def test_pending_action_exact_resume_and_stale_rejection():
    session, source = passive()
    d = session.commit_probe("subject", ORIGINAL)
    saved = session.subject_state("subject")
    resumed = fixture_session()
    resumed.restore_subject("subject", saved)
    assert resumed.subject_state("subject") == saved
    for s in (session, resumed):
        s.observe("subject", source.observe(d))
        s.fit("subject")
    assert session.subject_state("subject") == resumed.subject_state("subject")
    with pytest.raises(ValueError, match="committed"):
        resumed.observe("subject", source.observe(d))
    d = resumed.commit_probe("subject", ORIGINAL)
    resumed.model("subject", "temporal").weight.data[0] += .1
    with pytest.raises(ValueError, match="Stale"):
        resumed.observe("subject", source.observe(d))


@pytest.mark.parametrize("change", ["imagined", "source", "monitor", "actual"])
def test_invalid_evidence_does_not_change_state(change):
    session, source = passive()
    d = session.commit_probe("subject", ORIGINAL)
    row = source.observe(d)
    if change == "imagined":
        row["kind"] = "HYPOTHETICAL"
    elif change == "source":
        row["source"] = "other"
    elif change == "monitor":
        row["monitor"]["applied_pulse_shots"] = 0
    else:
        row["groups"][0]["applied"]["pulses"] = [3]
    before = session.state()
    with pytest.raises(ValueError):
        session.observe("subject", row)
    assert session.state() == before


def test_pulse_failure_is_an_observed_control_fact():
    session = fixture_session()
    source = Source("unreliable_pulses", 47001, "subject")
    session.start("subject", GOAL, source.identity, ORIGINAL)
    row = source.observe(session.commit_probe("subject", ORIGINAL))
    assert 0 < row["monitor"]["applied_pulse_shots"] < row["monitor"]["requested_pulse_shots"]
    assert any(not g["applied"]["pulses"] for g in row["groups"])
    assert checked_observation(row, "subject", source.identity, "acquisition") == row


def test_full_cycle_and_independent_fit(tmp_path):
    session, _ = completed(tmp_path)
    r = session.subjects["subject"]
    q = r["qualification"]
    assert q["accepted"] and q["selected"] == "pulse_loss"
    assert session.answer("subject")["goal_id"] == r["initial"]["goal_id"]
    w, _ = independent_fit(r["observations"], q["selected"])
    actual = session.model("subject", q["selected"]).weight.detach().numpy()
    np.testing.assert_allclose(predict(bank("selection"), w), predict(bank("selection"), actual), atol=.002)
    before = session.state()
    session.answer("subject", {"ticks": 12, "pulses": [6]})
    assert session.state() == before
    assert session.answer("subject", {"ticks": 24, "pulses": [12]})["status"] == "INVESTIGATION_REQUIRED"
    with pytest.raises(ValueError, match="Repeated"):
        session.qualify("subject", q["selection"], q["audit"])
    session.model("subject", q["selected"]).weight.data[0] += .1
    with pytest.raises(ValueError, match="stale"):
        session.answer("subject")


def test_omitted_mechanism_stays_open(tmp_path):
    session, _ = completed(tmp_path, "omitted")
    q = session.subjects["subject"]["qualification"]
    assert not q["accepted"]
    assert session.answer("subject")["status"] == "INVESTIGATION_REQUIRED"


def test_restoration_checks_qualification_and_goal(tmp_path):
    session, _ = completed(tmp_path)
    saved = session.subject_state("subject")
    resumed = fixture_session()
    resumed.restore_subject("subject", saved)
    assert resumed.answer("subject") == session.answer("subject")
    for change in ("goal", "qualification", "evidence"):
        value = copy.deepcopy(saved)
        if change == "goal":
            value["record"]["goal"] += " changed"
        elif change == "qualification":
            value["record"]["qualification"]["accepted"] = False
        else:
            value["record"]["observations"][0]["groups"][0]["plus"] += 1
        with pytest.raises(ValueError):
            fixture_session().restore_subject("subject", value)


def test_portable_receipts_keep_acceptance_exact(tmp_path):
    session, _ = completed(tmp_path)
    a = session.subjects["subject"]["qualification"]
    b = copy.deepcopy(a)
    b["adequacy"]["diagnostic_tail_probability"] += 1e-12
    b["receipt"] = digest({k: v for k, v in b.items() if k != "receipt"})
    assert same_qualification(a, b)
    b["accepted"] = not b["accepted"]
    b["receipt"] = digest({k: v for k, v in b.items() if k != "receipt"})
    assert not same_qualification(a, b)
