"""Safety, differentiation and evidence tests independent of large owner fixtures."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from experiments.gap_inquiry import digest
from experiments.structural_check import Source, independent_fit, prediction, same_qualification
from experiments.structural_field import FieldSession, PhysicalWeights
from experiments.structural_inquiry import investigate


def fixture_session():
    # A small lifecycle fixture only. Actual-owner qualification is a separate replay.
    session = FieldSession.__new__(FieldSession)
    session.parent_owner = "fixture"
    session.owner = nn.Module()
    session.owner.physical_fields = nn.ModuleDict()
    session.owner.field_integrals = (.5, .25)
    session.owner.export_config = lambda: {"type": "explicit-test-fixture"}
    session.base = SimpleNamespace()
    session.subjects, session.events, session.pending = {}, [], None
    return session


def test_retention_keeps_source_identity_outside_regenerated_query_receipts():
    from scripts.structural_study import substance
    first = {"owner": "earlier", "result": {"source_sha256": "original-source", "value": 3}}
    changed = {"owner": "successor", "result": {"source_sha256": "changed-source", "value": 3}}
    assert substance(first) != substance(changed)
    assert substance(first) == substance(first | {"owner": "successor"})


def run_fixture(family="directional", sink=None):
    session = fixture_session()
    source = Source(family, 46001, "subject")
    session.start("subject", "Predict changed direction", source.identity)
    investigate(session, "subject", source.bank("initial", 12, narrow=True), source.bank("candidates", 128),
                source.bank("selection", 16), source.bank("adequacy", 32), source.observe, "disagreement", sink or (lambda _: None))
    return session, source


def test_invalid_original_task_does_not_install_untracked_weights():
    session = fixture_session()
    before = session.state()
    with pytest.raises(ValueError, match="bounded finite"):
        session.start("bad-input", "Predict a changed state", "attributed-source", [float("nan")]*6)
    assert session.state() == before


def test_acquisition_changes_weights_and_agrees_with_independent_fit():
    session, source = run_fixture()
    rows = session.subjects["subject"]["observations"]
    for kind in ("radial", "directional"):
        model = session.model("subject", kind)
        independent = independent_fit(rows, kind)
        np.testing.assert_allclose(model.weight.detach(), independent, atol=1e-9)
    q = session.subjects["subject"]["qualification"]
    assert q["selected"] == "directional" and q["accepted"]
    point = source.bank("final", 1)[0]
    before = prediction(point, independent_fit(rows[:12], "radial"), "radial")
    after = session.view("subject").imagine(point)["acceleration"]
    assert np.linalg.norm(before-after) > .001


def test_imagination_and_planning_preserve_evidence_and_weights():
    session, _ = run_fixture()
    state = copy.deepcopy(session.state())
    session.view("subject").imagine([.2, .5, -.1, .1, 0., 0.])
    session.view("subject").plan([.2, .5, -.1, .1], [.3, .2])
    assert session.state() == state


@pytest.mark.parametrize("mutation", ["weight", "replace", "observation"])
def test_stale_views_are_rejected(mutation):
    session, source = run_fixture()
    view = session.view("subject")
    if mutation == "weight":
        with torch.no_grad():
            view.model.weight[0] += .01
    elif mutation == "replace":
        session.owner.physical_fields[session.key("subject", view.kind)] = copy.deepcopy(view.model)
    else:
        row = source.observe([.4, .5, .2, -.1, 0., 0.], "new", "acquisition")
        session.observe("subject", row)
    with pytest.raises(ValueError, match="Stale"):
        view.imagine([.2, .3, 0., 0., 0., 0.])


def test_imagined_labels_duplicate_evidence_and_repeated_qualification_are_rejected():
    session, source = run_fixture()
    record = session.subjects["subject"]
    before = copy.deepcopy(session.state())
    with pytest.raises(ValueError, match="Duplicate"):
        session.observe("subject", record["observations"][0])
    fake = source.observe([0.]*6, "fake", "acquisition")
    fake["kind"] = "IMAGINED"
    with pytest.raises(ValueError, match="not an observation"):
        session.observe("subject", fake)
    q = record["qualification"]
    with pytest.raises(ValueError, match="Repeated"):
        session.qualify("subject", q["selection"], q["audit"])
    assert session.state() == before


def test_assessment_overlap_cannot_qualify_a_model():
    session, _ = run_fixture()
    record = session.subjects["subject"]
    q = record["qualification"]
    audit = copy.deepcopy(q["audit"])
    audit[0]["id"] = record["observations"][0]["id"]
    with pytest.raises(ValueError, match="overlaps|Fresh independent"):
        session.qualify("subject", q["selection"], audit)


def test_new_view_cannot_reuse_a_receipt_after_unversioned_weight_mutation():
    session, _ = run_fixture()
    model = session.model("subject", "directional")
    model.weight.data[0] += .1
    with pytest.raises(ValueError, match="Stale qualification"):
        session.view("subject")


def test_numerical_replay_preserves_exact_evidence_and_acceptance():
    session, _ = run_fixture()
    original = session.subjects["subject"]["qualification"]
    replay = copy.deepcopy(original)
    replay["rmse"] += 1e-12
    replay["id"] = digest({k: v for k, v in replay.items() if k != "id"})
    assert same_qualification(original, replay)
    for field, value in (("accepted", False), ("source", "different-source"), ("rmse", .08)):
        changed = copy.deepcopy(replay)
        changed[field] = value
        changed["id"] = digest({k: v for k, v in changed.items() if k != "id"})
        assert not same_qualification(original, changed)
    replay["rmse"] += 1e-12
    assert not same_qualification(original, replay)  # Changed bytes need their own valid receipt.


def test_omitted_mechanism_reopens_the_original_goal():
    session, _ = run_fixture("omitted")
    q = session.subjects["subject"]["qualification"]
    assert not q["accepted"] and "missing mechanism" in q["next_action"]
    answer = session.view("subject").imagine([.2, .3, 0., 0., 0., 0.])
    assert answer["status"] == "INVESTIGATION_REQUIRED" and answer["goal"] == "Predict changed direction"


def test_state_gradient_and_continuous_energy_balance():
    model = PhysicalWeights("directional")
    model.weight.copy_(torch.tensor([1., 2., .2, .1, .1, .3], dtype=torch.float64))
    state = torch.tensor([.3, -.2, .5, .1], dtype=torch.float64, requires_grad=True)
    u = torch.tensor([.1, -.1], dtype=torch.float64)
    derivative = torch.cat((state[2:], model(torch.cat((state, u)))))
    gradient = torch.autograd.grad(model.energy(state), state, create_graph=True)[0]
    energy_rate = gradient @ derivative
    expected = -.1 * state[2:].square().sum() + state[2:] @ u
    torch.testing.assert_close(energy_rate, expected, atol=1e-12, rtol=1e-12)
    jac = torch.autograd.functional.jacobian(lambda z: model(torch.cat((z, u))), state).detach().numpy()
    h = 1e-6
    independent = np.stack([(prediction(np.r_[state.detach().numpy()+np.eye(4)[i]*h, u], model.weight, model.kind)-
                             prediction(np.r_[state.detach().numpy()-np.eye(4)[i]*h, u], model.weight, model.kind))/(2*h) for i in range(4)], -1)
    np.testing.assert_allclose(jac, independent, atol=1e-8)


def test_interruption_keeps_the_committed_query_before_the_observation(tmp_path):
    import json

    from scripts.structural_study import save_checkpoint
    saved = []
    def sink(value):
        saved.append(copy.deepcopy(value))
        save_checkpoint(tmp_path / "pending.json", value)
        if value["pending"] and value["pending"]["decision"] is not None:
            raise InterruptedError("After decision commit")
    with pytest.raises(InterruptedError):
        run_fixture(sink=sink)
    pending = saved[-1]["pending"]
    assert len(saved[-1]["subjects"]["subject"]["observations"]) == 12
    decision = pending["decision"]
    assert json.loads((tmp_path / "pending.json").read_text()) == saved[-1]
    assert decision["id"] == digest({k: v for k, v in decision.items() if k != "id"})
    assert not pending["used"] and not pending["trace"]
